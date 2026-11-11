import argparse
import csv
import json
import logging
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from tqdm import tqdm
from helper_functions import *
from utils.metrics import calculate_fid, calculate_uniqueness


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 42
EMBEDDING_DIM = 512


# =====================================================================
# Plotting
# =====================================================================


def plot_training_results(out_folder):
    # 1. Read CSV training history
    history_path = os.path.join(out_folder, "training_history.csv")
    if not os.path.exists(history_path):
        return
    with open(history_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    epochs_csv = [int(r["epoch"]) for r in rows]

    # 2. Read JSON evaluation metrics (JSON Lines format)
    metrics_path = os.path.join(out_folder, "metrics.jsonl")
    json_rows = []
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    json_rows.append(json.loads(line))

    # 3. Create a 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # --- Top Row: Training & Loss Metrics ---
    # Plot 1: VAE objective
    for key, label in [("train_loss", "Train total"), ("val_loss", "Validation total")]:
        axes[0].plot(epochs_csv, [float(r[key]) for r in rows], marker="o", label=label)
    axes[0].set_title("VAE Objective")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Reconstruction loss
    for key, label in [
        ("train_recon", "Train reconstruction"),
        ("val_recon", "Validation reconstruction"),
    ]:
        axes[1].plot(epochs_csv, [float(r[key]) for r in rows], marker="o", label=label)
    axes[1].set_title("Reconstruction Loss (summed MSE)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: KL divergence
    for key, label in [("train_kl", "Train KL"), ("val_kl", "Validation KL")]:
        axes[2].plot(epochs_csv, [float(r[key]) for r in rows], marker="o", label=label)
    axes[2].set_title("KL Divergence")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # --- Bottom Row: Evaluation Metrics from JSON ---
    if json_rows:
        epochs_json = [int(r["epoch"]) for r in json_rows]

        # Plot 4: FID
        axes[3].plot(
            epochs_json,
            [float(r["fid"]) for r in json_rows],
            marker="o",
            color="purple",
            label="FID",
        )
        axes[3].set_title("Fréchet Inception Distance (FID)")
        axes[3].set_xlabel("Epoch")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        # Plot 5: Uniqueness
        axes[4].plot(
            epochs_json,
            [float(r["uniqueness"]) for r in json_rows],
            marker="o",
            color="orange",
            label="Uniqueness",
        )
        axes[4].set_title("Sample Uniqueness")
        axes[4].set_xlabel("Epoch")
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)

        # Plot 6: Min/Max Similarity
        axes[5].plot(
            epochs_json,
            [float(r["min_similarity"]) for r in json_rows],
            marker="o",
            color="blue",
            label="Min Similarity",
        )
        axes[5].plot(
            epochs_json,
            [float(r["max_similarity"]) for r in json_rows],
            marker="o",
            color="green",
            label="Max Similarity",
        )
        axes[5].set_title("Similarity Range")
        axes[5].set_xlabel("Epoch")
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)
    else:
        # Hide bottom row if the JSON metrics file doesn't exist yet
        for i in range(3, 6):
            axes[i].set_visible(False)

    fig.tight_layout()
    path = os.path.join(out_folder, "training_results.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Training plots saved to %s", path)


# =====================================================================
# MODEL
# =====================================================================


class SwiGLU(nn.Module):
    """SwiGLU activation function to increase capacity and feature representation."""

    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

    def forward(self, x):
        return F.silu(self.w1(x)) * self.w2(x)


class HighCapacityResMLPBlock(nn.Module):
    """High-capacity residual MLP block with 4x expansion and SwiGLU."""

    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)

        self.norm = nn.LayerNorm(dim)
        # SwiGLU doubles the capacity of the inner projection layer
        self.swiglu = SwiGLU(dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.proj = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)

        # Zero-initialization for the final projection of deep residual blocks
        # This prevents gradient explosion during early training of heavy models.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        residual = x
        h = self.norm(x)
        h = self.swiglu(h)
        h = self.dropout1(h)
        h = self.proj(h)
        h = self.dropout2(h)
        return residual + h


class IDPrior_HighCapacity_VAE(nn.Module):
    """Ultra-high capacity VAE for 512-D face identity embeddings."""

    def __init__(
        self,
        input_dim=512,
        latent_dim=256,  # Increased latent capacity as well
        hidden_dim=4096,  # Massively scaled hidden width
        depth_encoder=8,  # Deep layers for hierarchical abstraction
        depth_decoder=8,
        dropout=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # --- High-Capacity Encoder ---
        self.encoder_in = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )

        self.encoder_body = nn.Sequential(
            *[
                HighCapacityResMLPBlock(hidden_dim, dropout=dropout)
                for _ in range(depth_encoder)
            ]
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # --- High-Capacity Decoder ---
        self.decoder_in = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )

        self.decoder_body = nn.Sequential(
            *[
                HighCapacityResMLPBlock(hidden_dim, dropout=dropout)
                for _ in range(depth_decoder)
            ]
        )

        self.decoder_out = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder_in(x)
        h = self.encoder_body(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = self.decoder_in(z)
        h = self.decoder_body(h)
        return F.normalize(self.decoder_out(h), p=2, dim=-1, eps=1e-12)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=-1, eps=1e-12)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z


class VAEForwardWrapper(nn.Module):
    """Small wrapper so DDP can wrap the VAE's forward pass normally."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        return self.vae(x)


# =====================================================================
# CHECKPOINTS
# =====================================================================


def save_checkpoint(model, optimizer, epoch, path, config, best_val_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = IDPrior_HighCapacity_VAE(
        input_dim=config.get("input_dim", EMBEDDING_DIM),
        latent_dim=config.get("latent_dim", 128),
        hidden_dim=config.get("hidden_dim", 2048),
        depth_encoder=config.get("depth_encoder", 8),
        depth_decoder=config.get("depth_decoder", 8),
        dropout=config.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logging.info(
        "Loaded checkpoint from epoch %s: %s", checkpoint.get("epoch", "?"), path
    )
    return model, checkpoint


# =====================================================================
# LOSS AND EVALUATION
# =====================================================================


def vae_loss(x_hat, x, mu, logvar, beta=1e-3):
    """Per-example summed MSE plus beta times standard-normal KL.

    Both components are summed over their feature dimensions and averaged over
    the batch. Since the decoder returns unit vectors, MSE is a natural
    reconstruction objective. beta can be tuned from the command line.
    """
    x = F.normalize(x, p=2, dim=-1, eps=1e-12)
    recon_per_sample = ((x_hat - x) ** 2).sum(dim=-1)
    # Clamp logvar for safe KL arithmetic, consistent with reparameterization.
    safe_logvar = torch.clamp(logvar, min=-30.0, max=20.0)
    kl_per_sample = -0.5 * torch.sum(
        1.0 + safe_logvar - mu.pow(2) - safe_logvar.exp(), dim=-1
    )
    recon = recon_per_sample.mean()
    kl = kl_per_sample.mean()
    total = recon + beta * kl
    return total, recon, kl


@torch.no_grad()
def evaluate_loss(model, embeddings, device, batch_size, beta):
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(embeddings)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    total_loss = total_recon = total_kl = 0.0
    total_n = 0
    for (batch,) in loader:
        batch = batch.to(device, non_blocking=True)
        x_hat, mu, logvar, _ = model(batch)
        loss, recon, kl = vae_loss(x_hat, batch, mu, logvar, beta)
        n = batch.shape[0]
        total_loss += loss.item() * n
        total_recon += recon.item() * n
        total_kl += kl.item() * n
        total_n += n
    return total_loss / total_n, total_recon / total_n, total_kl / total_n


@torch.no_grad()
def generate_embeddings(model, num_samples, device, sample_batch_size=256):
    model.eval()
    batches = []
    remaining = num_samples
    while remaining > 0:
        n = min(sample_batch_size, remaining)
        z = torch.randn(n, model.latent_dim, device=device)
        samples = model.decode(z)
        batches.append(samples.cpu().numpy())
        remaining -= n
    generated = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
    # The decoder already normalizes, but project once more to guarantee output norms.
    generated /= np.maximum(np.linalg.norm(generated, axis=1, keepdims=True), 1e-12)
    if not np.isfinite(generated).all():
        raise RuntimeError("Generated embeddings contain NaN or Inf values.")
    return generated


def evaluate_generation(
    model,
    real_embeddings,
    num_samples,
    device,
    out_folder,
    calculate_metrics=True,
    epoch=None,
):
    logging.info("Generating %d embeddings...", num_samples)
    generated = generate_embeddings(model, num_samples, device)

    # only save at inference
    if epoch is None:
        suffix = f"epoch_{epoch:04d}" if epoch is not None else "inference"
        os.makedirs(out_folder, exist_ok=True)
        output_path = os.path.join(out_folder, f"generated_embeddings_{suffix}.npy")
        np.save(output_path, generated)
        logging.info("Saved generated embeddings to %s", output_path)

    norms = np.linalg.norm(generated, axis=1)
    logging.info(
        "Generated shape=%s | norm mean=%.6f min=%.6f max=%.6f",
        generated.shape,
        float(norms.mean()),
        float(norms.min()),
        float(norms.max()),
    )

    if calculate_metrics:
        logging.info("Calculating generation metrics...")
        fid_score = calculate_fid(real_embeddings, generated)
        uniqueness_score, min_similarity, max_similarity = calculate_uniqueness(
            generated
        )
        record = {
            "epoch": epoch,
            "num_samples": int(num_samples),
            "fid": float(fid_score),
            "uniqueness": float(uniqueness_score),
            "min_similarity": float(min_similarity),
            "max_similarity": float(max_similarity),
        }
        with open(
            os.path.join(out_folder, "metrics.jsonl"), "a", encoding="utf-8"
        ) as f:
            f.write(json.dumps(record) + "\n")
        logging.info(
            "Metrics | FID %.6f | uniqueness %.6f | min cosine %.6f | max cosine %.6f",
            fid_score,
            uniqueness_score,
            min_similarity,
            max_similarity,
        )
    return generated


# =====================================================================
# TRAINING
# =====================================================================


def train(
    model,
    train_embeddings,
    val_embeddings,
    real_embeddings,
    optimizer,
    args,
    device,
    config,
    ddp_model=None,
    rank=0,
    world_size=1,
):
    dataset = TensorDataset(torch.from_numpy(train_embeddings))
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    checkpoint_dir = os.path.join(args.out_folder, "checkpoints")
    history_path = os.path.join(args.out_folder, "training_history.csv")
    best_val_loss = float("inf")
    if is_main_process():
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(history_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_recon",
                    "train_kl",
                    "val_loss",
                    "val_recon",
                    "val_kl",
                ]
            )

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        # Sum batch-averaged metrics weighted by the local batch size.
        running = torch.zeros(4, dtype=torch.float64, device=device)  # loss,recon,kl,n
        progress = tqdm(
            loader, desc=f"Epoch {epoch}/{args.num_epochs}", disable=(rank != 0)
        )
        for (batch,) in progress:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = ddp_model(batch) if ddp_model is not None else model(batch)
            x_hat, mu, logvar, _ = outputs
            loss, recon, kl = vae_loss(x_hat, batch, mu, logvar, args.beta)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite VAE loss at epoch {epoch}: loss={loss.item()}, "
                    f"recon={recon.item()}, KL={kl.item()}"
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient norm at epoch {epoch}.")
            optimizer.step()
            n = batch.shape[0]
            running[0] += loss.detach().double() * n
            running[1] += recon.detach().double() * n
            running[2] += kl.detach().double() * n
            running[3] += n
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                recon=f"{recon.item():.4f}",
                kl=f"{kl.item():.2f}",
            )

        if world_size > 1:
            dist.all_reduce(running, op=dist.ReduceOp.SUM)
        train_loss, train_recon, train_kl = (running[:3] / running[3]).tolist()

        if is_main_process():
            val_loss_value, val_recon_value, val_kl_value = evaluate_loss(
                model, val_embeddings, device, args.eval_batch_size, args.beta
            )
            val_stats = torch.tensor(
                [val_loss_value, val_recon_value, val_kl_value],
                dtype=torch.float64,
                device=device,
            )
        else:
            val_stats = torch.zeros(3, dtype=torch.float64, device=device)
        if world_size > 1:
            dist.broadcast(val_stats, src=0)
        val_loss, val_recon, val_kl = val_stats.tolist()

        if is_main_process():
            logging.info(
                "Epoch %d/%d | train loss %.6f (recon %.6f, KL %.4f) | "
                "val loss %.6f (recon %.6f, KL %.4f)",
                epoch,
                args.num_epochs,
                train_loss,
                train_recon,
                train_kl,
                val_loss,
                val_recon,
                val_kl,
            )
            with open(history_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [
                        epoch,
                        train_loss,
                        train_recon,
                        train_kl,
                        val_loss,
                        val_recon,
                        val_kl,
                    ]
                )
            if epoch % 10 == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    os.path.join(checkpoint_dir, f"epoch_{epoch}.pt"),
                    config,
                    best_val_loss,
                )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    os.path.join(checkpoint_dir, "best.pt"),
                    config,
                    best_val_loss,
                )
                logging.info(
                    "Saved new best checkpoint (validation loss %.6f)", best_val_loss
                )
            if args.calculate_metrics:
                evaluate_generation(
                    model,
                    real_embeddings,
                    args.num_generated_samples,
                    device,
                    args.out_folder,
                    calculate_metrics=True,
                    epoch=epoch,
                )

        if world_size > 1:
            best_tensor = torch.tensor(
                [best_val_loss if rank == 0 else 0.0],
                dtype=torch.float64,
                device=device,
            )
            dist.broadcast(best_tensor, src=0)
            best_val_loss = best_tensor.item()

    if is_main_process():
        logging.info("Training complete. Best validation loss: %.6f", best_val_loss)


# =====================================================================
# MAIN
# =====================================================================


def main(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))

    if torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda:0")
    else:
        if distributed:
            raise RuntimeError("Multi-process DDP training requires CUDA GPUs.")
        device = torch.device("cpu")

    if distributed:
        print(
            f"[Rank {rank}] Initializing process group: device={device}, "
            f"local_rank={local_rank}, world_size={world_size}",
            flush=True,
        )
        dist.init_process_group(
            backend="nccl", init_method="env://", timeout=timedelta(minutes=3)
        )
        print(f"[Rank {rank}] Process group initialized", flush=True)
        probe = torch.tensor([float(rank)], device=device, dtype=torch.float32)
        print(f"[Rank {rank}] Starting NCCL all_reduce test", flush=True)
        dist.all_reduce(probe, op=dist.ReduceOp.SUM)
        expected = float(world_size * (world_size - 1) // 2)
        if not torch.isclose(probe, torch.tensor([expected], device=device)).all():
            raise RuntimeError(
                f"Rank {rank}: NCCL test result {probe.item()}, expected {expected}"
            )
        print(f"[Rank {rank}] NCCL test passed: {probe.item()}", flush=True)

    set_seed(SEED + rank)
    os.makedirs(args.out_folder, exist_ok=True)
    logger = (
        setup_logging(args.out_folder) if rank == 0 else logging.getLogger(__name__)
    )

    try:
        if rank == 0:
            logger.info(
                "Device=%s | world_size=%d | per-GPU batch=%d | effective batch=%d",
                device,
                world_size,
                args.batch_size,
                args.batch_size * world_size,
            )
        embeddings = load_embeddings(args.embeddings_file)
        if rank == 0:
            logger.info("Dataset has %d embeddings of dimension %d", *embeddings.shape)

        if args.run_training:
            indices = np.random.RandomState(SEED).permutation(len(embeddings))
            split_idx = int(0.9 * len(indices))
            train_embeddings = embeddings[indices[:split_idx]]
            val_embeddings = embeddings[indices[split_idx:]]
            config = {
                "input_dim": EMBEDDING_DIM,
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
                "depth_encoder": args.depth_encoder,
                "depth_decoder": args.depth_decoder,
                "dropout": args.dropout,
                "beta": args.beta,
            }

            if rank == 0:
                logger.info(
                    "Building IDPrior_HighCapacity_VAE | latent_dim=%d hidden_dim=%d depth_encoder=%d depth_decoder=%d dropout=%.3f beta=%g",
                    args.latent_dim,
                    args.hidden_dim,
                    args.depth_encoder,
                    args.depth_decoder,
                    args.dropout,
                    args.beta,
                )
            model = IDPrior_HighCapacity_VAE(
                input_dim=EMBEDDING_DIM,
                latent_dim=args.latent_dim,
                hidden_dim=args.hidden_dim,
                depth_encoder=args.depth_encoder,
                depth_decoder=args.depth_decoder,
                dropout=args.dropout,
            ).to(device)
            total, trainable = count_parameters(model)
            print(
                f"[Rank {rank}] Model on {next(model.parameters()).device}; "
                f"parameters={total:,} total, {trainable:,} trainable",
                flush=True,
            )

            if distributed:
                print(f"[Rank {rank}] Before DDP constructor", flush=True)
                start = time.time()
                ddp_model = DDP(
                    VAEForwardWrapper(model),
                    device_ids=[local_rank],
                    output_device=local_rank,
                )
                print(
                    f"[Rank {rank}] DDP initialized in {time.time() - start:.2f}s",
                    flush=True,
                )
            else:
                ddp_model = None

            optimizer = optim.Adam(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            if rank == 0:
                logger.info("Starting VAE training for %d epochs", args.num_epochs)
            train(
                model=model,
                train_embeddings=train_embeddings,
                val_embeddings=val_embeddings,
                real_embeddings=embeddings,
                optimizer=optimizer,
                args=args,
                device=device,
                config=config,
                ddp_model=ddp_model,
                rank=rank,
                world_size=world_size,
            )
            if rank == 0:
                plot_training_results(args.out_folder)
        else:
            if distributed:
                raise ValueError(
                    "Run inference without torchrun (single process only)."
                )
            if not args.model_ckpt:
                raise ValueError("--model_ckpt is required when --run_training False")
            model, checkpoint = load_checkpoint(args.model_ckpt, device)
            evaluate_generation(
                model,
                embeddings,
                args.num_generated_samples,
                device,
                args.out_folder,
                calculate_metrics=args.calculate_metrics,
                epoch=None,
            )
    finally:
        if distributed and dist.is_initialized():
            print(f"[Rank {rank}] Destroying process group", flush=True)
            dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-GPU beta-VAE training for L2-normalized 512-D face embeddings."
    )
    parser.add_argument("--out_folder", type=str, required=True)
    parser.add_argument(
        "--run_training", type=lambda x: x.lower() in ("true", "1", "yes"), default=True
    )
    parser.add_argument("--embeddings_file", type=str, required=True)
    parser.add_argument("--model_ckpt", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--depth_encoder", type=int, default=8)
    parser.add_argument("--depth_decoder", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--beta",
        type=float,
        default=2e-5,
        help="Weight on KL term in total loss = summed MSE + beta * KL.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Per-GPU batch size under DDP."
    )
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_generated_samples", type=int, default=10000)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument(
        "--calculate_metrics",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
