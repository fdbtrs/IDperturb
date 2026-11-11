from utils.metrics import *
import time
from datetime import timedelta

import os
import csv
import json
import random
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from helper_functions import *

import normflows as nf


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
HIDDEN_DIM = 1024
NUM_FLOW_LAYERS = 12
NUM_FLOW_BLOCKS = 4

# =====================================================================
# Plotting
# =====================================================================


def plot_training_results(out_folder):
    """
    Plot training/validation NLL and generation metrics from:
        training_history.csv
        metrics.jsonl

    Saves:
        training_results.png
    """
    import csv
    import json
    import os
    import matplotlib.pyplot as plt

    history_path = os.path.join(out_folder, "training_history.csv")
    metrics_path = os.path.join(out_folder, "metrics.jsonl")

    # ---------------------------------------------------------
    # Load training history
    # ---------------------------------------------------------
    epochs = []
    train_nll = []
    val_nll = []

    if os.path.exists(history_path):
        with open(history_path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                epochs.append(int(row["epoch"]))
                train_nll.append(float(row["train_nll"]))
                val_nll.append(float(row["val_nll"]))
    else:
        logging.warning("Training history not found: %s", history_path)
        return

    # ---------------------------------------------------------
    # Load generation metrics (JSON Lines)
    # ---------------------------------------------------------
    metrics = []

    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    metrics.append(json.loads(line))

    # ---------------------------------------------------------
    # Create plots
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Training and validation NLL
    axes[0].plot(epochs, train_nll, marker="o", label="Train NLL")
    axes[0].plot(epochs, val_nll, marker="o", label="Validation NLL")
    axes[0].set_title("Negative Log-Likelihood")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("NLL")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: FID and uniqueness
    if metrics:
        metric_epochs = [m["epoch"] for m in metrics if m.get("epoch") is not None]
        fid_values = [m["fid"] for m in metrics if m.get("epoch") is not None]
        uniqueness_values = [
            m["uniqueness"] for m in metrics if m.get("epoch") is not None
        ]

        axes[1].plot(metric_epochs, fid_values, marker="o", label="FID")
        axes[1].plot(metric_epochs, uniqueness_values, marker="s", label="Uniqueness")

        axes[1].set_title("Generation Quality")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].set_title("Generation Quality")
        axes[1].text(
            0.5,
            0.5,
            "No generation metrics available",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )

    # Plot 3: Minimum and maximum cosine similarity
    if metrics:
        similarity_metrics = [m for m in metrics if m.get("epoch") is not None]

        sim_epochs = [m["epoch"] for m in similarity_metrics]
        min_sim = [m["min_similarity"] for m in similarity_metrics]
        max_sim = [m["max_similarity"] for m in similarity_metrics]

        axes[2].plot(sim_epochs, min_sim, marker="o", label="Minimum similarity")
        axes[2].plot(sim_epochs, max_sim, marker="o", label="Maximum similarity")

        axes[2].set_title("Generated Embedding Similarities")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Cosine similarity")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].set_title("Generated Embedding Similarities")
        axes[2].text(
            0.5,
            0.5,
            "No generation metrics available",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )

    fig.suptitle("Normalizing Flow Training Results")
    fig.tight_layout()

    output_path = os.path.join(out_folder, "training_results.png")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logging.info("Training plots saved to %s", output_path)


# =====================================================================
# MODEL
# =====================================================================


class BoundedScaleNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_blocks=4):
        super().__init__()

        self.net = nf.nets.ResidualNet(
            in_features=embedding_dim,
            out_features=embedding_dim,
            hidden_features=hidden_dim,
            num_blocks=num_blocks,
        )

    def forward(self, x):
        return 2.0 * torch.tanh(self.net(x))


def create_flow_model(
    embedding_dim=EMBEDDING_DIM, hidden_dim=512, num_layers=8, num_blocks=4
):
    base_distribution = nf.distributions.base.DiagGaussian(embedding_dim)

    flows = []

    for i in range(num_layers):
        # 1. Invertible linear mixing.
        flows.append(nf.flows.InvertibleAffine(num_channels=embedding_dim, use_lu=True))

        # 2. Alternating binary mask.
        mask = torch.tensor(
            [float((j + i) % 2) for j in range(embedding_dim)], dtype=torch.float32
        )

        # 3. Scale
        scale_net = BoundedScaleNet(
            embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_blocks=num_blocks
        )

        # 4. Translation
        translation_net = nf.nets.ResidualNet(
            in_features=embedding_dim,
            out_features=embedding_dim,
            hidden_features=hidden_dim,
            num_blocks=num_blocks,
        )

        flows.append(nf.flows.MaskedAffineFlow(b=mask, s=scale_net, t=translation_net))

    model = nf.NormalizingFlow(q0=base_distribution, flows=flows)
    total_params, trainable_params = count_parameters(model)

    print(
        f"Model parameters: {total_params:,} total, " f"{trainable_params:,} trainable",
        flush=True,
    )

    return model


class FlowKLDWrapper(nn.Module):
    """Expose normflows' forward_kld as forward so DDP can track gradients."""

    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def forward(self, x):
        return self.flow.forward_kld(x)


# =====================================================================
# CHECKPOINTS
# =====================================================================


def save_checkpoint(
    model, optimizer, epoch, checkpoint_path, config, best_val_nll=None
):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "best_val_nll": best_val_nll,
    }

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path, device):
    """
    Reconstructs the model using the saved configuration.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint["config"]

    model = create_flow_model(
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_flow_layers"],
        num_blocks=config["num_flow_blocks"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logging.info(
        "Loaded checkpoint from epoch %s: %s", checkpoint["epoch"], checkpoint_path
    )

    return model, checkpoint


# =====================================================================
# LOG-LIKELIHOOD
# =====================================================================


def calculate_nll(model, embeddings, device, batch_size=512):
    """
    Calculates mean negative log-likelihood.

    Lower NLL generally indicates better density modeling.
    """
    model.eval()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(embeddings)),
        batch_size=batch_size,
        shuffle=False,
    )

    total_nll = 0.0
    total_samples = 0

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)

            log_prob = model.log_prob(batch)

            total_nll += -log_prob.sum().item()
            total_samples += batch.shape[0]

    return total_nll / total_samples


# =====================================================================
# EVALUATION AND SAMPLE GENERATION
# =====================================================================


def evaluate(
    model,
    real_dataset_embeddings,
    num_samples,
    device,
    out_folder,
    calculate_metrics=True,
    epoch=None,
):
    """
    Generates embeddings and optionally evaluates their quality.

    Args:
        model:
            Trained normalizing flow.

        real_dataset_embeddings:
            Original real embeddings, shape (N, 512).

        data_mean, data_std:
            Preprocessing statistics from training.

        num_samples:
            Number of embeddings to generate.

        device:
            torch.device.

        out_folder:
            Output directory.

        calculate_metrics:
            Whether to compute FID and uniqueness.

        epoch:
            Optional epoch number for output filenames.

    Returns:
        generated_embeddings: NumPy array of shape (num_samples, 512)
    """
    model.eval()

    logging.info("Generating %d embeddings...", num_samples)

    generated_batches = []
    remaining = num_samples

    # Generate in batches to avoid unnecessary GPU memory usage.
    sample_batch_size = 512

    with torch.no_grad():
        while remaining > 0:
            current_batch_size = min(sample_batch_size, remaining)

            # normflows returns generated samples and their log-probabilities.
            samples, _ = model.sample(current_batch_size)

            generated_batches.append(samples.cpu().numpy())

            remaining -= current_batch_size

    generated_embeddings = np.concatenate(generated_batches, axis=0).astype(np.float32)

    # Project generated samples onto the unit sphere.
    generated_embeddings /= np.maximum(
        np.linalg.norm(generated_embeddings, axis=1, keepdims=True), 1e-12
    )

    if not np.isfinite(generated_embeddings).all():
        raise RuntimeError("Generated embeddings contain NaN or Inf values.")

    # only save in evaluation
    if epoch is None:
        suffix = f"epoch_{epoch:04d}" if epoch is not None else "inference"

        os.makedirs(out_folder, exist_ok=True)

        output_path = os.path.join(out_folder, f"generated_embeddings_{suffix}.npy")

        np.save(output_path, generated_embeddings)

        logging.info("Saved generated embeddings to %s", output_path)

        logging.info(
            "Generated shape=%s, mean=%.6f, std=%.6f",
            generated_embeddings.shape,
            generated_embeddings.mean(),
            generated_embeddings.std(),
        )

    if calculate_metrics:
        logging.info("Calculating generation metrics...")

        # These functions should already exist in your project.
        fid_score = calculate_fid(real_dataset_embeddings, generated_embeddings)

        uniqueness_score, min_similarity, max_similarity = calculate_uniqueness(
            generated_embeddings
        )

        metrics = {
            "epoch": epoch,
            "num_samples": num_samples,
            "fid": float(fid_score),
            "uniqueness": float(uniqueness_score),
            "min_similarity": float(min_similarity),
            "max_similarity": float(max_similarity),
        }

        logging.info(
            "Metrics | FID: %.6f | Uniqueness: %.6f | "
            "Min similarity: %.6f | Max similarity: %.6f",
            fid_score,
            uniqueness_score,
            min_similarity,
            max_similarity,
        )

        metrics_path = os.path.join(out_folder, "metrics.jsonl")

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

    return generated_embeddings


# =====================================================================
# TRAINING
# =====================================================================


def train(
    model,
    train_embeddings,
    val_embeddings,
    real_dataset_embeddings,
    optimizer,
    num_epochs,
    batch_size,
    device,
    out_folder,
    num_generated_samples,
    calculate_metrics=True,
    config=None,
    ddp_model=None,
    rank=0,
    world_size=1,
):
    """Maximum-likelihood training; batch_size is per GPU under DDP."""
    dataset = TensorDataset(torch.from_numpy(train_embeddings))
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    best_val_nll = float("inf")
    checkpoint_dir = os.path.join(out_folder, "checkpoints")
    if is_main_process():
        os.makedirs(checkpoint_dir, exist_ok=True)
        history_path = os.path.join(out_folder, "training_history.csv")
        with open(history_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_nll", "val_nll"])
    for epoch in range(1, num_epochs + 1):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        total_loss, total_samples = 0.0, 0
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{num_epochs}",
            disable=not is_main_process(),
        )
        for (batch,) in progress:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            # DDP must see the forward call; calling model.forward_kld directly bypasses DDP.
            loss = (
                ddp_model(batch) if ddp_model is not None else model.forward_kld(batch)
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch}. Check data and learning rate."
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            n = batch.shape[0]
            total_loss += loss.detach().item() * n
            total_samples += n
            progress.set_postfix(nll=f"{loss.item():.4f}")

        stats = torch.tensor(
            [total_loss, total_samples], dtype=torch.float64, device=device
        )
        if world_size > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        train_nll = (stats[0] / stats[1]).item()

        # Rank 0 performs evaluation and file writes; all ranks receive the same val score.
        if is_main_process():
            val_nll_value = calculate_nll(
                model, val_embeddings, device, batch_size=batch_size
            )
            val_tensor = torch.tensor(
                [val_nll_value], dtype=torch.float64, device=device
            )
        else:
            val_tensor = torch.zeros(1, dtype=torch.float64, device=device)
        if world_size > 1:
            dist.broadcast(val_tensor, src=0)
        val_nll = val_tensor.item()

        if is_main_process():
            logging.info(
                "Epoch %d/%d | Train NLL: %.6f | Val NLL: %.6f",
                epoch,
                num_epochs,
                train_nll,
                val_nll,
            )
            with open(
                os.path.join(out_folder, "training_history.csv"),
                "a",
                newline="",
                encoding="utf-8",
            ) as f:
                csv.writer(f).writerow([epoch, train_nll, val_nll])
            if epoch % 10 == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    os.path.join(checkpoint_dir, f"epoch_{epoch}.pt"),
                    config,
                    best_val_nll,
                )
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    os.path.join(checkpoint_dir, "best.pt"),
                    config,
                    best_val_nll,
                )
                logging.info("Saved new best checkpoint (Val NLL: %.6f)", best_val_nll)
            if calculate_metrics:
                evaluate(
                    model,
                    real_dataset_embeddings,
                    num_generated_samples,
                    device,
                    out_folder,
                    True,
                    epoch,
                )
        if world_size > 1:
            # Keep best_val_nll consistent for logging/control flow on all ranks.
            best_tensor = torch.tensor(
                [best_val_nll if is_main_process() else 0.0],
                dtype=torch.float64,
                device=device,
            )
            dist.broadcast(best_tensor, src=0)
            best_val_nll = best_tensor.item()

    if is_main_process():
        logging.info("Training complete. Best validation NLL: %.6f", best_val_nll)


# =====================================================================
# MAIN PIPELINE
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
            raise RuntimeError("Multi-process DDP requires CUDA GPUs.")
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
                f"[Rank {rank}] NCCL test returned {probe.item()}, expected {expected}"
            )
        print(f"[Rank {rank}] NCCL test passed: {probe.item()}", flush=True)

    # Use a rank-specific seed for other rank-local randomness. DDP's default
    # initialization synchronization broadcasts rank 0's model parameters.
    set_seed(SEED + rank)
    os.makedirs(args.out_folder, exist_ok=True)
    logger = (
        setup_logging(args.out_folder) if rank == 0 else logging.getLogger(__name__)
    )

    try:

        if rank == 0:
            logger.info(
                "Device: %s | world_size=%d | per-GPU batch=%d | effective batch=%d",
                device,
                world_size,
                args.batch_size,
                args.batch_size * world_size,
            )

        embeddings = load_embeddings(args.embeddings_file)
        real_dataset_embeddings = embeddings
        if rank == 0:
            logger.info("Dataset: %d embeddings, %d dimensions", *embeddings.shape)

        if args.run_training:
            indices = np.random.RandomState(SEED).permutation(len(embeddings))
            split_idx = int(0.9 * len(indices))
            train_embeddings = embeddings[indices[:split_idx]]
            val_embeddings = embeddings[indices[split_idx:]]

            config = {
                "embedding_dim": EMBEDDING_DIM,
                "hidden_dim": args.hidden_dim,
                "num_flow_layers": args.num_flow_layers,
                "num_flow_blocks": args.num_flow_blocks,
            }

            if rank == 0:
                logger.info("Constructing flow model on %s", device)
            if args.model_ckpt:
                model, _ = load_checkpoint(args.model_ckpt, device)
            else:
                model = create_flow_model(
                    EMBEDDING_DIM,
                    args.hidden_dim,
                    args.num_flow_layers,
                    args.num_flow_blocks,
                ).to(device)
            print(
                f"[Rank {rank}] Model constructed on {next(model.parameters()).device}; "
                f"parameters={sum(p.numel() for p in model.parameters()):,}",
                flush=True,
            )

            if distributed:
                print(f"[Rank {rank}] Before DDP constructor", flush=True)
                start_time = time.time()
                ddp_model = DDP(
                    FlowKLDWrapper(model),
                    device_ids=[local_rank],
                    output_device=local_rank,
                    # Keep the default init_sync=True: it synchronizes initial
                    # parameters/buffers so every rank starts with the same model.
                )
                print(
                    f"[Rank {rank}] DDP initialized in {time.time() - start_time:.2f}s",
                    flush=True,
                )
            else:
                ddp_model = None

            optimizer = optim.Adam(
                model.parameters(), lr=args.learning_rate, weight_decay=1e-6
            )
            if rank == 0:
                logger.info(
                    "Starting training | lr=%g | batch/GPU=%d | epochs=%d",
                    args.learning_rate,
                    args.batch_size,
                    args.num_epochs,
                )

            train(
                model,
                train_embeddings,
                val_embeddings,
                real_dataset_embeddings,
                optimizer,
                args.num_epochs,
                args.batch_size,
                device,
                args.out_folder,
                args.num_generated_samples,
                args.calculate_metrics,
                config,
                ddp_model=ddp_model,
                rank=rank,
                world_size=world_size,
            )
            if rank == 0:
                plot_training_results(args.out_folder)
        else:
            if distributed:
                raise ValueError("Run inference without torchrun (single process).")
            if not args.model_ckpt:
                raise ValueError(
                    "--model_ckpt is required when --run_training is False"
                )
            model, _ = load_checkpoint(args.model_ckpt, device)
            evaluate(
                model,
                real_dataset_embeddings,
                args.num_generated_samples,
                device,
                args.out_folder,
                args.calculate_metrics,
                epoch=None,
            )
    finally:
        if distributed and dist.is_initialized():
            print(f"[Rank {rank}] Destroying process group", flush=True)
            dist.destroy_process_group()


# =====================================================================
# ARGUMENTS
# =====================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or sample a RealNVP normalizing flow "
        "for 512-dimensional embeddings."
    )

    parser.add_argument("--out_folder", type=str, required=True)

    parser.add_argument(
        "--run_training", type=lambda x: x.lower() in ("true", "1", "yes"), default=True
    )

    parser.add_argument("--embeddings_file", type=str, required=True)

    parser.add_argument("--model_ckpt", type=str, default="")

    parser.add_argument("--learning_rate", type=float, default=1e-4)

    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_flow_layers", type=int, default=12)
    parser.add_argument("--num_flow_blocks", type=int, default=2)

    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--num_epochs", type=int, default=100)

    parser.add_argument("--num_generated_samples", type=int, default=10000)

    parser.add_argument(
        "--calculate_metrics",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
