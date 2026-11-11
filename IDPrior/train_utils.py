import os
import random
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader

from evaluation_utils import evaluate
from helper_functions import *
from models import predict_model_output

def diffusion_forward(
    model, clean, config, noise_scheduler, loss_fn, amp_enabled,
                amp_dtype
):
    # 1. Sample Noise
    noise = torch.randn_like(clean)

    # 2. Sample Timesteps
    timesteps = torch.randint(
        0,
        config.training.timesteps,
        (clean.shape[0],),
        device=config["device"],
        dtype=torch.long,
    )

    # 3. Add Noise
    noisy = noise_scheduler.add_noise(clean, noise, timesteps)

    # 4. Predict
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
        noise_pred = predict_model_output(
            model, noisy, timesteps
        )

    # 5. Diffusion Loss (Prediction vs Noise)
    loss = loss_fn(noise_pred.float(), noise.float())

    return loss


def train(
    model,
    noise_scheduler,
    eval_noise_scheduler,
    optimizer,
    dataset,
    eval_dataset,
    scheduler,
    config,
    **kwargs,
):
    local_rank = dist.get_rank()

    # Mixed precision configuration
    precision = getattr(config.training, "mixed_precision", "no")

    amp_enabled = precision in ["fp16", "bf16"]

    amp_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "no": torch.float32,
    }.get(precision)

    if amp_dtype is None:
        raise ValueError(f"Unsupported mixed precision: {precision}")

    # GradScaler is needed for FP16, but not BF16 or FP32
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))

    best_fid = float("inf")
    patience_counter = 0
    stop_training = torch.tensor(
        0, device=config["device"]
    )  # tensor to synchronize parallel GPU early stopping
    loss_fn = nn.MSELoss()

    sampler = DistributedSampler(
        dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
    )
    dataloader = DataLoader(
        dataset, batch_size=config.training.batch_size, sampler=sampler
    )

    for epoch in range(config.training.num_epoch):
        sampler.set_epoch(epoch)

        total_epochs = config.training.num_epoch
        p10 = total_epochs * 0.1
        p90 = total_epochs * 0.9

        is_in_outer_percentiles = epoch < p10 or epoch >= p90

        is_save_epoch = (
            epoch == total_epochs - 1
            or (is_in_outer_percentiles and epoch % 5 == 0)
            or (not is_in_outer_percentiles and epoch % 10 == 0)
        )

        total_epoch_loss = 0

        # model stays at training after eval so we need to switch it to train at every loop
        model.train()

        for i, batch in tqdm(enumerate(dataloader), desc=f"Training epoch: {epoch}"):
            clean = batch
            clean = clean.to(config["device"])

            loss = diffusion_forward(
                model,
                clean,
                config,
                noise_scheduler,
                loss_fn,
                amp_enabled,
                amp_dtype,
            )

            # Standard optimization step
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_epoch_loss += loss.item()

        if scheduler:
            scheduler.step()

        avg_epoch_loss = total_epoch_loss / len(dataloader)
        print(f"RANK[{local_rank}] Epoch {epoch}: average loss {avg_epoch_loss}")

        _, fid, _, (_, _), _, (_, _) = evaluate(
            model, eval_noise_scheduler, eval_dataset, config, **kwargs
        )

        # RANK0 does the model save and early stop
        dist.barrier()
        if local_rank == 0:
            if is_save_epoch:
                torch.save(
                    model.state_dict(),
                    os.path.join(config.paths.output_path, f"model_{epoch}.pt"),
                )
            if fid < best_fid:
                best_fid = fid
                patience_counter = 0
                stop_training.fill_(0)
            else:
                patience_counter += 1
                if patience_counter >= config.training.patience:
                    print("Early stopping.")
                    stop_training.fill_(1)

        # send stop flag to all GPUs
        dist.broadcast(stop_training, src=0)
        # and wait here so every GPU has the most recent stop flag
        dist.barrier()
        # for log file readability
        if local_rank == 0:
            print()
        # check for stopping
        if stop_training.item() == 1:
            break
