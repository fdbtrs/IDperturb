# libraries needed to be seeded/configured at the start
import random
import numpy as np
import torch
import os
import torch.distributed as dist

if not dist.is_initialized():
    dist.init_process_group("nccl")
local_rank = dist.get_rank()
BASE_SEED = 1337
kwargs = {"BASE_SEED": BASE_SEED}
seed_value = BASE_SEED + local_rank
torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

# set a unique seed to every GPU
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(seed_value)

import sys
import shutil
import importlib
from diffusers import DDPMScheduler, DDIMScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import *
from models import *
from train_utils import train
from evaluation_utils import evaluate
from utils.merge_outputs import *

# CONFIG
if len(sys.argv) < 2:
    raise ValueError("Please provide the config name as an arguman")

config_name = sys.argv[1]
if config_name.endswith(".py"):
    config_name = config_name[:-3]

config = importlib.import_module(f"configs.{config_name}").config
config["device"] = torch.device(f"cuda:{local_rank}")
if local_rank == 0:
    print(config)

if local_rank == 0:
    os.makedirs(config.paths.output_path, exist_ok=True)
    shutil.copy(
        os.path.join("configs", f"{config_name}.py"),
        os.path.join(config.paths.output_path, f"{config_name}.py"),
    )

# DATASET
dataset = import_dataset(config)

# DATASET SPLIT
if config.training.run_training:
    train_size = int(0.90 * len(dataset))
    eval_size = len(dataset) - train_size

    # Same split on every DDP process
    generator = torch.Generator().manual_seed(BASE_SEED)

    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size], generator=generator
    )

    if local_rank == 0:
        print(f"Training samples: {len(train_dataset)}")
        print(f"Evaluation samples: {len(eval_dataset)}")

else:
    train_dataset = None
    eval_dataset = dataset


# DIFF MODEL
model = import_model(config)
load_model(model, config, local_rank)
model.to(config["device"])
model = DDP(model, device_ids=[local_rank])

if local_rank == 0:
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

# SCHEDULERS
train_noise_scheduler = DDPMScheduler(
    num_train_timesteps=config.training.timesteps,
    beta_start=0.0005,
    beta_end=0.01,
    beta_schedule="scaled_linear",
)
train_noise_scheduler.set_timesteps(config.training.timesteps, device=config["device"])
if config.eval.noise_scheduler == "DDIM":
    eval_noise_scheduler = DDIMScheduler(
        num_train_timesteps=config.training.timesteps,
        beta_start=0.0005,
        beta_end=0.01,
        beta_schedule="scaled_linear",
        clip_sample=False,
    )
    eval_noise_scheduler.set_timesteps(
        num_inference_steps=config.eval.timesteps, device=config["device"]
    )
else:
    eval_noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.training.timesteps,
        beta_start=0.0005,
        beta_end=0.01,
        beta_schedule="scaled_linear",
    )
    eval_noise_scheduler.set_timesteps(
        config.training.timesteps, device=config["device"]
    )

config.real_embeddings = []

# OPTIMIZER AND LR SCHEDULER
optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)

if config.scheduler.type == "step":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.scheduler.step_size, gamma=config.scheduler.gamma
    )
elif config.scheduler.type == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.scheduler.T_max
    )
else:
    scheduler = None

# START TRAINING OR EVALUATION
dist.barrier()
if config.training.run_training:
    train(
        model,
        train_noise_scheduler,
        eval_noise_scheduler,
        optimizer,
        train_dataset,
        eval_dataset,
        scheduler,
        config,
        **kwargs,
    )
else:
    embeddings, _, _, (_, _), _, (_, _) = evaluate(
        model, eval_noise_scheduler, eval_dataset, config, **kwargs
    )
    np.save(
        os.path.join(
            config.paths.output_path,
            f"generated_embeddings_epoch0_rank{local_rank}.npy",
        ),
        embeddings,
    )

dist.barrier()
if local_rank == 0:
    merge_epoch_rank_embeddings(config.paths.output_path)
