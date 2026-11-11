from easydict import EasyDict as edict

config = edict()

"""
For inference:
1- pass model checkpoint to model_ckpt
2- set run_training to False
3- set num_generated to number of IDs, count_per_id to 1.
4- (optional) set calculate_metrics to False otherwise loading all dataset embeddings may take a while
"""

# Paths
config.paths = edict()
config.paths.output_path = f"./runs/OUTPUTFOLDER"
config.paths.vector_folder = "embeddings.npy"
config.paths.model_ckpt = None

# Training
config.training = edict()
config.training.lr = 1e-4
config.training.batch_size = 512
config.training.num_epoch = 100
config.training.patience = 50  # early stopping patience
config.training.timesteps = 1000  # Diffusion timesteps
config.training.run_training = True  # Toggle training on/off

# LR Scheduler
config.scheduler = edict()
config.scheduler.type = (
    "cosine"  # options: "step" or "cosine", otherwise it will be set to None
)
config.scheduler.step_size = 10  # StepLR
config.scheduler.gamma = 0.5  # StepLR
config.scheduler.T_max = config.training.num_epoch  # CosineAnnealingLR

# Model
config.model = edict()
config.model.embedding_size = (16, 32)  # set to 512 for 1d model
config.model.in_channels = 1
config.model.out_channels = 1
config.model.block_out_channels = [64, 128, 256, 512, 512]
config.model.down_block_types = [
    "ResnetDownsampleBlock2D",
    "ResnetDownsampleBlock2D",
    "ResnetDownsampleBlock2D",
    "AttnDownBlock2D",
    "AttnDownBlock2D",
]
config.model.up_block_types = [
    "AttnUpBlock2D",
    "AttnUpBlock2D",
    "ResnetUpsampleBlock2D",
    "ResnetUpsampleBlock2D",
    "ResnetUpsampleBlock2D",
]
config.model.layers_per_block = 2
config.model.attention_head_dim = 8
config.model.dropout = 0.1
config.model.downsample_type = "resnet"
config.model.upsample_type = "resnet"
config.model.use_deterministic_randomness = False # for easier debug when needed

# Dataset
config.dataset = edict()
config.dataset.name = "EmbeddingFileDataset"

# Evaluation / Generation
config.eval = edict()
config.eval.num_generated = 100
config.eval.noise_scheduler = "DDIM"
config.eval.timesteps = (  # defines the stride for inference. Scaled for DDIM, kept same for DDPM.
    # 50 means we take 50 steps from 1000 to 0th timestep each being 20 strided (like 0, 20, 40...)
    50
    if config.eval.noise_scheduler == "DDIM"
    else config.training.timesteps
)
config.eval.count_per_id = 20 # can used for intra-ID related-things, currently its just an insignificant multiplier on num_generated. For more information check how 'use_deterministic_randomness' is used
config.eval.calculate_metrics = True
