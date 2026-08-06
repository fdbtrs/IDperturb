import math
import os
import random
import sys
from typing import Any

import hydra
import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from diffusion.ddpm import DenoisingDiffusionProbabilisticModel
from hydra.utils import instantiate
from models.autoencoder.vqgan import VQDecoderInterface, VQEncoderInterface
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.lite import LightningLite
from torchvision.utils import save_image
from utils.helpers import denormalize_to_zero_to_one, ensure_path_join
import torch.nn.functional as F

sys.path.insert(1, "../")


class DiffusionSamplerLite(LightningLite):
    def compute_lb_eq6_single(self,V_norm, all_vectors, base_lb=0.0, eps=1e-8):
        """
        Exact implementation of Eq (6) for a single vector

        Args:
            V_norm: (D,) or (1, D) normalized vector
            all_vectors: (N, D) normalized vectors
            base_lb: scalar lower bound
        Returns:
            lb: scalar
        """

        if V_norm.dim() == 2:
            V_norm = V_norm.squeeze(0)  # (D,)

        # --- cosine similarity (N,) ---
        sim = torch.matmul(all_vectors, V_norm)

        # --- numerical stability ---
        sim = torch.clamp(sim, -1.0 + eps, 1.0 - eps)

        # --- half-angle cosine ---
        cos_half = torch.sqrt((1.0 + sim) * 0.5)

        # --- exclude self if needed ---
        # (only if V_norm is inside all_vectors)
        idx = torch.argmax(sim)
        cos_half[idx] = -1.0

        # --- max over j ---
        max_half_cos = cos_half.max()

        # --- Eq (6) ---
        lb = max(base_lb, max_half_cos.item())

        return lb
    def smooth_batch_vectors(self, V_norm, min_cos=0.8, max_cos=1.0, batch_size=16,nosie_injection=False,margin=0.01):
        """
        V: (n, 512) batch of vectors
        t: scalar or tensor in [0,1] controlling progress
        Returns smoothed vectors with target cosine similarity.
        """
        s=torch.rand(batch_size)

        if (nosie_injection):
            noise_t = torch.randn_like(V_norm)
            V_norm+=noise_t
            return torch.nn.functional.normalize(V_norm)
        cos_target = min_cos + s * (max_cos - min_cos)  # scalar or (n,)

        # If scalar t, expand to (n,)
        if cos_target.ndim == 0:
            cos_target = cos_target.expand(V_norm.size(0))

        # 3. Compute corresponding sin(θ)
        sin_target = torch.sqrt(1.0 - cos_target**2 + 1e-8)  # add epsilon for safety

        # 4. Sample random noise
        noise = torch.randn_like(V_norm)

        # 5. Remove projection on V (orthogonalize)
        proj = (noise * V_norm).sum(dim=1, keepdim=True) * V_norm
        noise_orth = noise - proj

        # 6. Normalize noise
        noise_orth_norm = F.normalize(noise_orth, dim=1)

        # 7. Combine
        cos_target = cos_target.unsqueeze(1)  # (n, 1)
        sin_target = sin_target.unsqueeze(1)  # (n, 1)

        V_smooth = cos_target * V_norm + sin_target * noise_orth_norm
        return V_smooth
    def run(self, cfg) -> Any:

        # load diffusion cfg
        train_cfg = omegaconf.OmegaConf.load(cfg.diffusion_cfg_path)

        # do not set seed to get different samples from each device
        self.seed_everything(cfg.sampling.seed * (1 + self.global_rank))

        # instantiate stuff from restoration config
        diffusion_model = instantiate(train_cfg)

        checkpoint_path = cfg.checkpoint.path

        # loading
        weights = torch.load(checkpoint_path, map_location="cpu")
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace("eps_model.", "") if "eps_model" in k else k
            weights_dict[new_k] = v
        weights_dict_2 = {}
        for k, v in weights_dict.items():
            new_k = k.replace("module.", "") if "module." in k else k
            weights_dict_2[new_k] = v
        diffusion_model.load_state_dict(weights_dict_2, strict=False)
        diffusion_model = DenoisingDiffusionProbabilisticModel(
            eps_model=diffusion_model
        )

        # registrate model in lite
        diffusion_model = self.setup(diffusion_model)

        # sample size
        size = (3, 128, 128)
        # create VQGAN encoder and decoder for training in its latent space
        latent_encoder = VQEncoderInterface(
            first_stage_config_path=os.path.join(
                "/workspace/IDPerturb", "models", "autoencoder", "first_stage_config.yaml"
            ),
            encoder_state_dict_path=os.path.join(cfg.VQEncoder_path),
        )

        size = latent_encoder(torch.ones([1, *size])).shape[-3:]
        latent_encoder = self.setup(latent_encoder)
        latent_encoder.eval()
        latent_decoder = VQDecoderInterface(
            first_stage_config_path=os.path.join(
                "/workspace/IDPerturb", "models", "autoencoder", "first_stage_config.yaml"
            ),
            decoder_state_dict_path=os.path.join(cfg.VQDecoder_path),
        )
        latent_decoder = self.setup(latent_decoder)
        latent_decoder.eval()

        # load identity contexts
        if cfg.sampling.contexts_file != "":
            if "random_synthetic_extracted_15000.npy" in cfg.sampling.contexts_file:
                contexts = torch.load(cfg.sampling.contexts_file)
                contexts = list(contexts.values())
                contexts = contexts[: cfg.sampling.n_contexts]
                contexts =torch.tensor(contexts).float()
            else:
                contexts = np.load(cfg.sampling.contexts_file)
                contexts = contexts[: cfg.sampling.n_contexts]
                contexts = torch.from_numpy(contexts).float()


        else:
            contexts = np.random.randn(cfg.sampling.n_contexts, 512)
            contexts_norm = np.linalg.norm(contexts, axis=1)
            contexts = contexts / contexts_norm[:, np.newaxis]
        


       
        print(f"contexts.shape: {contexts.shape}")

        context_ids = list(i for i in range(0, contexts.shape[0]))

        model_name = cfg.checkpoint.path.split("/")[-1]
        if cfg.checkpoint.use_non_ema:
            model_name += "_non_ema"
        elif cfg.checkpoint.global_step is not None:
            model_name += f"_{cfg.checkpoint.global_step}"

        samples_dir = cfg.sampling.save_dir

        if not os.path.exists(samples_dir):
            os.mkdir(samples_dir)

        context_ids = self.split_across_devices(context_ids)

        if self.global_rank == 0:
            with open(ensure_path_join(f"{samples_dir}.yaml"), "w+") as f:
                OmegaConf.save(config=cfg, f=f.name)

        np.random.seed(1337)

        for id_index in range(0, len(context_ids)):

            prefix = str(context_ids[id_index])
            context=contexts[context_ids[id_index]]
            if (cfg.sampling.safe_min_cos):
                min_cos=self.compute_lb_eq6_single(context,contexts,base_lb=cfg.sampling.min_cos)
            else:
                min_cos=cfg.sampling.min_cos

            if (cfg.sampling.context_smoothing):
                context = context.repeat(cfg.sampling.batch_size, 1)
                context=self.smooth_batch_vectors(context,min_cos=min_cos,max_cos=cfg.sampling.max_cos, batch_size=cfg.sampling.batch_size, nosie_injection=cfg.sampling.nosie_injection).cuda()
            else:
                context = context.repeat(cfg.sampling.batch_size, 1).cuda()
            while not isinstance(diffusion_model, DenoisingDiffusionProbabilisticModel):
                diffusion_model = diffusion_model.module

            n_samples = cfg.sampling.n_samples_per_context
            start_batch = None
            self.perform_sampling(
                diffusion_model=diffusion_model,
                n_samples=n_samples,
                size=size,
                batch_size=cfg.sampling.batch_size,
                samples_dir=samples_dir,
                prefix=prefix,
                context=context,
                latent_encoder=latent_encoder,
                latent_decoder=latent_decoder,
                start_batch=start_batch,
                sample_config=cfg.sampling.sample_config,
                save_mode=cfg.sampling.save_mode,
            )

    @staticmethod
    def perform_sampling(
        diffusion_model,
        n_samples,
        size,
        batch_size,
        samples_dir,
        prefix: str = None,
        context: torch.Tensor = None,
        latent_encoder: torch.nn.Module = None,
        latent_decoder: torch.nn.Module = None,
        start_batch: torch.Tensor = None,
        sample_config=None,
        save_mode=None
    ):

        n_batches = math.ceil(n_samples / batch_size)

        samples_for_grid = []

        if context is not None:
            assert prefix is not None

        for _ in range(n_batches):
            batch_samples = (
                diffusion_model.sample_ddim(
                    batch_size,
                    size,
                    x_T=start_batch,
                    context=context,
                    cfg_scale=sample_config.cfg_scale))

            save_path = os.path.join(samples_dir, prefix)
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            with torch.no_grad():
                if latent_decoder:
                    batch_samples = latent_decoder(batch_samples).cpu()
            batch_samples = denormalize_to_zero_to_one(batch_samples)
            samples_for_grid.append(batch_samples)
            samples = torch.cat(samples_for_grid, dim=0)[:n_samples]

            samples = F.interpolate(samples, size=[112, 112], mode="bilinear")

            for sample_index in range(len(samples)):
                save_image(
                    samples[sample_index],
                    ensure_path_join(save_path, str(sample_index) + ".png"),
                )

    def split_across_devices(self, L):
        if isinstance(L, int):
            L = list(range(L))

        chunk_size = math.ceil(len(L) / self.world_size)
        L_per_device = [
            L[idx : idx + chunk_size] for idx in range(0, len(L), chunk_size)
        ]
        while len(L_per_device) < self.world_size:
            L_per_device.append([])

        return L_per_device[self.global_rank]


@hydra.main(
    config_path="./configs", config_name="sample_ddim_config", version_base=None
)
def sample(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    sampler = DiffusionSamplerLite(devices="auto", accelerator="auto")
    sampler.run(cfg)


if __name__ == "__main__":
    sample()
