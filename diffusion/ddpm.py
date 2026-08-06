import math

import torch
from tqdm import tqdm
import torch.nn.functional as F

class DenoisingDiffusionProbabilisticModel(torch.nn.Module):
    def __init__(
        self,
        eps_model: torch.nn.Module,
        T: int = 1000,
        criterion: torch.nn.Module = torch.nn.MSELoss(),
        schedule_type: str = "linear",
        schedule_k: float = 1.0,
        schedule_beta_min: float = 0.0,
        schedule_beta_max: float = 0.1,
    ) -> None:

        super(DenoisingDiffusionProbabilisticModel, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It
        # helps device placement.
        betas = compute_beta_schedule(
            T,
            schedule_type,
            k=schedule_k,
            beta_min=schedule_beta_min,
            beta_max=schedule_beta_max,
        )
        for k, v in precompute_schedule_constants(betas).items():
            self.register_buffer(k, v)

        self.T = T
        self.criterion = criterion
        self.schedule_type = schedule_type
        self.schedule_k = schedule_k

        # for ddim sample
        self.alphas_prev = torch.tensor([1.0]).float()
        self.alphas_prev = torch.cat((self.alphas_prev, self.alpha_bars[:-1]), 0)
        # for ddim reverse sample
        self.alphas_next = torch.tensor([0.0]).float()
        self.alphas_next = torch.cat((self.alphas_next, self.alpha_bars[1:]), 0)

    def forward(
        self,
        x0: torch.Tensor,
        context: torch.Tensor = None,
        score: torch.Tensor = None,
        dropout_mask: torch.Tensor = None,
        latent_decoder=None,
        FR_model=None,
        FIQA_model=None,
    ) -> torch.Tensor:
        # t ~ U(0, T)

        t = torch.randint(0, self.T, (x0.shape[0],)).to(x0.device)
        # eps ~ N(0, 1)
        eps = torch.randn_like(x0)

        mean = self.sqrt_alpha_bars[t, None, None, None] * x0
        sd = self.sqrt_one_minus_alpha_bars[t, None, None, None]

        x_t = mean + sd * eps
        noise_pred, _, _ = self.eps_model(x_t, t, context, score, dropout_mask)
        return self.criterion(eps, noise_pred)
    
    def cosine_cads_noise_schedule(self,timestep: int, total_timesteps: int, max_cads_strength: float = 1.0, threshold= 0.5):
        """
        A cosine annealing schedule for CADS noise.
        Provides a smoother transition.
        """
        alpha = timestep / (total_timesteps - 1)
        # Cosine goes from 1 to 0 over 0 to pi/2
        ths=max_cads_strength * (torch.cos(alpha * torch.pi / 2.0))
        capped_val = torch.where(alpha > threshold, torch.tensor(1.0, device=alpha.device, dtype=alpha.dtype), ths)

        return capped_val
    
    
    def cosine_annealing(self,timestep: int, total_timesteps: int):
        alpha = timestep / (total_timesteps - 1) # Normalizes timestep from 0 to 1

        return 0.5 * (1 + torch.cos(torch.pi * alpha))
    
    def smooth_batch_vectors(self, V_norm, s, min_cos=0.95, max_cos=1.0):
        """
        V: (n, 512) batch of vectors
        t: scalar or tensor in [0,1] controlling progress
        Returns smoothed vectors with target cosine similarity.
        """

        # 1. Normalize input vectors
        #V_norm = F.normalize(V, dim=1)  # (n, 512)

        # 2. Compute target cosine similarity in [-1, 1]
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

    def sample_ddim(
        self,
        n_samples,
        size,
        x_T: torch.Tensor = None,
        context: torch.Tensor = None,
        score: torch.Tensor = None,
        dropout_mask: torch.Tensor = None,
        eta=0,
        ddim_step=50,fixe_context=True,cfg_scale=1.0
    ) -> torch.Tensor:
        x_t = x_T if x_T is not None else self.sample_prior(n_samples, size).cuda()

        skip = self.T // ddim_step
        print("DDIM Sampling")
        print("skip: %d" % skip)
        self.eval()
        with torch.no_grad():
            for i in reversed(range(0, self.T, skip)):
                t = torch.tensor(i).repeat(n_samples).cuda()
                #score = score.cuda()
                model_output_cond= self.eps_model(
                    x_t, t, context,  dropout_mask
                )
                model_output_uncond= (
                        self.eps_model(x_t, t, None, dropout_mask))
                model_output = (
                        1 + cfg_scale
                    ) * model_output_cond - cfg_scale * model_output_uncond
    

                prev_timestep = i - skip
                alpha_prod_t = self.alpha_bars[i]
                alpha_prod_t_prev = (
                    self.alphas_prev[prev_timestep]
                    if prev_timestep >= 0
                    else torch.tensor(1.0).cuda()
                )
                beta_prod_t = 1 - alpha_prod_t
                beta_prod_t_prev = 1 - alpha_prod_t_prev
                pred_original_sample = (
                    x_t - beta_prod_t ** (0.5) * model_output
                ) / alpha_prod_t ** (0.5)
                variance = (beta_prod_t_prev / beta_prod_t) * (
                    1 - alpha_prod_t / alpha_prod_t_prev
                )
                std_dev_t = eta * variance ** (0.5)
                model_output = (
                    x_t - alpha_prod_t ** (0.5) * pred_original_sample
                ) / beta_prod_t ** (0.5)

                pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
                    0.5
                ) * model_output

                x_t = (
                    alpha_prod_t_prev ** (0.5) * pred_original_sample
                    + pred_sample_direction
                )

                if eta > 0:
                    device = (
                        model_output.device if torch.is_tensor(model_output) else "cpu"
                    )
                    noise = torch.randn(n_samples, *size).cuda()
                    variance = std_dev_t * noise
                    if not torch.is_tensor(model_output):
                        variance = variance.numpy()
                    x_t = x_t + variance

        self.train()

        return x_t
    def sample_ddim_unconditional(
        self,
        n_samples,
        size,
        x_T: torch.Tensor = None,
        dropout_mask: torch.Tensor = None,
        eta=0,
        ddim_step=50
    ) -> torch.Tensor:
        x_t = x_T if x_T is not None else self.sample_prior(n_samples, size).cuda()

        skip = self.T // ddim_step
        print("DDIM Sampling")
        print("skip: %d" % skip)
        self.eval()
        with torch.no_grad():
            for i in reversed(range(0, self.T, skip)):
                t = torch.tensor(i).repeat(n_samples).cuda()

                model_output= (
                        self.eps_model(x_t, t, None, dropout_mask))

                prev_timestep = i - skip
                alpha_prod_t = self.alpha_bars[i]
                alpha_prod_t_prev = (
                    self.alphas_prev[prev_timestep]
                    if prev_timestep >= 0
                    else torch.tensor(1.0).cuda()
                )
                beta_prod_t = 1 - alpha_prod_t
                beta_prod_t_prev = 1 - alpha_prod_t_prev
                pred_original_sample = (
                    x_t - beta_prod_t ** (0.5) * model_output
                ) / alpha_prod_t ** (0.5)
                variance = (beta_prod_t_prev / beta_prod_t) * (
                    1 - alpha_prod_t / alpha_prod_t_prev
                )
                std_dev_t = eta * variance ** (0.5)
                model_output = (
                    x_t - alpha_prod_t ** (0.5) * pred_original_sample
                ) / beta_prod_t ** (0.5)

                pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
                    0.5
                ) * model_output

                x_t = (
                    alpha_prod_t_prev ** (0.5) * pred_original_sample
                    + pred_sample_direction
                )

                if eta > 0:
                    device = (
                        model_output.device if torch.is_tensor(model_output) else "cpu"
                    )
                    noise = torch.randn(n_samples, *size).cuda()
                    variance = std_dev_t * noise
                    if not torch.is_tensor(model_output):
                        variance = variance.numpy()
                    x_t = x_t + variance

        self.train()

        return x_t

    @staticmethod
    def calculate_rmse(tensor1, tensor2):
        mse = torch.nn.functional.mse_loss(tensor1, tensor2, reduction="none")
        rmse = torch.mean(torch.sqrt(mse), dim=(1, 2))
        return rmse

    @staticmethod
    def get_attn_map(attn_map_list, bs, img_size=32):
        ave_attn_map = []
        attn_map_list = list(filter(lambda x: x is not None, attn_map_list))
        statistic = {}
        statistic["mean"] = []
        statistic["std"] = []
        for idx, attn_map in enumerate(attn_map_list):
            b, lq, lk = attn_map.shape
            h = int(math.sqrt(lq))
            w = int(math.sqrt(lq))

            attn_map = attn_map.reshape(bs, -1, lq, lk)

            statistic["mean"].append(attn_map.mean([1, 2, 3])[0])
            statistic["std"].append(attn_map.std())

            attn_map = torch.softmax(attn_map, dim=-2)
            attn_map = attn_map.mean(dim=-1)  # (bs, n_hean, lq)
            attn_map = attn_map.mean(dim=1)  # (bs, lq)

            attn_map = attn_map / attn_map.max(dim=1, keepdim=True)[0]  # (bs, lq)

            attn_map = attn_map.reshape(bs, 1, h, w)
            attn_map = torch.nn.functional.interpolate(
                attn_map,
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            attn_map = torch.clamp(attn_map, min=0, max=1)

            ave_attn_map.append(attn_map)

        statistic["mean"] = torch.stack(statistic["mean"])
        ave_attn_map = torch.stack(ave_attn_map, dim=1)
        ave_attn_map = ave_attn_map.mean(dim=1)  # (bs, img_size, img_size)
        max_value = torch.max(ave_attn_map.reshape(bs, -1), dim=-1)[0]
        min_value = torch.min(ave_attn_map.reshape(bs, -1), dim=-1)[0]
        ave_attn_map = (ave_attn_map - min_value.unsqueeze(-1).unsqueeze(-1)) / (
            max_value.unsqueeze(-1).unsqueeze(-1)
            - min_value.unsqueeze(-1).unsqueeze(-1)
        )
        return ave_attn_map

   
    @staticmethod
    def sample_prior(n_samples, size):
        return torch.randn(n_samples, *size)


def compute_beta_schedule(
    T: int,
    schedule_type: str = "linear",
    k: float = 1.0,
    beta_min: float = None,
    beta_max: float = None,
) -> torch.Tensor:

    if schedule_type.lower() == "linear":
        scale = 1000 / T
        beta_1 = scale * 0.0001
        beta_T = scale * 0.02
        return torch.linspace(beta_1, beta_T, T, dtype=torch.float32)

    elif schedule_type.lower() in ["cosine", "cosine_warped"]:

        s = 0.008
        beta_min = 0.0 if schedule_type.lower() == "cosine" else beta_min
        k = 1 if schedule_type.lower() == "cosine" else k

        return betas_for_alpha_bar(
            T,
            lambda t: math.cos(math.pi / 2 * (t + s) / (1 + s) ** k) ** 2,
            beta_min=beta_min,
            beta_max=beta_max,
        )

    raise NotImplementedError


def betas_for_alpha_bar(T, alpha_bar, beta_min=0.0, beta_max=1.0):
    betas = []
    for i in range(T):
        t1 = i / T
        t2 = (i + 1) / T
        betas.append(min(max(1 - alpha_bar(t2) / alpha_bar(t1), beta_min), beta_max))
    return torch.tensor(betas).float()


def precompute_schedule_constants(betas: torch.Tensor):
    alphas = 1 - betas
    sqrt_alphas_inv = 1 / alphas.sqrt()

    sigmas = betas.sqrt()

    alpha_bars = torch.cumsum(torch.log(alphas), dim=0).exp()
    sqrt_alpha_bars = alpha_bars.sqrt()

    sqrt_one_minus_alpha_bars = (1 - alpha_bars).sqrt()
    one_minus_alphas_over_sqrt_one_minus_alpha_bars = (
        1 - alphas
    ) / sqrt_one_minus_alpha_bars

    """
    import matplotlib.pyplot as plt
    plt.title("Variance Schedule")
    plt.plot(betas, label="betas")
    plt.plot(alpha_bars, label="alpha_bars")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
    """

    return {
        "betas": betas,
        "alphas": alphas,
        "sigmas": sigmas,
        "sqrt_alphas_inv": sqrt_alphas_inv,
        "alpha_bars": alpha_bars,
        "sqrt_alpha_bars": sqrt_alpha_bars,
        "sqrt_one_minus_alpha_bars": sqrt_one_minus_alpha_bars,
        "one_minus_alphas_over_sqrt_one_minus_alpha_bars": one_minus_alphas_over_sqrt_one_minus_alpha_bars,
    }
