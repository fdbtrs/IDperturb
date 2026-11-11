import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel


def import_model(config):
    m = config.model    
    model = UNet2DModel(
            sample_size=m.embedding_size,
            in_channels=m.in_channels,
            out_channels=m.out_channels,
            block_out_channels=tuple(m.block_out_channels),
            down_block_types=tuple(m.down_block_types),
            up_block_types=tuple(m.up_block_types),
            layers_per_block=m.layers_per_block,
            attention_head_dim=m.attention_head_dim,
            dropout=m.dropout,
            downsample_type=m.downsample_type,
            upsample_type=m.upsample_type,
        )

    return model


def load_model(model, config, local_rank):
    # load the checkpoint file if given
    if config.paths.model_ckpt:
        state_dict = torch.load(config.paths.model_ckpt)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            if local_rank == 0:
                print(
                    "Failed to load the checkpoint, removing existing 'module.' prefixes in the state dict keys"
                )
            state_dict_no_module_prefix = {}
            for k, v in state_dict.items():
                new_k = k.replace("module.", "") if "module." in k else k
                state_dict_no_module_prefix[new_k] = v
            try:
                model.load_state_dict(state_dict_no_module_prefix)
                if local_rank == 0:
                    print("Loaded successfuly")
            except Exception as inner_e:
                if local_rank == 0:
                    print("Still failed to load", inner_e)
                    raise Exception(f"{config.paths.model_ckpt} load failed.")


def predict_model_output(model, noisy, timesteps):
    prediction = model(noisy, timesteps).sample
    return prediction

