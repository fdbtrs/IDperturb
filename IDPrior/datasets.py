import os
import pickle
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

def read_numpy_embedding(file_path):
    n = np.load(file_path, allow_pickle=True)

    if isinstance(n, np.ndarray):
        return n

    # if the file wasn't a numpy file it was a pickle
    pkl_key = [k for k in n.files if k.endswith(".pkl")][0]  # Extract pickled bytes
    pkl_bytes = n[pkl_key]
    arr = pickle.loads(pkl_bytes)  # numpy ndarray (embedding_size,)
    return arr

# TODO: final tensor shape could be set using config.model.embedding_size
class EmbeddingFileDataset(Dataset):
    def __init__(self, embed_file):
        self.embeds = np.load(embed_file)
    def __len__(self):
        return len(self.embeds)

    def __getitem__(self, idx):
        tensor = torch.from_numpy(self.embeds[idx]).float()
        tensor = F.normalize(tensor, p=2, dim=0)
        tensor = tensor.view(1, 16, 32)

        return tensor


def import_dataset(config):
    return EmbeddingFileDataset(config.paths.vector_folder)


def get_initial_noise(n, config):
    return torch.randn(
            n, 1, *config.model.embedding_size, device=config["device"]
        )


def repeated_samples(id_sample, count_per_id, config):
    return id_sample.repeat(count_per_id, 1, 1)