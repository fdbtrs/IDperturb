import os
import logging


def setup_logging(out_folder):
    os.makedirs(out_folder, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(out_folder, "training.log"), mode="a"),
        ],
        force=True,
    )

    return logging.getLogger(__name__)


import numpy as np


def load_embeddings(embeddings_file):
    """
    Loads embeddings from a NumPy file.

    Expected shape:
        (N, 512)

    Returns:
        embeddings: float32 NumPy array of shape (N, 512)
    """
    embeddings = np.load(embeddings_file)

    if embeddings.ndim == 3:
        embeddings = embeddings[:, 0, :]

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding matrix, got shape {embeddings.shape}"
        )

    if embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Expected embedding dimension {EMBEDDING_DIM}, "
            f"got {embeddings.shape[1]}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings contain NaN or Inf values. "
            "Clean the dataset before training."
        )

    embeddings = embeddings.astype(np.float32, copy=False)

    norms = np.linalg.norm(embeddings, axis=1)

    logging.info(
        "Input embedding norms: mean=%.6f, min=%.6f, max=%.6f",
        norms.mean(),
        norms.min(),
        norms.max(),
    )

    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError(
            "Input embeddings are not all L2-normalized. "
            "Check your embedding extraction pipeline."
        )

    logging.info("Loaded embeddings: %s", embeddings.shape)
    logging.info(
        "Original data statistics: mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
        embeddings.mean(),
        embeddings.std(),
        embeddings.min(),
        embeddings.max(),
    )

    return embeddings


import torch.distributed as dist


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return not is_distributed() or dist.get_rank() == 0


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
