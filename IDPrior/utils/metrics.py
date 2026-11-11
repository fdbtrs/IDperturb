import numpy as np
from scipy import linalg


# FID metric
def calculate_fid(real_embeddings, fake_embeddings):
    # use the first sample of each id if there are multiple sample per id
    fake_embeddings = (
        fake_embeddings[:, 0, :] if len(fake_embeddings.shape) == 3 else fake_embeddings
    )

    mu_real = np.mean(real_embeddings, axis=0)
    sigma_real = np.cov(real_embeddings, rowvar=False)
    mu_fake = np.mean(fake_embeddings, axis=0)
    sigma_fake = np.cov(fake_embeddings, rowvar=False)
    diff = mu_real - mu_fake
    cov_prod = sigma_real @ sigma_fake

    try:
        covmean = linalg.sqrtm(cov_prod)
    except Exception:
        covmean = linalg.sqrtm(cov_prod + np.eye(sigma_real.shape[0]) * 0.000001)

    if np.iscomplexobj(covmean):
        if not np.allclose(np.imag(covmean), 0, atol=1e-3):
            print(
                "Imaginary component too large in sqrtm, continuing FID calculation anyways"
            )
        covmean = np.real(covmean)

    # Compute FID
    fid = (diff @ diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid


# average cosine similarity
def calculate_uniqueness(fake_embeddings):
    # use the first sample of each id if there are multiple sample per id
    fake_embeddings = (
        fake_embeddings[:, 0, :] if len(fake_embeddings.shape) == 3 else fake_embeddings
    )

    sim_matrix = fake_embeddings @ fake_embeddings.T

    min_similarity = np.min(sim_matrix)
    # filling with -1 temporarly to get the max similarity
    np.fill_diagonal(sim_matrix, -1.0)
    max_similarity = np.max(sim_matrix)
    # now filling with 0 so (embed_i, embed_i) pairs don't contribute to the similarity calculation
    np.fill_diagonal(sim_matrix, 0.0)

    avg_similarity = sim_matrix.sum() / (sim_matrix.shape[0] ** 2 - sim_matrix.shape[0])
    return avg_similarity, min_similarity, max_similarity


def calculate_intra_class_similarity(fake_embeddings):
    """
    Calculates the average similarity within each class.
    Input shape: (num_classes, num_samples, vector_size)
    """
    if len(fake_embeddings.shape) != 3:
        return 0, 0, 0

    class_averages = []
    all_min = []
    all_max = []

    for i in range(fake_embeddings.shape[0]):
        # Extract all samples for the current class: (num_samples, vector_size)
        class_samples = fake_embeddings[i]

        # Compute pairwise similarity matrix for this class
        sim_matrix = class_samples @ class_samples.T

        # Mask the diagonal (self-similarity) to avoid bias
        mask = ~np.eye(fake_embeddings.shape[1], dtype=bool)
        valid_similarities = sim_matrix[mask]

        class_averages.append(np.mean(valid_similarities))
        all_min.append(np.min(valid_similarities))
        all_max.append(np.max(valid_similarities))

    # Aggregate results across all classes
    avg_intra = np.mean(class_averages)
    min_intra = np.min(all_min)
    max_intra = np.max(all_max)

    return avg_intra, min_intra, max_intra
