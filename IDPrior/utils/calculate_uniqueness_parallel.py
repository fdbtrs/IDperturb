import os
import pickle
import numpy as np
from datetime import datetime
import torch
import torch.distributed as dist


def setup_distributed():
    dist.init_process_group("nccl")  # GPU backend
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())


setup_distributed()
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device(f"cuda:{rank}")

"""
we can't fit MATRIX = 10Mx10M similarity matrix in memory, so first we split into chunks:

CHUNK_MATRIX (each chunk Cij is chunk_size x chunk_size):
[C00, C01, C02, C03, ...]
[C10, C11, C12, C13, ...]
[C20, C21, C22, C23, ...]
[..., ..., ..., ..., ...]

Then we calculate the sum of similarities in these chunks:

CHUNK_SUMMED_MATRIX (each Cij_sim is scalar=Cij.sum()):
[C00_sim_sum, C01_sim_sum, C02_sim_sum, C03_sim_sum, ...]
[C10_sim_sum, C11_sim_sum, C12_sim_sum, C13_sim_sum, ...]
[C20_sim_sum, C21_sim_sum, C22_sim_sum, C23_sim_sum, ...]
[..........., ..........., ..........., ..........., ...]

after that, we calculate the averaged similarity from CHUNK_SUMMED_MATRIX

* same chunking and summing logic also applied for the squared sums for std calculation of the 10Mx10M matrix.
* same chunking logic also applied for histograms. but instead of summing  similarities into scallars, they are distributed to 100 bins.


for example we have 8K ids and our chunk_size is 5K:
than we have to calculate (duplicate calculation is included for easier explanation)
[C00, C01]
[C10, C11]
C00 -> first 5K * first 5K (matrix[:5000] @ matrix[:5000])
C01 -> first 5K * last 3K (matrix[:5000] @ matrix[5000:8000])
C10 -> last 3K * first 5K (matrix[5000:8000] @ matrix[:5000])
C11 -> last 3K * last 3K (matrix[5000:8000] @ matrix[5000:8000])

torch indexing handles the out of bounds indexing so we don't need to do anything extra for multiplication part
but to use our diagonal mask with the histograms: "chunk_matrix[mask]" we need to resize the mask as done in:
mask = mask[:chunk_matrix.shape[0], :chunk_matrix.shape[1]]

"""

matrix_path = "***.npy"

if isinstance(matrix_path, str):
    print("Loading the matrix")
    MAIN_PATH = os.path.split(matrix_path)[0]
    matrix = np.load(matrix_path)

    # if matrix is in shape (N,E,D) where N:number of identities, E: number of embeddings per ID, D: dimension size of the embeddings
    if len(matrix.shape) == 3:
        # only get the first sample from each id
        matrix = matrix[:, 0, :]
    matrix = torch.from_numpy(matrix).float().to(device)

elif isinstance(matrix_path, int):
    print(f"USING RANDOM WITH {matrix_path} IDs")
    matrix = np.random.randn(matrix_path, 512).astype(np.float32)
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = torch.from_numpy(matrix).float().to(device)
    MAIN_PATH = "random_uniqueness"
    os.makedirs(MAIN_PATH, exist_ok=True)

else:
    print("NO MATRIX PATH")
    exit(0)

chunk_size = 500
chunk_sizes = range(0, matrix.shape[0], chunk_size)
# check the plot_uniqueness_optimized_seperate_images.py file for the calculation.
# This code just calculates the similarities, sums them, bins them for the histogram and saves the result matrices
# these parameters are saved to pass to the plot calculation as a dict

# empty matrices to store chunk results
chunk_sim_sums = torch.zeros(
    (len(chunk_sizes), len(chunk_sizes)), device=device
).float()
chunk_squared_sim_sums = torch.zeros(
    (len(chunk_sizes), len(chunk_sizes)), device=device
).float()
# bins to store the chunk distributions
bin_counts = 101
bins = torch.linspace(-1, 1, bin_counts, device=device).float()
hist_counts = torch.zeros(
    (len(chunk_sizes), len(chunk_sizes), len(bins)), device=device
).float()

for i, chunk1_start in enumerate(chunk_sizes):
    if i % 100 == 0:
        print(
            f"RANK[{rank}] [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] processing row {i} out of {len(chunk_sizes)}"
        )
    for j, chunk2_start in enumerate(chunk_sizes):
        if j % world_size != rank or i > j:
            # skip work not assigned to this GPU or no need to compute [i,j] we have the symmetric [i,j]
            # each GPU computes the RANKth colummn at the upper triangle and mirrors the results to the lower triangle
            continue

        # calculate pair-wise chunk similarities sum
        chunk_matrix = torch.matmul(
            matrix[chunk1_start : chunk1_start + chunk_size],
            matrix[chunk2_start : chunk2_start + chunk_size].T,
        )

        # store the sum
        chunk_sim_sums[i, j] = chunk_sim_sums[j, i] = chunk_matrix.sum()

        # store the squared mean for std calculation at the distribution
        chunk_squared_sim_sums[i, j] = chunk_squared_sim_sums[j, i] = (
            chunk_matrix**2
        ).sum()

        # removing diagonal from the summing is done at the aggregation step
        # but removing diagonal from the hist bins are done in this code below:

        if i == j:  # remove the diagonal similarities if chunk is on the diagonal
            mask = ~torch.eye(chunk_size, dtype=torch.bool, device=device)
            mask = mask[
                : chunk_matrix.shape[0], : chunk_matrix.shape[1]
            ]  # for out of bounds part
            idx = torch.bucketize(chunk_matrix[mask], bins, right=True) - 1
        else:
            idx = torch.bucketize(chunk_matrix.view(-1), bins, right=True) - 1

        # calculate histogram bin distribution
        h = torch.bincount(idx, minlength=hist_counts.shape[2])
        hist_counts[i, j] = hist_counts[j, i] = h

# all_reduce each tensor so every rank has the full merged matrix
print("Reducing chunk_sim_sums...")
dist.all_reduce(chunk_sim_sums, op=dist.ReduceOp.SUM)
print("Reducing chunk_squared_sim_sums...")
dist.all_reduce(chunk_squared_sim_sums, op=dist.ReduceOp.SUM)
print("Reducing hist_counts...")
dist.all_reduce(hist_counts, op=dist.ReduceOp.SUM)
dist.barrier()
print("Reducing done.")

if rank == 0:
    with open(os.path.join(MAIN_PATH, "config_dict.pkl"), "wb") as f:
        pickle.dump(
            {"shape": matrix.shape, "chunk_size": chunk_size, "bin_counts": bin_counts},
            f,
        )
    f.close()
    np.save(
        os.path.join(MAIN_PATH, "chunk_squared_sim_sums.npy"),
        chunk_squared_sim_sums.cpu().numpy(),
    )
    np.save(os.path.join(MAIN_PATH, "chunk_sim_sums.npy"), chunk_sim_sums.cpu().numpy())
    np.save(os.path.join(MAIN_PATH, "hist_counts.npy"), hist_counts.cpu().numpy())
