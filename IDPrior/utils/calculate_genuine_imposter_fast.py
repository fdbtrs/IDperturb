import os
from tqdm import tqdm
import numpy as np

np.random.seed(1337)

verbose = True
matrix_to_load = "***.npy"
RUN_FOLDER = os.path.split(matrix_to_load)[0]

list_of_percentages = [100]  # chunk cuts to divide the matrix processing in
genuine_sample_count = 10  # selecting 10 embeddings: 100 scores for each imposter pair
imposter_sample_count = 3  # selecting 3 embeddings: 9 scores for each imposter pair
imposter_comparison_percentage = 100  # percentage of the embeddings each embedding is compared with in the imposter calculation

imposter_percentage_starts = list(range(0, 100, imposter_comparison_percentage))
imposter_percentage_slices = list(
    zip(imposter_percentage_starts, (imposter_percentage_starts[1:] + [100]))
)
if verbose:
    print(
        "ID embedding group percentage slices:", imposter_percentage_slices, flush=True
    )
"""
if we have a total of 100 embeddings and imposter_comparison_percentage = 25:
    each embedding is compared with 25 other embedding
so instead of computing a 100x100 matrix, we calculate 25x25 matrix 4 times (diagonal and upper triangle discarded)

if normal set up:
    100 embedding -> 10K scores -> discarding diagonal and upper triangle -> 4950 score
if imposter_comparison_percentage = 25
    100 embedding -> split groups of (25,25,25,25) -> each group generates 25x25=625 score discarding diagonal and upper triangle -> 300 scores x 4 groups -> a total of 1200 scores

Still all 100 of the embeddings contribute to the computation so when we say "percent_to_calculate_for% of the id's from 300K is used" that holds
"""

if verbose:
    print(f"loading the matrix ({matrix_to_load})...", end=" ", flush=True)
loaded_matrix = np.load(matrix_to_load)
if verbose:
    print(loaded_matrix.shape)

# defines the portion of the matrix we'll do the calculation for
for percent_to_calculate_for in list_of_percentages:

    save_folder = os.path.join(
        RUN_FOLDER,
        f"impsample{imposter_sample_count}_with_{imposter_comparison_percentage}%comparison_gensample{genuine_sample_count}_{percent_to_calculate_for}%",
    )

    os.makedirs(save_folder, exist_ok=True)
    txt_file_path_to_write_the_imposters = os.path.join(save_folder, "imposter.txt")
    txt_file_path_to_write_the_genuines = os.path.join(save_folder, "genuine.txt")

    n = loaded_matrix[: int(len(loaded_matrix) * percent_to_calculate_for / 100)]
    if verbose:
        print(f"using the first {percent_to_calculate_for}% of the matrix: ", n.shape)

    # ===== IMPOSTER CALCULATION BETWEEN 3 RANDOM EMBEDDING FROM EACH ID ===== #

    # get 3 random embeding from each id for imposter calculation
    random_indices = np.array(
        [
            np.random.choice(n.shape[1], size=(imposter_sample_count), replace=False)
            for _ in range(n.shape[0])
        ]
    )
    embeddings_selected_for_imposter = n[
        np.arange(n.shape[0])[:, np.newaxis], random_indices
    ]

    # we have a (N, imposter_sample_count, 512) matrix
    # (N, N, 3, 3) is the result shape where first 2 index are the id index and second two index are embedding index
    with open(txt_file_path_to_write_the_imposters, "w") as imp_file:
        for group_start, group_end in imposter_percentage_slices:
            group = embeddings_selected_for_imposter[
                int(len(embeddings_selected_for_imposter) * group_start / 100) : int(
                    len(embeddings_selected_for_imposter) * group_end / 100
                ),
                :,
                :,
            ]

            if verbose:
                print(
                    f"calculating the imposters of group [{group_start} - {group_end})% with shape {group.shape}...",
                    end=" ",
                    flush=True,
                )
            R = np.einsum("ikm,jlm->ijkl", group, group)
            if verbose:
                print(f"shape of calculated imposter matrix: {R.shape}")

            # now we are getting only the imposter ones (masking the duplicate imposters in the upper triangle and masking the genuines in the diagonal)
            if verbose:
                print(
                    "removing diagonals and upper triangle from imposters...",
                    end=" ",
                    flush=True,
                )
            R = R[
                np.tril_indices(group.shape[0], k=-1)
            ]  # remove the diagonal and duplicate imposter id pairs
            if verbose:
                print(f"shape of imposter matrix: {R.shape}")

            print(
                f"writing the imposters of group [{group_start} - {group_end})% in to {txt_file_path_to_write_the_imposters}",
                flush=True,
            )
            [imp_file.write(str(value) + "\n") for value in tqdm(R.flatten())]
    imp_file.close()

    # ===== GENUINE CALCULATION BETWEEN 10 RANDOM EMBEDDING FROM EACH ID ===== #

    # get 10 random embeding from each id for genuine calculation
    random_indices = np.array(
        [
            np.random.choice(n.shape[1], size=(genuine_sample_count), replace=False)
            for _ in range(n.shape[0])
        ]
    )
    embeddings_selected_for_genuine = n[
        np.arange(n.shape[0])[:, np.newaxis], random_indices
    ]

    # we have a (N, genuine_sample_count, 512) matrix
    # (N, 10, 10) is the result shape where first index is the id index and next other two are embeddings
    # But we remove the diagonal and the duplicate results by only taking the lower triangle
    if verbose:
        print("calculating the genuines...", end=" ", flush=True)
    R = embeddings_selected_for_genuine @ embeddings_selected_for_genuine.transpose(
        0, 2, 1
    )
    if verbose:
        print(f"shape of calculated genuine matrix: {R.shape}")

    if verbose:
        print(
            "removing diagonals and upper triangles from genuines...",
            end=" ",
            flush=True,
        )
    R = R[
        :,
        np.tril(
            np.ones((genuine_sample_count, genuine_sample_count), dtype=bool), k=-1
        ),
    ]  # only get the lower triange of each genuine id
    if verbose:
        print(f"shape of genuine matrix: {R.shape}")

    print(
        f"writing the genuines in to {txt_file_path_to_write_the_genuines}", flush=True
    )
    with open(txt_file_path_to_write_the_genuines, "w") as gen_file:
        [gen_file.write(str(value) + "\n") for value in tqdm(R.flatten())]
    gen_file.close()

    if verbose:
        print(
            f"percent_to_calculate_for {percent_to_calculate_for}% done out of",
            list_of_percentages,
        )
