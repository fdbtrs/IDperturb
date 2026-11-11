import os
import gc
import math
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.manifold import TSNE

# --- Global Font Size Configurations ---
plt.rcParams.update(
    {
        "font.size": 14,  # Base text size
        "axes.titlesize": 16,  # Title size
        "axes.labelsize": 14,  # X and Y label size
        "xtick.labelsize": 12,  # X-tick labels
        "ytick.labelsize": 12,  # Y-tick labels
        "legend.fontsize": 10,  # Legend text size
        "figure.titlesize": 18,  # Main figure title size (if used)
    }
)

save_path = "UNIQUENESS_PLOT"
os.makedirs(save_path, exist_ok=True)
csv_save_path = os.path.join(save_path, "uniqueness_analysis_optimized.csv")

LIMIT = 1000000
TSNE_LIMIT = 100
SMALLEST_COUNT_OF_WELL_SEPERATED_IDS = LIMIT
experiment_dir_label_file_type = [
    # Random
    ["random_uniqueness", "Random", "Random", "raw_benchmark"],
    # benchmark
    ["FOLDER", "label1", "***.npy", "raw_benchmark"],
    # synthetic benchmark
    ["FOLDER", "label2", "***.npy", "generated_benchmark"],
    # synthetic data
    ["FOLDER", "label3", "***.npy", "generated"],
]

num_of_experiments = len(experiment_dir_label_file_type)
marker_dict = {"raw_benchmark": "o", "generated_benchmark": "s", "generated": "x"}
marker_list = [marker_dict[p_l_f_t[3]] for p_l_f_t in experiment_dir_label_file_type]

color_map = {"Random": "black", "label1": "orange", "label2": "green", "label3": "red"}

thresholds_for_percentages = np.array(
    [
        0.05,
        0.065,
        0.08,
        0.095,
        0.11,
        0.125,
        0.14,
        0.155,
        0.17,
        0.185,
        0.2,
        0.215,
        0.23,
        0.245,
        0.26,
        0.275,
        0.29,
        0.305,
        0.32,
        0.335,
        0.35,
        0.365,
        0.38,
        0.395,
        0.41,
        0.425,
        0.44,
        0.455,
        0.47,
        0.485,
        0.5,
        0.515,
        0.56,
        0.605,
        0.65,
        0.695,
        0.74,
        0.785,
        0.83,
        0.875,
        0.905,
    ]
)
threshold_for_scaling1 = 0.4
threshold_for_scaling2 = 0.3
threshold_for_scaling3 = 0.25
thresholds_and_axises = [
    {"thresh": threshold_for_scaling1, "ax_idx": 0},
    {"thresh": threshold_for_scaling2, "ax_idx": 1},
    {"thresh": threshold_for_scaling3, "ax_idx": 2},
]
samples_to_check_for_scaling = [
    1000,
    2500,
    5000,
    7500,
    10000,
    15000,
    20000,
    25000,
    30000,
    40000,
    50000,
    75000,
    100000,
    125000,
    150000,
    200000,
    250000,
    500000,
    750000,
    1000000,
]


def get_matrix(dir_path, filename):
    if filename.lower() == "random":
        print(f"CREATING MATRIX WITH {LIMIT} RANDOM ID")
        matrix = np.random.randn(LIMIT, 512).astype(np.float32)
    else:
        path = os.path.join(dir_path, filename)
        if not os.path.exists(
            path
        ):  # Fallback if filename is already absolute in config
            path = filename
        matrix = np.load(path).astype(np.float32)
        if matrix.ndim == 3:
            matrix = matrix[:, 0, :]

        if matrix.shape[0] > LIMIT:
            print(f"CAPPING MATRIX TO FIRST {LIMIT} ID")
            matrix = matrix[:LIMIT]

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / (norms + 1e-12)


def calculate_distribution_data(path):
    chunk_sim_sums = np.load(os.path.join(path, "chunk_sim_sums.npy"))
    chunk_squared_sim_sums = np.load(os.path.join(path, "chunk_squared_sim_sums.npy"))
    hist_counts = np.load(os.path.join(path, "hist_counts.npy"))
    with open(os.path.join(path, "config_dict.pkl"), "rb") as f:
        config = pickle.load(f)

    matrix_shape, chunk_size = config["shape"], config["chunk_size"]
    subset_size = 2500
    bins = np.linspace(-1, 1, config["bin_counts"])
    subset_sizes = list(range(subset_size, matrix_shape[0], subset_size)) + [
        matrix_shape[0]
    ]

    means = []
    for s_size in subset_sizes:
        num_chunks = math.ceil(
            s_size / chunk_size
        )  # ceil used to handle the possibility of last part being out of bounds
        num_elements = (
            s_size**2 - s_size
        )  # diagonals are excluded during the calculations
        summed = chunk_sim_sums[:num_chunks, :num_chunks].sum() - s_size
        squared_summed = chunk_squared_sim_sums[:num_chunks, :num_chunks].sum() - s_size

        # average it
        subset_mean = summed / num_elements
        means.append(subset_mean)

        # Var(X) = E[X^2] - (E[X])^2
        subset_var = (squared_summed / num_elements) - (subset_mean) ** 2
        subset_std = subset_var**0.5

    hist_values = hist_counts.sum(axis=(0, 1))
    return subset_sizes, means, hist_values, bins


def draw_line_with_markers(ax, X, Y, label, e, emb_type, axis_scale=False):
    global SMALLEST_COUNT_OF_WELL_SEPERATED_IDS

    if axis_scale:
        X = np.log10(X)
        Y = np.log10(Y)
        if np.min(Y) < SMALLEST_COUNT_OF_WELL_SEPERATED_IDS:
            SMALLEST_COUNT_OF_WELL_SEPERATED_IDS = np.min(Y)

    style_map = {
        "raw_benchmark": "solid",  # Standard solid line
        "generated": (0, (1, 10)),  # Custom: 1pt dot, 10pt space (Separated)
        "generated_benchmark": (0, (10, 5)),  # Custom: 10pt dash, 5pt space (Longer)
    }
    chosen_style = style_map.get(emb_type, "solid")

    chosen_color = color_map.get(label, "black")

    # first we draw the lines with small markers on top
    line = ax.plot(
        X,
        Y,
        marker="o",
        markersize=2,
        linewidth=1.5,
        alpha=0.4,
        linestyle=chosen_style,
        color=chosen_color,
    )

    # then we draw only separated markers to distinguish the overlapping lines
    ax.plot(
        X,
        Y,
        label=label,
        color=chosen_color,
        marker=marker_list[e % len(marker_list)],
        markevery=(
            e % (min(num_of_experiments, len(X) // 3)),
            min(num_of_experiments, len(X) // 3),
        ),
        markersize=6,
        linestyle="None",
        alpha=0.8,
    )


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_row_max_fully_vectorized(matrix_np, block_size=4096, device="cuda:0"):
    # 1. Transfer matrix to GPU once
    matrix = torch.from_numpy(matrix_np).to(device).to(torch.float16)
    N = matrix.shape[0]

    # Initialize row_max on GPU
    row_max = torch.full((N,), float("-inf"), device=device, dtype=torch.float16)

    # We use a context manager to disable gradient tracking (saves memory/time)
    with torch.no_grad():
        for i in tqdm(range(0, N, block_size), desc="GPU Row Max", leave=False):
            end_i = min(i + block_size, N)
            block_data = matrix[i:end_i]

            # 1. Compare current block against all rows BEFORE this block
            if i > 0:
                past_rows = matrix[:i]
                # res_past shape: (block_size, i)
                res_past = torch.mm(block_data, past_rows.t())
                # Update row_max with max from previous rows
                row_max[i:end_i] = torch.max(res_past, dim=1).values
                del past_rows, res_past  # Explicitly free references

            # 2. Compare rows WITHIN the current block
            res_internal = torch.mm(block_data, block_data.t())

            # 3. Mask lower triangle (k=-1)
            # Create a boolean mask on the fly
            mask = torch.tril(
                torch.ones((res_internal.shape), device=device, dtype=torch.bool),
                diagonal=-1,
            )

            # Fill non-lower-triangle with -inf
            res_internal = torch.where(
                mask, res_internal, torch.tensor(float("-inf"), device=device)
            )

            # 4. Final max update for this block
            internal_maxes = torch.max(res_internal, dim=1).values
            row_max[i:end_i] = torch.maximum(row_max[i:end_i], internal_maxes)

            del res_internal, mask, internal_maxes

    # Transfer only the final 1D array back to CPU
    return row_max.cpu().numpy()


# --- MAIN PLOTTING ---
results = []  # for scaling percentages

# Create separate figures/axes
figs = []
axs = []

for i in range(8):
    fig, ax = plt.subplots(figsize=(8, 6))
    figs.append(fig)
    axs.append(ax)

all_tsne_embeddings = []
tsne_metadata = []  # will store tuples of (label, emb_type)

for e, (dir_path, label, filename, emb_type) in enumerate(
    experiment_dir_label_file_type
):
    clear_gpu_memory()
    print(f"Processing ({e+1}/{num_of_experiments}): {label}")

    matrix = get_matrix(dir_path, filename)
    N = matrix.shape[0]
    row_max = get_row_max_fully_vectorized(matrix)

    #
    #
    #
    #
    # ===== tsne plotting ===== # # ===== tsne plotting ===== #
    tsne_chunk = matrix[:TSNE_LIMIT].astype(np.float32)
    all_tsne_embeddings.append(tsne_chunk)

    # Store metadata for each embedding vector in this chunk
    for _ in range(tsne_chunk.shape[0]):
        tsne_metadata.append((label, emb_type))
    # ===== tsne plotting ===== # # ===== tsne plotting ===== #
    #
    #
    #
    #
    # ===== scaling MERGED FOR SPEED ===== # # ===== scaling MERGED FOR SPEED ===== #
    threshold_to_counts_dict = {exp["thresh"]: [] for exp in thresholds_and_axises}
    valid_samples = [s for s in samples_to_check_for_scaling if s <= N]

    for n in tqdm(valid_samples, desc="scaling", leave=False):
        for exp in thresholds_and_axises:
            t = exp["thresh"]
            count = 1 + np.sum(row_max[1:n] <= t)
            threshold_to_counts_dict[t].append(count)

            results.append(
                {
                    "method": label,
                    "threshold": t,
                    "samples": n,
                    "percentage": count * 100 / n,
                }
            )

    for exp in thresholds_and_axises:
        t = exp["thresh"]
        draw_line_with_markers(
            axs[exp["ax_idx"]],
            valid_samples,
            threshold_to_counts_dict[t],
            label,
            e,
            emb_type,
            axis_scale=True,
        )
    # ===== scaling MERGED FOR SPEED ===== # # ===== scaling MERGED FOR SPEED ===== #
    #
    #
    #
    #
    # ===== percentages ===== # # ===== percentages ===== # # ===== percentages ===== #
    t_percentages = []

    for t in tqdm(thresholds_for_percentages, desc="percentages", leave=False):
        count = 1 + np.sum(row_max[1:] <= t)
        t_percentages.append((count / N) * 100)
    draw_line_with_markers(
        axs[3], thresholds_for_percentages, t_percentages, label, e, emb_type
    )
    # ===== percentages ===== # # ===== percentages ===== # # ===== percentages ===== #
    #
    #
    #
    #
    # ===== Chunk-based ===== # # ===== Chunk-based ===== # # ===== Chunk-based ===== #
    try:
        x_avg, y_avg, hist_vals, bins = calculate_distribution_data(dir_path)
        draw_line_with_markers(axs[4], x_avg, y_avg, label, e, emb_type)

        # Only plot the non-zero data
        mask = hist_vals > 0
        total_pairs = hist_vals.sum()
        hist_percentages = (hist_vals[mask] / total_pairs) * 100
        draw_line_with_markers(axs[5], bins[mask], hist_percentages, label, e, emb_type)
    except FileNotFoundError:
        print(f"Skipping distribution plots for {label} (chunk files not found)")
    # ===== Chunk-based ===== # # ===== Chunk-based ===== # # ===== Chunk-based ===== #
    #
    #
    #
    #
    # ===== row max hist ===== # # ===== row max hist ===== # # ===== row max hist ===== #
    # Define 100 bins spanning from -1 to 1 for cosine similarity
    row_max_bins = np.linspace(-1, 1, 101)
    counts, edges = np.histogram(row_max, bins=row_max_bins)

    # Calculate bin centers to use as X coordinates for the line plot
    bin_centers = (edges[:-1] + edges[1:]) / 2

    # Convert counts to percentages (Pair Frequency %)
    total_samples = counts.sum()
    hist_pct = (counts / total_samples) * 100 if total_samples > 0 else counts

    # Filter to only plot non-zero data points for a cleaner look
    mask = hist_pct > 0
    draw_line_with_markers(
        axs[6], bin_centers[mask], hist_pct[mask], label, e, emb_type
    )
    # ===== row max hist ===== # # ===== row max hist ===== # # ===== row max hist ===== #
    #
    #
    #
    #


# ==========================================
# 2. Concat & fit the Global t-SNE Model
# ==========================================
print("Calculating global combined 2D t-SNE for all experiments...")
global_tsne_input = np.concatenate(all_tsne_embeddings, axis=0)

tsne_model = TSNE(
    n_components=2, random_state=42
)  # random_state added for reproducibility
global_tsne_results = tsne_model.fit_transform(global_tsne_input)

# ==========================================
# 3. Plot the global t-SNE onto Axis [7]
# ==========================================
df_metadata = pd.DataFrame(tsne_metadata, columns=["label", "emb_type"])
# Plot experiment by experiment to preserve clean legends
for label in color_map.keys():
    indices = df_metadata[df_metadata["label"] == label].index.tolist()
    if not indices:
        continue

    # Get the embedding type for the current label to deduce the marker
    emb_type = df_metadata.loc[indices[0], "emb_type"]

    axs[7].scatter(
        global_tsne_results[indices, 0],
        global_tsne_results[indices, 1],
        label=label,
        color=color_map.get(label, "black"),
        marker=marker_dict.get(emb_type, "o"),
        alpha=0.8,
        s=15,
        edgecolors="none",
    )

# Formatting
titles = [
    f"Scaling Analysis (Threshold {threshold_for_scaling1})",
    f"Scaling Analysis (Threshold {threshold_for_scaling2})",
    f"Scaling Analysis (Threshold {threshold_for_scaling3})",
    "Threshold vs Uniqueness Analysis",
    "Avg Similarity vs Sample Size",
    "Similarity Histograms",
    "inter-ID Max Similarity Histograms",
    "Embedding Space",
]
x_labels = [
    "Number of IDs",
    "Number of IDs",
    "Number of IDs",
    "Threshold",
    "Number of Samples",
    "Cosine Similarity",
    "Cosine Similarity",
    "t-SNE Dimension 1",
]
y_labels = [
    "Well Separated IDs",
    "Well Separated IDs",
    "Well Separated IDs",
    "% of Unique IDs",
    "Avg Similarity",
    "Pair Frequency %",
    "Pair Frequency %",
    "t-SNE Dimension 2",
]
file_names = [
    "scaling_thresh_04.png",
    "1M_ID_scaling_thresh_03.pdf",
    "1M_ID_scaling_thresh_025.pdf",
    "threshold_vs_uniques_curve.pdf",
    "avg_similarity_vs_samples.png",
    "similarity_histograms.png",
    "max_similarity_histograms.pdf",
    "tsne_embedding_visualization.pdf",
]

x_tick_mapping = {
    np.log10(1000): "1K",
    np.log10(5000): "5K",
    np.log10(10000): "10K",
    np.log10(25000): "25K",
    np.log10(50000): "50K",
    np.log10(100000): "100K",
    np.log10(250000): "250K",
    np.log10(500000): "500K",
    np.log10(1000000): "1M",
}

y_tick_mapping = {
    # np.log10(10): "10",
    # np.log10(100): "100",
    np.log10(1000): "1K",
    np.log10(10000): "10K",
    np.log10(100000): "100K",
    np.log10(1000000): "1M",
}

for i in range(8):
    axs[i].set_title(titles[i])
    axs[i].set_xlabel(x_labels[i])
    axs[i].set_ylabel(y_labels[i])
    axs[i].grid(True, linestyle="--", alpha=0.6)

    if i in [0, 1, 2]:
        axs[i].set_xticks(list(x_tick_mapping.keys()))
        axs[i].set_xticklabels(list(x_tick_mapping.values()))

        axs[i].set_yticks(list(y_tick_mapping.keys()))
        axs[i].set_yticklabels(list(y_tick_mapping.values()))

        axs[i].set_ylim(
            SMALLEST_COUNT_OF_WELL_SEPERATED_IDS * 0.8, np.log10(LIMIT * 1.2)
        )
        axs[i].legend(loc="upper left", fontsize=11)
    elif i == 3:
        axs[i].legend(loc="lower right", fontsize=11)
    elif i == 7:
        # Move legend out of layout or resize markers inside it if points overlap densely
        axs[i].legend(fontsize=11, markerscale=2.5, loc="best")
    else:
        axs[i].legend(fontsize=11)
    figs[i].tight_layout()
    save_path_individual = os.path.join(save_path, file_names[i])
    figs[i].savefig(save_path_individual, dpi=300)
    plt.close(figs[i])

df = pd.DataFrame(results)
pivot_df = df.pivot_table(
    index=["method", "samples"], columns="threshold", values="percentage"
)
pivot_df.to_csv(csv_save_path)
print("saved to", save_path)
