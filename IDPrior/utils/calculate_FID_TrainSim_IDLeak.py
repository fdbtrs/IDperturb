import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

l1 = "experiment1"  # label of experiment1
f1 = "***.npy"  # path to synthetic embeddings
r1 = "***.npy"  # path to real embeddings

l2 = "experiment2"  # label of experiment2
f2 = "***.npy"  # path to synthetic embeddings
r2 = "***.npy"  # path to real embeddings


# FID metric
def calculate_fid(real_embeddings, fake_embeddings):
    fake_embeddings = (
        fake_embeddings[:, 0, :] if len(fake_embeddings.shape) == 3 else fake_embeddings
    )
    real_embeddings = (
        real_embeddings[:, 0, :] if len(real_embeddings.shape) == 3 else real_embeddings
    )
    real_embeddings = real_embeddings / np.linalg.norm(
        real_embeddings, axis=1, keepdims=True
    )
    fake_embeddings = fake_embeddings / np.linalg.norm(
        fake_embeddings, axis=1, keepdims=True
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


def calculate_train_similarity_and_IDleak(
    real_embeddings, fake_embeddings, leak_thresholds=[0.3]
):
    fake_embeddings = (
        fake_embeddings[:, 0, :] if len(fake_embeddings.shape) == 3 else fake_embeddings
    )
    real_embeddings = (
        real_embeddings[:, 0, :] if len(real_embeddings.shape) == 3 else real_embeddings
    )

    real_embeddings = real_embeddings.astype(np.float32)
    fake_embeddings = fake_embeddings.astype(np.float32)

    # L2 normalize the embeddings to compute accurate Cosine Similarity
    real_norm = real_embeddings / np.linalg.norm(real_embeddings, axis=1, keepdims=True)
    fake_norm = fake_embeddings / np.linalg.norm(fake_embeddings, axis=1, keepdims=True)

    # Matrix multiplication: shape (num_fake, num_real)
    # Each row represents a generated image's similarity to all real training images
    similarity_matrix = fake_norm @ real_norm.T

    # Get the maximum similarity for every generated image
    max_sim_per_row = np.max(similarity_matrix, axis=1)

    # Average the max
    mean_sim = np.mean(max_sim_per_row)
    median_sim = np.median(max_sim_per_row)
    max_sim = np.max(max_sim_per_row)
    p95_sim = np.percentile(max_sim_per_row, 95)
    p99_sim = np.percentile(max_sim_per_row, 99)

    results_dict = {
        "mean": float(mean_sim),
        "median": float(median_sim),
        "max": float(max_sim),
        "p95": float(p95_sim),
        "p99": float(p99_sim),
    }

    for th in leak_thresholds:
        # Indicator function I(max_sim > threshold)
        leakage_mask = max_sim_per_row > th
        # Leakage Rate: L = (1/M) * sum(indicator)
        results_dict[f"{th:.2f}"] = float(np.mean(leakage_mask)) * 100

    return results_dict, max_sim_per_row


LIMIT = 10000
print(LIMIT)
leak_thresholds = np.round(np.arange(0.25, 0.65, 0.05), 2)
real_and_fakes = [(l1, r1, f1), (l2, r2, f2)]
experiment_sim_scores = {}
experiment_leak_rates = {}
for l, r, f in real_and_fakes:
    print(f"\n{l}")
    real_embeddings = np.load(r)[:LIMIT]
    fake_embeddings = (
        np.random.randn(LIMIT, 512).astype(np.float32)
        if f == "random"
        else np.load(f)[:LIMIT]
    )

    fid = calculate_fid(real_embeddings, fake_embeddings)
    results_dict, max_sim_per_row = calculate_train_similarity_and_IDleak(
        real_embeddings, fake_embeddings, leak_thresholds
    )
    experiment_sim_scores[l] = max_sim_per_row
    experiment_leak_rates[l] = [results_dict[f"{th:.2f}"] for th in leak_thresholds]

    print(f"-> FID: {fid:.4f}")
    print(f"-> Train Sim (mean_sim): {results_dict['mean']:.4f}")
    print(f"-> Train Sim (median_sim): {results_dict['median']:.4f}")
    print(f"-> Train Sim (max_sim): {results_dict['max']:.4f}")
    print(f"-> Train Sim (p95_sim): {results_dict['p95']:.4f}")
    print(f"-> Train Sim (p99_sim): {results_dict['p99']:.4f}")
    for th in leak_thresholds:
        print(f"-> ID-Leak ({th:.2f}): {results_dict[f'{th:.2f}']:.6f}")

import numpy as np
import matplotlib.pyplot as plt
import itertools

# Define distinct line styles to cycle through
line_styles = ["-", "--", "-.", ":"]

# Define distinct markers for Plot 1 and Plot 2
markers = ["o", "s", "^"]

# ==========================================
# PLOT 1: Cosine Similarity Distribution
# ==========================================
plt.figure(figsize=(8, 6))

num_bins = 150
bins = np.linspace(-0.4, 1.0, num_bins + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2

num_exps = len(experiment_sim_scores)

# Generate N visually distinct colors using a large colormap
# 'tab20' works great for <= 20 lines; use 'turbo' or 'viridis' for > 20
colors = [plt.cm.tab20(i / max(num_exps, 1)) for i in range(num_exps)]

for i, (label, scores) in enumerate(experiment_sim_scores.items()):
    counts, _ = np.histogram(scores, bins=bins)
    percentages = (counts / len(scores)) * 100

    percentages = np.where(percentages == 0, np.nan, percentages)

    # Assign distinct color, linestyle, and marker
    color = colors[i]
    ls = line_styles[i % len(line_styles)]
    marker = markers[i % len(markers)]

    # markevery=15 prevents markers from overcrowding the dense plot
    plt.plot(
        bin_centers,
        percentages,
        label=label,
        color=color,
        linestyle=ls,
        marker=marker,
        markevery=(i, 8),
        markersize=5,
        linewidth=1.8,
    )

# 3. Add vertical dashed lines for leak thresholds (kept in black to avoid line confusion)
for th in [0.25, 0.3, 0.4]:
    plt.axvline(
        x=th, color="black", linestyle="--", alpha=0.7, label=f"Leak Threshold ({th})"
    )

# 4. Styling
plt.xlabel("Cosine Similarity of Nearest Training Neighbor", fontsize=12)
plt.ylabel("Frequency (%)", fontsize=12)
plt.xlim([-0.1, 1.0])
plt.ylim(bottom=0)
plt.grid(True, linestyle=":", alpha=0.6)

# Move legend outside if there are many experiments
plt.legend(loc="upper right", fontsize=9)
plt.tight_layout()

plt.savefig("similarity_distribution_comparison.pdf", dpi=300, bbox_inches="tight")
plt.close()


# ==========================================
# PLOT 2: ID-Leak Rate vs Threshold
# ==========================================
plt.figure(figsize=(8, 6))

num_exps_leak = len(experiment_leak_rates)
colors_leak = [plt.cm.tab20(i / max(num_exps_leak, 1)) for i in range(num_exps_leak)]

for i, (label, rates) in enumerate(experiment_leak_rates.items()):
    color = colors_leak[i]
    ls = line_styles[i % len(line_styles)]
    marker = markers[i % len(markers)]

    rates_pct = [r for r in rates]

    plt.plot(
        leak_thresholds,
        rates_pct,
        label=label,
        color=color,
        linestyle=ls,
        markevery=(i, 3),
        marker=marker,
        markersize=6,
        linewidth=2,
    )

plt.xlabel("Threshold", fontsize=12)
plt.ylabel("ID-Leak Rate (%)", fontsize=12)
plt.ylim(bottom=-5)
plt.xticks(leak_thresholds)
plt.grid(True, linestyle=":", alpha=0.6)

plt.legend(loc="upper right", fontsize=10)
plt.tight_layout()

plt.savefig("id_leak_vs_threshold_in_percentage.pdf", dpi=300, bbox_inches="tight")
plt.close()
