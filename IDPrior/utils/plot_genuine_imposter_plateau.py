import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

column_names = [
    "Experiment ID",
    "GMean",
    "GSTD",
    "IMean",
    "ISTD",
    "Sensitivity index (d')",
    "AUC",
    "J-Index",
    "J-Index_TH",
    "MCC",
    "MCC_TH",
    "EERlow",
    "EERhigh",
    "EER",
    "ZeroFMR",
    "FMR1000",
    "FMR100",
    "FMR20",
    "FMR10",
    "ZeroFNMR",
    "EER_TH",
    "ZeroFMR_TH",
    "FMR1000_TH",
    "FMR100_TH",
    "FMR20_TH",
    "FMR10_TH",
    "ZeroFNMR_TH",
]
columns_to_exclude_in_the_graph = ["Experiment ID"]

# genuine imposter result csvs with their experiment names from PyEER library
report_csv_paths = [["***.csv", "exp1"], ["***.csv", "exp2"]]


OUT_PATH = "OUT_FOLDER"
os.makedirs(OUT_PATH, exist_ok=True)
all_in_one_path = os.path.join(OUT_PATH, "all_experiment_all_metrics.png")
seperate_metrics_path = os.path.join(OUT_PATH, "all_experiment_separate_metrics.png")

data_rows = []
x_values = []
for csv_path, experiment_name in report_csv_paths:
    f = open(csv_path)
    lines = f.readlines()
    f.close()
    data = [x if e == 0 else float(x) for e, x in enumerate(lines[2][:-1].split(","))]
    data_rows.append(data)
    x_values.append(experiment_name)

df = pd.DataFrame(data_rows, columns=column_names)

plt.figure(figsize=(14, 10))

for column in [col for col in df.columns if col not in columns_to_exclude_in_the_graph]:
    y_values = df[column]

    if column == "Sensitivity index (d')":
        normalized_y_values = y_values / max(y_values)
        plt.plot(
            x_values,
            normalized_y_values,
            marker="o",
            label=column + " normalized to fit 0-1 range",
        )

        final_y = (
            normalized_y_values.iloc[-1]
            if isinstance(normalized_y_values, pd.Series)
            else normalized_y_values[-1]
        )

        plt.text(x_values[-1], final_y, column, fontsize=12, va="center")
    else:
        plt.plot(x_values, y_values, marker="o", label=column)
        plt.text(
            x_values[-1], y_values.iloc[-1], column, fontsize=12, ha="left", va="center"
        )

plt.title("Genuine Imposter Metric Comparison")
plt.xlabel("Experiment Name")
plt.ylabel("Metric Value")
plt.xticks(x_values)  # Ensure only the specified x-values are shown
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
plt.grid(True)
plt.tight_layout()
plt.savefig(all_in_one_path)
plt.close()


# Filter columns to plot
columns_to_plot = [
    col for col in df.columns if col not in columns_to_exclude_in_the_graph
]

# Prepare grid layout
n_rows, n_cols = 4, 7
fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, 16))
axes = axes.flatten()

for i, column in enumerate(tqdm(columns_to_plot)):
    ax = axes[i]
    y_values = df[column]

    # Handle normalization for Sensitivity index
    if column == "Sensitivity index (d')":
        normalized_y_values = y_values / max(y_values)
        ax.plot(
            x_values, normalized_y_values, marker="o", label=column + " (normalized)"
        )
        final_y = (
            normalized_y_values.iloc[-1]
            if isinstance(normalized_y_values, pd.Series)
            else normalized_y_values[-1]
        )
        ax.set_ylim(min(normalized_y_values) - 0.05, max(normalized_y_values) + 0.05)
    else:
        ax.plot(x_values, y_values, marker="o", label=column)
        ax.set_ylim(min(y_values) - 0.05, max(y_values) + 0.05)

    ax.set_ylabel("Metric Value", fontsize=8)
    ax.set_xticks([])  # remove tick labels to not make plot crowded
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True)
    ax.legend(fontsize=10, loc="upper left")

# Hide unused subplots (if any)
for j in range(len(columns_to_plot), n_rows * n_cols):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(seperate_metrics_path)
plt.close()
