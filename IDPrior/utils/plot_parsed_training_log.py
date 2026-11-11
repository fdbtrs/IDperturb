import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(file_path):
    # Read the entire file content into a string
    with open(file_path, "r") as f:
        log_data = f.read()

    # ---------------------------------------------------------
    # 1. Define Regex Patterns
    # ---------------------------------------------------------
    # Pattern for Metrics: Handles float, scientific notation, and negatives
    # e.g. RANK[0] --> FID: 0.62..., Uniqueness: 0.00..., Min/Max Similarity: (-0.2, 0.2)
    metric_pattern = re.compile(
        r"RANK\[(\d+)\] --> FID: ([\d\.eE\-]+), "
        r"Uniqueness: ([\d\.eE\-]+), "
        r"Min/Max Similarity: \(([\-\d\.eE]+), ([\-\d\.eE]+)\), "
        r"Same ID sim: ([\d\.eE\-]+), "
        r"Same ID min/max sim: \(([\-\d\.eE]+), ([\-\d\.eE]+)\)"
    )

    # Pattern for Loss: Handles the epoch and loss value
    # e.g. RANK[0] Epoch 64: average loss 0.0022...
    loss_pattern = re.compile(r"RANK\[(\d+)\] Epoch (\d+): average loss ([\d\.eE\-]+)")

    # ---------------------------------------------------------
    # 2. Extract Events
    # ---------------------------------------------------------
    events = []

    # Find all Metric events
    for m in metric_pattern.finditer(log_data):
        events.append(
            {
                "type": "metric",
                "pos": m.start(),  # Byte position to sort events chronologically
                "rank": int(m.group(1)),
                "fid": float(m.group(2)),
                "uniqueness": float(m.group(3)),
                "min_sim": float(m.group(4)),
                "max_sim": float(m.group(5)),
                "same_id_avg": float(m.group(6)),
                "same_id_min": float(m.group(7)),
                "same_id_max": float(m.group(8)),
            }
        )

    # Find all Loss events
    for m in loss_pattern.finditer(log_data):
        events.append(
            {
                "type": "loss",
                "pos": m.start(),
                "rank": int(m.group(1)),
                "epoch": int(m.group(2)),
                "loss": float(m.group(3)),
            }
        )

    # Sort all events by their position in the file
    events.sort(key=lambda x: x["pos"])

    # ---------------------------------------------------------
    # 3. Correlate Metrics with Epochs
    # ---------------------------------------------------------
    parsed_rows = []
    metric_buffer = []

    for event in events:
        if event["type"] == "metric":
            # Buffer metrics until we encounter a Loss event that tells us the Epoch
            metric_buffer.append(event)
        elif event["type"] == "loss":
            current_epoch = event["epoch"]

            # Add the loss entry
            parsed_rows.append(
                {
                    "epoch": current_epoch,
                    "rank": event["rank"],
                    "loss": event["loss"],
                    "fid": np.nan,
                    "uniqueness": np.nan,
                    "min_sim": np.nan,
                    "max_sim": np.nan,
                    "same_id_avg": np.nan,
                    "same_id_min": np.nan,
                    "same_id_max": np.nan,
                }
            )

            # Assign the current epoch to all buffered metrics and flush
            # (Assuming metrics correspond to the nearest subsequent epoch log)
            if metric_buffer:
                for m in metric_buffer:
                    parsed_rows.append(
                        {
                            "epoch": current_epoch,
                            "rank": m["rank"],
                            "loss": np.nan,
                            "fid": m["fid"],
                            "uniqueness": m["uniqueness"],
                            "min_sim": m["min_sim"],
                            "max_sim": m["max_sim"],
                            "same_id_avg": m["same_id_avg"],
                            "same_id_min": m["same_id_min"],
                            "same_id_max": m["same_id_max"],
                        }
                    )
                metric_buffer = []

    # Create DataFrame
    df = pd.DataFrame(parsed_rows)

    # Merge rows: Group by Epoch and Rank to combine Loss and Metric columns
    df_merged = df.groupby(["epoch", "rank"]).first().reset_index()

    return df_merged


def plot_metrics(df, out_path):
    # Calculate averages per epoch for plotting
    df_avg = df.groupby("epoch").mean().reset_index()

    num_gpus = df["rank"].nunique()
    num_epochs = df["epoch"].nunique()

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.ravel()

    # 1. FID
    axs[0].plot(df_avg["epoch"], df_avg["fid"], color="tab:blue")
    axs[0].set_title("Average FID")
    axs[0].set_ylabel("FID")
    axs[0].grid(True, alpha=0.3)

    # 2. Uniqueness
    axs[1].plot(
        df_avg["epoch"],
        df_avg["uniqueness"],
        label="inter-class Avg Sim",
        color="tab:orange",
    )
    axs[1].set_title("Uniqueness")
    axs[1].set_ylabel("inter-class Avg Sim")
    axs[1].grid(True, alpha=0.3)

    # 3. Similarities
    axs[2].plot(
        df_avg["epoch"],
        df_avg["same_id_avg"],
        label="intra-class Avg Sim",
        color="tab:orange",
    )
    axs[2].plot(
        df_avg["epoch"],
        df_avg["same_id_max"],
        label="intra-class Max Sim",
        color="tab:blue",
    )
    axs[2].plot(
        df_avg["epoch"],
        df_avg["same_id_min"],
        label="intra-class Min Sim",
        color="tab:pink",
    )
    axs[2].plot(
        df_avg["epoch"], df_avg["max_sim"], label="inter-class Max Sim", color="tab:red"
    )
    axs[2].plot(
        df_avg["epoch"],
        df_avg["min_sim"],
        label="inter-class Min Sim",
        color="tab:green",
    )
    axs[2].set_title("Similarities")
    axs[2].set_ylabel("Similarity")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    # 4. Loss
    axs[3].plot(df_avg["epoch"], df_avg["loss"], color="tab:purple")
    axs[3].set_title("Average Training Loss")
    axs[3].set_ylabel("Loss")
    axs[3].grid(True, alpha=0.3)

    for ax in axs:
        ax.set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(out_path)


log_path = "RUN_FOLDER/training.log"
out_path = "RUN_FOLDER/training.png"
df = parse_log_file(log_path)
plot_metrics(df, out_path)
