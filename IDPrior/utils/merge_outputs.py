import numpy as np
import glob
import re
import os


def merge_epoch_rank_embeddings(path):
    # find all .npy files
    files = glob.glob(os.path.join(path, "generated_embeddings_epoch*_rank*.npy"))

    # extract epoch and rank
    pattern = re.compile(r"epoch(\d+)_rank(\d+)")

    # group files by epoch
    epoch_files = {}
    for f in files:
        match = pattern.search(f)
        if match:
            epoch = int(match.group(1))
            rank = int(match.group(2))
            epoch_files.setdefault(epoch, []).append((rank, f))

    # merge and save per epoch
    for epoch, file_list in epoch_files.items():
        file_list.sort(key=lambda x: x[0])

        arrays = [np.load(f) for _, f in file_list]
        merged = np.concatenate(arrays, axis=0)

        save_path = os.path.join(path, f"merged_embeddings_epoch{epoch}.npy")
        np.save(save_path, merged)

        for file in file_list:
            os.remove(file[1])

        print(
            f"Epoch {epoch}: merged {len(file_list)} files into {merged.shape}, saved to {save_path}"
        )
