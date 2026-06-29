import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d

from configs.paths import RAW_DEVICE_DIR, PROC_DEVICE_DIR, METADATA_DIR
from configs.frequencies import TARGET_FREQ


def group_device_files():
    txt_files = glob.glob(os.path.join(RAW_DEVICE_DIR, "*.txt"))
    device_groups = {}

    for path in txt_files:
        filename = os.path.basename(path)
        device_name = re.sub(r"\s*\[\d+\]", "", filename)
        device_name = device_name.replace(".txt", "")
        device_groups.setdefault(device_name, []).append(path)

    return device_groups


def preprocess_device_group(device_name, paths):
    print(f"Processing: {device_name}")
    responses = []
    freq = None

    for txt_path in paths:
        try:
            df = pd.read_csv(txt_path, sep=r"\s+", header=None)
            freq = df.iloc[:, 0].values
            response = df.iloc[:, 1].values
            responses.append(response)
        except Exception as exc:
            print(f"  -> Error loading {txt_path}: {exc}")

    if not responses:
        return None

    responses = np.array(responses)
    avg_response = np.mean(responses, axis=0)

    idx_1k = np.argmin(np.abs(freq - 1000))
    avg_response = avg_response - avg_response[idx_1k]

    interp_func = interp1d(
        freq,
        avg_response,
        kind="linear",
        bounds_error=False,
        fill_value=(avg_response[0], avg_response[-1])
    )
    interp_response = interp_func(TARGET_FREQ)
    smooth_response = uniform_filter1d(interp_response, size=3)
    smooth_response = np.clip(smooth_response, -20, 20)

    save_path = PROC_DEVICE_DIR / f"{device_name}.npy"
    np.save(save_path, smooth_response.astype(np.float32))

    return {
        "device_name": device_name,
        "num_measurements": len(paths),
        "output_path": str(save_path),
        "min_db": float(np.min(smooth_response)),
        "max_db": float(np.max(smooth_response)),
        "mean_db": float(np.mean(smooth_response))
    }


def main():
    PROC_DEVICE_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    metadata_device = []
    for device_name, paths in group_device_files().items():
        metadata = preprocess_device_group(device_name, paths)
        if metadata is not None:
            metadata_device.append(metadata)

    meta_device_path = METADATA_DIR / "device_metadata.csv"
    pd.DataFrame(metadata_device).to_csv(meta_device_path, index=False)

    print("\nDone preprocessing device dataset.")


if __name__ == "__main__":
    main()
