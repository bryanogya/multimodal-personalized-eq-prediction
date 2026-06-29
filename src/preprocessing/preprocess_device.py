import glob
import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
import re

# Path File & Folder
from configs.paths import RAW_DEVICE_DIR, PROC_DEVICE_DIR, METADATA_DIR
from configs.frequencies import TARGET_FREQ

PROC_DEVICE_DIR.parent.mkdir(exist_ok=True)

# PENGELOMPOKAN FILE BERDASARKAN PERANGKAT
txt_files = glob.glob(os.path.join(RAW_DEVICE_DIR, "*.txt"))
device_groups = {}

for path in txt_files:
    filename = os.path.basename(path)

    # Hapus indeks seperti [1], [2], dst., lalu hapus ekstensi .txt
    device_name = re.sub(r"\s*\[\d+\]", "", filename)
    device_name = device_name.replace(".txt", "")

    if device_name not in device_groups:
        device_groups[device_name] = []

    device_groups[device_name].append(path)

# PEMROSESAN SETIAP PERANGKAT
metadata_device = []

for device_name, paths in device_groups.items():
    print(f"Processing: {device_name}")
    responses = []
    freq = None

    for txt_path in paths:
        try:
            # Load data tabular (.txt)
            df = pd.read_csv(txt_path, sep=r'\s+', header=None)
            freq = df.iloc[:, 0].values
            response = df.iloc[:, 1].values
            responses.append(response)
        except Exception as e:
            print(f"  -> Error loading {txt_path}: {e}")

    # Lewati perangkat jika tidak ada respons yang berhasil dimuat
    if not responses:
        continue

    # Rata-rata semua respons pengukuran untuk perangkat ini
    responses = np.array(responses)
    avg_response = np.mean(responses, axis=0)

    # Normalisasi amplitudo ke titik 1kHz (menjadikan respons di 1kHz = 0 dB)
    idx_1k = np.argmin(np.abs(freq - 1000))
    avg_response = avg_response - avg_response[idx_1k]

    # Interpolasi ke 128 titik frekuensi target
    interp_func = interp1d(
        freq,
        avg_response,
        kind="linear",
        bounds_error=False,
        fill_value=(avg_response[0], avg_response[-1])
    )
    interp_response = interp_func(TARGET_FREQ)

    # Smoothing (mengurangi noise grafik)
    smooth_response = uniform_filter1d(interp_response, size=3)

    # Clipping (membatasi nilai maksimum/minimum agar tidak ada anomali ekstrim)
    smooth_response = np.clip(smooth_response, -20, 20)

    # Simpan hasil preprocessing perangkat ke format .npy
    save_name = f"{device_name}.npy"
    save_path = PROC_DEVICE_DIR / save_name
    np.save(save_path, smooth_response)

    # Rekap metadata
    metadata_device.append({
        "device_name": device_name,
        "num_measurements": len(paths),
        "output_path": str(save_path),
        "min_db": np.min(smooth_response),
        "max_db": np.max(smooth_response),
        "mean_db": np.mean(smooth_response)
    })

# SIMPAN METADATA
meta_df = pd.DataFrame(metadata_device)
meta_device_path = METADATA_DIR / "device_metadata.csv"
meta_df.to_csv(meta_device_path, index=False)

print("\nDone preprocessing device dataset.")