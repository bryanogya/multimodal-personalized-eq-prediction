import numpy as np
import pandas as pd

from configs.paths import RAW_HARMAN_FILE, PROC_HARMAN_FILE
from configs.frequencies import TARGET_FREQ

PROC_HARMAN_FILE.parent.mkdir(exist_ok=True)

data_harman = pd.read_csv(
    RAW_HARMAN_FILE,
    sep=r'\s+',
    header=None
)

data_harman.columns = [
    "frequency",
    "target"
]

harman_freqs = data_harman["frequency"].values
harman_target = data_harman["target"].values

idx_1k = np.argmin(
    np.abs(harman_freqs - 1000)
)

harman_target = (
    harman_target -
    harman_target[idx_1k]
)

harman_interp = np.interp(
    TARGET_FREQ,
    harman_freqs,
    harman_target
)

np.save(
    PROC_HARMAN_FILE,
    harman_interp.astype(np.float32)
)

print(
    f"Harman target saved: {PROC_HARMAN_FILE}"
)