import numpy as np

NUM_BANDS = 128

MIN_FREQ = 20
MAX_FREQ = 20000

TARGET_FREQ = np.logspace(
    np.log10(MIN_FREQ),
    np.log10(MAX_FREQ),
    NUM_BANDS
)