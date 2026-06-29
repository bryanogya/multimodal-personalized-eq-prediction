import numpy as np

# User Preference Influence
PREFERENCE_BASS_GAIN = 8.0
PREFERENCE_MID_GAIN = 4.0
PREFERENCE_TREBLE_GAIN = 8.0

# Audio Character Influence
AUDIO_BASS_ADJUST = 2.0
AUDIO_TREBLE_ADJUST = 2.0

# EQ Output Constraints
EQ_MIN_GAIN = -12.0
EQ_MAX_GAIN = 12.0

# Frequency Regions
BASS_CUTOFF = 250
MID_CUTOFF = 4000

# Audio Character Thresholds
BASS_HEAVY_THRESHOLD = 0.40
BASS_LIGHT_THRESHOLD = 0.25

TREBLE_HEAVY_THRESHOLD = 0.40
TREBLE_LIGHT_THRESHOLD = 0.20

EQ_FREQS = np.array([               # Center frequencies of EQ bands
    60,
    150,
    400,
    1000,
    2400,
    15000
])