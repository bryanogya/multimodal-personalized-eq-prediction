NUM_BANDS = 128
NUM_EQ_BANDS = 6

# Audio
AUDIO_EMBED_DIM = 256           # Audio encoder output dimension
AUDIO_PATCH_SIZE = 16
AUDIO_NUM_HEADS = 8
AUDIO_NUM_LAYERS = 4
AUDIO_DROPOUT = 0.1

# Device
DEVICE_EMBED_DIM = 128          # Device response encoder output dimension
DEVICE_INPUT_DIM = 128
DEVICE_HIDDEN_DIM = 256
DEVICE_DROPOUT = 0.3

# Preference
PREFERENCE_EMBED_DIM = 128      # Preference encoder output dimension
PREFERENCE_INPUT_DIM = 3
PREFERENCE_HIDDEN_DIM = 64
PREFERENCE_DROPOUT = 0.3

# Fusion
FUSION_DROPOUT = 0.3            # Dropout probability for regularization
FUSION_HIDDEN_DIM = 256         # Hidden layer dimension after feature fusion