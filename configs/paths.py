from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]                                  # Root directory of the project

DATA_DIR = PROJECT_ROOT / "data"                                                    # Main data directory
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "outputs"                                               # Generated outputs
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"                                       # Saved model checkpoints and metrics

# DIR
RAW_DIR = DATA_DIR / "raw"                                                          # Raw dataset directory
PROCESSED_DIR = DATA_DIR / "processed"                                              # Processed data directory
METADATA_DIR = DATA_DIR / "metadata"                                                # Metadata directory
SPLITS_DIR = DATA_DIR / "splits"                                                    # Train, validation, and test split metadata

# Audio
RAW_AUDIO_DIR = RAW_DIR / "audio"
PROC_AUDIO_DIR = PROCESSED_DIR / "audio"
PROC_AUDIO_PROFILE_DIR = (PROCESSED_DIR / "audio_profiles")
AUDIO_METADATA_FILE = (METADATA_DIR / "audio_metadata.csv")

# Device
RAW_DEVICE_DIR = RAW_DIR / "device"
PROC_DEVICE_DIR = PROCESSED_DIR / "device"

# Harman
RAW_HARMAN_FILE = RAW_DIR / "harman" / "Harman_Target.txt"
PROC_HARMAN_FILE = PROCESSED_DIR / "harman" / "harman_target.npy"

# Preference
PREFERENCE_FILE = PROCESSED_DIR / "preferences" / "user_preferences.npy"

# Model
BEST_MODEL_FILE = CHECKPOINT_DIR / "full_model" / "best_model.pth"

# Output
FIGURE_DIR = OUTPUT_DIR / "figures"                                         # Generated figures
EVAL_DIR = OUTPUT_DIR / "evaluation"
BASELINE_DIR = OUTPUT_DIR / "baseline"
ABLATION_DIR = OUTPUT_DIR / "ablation"

if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATA_DIR:", DATA_DIR)
    print("RAW_DATA_DIR:", RAW_DIR)
    print("PROCESSED_DATA_DIR:", PROCESSED_DIR)
    print("CHECKPOINT_DIR:", CHECKPOINT_DIR)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("FIGURE_DIR:", FIGURE_DIR)