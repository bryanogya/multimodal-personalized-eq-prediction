from pathlib import Path
import pandas as pd
import numpy as np

from configs.paths import PROJECT_ROOT, METADATA_DIR

metadata_path = METADATA_DIR / "audio_metadata.csv"

df = pd.read_csv(metadata_path)

print(df.head())
print("Total rows:", len(df))

first_audio_path = Path(df.loc[0, "audio_path"])

if not first_audio_path.is_absolute():
    first_audio_path = PROJECT_ROOT / first_audio_path

    print("First audio path:", first_audio_path)
    print("Exists:", first_audio_path.exists())

    audio = np.load(first_audio_path)
    print("Audio shape:", audio.shape)