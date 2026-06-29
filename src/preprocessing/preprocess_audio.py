import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
import glob
import random

# Path File & Folder
from configs.paths import PROJECT_ROOT, RAW_AUDIO_DIR, PROC_AUDIO_DIR, METADATA_DIR
from configs.audio import *

# Random Prep
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Kalkulasi sample segmentasi
segment_samples = SAMPLE_RATE * SEGMENT_DURATION
hop_samples = int(segment_samples * (1 - OVERLAP))

# SELEKSI SUBSET DATASET
selected_files = []
genres = sorted(os.listdir(RAW_AUDIO_DIR))

for genre in genres:
    genre_path = os.path.join(RAW_AUDIO_DIR, genre, "*.wav")
    files = sorted(glob.glob(genre_path))  # sorted agar urutan konsisten
    sampled_files = random.sample(files, min(FILES_PER_GENRE, len(files)))
    selected_files.extend(sampled_files)

print(f"Total selected audio files: {len(selected_files)}")

# PEMROSESAN & EKSTRAKSI FITUR
metadata_audio = []
sample_index = 0
for audio_path in tqdm(selected_files, desc="Processing Audio"):
    try:
        y, sr = sf.read(audio_path)

        for start in range(0, len(y) - segment_samples + 1, hop_samples):
            segment = y[start : start + segment_samples]

            # Ekstraksi Mel-Spectrogram
            mel = librosa.feature.melspectrogram(
                y=segment, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)  # Log-scale

            # Z-Score Normalization 
            mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)

            # Simpan fitur ke format .npy
            save_name = f"sample_{sample_index:04d}.npy"
            save_path = PROC_AUDIO_DIR / save_name
            PROC_AUDIO_DIR.parent.mkdir(exist_ok=True)
            np.save(save_path, mel_db)

            metadata_audio.append({
                "sample_id": sample_index,
                "audio_path": save_path.relative_to(PROJECT_ROOT),
                "original_file": os.path.basename(audio_path),
                "genre": os.path.basename(os.path.dirname(audio_path)),
                "start_sec": start / SAMPLE_RATE,
                "shape": str(mel_db.shape)
            })
            sample_index += 1

    except Exception as e:
        import traceback
        traceback.print_exc()
        
# SIMPAN METADATA
meta_df = pd.DataFrame(metadata_audio)
meta_audio_path = METADATA_DIR / "audio_metadata.csv"
meta_df.to_csv(meta_audio_path, index=False)

# INFORMASI FINAL
print(f"\nTotal generated samples: {sample_index}")
print(f"Saved to: {PROC_AUDIO_DIR}")
print(f"Metadata saved to: {meta_audio_path}")