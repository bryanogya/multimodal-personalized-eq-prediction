import glob
import os
import random

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from configs.paths import PROJECT_ROOT, RAW_AUDIO_DIR, PROC_AUDIO_DIR, METADATA_DIR
from configs.audio import *


RANDOM_SEED = 42


def select_audio_files():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    selected_files = []
    genres = sorted(os.listdir(RAW_AUDIO_DIR))

    for genre in genres:
        genre_path = os.path.join(RAW_AUDIO_DIR, genre, "*.wav")
        files = sorted(glob.glob(genre_path))
        sampled_files = random.sample(files, min(FILES_PER_GENRE, len(files)))
        selected_files.extend(sampled_files)

    return selected_files


def extract_segments(selected_files):
    segment_samples = SAMPLE_RATE * SEGMENT_DURATION
    hop_samples = int(segment_samples * (1 - OVERLAP))

    metadata_audio = []
    sample_index = 0

    for audio_path in tqdm(selected_files, desc="Processing Audio"):
        try:
            y, sr = sf.read(audio_path)

            for start in range(0, len(y) - segment_samples + 1, hop_samples):
                segment = y[start:start + segment_samples]

                mel = librosa.feature.melspectrogram(
                    y=segment,
                    sr=sr,
                    n_mels=N_MELS,
                    n_fft=N_FFT,
                    hop_length=HOP_LENGTH
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)
                mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)

                save_name = f"sample_{sample_index:04d}.npy"
                save_path = PROC_AUDIO_DIR / save_name
                np.save(save_path, mel_db.astype(np.float32))

                metadata_audio.append({
                    "sample_id": sample_index,
                    "audio_path": save_path.relative_to(PROJECT_ROOT),
                    "original_file": os.path.basename(audio_path),
                    "genre": os.path.basename(os.path.dirname(audio_path)),
                    "start_sec": start / SAMPLE_RATE,
                    "shape": str(mel_db.shape)
                })
                sample_index += 1

        except Exception as exc:
            print(f"Failed to process {audio_path}: {exc}")

    return metadata_audio, sample_index


def main():
    PROC_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    selected_files = select_audio_files()
    print(f"Total selected audio files: {len(selected_files)}")

    metadata_audio, sample_index = extract_segments(selected_files)

    meta_audio_path = METADATA_DIR / "audio_metadata.csv"
    pd.DataFrame(metadata_audio).to_csv(meta_audio_path, index=False)

    print(f"\nTotal generated samples: {sample_index}")
    print(f"Saved to: {PROC_AUDIO_DIR}")
    print(f"Metadata saved to: {meta_audio_path}")


if __name__ == "__main__":
    main()
