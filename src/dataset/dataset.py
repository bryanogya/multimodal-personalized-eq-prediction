import random
import numpy as np
import torch
from torch.utils.data import Dataset

from configs.paths import PROJECT_ROOT, PROC_DEVICE_DIR, PROC_HARMAN_FILE, PREFERENCE_FILE
from configs.training import RANDOM_SEED
from src.utils.generate_target_eq import generate_target_eq


class AudioEQDataset(Dataset):
    """
    PyTorch Dataset for loading audio features, device responses,
    user preferences, and generated EQ targets.

    Each sample contains a mel-spectrogram, a randomly selected device
    frequency response, a randomly selected preference vector, and
    generated target EQ values.
    """
    def __init__(self, metadata_df, deterministic=False, seed=RANDOM_SEED):
        """
        Initialize the dataset from audio metadata and load shared target data.

        Args:
            metadata_df: DataFrame containing audio feature paths and metadata.
            deterministic: Whether to use fixed device/preference choices.
            seed: Seed used for deterministic device/preference assignment.
        """
        self.df = metadata_df.reset_index(drop=True)
        self.deterministic = deterministic
        
        self.harman_target = np.load(PROC_HARMAN_FILE).astype(np.float32)
        self.preferences = np.load(PREFERENCE_FILE).astype(np.float32)
        self.device_files = sorted(PROC_DEVICE_DIR.glob("*.npy"))
        self.device_cache = {}

        if not self.device_files:
            raise FileNotFoundError(f"No processed device files found in {PROC_DEVICE_DIR}")

        if len(self.preferences) == 0:
            raise ValueError(f"No preferences found in {PREFERENCE_FILE}")

        if self.deterministic:
            rng = np.random.default_rng(seed)
            self.device_indices = rng.integers(
                low=0,
                high=len(self.device_files),
                size=len(self.df)
            )
            self.preference_indices = rng.integers(
                low=0,
                high=len(self.preferences),
                size=len(self.df)
            )

    def __len__(self):
        """
        Return the number of audio samples in the dataset.

        Returns:
            Number of rows in the metadata DataFrame.
        """
        return len(self.df)

    def load_device(self, device_path):
        """
        Load a device frequency response and cache it in memory.

        Args:
            device_path: Path to the processed device response file.

        Returns:
            Device frequency response as a float32 NumPy array.
        """
        if device_path not in self.device_cache:
            self.device_cache[device_path] = np.load(device_path).astype(np.float32)
            
        return self.device_cache[device_path]

    def __getitem__(self, idx):
        """
        Load one dataset sample and generate its target EQ data.

        Args:
            idx: Index of the sample in the metadata DataFrame.

        Returns:
            Dictionary containing audio, device response, preference vector,
            target EQ, target curve, raw EQ curve, and personalized target.
        """
        row = self.df.iloc[idx]
        
        # AUDIO
        mel = np.load(PROJECT_ROOT / row["audio_path"]).astype(np.float32)        
        audio_tensor = torch.tensor(
            mel,
            dtype=torch.float32
        ).unsqueeze(0)

        # DEVICE
        if self.deterministic:
            device_path = self.device_files[self.device_indices[idx]]
        else:
            device_path = random.choice(self.device_files)

        device_response = self.load_device(device_path)
        device_tensor = torch.tensor(
            device_response,
            dtype=torch.float32
        )

        # PREFERENCE
        if self.deterministic:
            pref_idx = self.preference_indices[idx]
        else:
            pref_idx = random.randint(0, len(self.preferences) - 1)

        preference = self.preferences[pref_idx].astype(np.float32)
        preference_tensor = torch.tensor(
            preference,
            dtype=torch.float32
        )

        # TARGET EQ
        targets = generate_target_eq(
            mel_spectrogram=mel,
            device_response=device_response,
            preference=preference,
            harman_target=self.harman_target
        )
        target_eq = torch.tensor(targets["target_eq"], dtype=torch.float32)
        target_curve = torch.tensor(targets["target_curve"], dtype=torch.float32)
        target_curve_raw = torch.tensor(targets["eq_curve_raw"], dtype=torch.float32)
        
        # PERSONALIZED TARGET
        personalized_target_tensor = torch.tensor(targets["personalized_target"], dtype=torch.float32)

        return {
            "audio": audio_tensor,
            "device": device_tensor,
            "preference": preference_tensor,
            "target_eq": target_eq,
            "target_curve": target_curve,
            "target_curve_raw": target_curve_raw,
            "personalized_target": personalized_target_tensor
        }
        
