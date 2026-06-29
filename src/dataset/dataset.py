import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from configs.paths import PROJECT_ROOT, PROC_DEVICE_DIR, PROC_HARMAN_FILE, PREFERENCE_FILE
from src.utils.generate_target_eq import generate_target_eq


class AudioEQDataset(Dataset):
    """
    PyTorch Dataset for loading audio features, device responses,
    user preferences, and generated EQ targets.

    Each sample contains a mel-spectrogram, a randomly selected device
    frequency response, a randomly selected preference vector, and
    generated target EQ values.
    """
    def __init__(self,metadata_df):
        """
        Initialize the dataset from audio metadata and load shared target data.

        Args:
            metadata_df: DataFrame containing audio feature paths and metadata.
        """
        self.df = metadata_df.reset_index(drop=True)
        
        self.harman_target = np.load(PROC_HARMAN_FILE).astype(np.float32)
        self.preferences = np.load(PREFERENCE_FILE).astype(np.float32)
        self.device_files = list(PROC_DEVICE_DIR.glob("*.npy"))
        self.device_cache = {}

    def __len__(self):
        """
        Return the number of audio samples in the dataset.

        Returns:
            Number of rows in the metadata DataFrame.
        """
        return len(self.df)

    def load_device(self,device_path):
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

        # RANDOM DEVICE
        device_path = random.choice(self.device_files)
        device_response = self.load_device(device_path)
        device_tensor = torch.tensor(
            device_response,
            dtype=torch.float32
        )

        # RANDOM PREFERENCE
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
        
