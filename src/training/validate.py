import json

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from configs.paths import CHECKPOINT_DIR
from configs.eq import EQ_FREQS
from configs.training import MSE_WEIGHT, SPEC_WEIGHT

from src.dataset.data_loader import create_dataloaders
from src.models.multimodal_model import MultimodalEQModel
from src.training.losses import SpectralLoss
from src.utils.interpolate_eq import LogFrequencyInterpolator
from src.evaluation.test import *


def main():
    run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== MODEL VALIDATION / TESTING ===")
    print(f"Device: {run_device}")

    _, _, test_loader = create_dataloaders()

    model = MultimodalEQModel().to(run_device)

    checkpoint_path = (
        CHECKPOINT_DIR /
        "full_model" /
        "best_model.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=run_device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Best Val Loss: {checkpoint['best_val_loss']:.4f}")

    interpolator = LogFrequencyInterpolator().to(run_device)

    spectral_criterion = SpectralLoss().to(run_device)

    results = predict_test_set(
        model=model,
        loader=test_loader,
        interpolator=interpolator,
        run_device=run_device,
    )

    metrics = compute_test_metrics(
        results=results,
        spectral_criterion=spectral_criterion,
        run_device=run_device,
    )

    save_test_results(results)

    save_test_metrics(metrics)

    print_metrics(metrics)


if __name__ == "__main__":
    main()