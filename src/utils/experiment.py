import json
from datetime import datetime

import torch

from configs.paths import CHECKPOINT_DIR
from configs.training import (
    RANDOM_SEED,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    MSE_WEIGHT,
    SPEC_WEIGHT
)


def save_experiment_config(
    train_loader,
    val_loader,
    test_loader,
    output_dir=None,
):
    """
    Save experiment configuration into a JSON file.

    Args:
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        test_loader: DataLoader for the test set.
        output_dir: Directory used to save the configuration file.
    """
    if output_dir is None:
        output_dir = CHECKPOINT_DIR / "full_model"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    experiment_config = {
        "experiment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "random_seed": RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "mse_weight": MSE_WEIGHT,
        "spec_weight": SPEC_WEIGHT,
        "num_train_samples": len(train_loader.dataset),
        "num_val_samples": len(val_loader.dataset),
        "num_test_samples": len(test_loader.dataset),
        "checkpoint_dir": str(output_dir),
    }

    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump(
            experiment_config,
            f,
            indent=4
        )

    print(f"Saved: {output_dir / 'experiment_config.json'}")