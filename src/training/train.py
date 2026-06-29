import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from configs.paths import CHECKPOINT_DIR
from configs.training import *

from src.dataset.data_loader import create_dataloaders
from src.models.multimodal_model import MultimodalEQModel
from src.training.losses import SpectralLoss
from src.utils.interpolate_eq import LogFrequencyInterpolator
from src.utils.experiment import save_experiment_config


def set_seed(seed):
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed used to make experiments more reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model,
    train_loader,
    mse_criterion,
    spectral_criterion,
    interpolator,
    optimizer,
    run_device,
):
    """
    Train the model for one epoch using EQ and spectral losses.

    Args:
        model: Multimodal EQ prediction model.
        train_loader: DataLoader for the training set.
        mse_criterion: Loss function for EQ band prediction.
        spectral_criterion: Loss function for frequency response matching.
        interpolator: Module for converting EQ bands into response curves.
        optimizer: Optimizer used to update model parameters.
        run_device: Device used for computation.

    Returns:
        Dictionary containing average training loss and metrics.
    """
    model.train()

    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_spec = 0.0
    total_sc = 0.0
    total_lsd = 0.0
    n_batches = 0

    pbar = tqdm(
        train_loader,
        desc="Training",
        leave=False
    )

    for batch in pbar:
        audio = batch["audio"].to(run_device)
        device_fr = batch["device"].to(run_device)
        preference = batch["preference"].to(run_device)

        target_eq = batch["target_eq"].to(run_device)
        personalized_target = batch["personalized_target"].to(run_device)

        optimizer.zero_grad()

        pred_eq = model(
            audio=audio,
            device=device_fr,
            preference=preference
        )

        pred_curve = interpolator(pred_eq)
        predicted_response  = device_fr + pred_curve
        target_response = personalized_target 
        
        mse_loss = mse_criterion(pred_eq, target_eq)
        spec_loss = spectral_criterion(predicted_response, target_response)
        sc, lsd = spectral_criterion.components(predicted_response, target_response)

        loss = (MSE_WEIGHT * mse_loss + SPEC_WEIGHT * spec_loss)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRAD_CLIP
        )

        optimizer.step()

        mae = torch.mean(torch.abs(pred_eq - target_eq))

        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_spec += spec_loss.item()
        total_sc +=  sc.item()
        total_lsd += lsd.item()
        total_mae += mae.item()
        n_batches += 1

        pbar.set_postfix(
            loss=f"{total_loss / n_batches:.4f}",
            mae=f"{total_mae / n_batches:.4f}"
        )

    return {
        "loss": total_loss / n_batches,
        "mse": total_mse / n_batches,
        "mae": total_mae / n_batches,
        "spec": total_spec / n_batches,
        "sc": total_sc / n_batches,
        "lsd": total_lsd / n_batches
    }


def validate_one_epoch(
    model,
    val_loader,
    mse_criterion,
    spectral_criterion,
    interpolator,
    run_device,
):
    """
    Evaluate the model for one epoch on the validation set.

    Args:
        model: Multimodal EQ prediction model.
        val_loader: DataLoader for the validation set.
        mse_criterion: Loss function for EQ band prediction.
        spectral_criterion: Loss function for frequency response matching.
        interpolator: Module for converting EQ bands into response curves.
        run_device: Device used for computation.

    Returns:
        Dictionary containing average validation loss, metrics,
        and per-band MAE.
    """
    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_spec = 0.0
    total_sc = 0.0
    total_lsd = 0.0
    n_batches = 0

    per_band_mae_list = []

    pbar = tqdm(
        val_loader,
        desc="Validating",
        leave=False
    )

    with torch.no_grad():
        for batch in pbar:
            audio = batch["audio"].to(run_device)
            device_fr = batch["device"].to(run_device)
            preference = batch["preference"].to(run_device)

            target_eq = batch["target_eq"].to(run_device)
            personalized_target = batch["personalized_target"].to(run_device)

            pred_eq = model(
                audio=audio,
                device=device_fr,
                preference=preference
            )

            pred_curve = interpolator(pred_eq)
            predicted_response = device_fr + pred_curve
            target_response = personalized_target
            mse_loss = mse_criterion(pred_eq, target_eq)
            spec_loss = spectral_criterion(predicted_response, target_response)
            sc, lsd = spectral_criterion.components(predicted_response, target_response)
            loss = (MSE_WEIGHT * mse_loss + SPEC_WEIGHT * spec_loss)
            mae = torch.mean(torch.abs(pred_eq - target_eq))

            batch_per_band_mae = torch.mean(torch.abs(pred_eq - target_eq), dim=0)
            per_band_mae_list.append(batch_per_band_mae.detach().cpu())

            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_spec += spec_loss.item()
            total_sc += sc.item()
            total_lsd += lsd.item()
            total_mae += mae.item()
            n_batches += 1

            pbar.set_postfix(
                val_loss=f"{total_loss / n_batches:.4f}",
                val_mae=f"{total_mae / n_batches:.4f}"
            )

    per_band_mae = torch.stack(
        per_band_mae_list
    ).mean(dim=0).numpy()

    return {
        "loss": total_loss / n_batches,
        "mse": total_mse / n_batches,
        "mae": total_mae / n_batches,
        "spec": total_spec / n_batches,
        "sc": total_sc / n_batches,
        "lsd": total_lsd / n_batches,
        "per_band_mae": per_band_mae,
    }


def main():
    """
    Run the full training pipeline.

    This function sets the seed, prepares DataLoaders, initializes the model,
    optimizer, scheduler, losses, runs training and validation, saves the best
    checkpoint, and stores the training history.
    """
    set_seed(RANDOM_SEED)

    run_device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=== TRAINING SETUP ===")
    print(f"Device       : {run_device}")
    print(f"Batch Size   : {BATCH_SIZE}")
    print(f"Epochs       : {NUM_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Weight Decay : {WEIGHT_DECAY}")
    print(f"Loss         : {MSE_WEIGHT} MSE + {SPEC_WEIGHT} Spectral")

    train_loader, val_loader, test_loader = create_dataloaders()
    
    save_experiment_config(
        train_loader,
        val_loader,
        test_loader
    )

    model = MultimodalEQModel().to(run_device)

    mse_criterion = nn.MSELoss()
    spectral_criterion = SpectralLoss().to(run_device)
    interpolator = LogFrequencyInterpolator().to(run_device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=7
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mse": [],
        "val_mse": [],
        "train_mae": [],
        "val_mae": [],
        "train_sc":[],
        "val_sc": [],
        "train_lsd": [],
        "val_lsd": [],
        "train_spec": [],
        "val_spec": [],
        "val_per_band_mae": [],
        "learning_rate": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = None
    patience_counter = 0

    checkpoint_dir = CHECKPOINT_DIR / "full_model"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting training...\n")

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            mse_criterion=mse_criterion,
            spectral_criterion=spectral_criterion,
            interpolator=interpolator,
            optimizer=optimizer,
            run_device=run_device,
        )

        val_metrics = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            mse_criterion=mse_criterion,
            spectral_criterion=spectral_criterion,
            interpolator=interpolator,
            run_device=run_device,
        )

        scheduler.step(val_metrics["loss"])

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])

        history["train_mse"].append(train_metrics["mse"])
        history["val_mse"].append(val_metrics["mse"])

        history["train_spec"].append(train_metrics["spec"])
        history["val_spec"].append(val_metrics["spec"])
        
        history["train_sc"].append(train_metrics["sc"])
        history["val_sc"].append(val_metrics["sc"])

        history["train_lsd"].append(train_metrics["lsd"])
        history["val_lsd"].append(val_metrics["lsd"])

        history["train_mae"].append(train_metrics["mae"])
        history["val_mae"].append(val_metrics["mae"])

        history["val_per_band_mae"].append(val_metrics["per_band_mae"].tolist())

        history["learning_rate"].append(current_lr)

        print(
            f"Train | "
            f"Loss: {train_metrics['loss']:.4f} | "
            f"MSE: {train_metrics['mse']:.4f} | "
            f"MAE: {train_metrics['mae']:.4f} |"
            f"Spec: {train_metrics['spec']:.4f} | "
            f"SC: {train_metrics['sc']:.4f} | "
            f"LSD: {train_metrics['lsd']:.4f}"
        )

        print(
            f"Val   | "
            f"Loss: {val_metrics['loss']:.4f} | "
            f"MSE: {val_metrics['mse']:.4f} | "
            f"MAE: {val_metrics['mae']:.4f} |"
            f"Spec: {val_metrics['spec']:.4f} | "
            f"SC: {val_metrics['sc']:.4f} | "
            f"LSD: {val_metrics['lsd']:.4f}"
        )

        print(f"LR    : {current_lr:.2e}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(
                model.state_dict()
            )
            patience_counter = 0

            checkpoint = {
                "epoch": epoch + 1,
                "best_epoch": best_epoch,
                "model_state_dict": best_model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "best_val_loss": best_val_loss,
                "config": {
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "mse_weight": MSE_WEIGHT,
                    "spec_weight": SPEC_WEIGHT,
                    "grad_clip": GRAD_CLIP,
                }
            }

            torch.save(checkpoint, checkpoint_dir / "best_model.pth")
            print("New best model saved.")

        else:
            patience_counter += 1

            print(
                f"No improvement "
                f"({patience_counter}/{EARLY_STOPPING_PATIENCE})"
            )

        print("-" * 50)
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss

    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(
            history,
            f,
            indent=4
        )

    print("\nTraining complete.")
    print(f"Best Epoch   : {best_epoch}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {checkpoint_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()