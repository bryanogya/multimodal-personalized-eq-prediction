import copy
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from configs.paths import CHECKPOINT_DIR
from configs.training import *

from src.dataset.data_loader import create_dataloaders
from src.models.multimodal_model_aux import MultimodalEQModel
from src.training.losses import SpectralLoss
from src.utils.interpolate_eq import LogFrequencyInterpolator


def set_seed(seed):
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
    model.train()

    total_loss = 0.0
    total_mse = 0.0
    total_aux = 0.0
    total_spec = 0.0
    total_mae = 0.0

    total_sc_eq = 0.0
    total_lsd_eq = 0.0
    total_sc_aux = 0.0
    total_lsd_aux = 0.0

    n_batches = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)

    for batch in pbar:
        audio = batch["audio"].to(run_device)
        device_fr = batch["device"].to(run_device)
        preference = batch["preference"].to(run_device)

        target_eq = batch["target_eq"].to(run_device)
        personalized_target = batch["personalized_target"].to(run_device)

        optimizer.zero_grad()

        outputs = model(
            audio=audio,
            device=device_fr,
            preference=preference
        )

        pred_eq = outputs["eq"]
        pred_response_curve = outputs["response_curve"]

        pred_curve = interpolator(pred_eq)
        predicted_response_from_eq = device_fr + pred_curve

        mse_loss = mse_criterion(
            pred_eq,
            target_eq
        )

        aux_curve_loss = mse_criterion(
            pred_response_curve,
            personalized_target
        )

        spec_loss = spectral_criterion(
            pred_response_curve,
            personalized_target
        )

        sc_eq, lsd_eq = spectral_criterion.components(
            predicted_response_from_eq,
            personalized_target
        )

        sc_aux, lsd_aux = spectral_criterion.components(
            pred_response_curve,
            personalized_target
        )

        loss = (
            EQ_LOSS_WEIGHT * mse_loss
            +
            AUX_CURVE_WEIGHT * aux_curve_loss
            +
            SPEC_WEIGHT * spec_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRAD_CLIP
        )

        optimizer.step()

        mae = torch.mean(
            torch.abs(pred_eq - target_eq)
        )

        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_aux += aux_curve_loss.item()
        total_spec += spec_loss.item()
        total_mae += mae.item()

        total_sc_eq += sc_eq.item()
        total_lsd_eq += lsd_eq.item()
        total_sc_aux += sc_aux.item()
        total_lsd_aux += lsd_aux.item()

        n_batches += 1

        pbar.set_postfix(
            loss=f"{total_loss / n_batches:.4f}",
            mae=f"{total_mae / n_batches:.4f}"
        )

    return {
        "loss": total_loss / n_batches,
        "mse": total_mse / n_batches,
        "aux": total_aux / n_batches,
        "spec": total_spec / n_batches,
        "mae": total_mae / n_batches,
        "sc_eq": total_sc_eq / n_batches,
        "lsd_eq": total_lsd_eq / n_batches,
        "sc_aux": total_sc_aux / n_batches,
        "lsd_aux": total_lsd_aux / n_batches,
    }


def validate_one_epoch(
    model,
    val_loader,
    mse_criterion,
    spectral_criterion,
    interpolator,
    run_device,
):
    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_aux = 0.0
    total_spec = 0.0
    total_mae = 0.0

    total_sc_eq = 0.0
    total_lsd_eq = 0.0
    total_sc_aux = 0.0
    total_lsd_aux = 0.0

    n_batches = 0
    per_band_mae_list = []

    pbar = tqdm(val_loader, desc="Validating", leave=False)

    with torch.no_grad():
        for batch in pbar:
            audio = batch["audio"].to(run_device)
            device_fr = batch["device"].to(run_device)
            preference = batch["preference"].to(run_device)

            target_eq = batch["target_eq"].to(run_device)
            personalized_target = batch["personalized_target"].to(run_device)

            outputs = model(
                audio=audio,
                device=device_fr,
                preference=preference
            )

            pred_eq = outputs["eq"]
            pred_response_curve = outputs["response_curve"]

            pred_curve = interpolator(pred_eq)
            predicted_response_from_eq = device_fr + pred_curve

            mse_loss = mse_criterion(
                pred_eq,
                target_eq
            )

            aux_curve_loss = mse_criterion(
                pred_response_curve,
                personalized_target
            )

            spec_loss = spectral_criterion(
                pred_response_curve,
                personalized_target
            )

            sc_eq, lsd_eq = spectral_criterion.components(
                predicted_response_from_eq,
                personalized_target
            )

            sc_aux, lsd_aux = spectral_criterion.components(
                pred_response_curve,
                personalized_target
            )

            loss = (
                EQ_LOSS_WEIGHT * mse_loss
                +
                AUX_CURVE_WEIGHT * aux_curve_loss
                +
                SPEC_WEIGHT_2 * spec_loss
            )

            mae = torch.mean(
                torch.abs(pred_eq - target_eq)
            )

            batch_per_band_mae = torch.mean(
                torch.abs(pred_eq - target_eq),
                dim=0
            )

            per_band_mae_list.append(
                batch_per_band_mae.detach().cpu()
            )

            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_aux += aux_curve_loss.item()
            total_spec += spec_loss.item()
            total_mae += mae.item()

            total_sc_eq += sc_eq.item()
            total_lsd_eq += lsd_eq.item()
            total_sc_aux += sc_aux.item()
            total_lsd_aux += lsd_aux.item()

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
        "aux": total_aux / n_batches,
        "spec": total_spec / n_batches,
        "mae": total_mae / n_batches,
        "sc_eq": total_sc_eq / n_batches,
        "lsd_eq": total_lsd_eq / n_batches,
        "sc_aux": total_sc_aux / n_batches,
        "lsd_aux": total_lsd_aux / n_batches,
        "per_band_mae": per_band_mae,
    }


def main():
    set_seed(RANDOM_SEED)

    run_device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("=== AUXILIARY CURVE TRAINING SETUP ===")
    print(f"Device       : {run_device}")
    print(f"Batch Size   : {BATCH_SIZE}")
    print(f"Epochs       : {NUM_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Weight Decay : {WEIGHT_DECAY}")
    print(
        f"Loss         : "
        f"{EQ_LOSS_WEIGHT} EQ MSE + "
        f"{AUX_CURVE_WEIGHT} AUX Curve + "
        f"{SPEC_WEIGHT_2} Spectral"
    )

    train_loader, val_loader, test_loader = create_dataloaders()

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

        "train_aux": [],
        "val_aux": [],

        "train_spec": [],
        "val_spec": [],

        "train_mae": [],
        "val_mae": [],

        "train_sc_eq": [],
        "val_sc_eq": [],

        "train_lsd_eq": [],
        "val_lsd_eq": [],

        "train_sc_aux": [],
        "val_sc_aux": [],

        "train_lsd_aux": [],
        "val_lsd_aux": [],

        "val_per_band_mae": [],
        "learning_rate": [],
    }

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    checkpoint_dir = CHECKPOINT_DIR / "aux_model"
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

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

        scheduler.step(
            val_metrics["loss"]
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])

        history["train_mse"].append(train_metrics["mse"])
        history["val_mse"].append(val_metrics["mse"])

        history["train_aux"].append(train_metrics["aux"])
        history["val_aux"].append(val_metrics["aux"])

        history["train_spec"].append(train_metrics["spec"])
        history["val_spec"].append(val_metrics["spec"])

        history["train_mae"].append(train_metrics["mae"])
        history["val_mae"].append(val_metrics["mae"])

        history["train_sc_eq"].append(train_metrics["sc_eq"])
        history["val_sc_eq"].append(val_metrics["sc_eq"])

        history["train_lsd_eq"].append(train_metrics["lsd_eq"])
        history["val_lsd_eq"].append(val_metrics["lsd_eq"])

        history["train_sc_aux"].append(train_metrics["sc_aux"])
        history["val_sc_aux"].append(val_metrics["sc_aux"])

        history["train_lsd_aux"].append(train_metrics["lsd_aux"])
        history["val_lsd_aux"].append(val_metrics["lsd_aux"])

        history["val_per_band_mae"].append(
            val_metrics["per_band_mae"].tolist()
        )

        history["learning_rate"].append(current_lr)

        print(
            f"Train | "
            f"Loss: {train_metrics['loss']:.4f} | "
            f"MSE: {train_metrics['mse']:.4f} | "
            f"Aux: {train_metrics['aux']:.4f} | "
            f"Spec: {train_metrics['spec']:.4f} | "
            f"MAE: {train_metrics['mae']:.4f} | "
            f"SC_EQ: {train_metrics['sc_eq']:.4f} | "
            f"LSD_EQ: {train_metrics['lsd_eq']:.4f} | "
            f"SC_AUX: {train_metrics['sc_aux']:.4f} | "
            f"LSD_AUX: {train_metrics['lsd_aux']:.4f}"
        )

        print(
            f"Val   | "
            f"Loss: {val_metrics['loss']:.4f} | "
            f"MSE: {val_metrics['mse']:.4f} | "
            f"Aux: {val_metrics['aux']:.4f} | "
            f"Spec: {val_metrics['spec']:.4f} | "
            f"MAE: {val_metrics['mae']:.4f} | "
            f"SC_EQ: {val_metrics['sc_eq']:.4f} | "
            f"LSD_EQ: {val_metrics['lsd_eq']:.4f} | "
            f"SC_AUX: {val_metrics['sc_aux']:.4f} | "
            f"LSD_AUX: {val_metrics['lsd_aux']:.4f}"
        )

        print(f"LR    : {current_lr:.2e}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_model_state = copy.deepcopy(
                model.state_dict()
            )
            patience_counter = 0

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": best_model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "best_val_loss": best_val_loss,
                "config": {
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "eq_loss_weight": EQ_LOSS_WEIGHT,
                    "aux_curve_weight": AUX_CURVE_WEIGHT,
                    "spec_weight": SPEC_WEIGHT_2,
                    "grad_clip": GRAD_CLIP,
                }
            }

            torch.save(
                checkpoint,
                checkpoint_dir / "best_model.pth"
            )

            print("New best model saved.")

        else:
            patience_counter += 1

            print(
                f"No improvement "
                f"({patience_counter}/{EARLY_STOPPING_PATIENCE})"
            )

        print("-" * 50)

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping triggered at epoch {epoch + 1}"
            )
            break

    if best_model_state is not None:
        model.load_state_dict(
            best_model_state
        )

    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(
            history,
            f,
            indent=4
        )

    print("\nTraining complete.")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(
        f"Checkpoint saved to: "
        f"{checkpoint_dir / 'best_model.pth'}"
    )


if __name__ == "__main__":
    main()