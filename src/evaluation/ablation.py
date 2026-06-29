import json
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from configs.paths import CHECKPOINT_DIR
from configs.eq import EQ_FREQS
from configs.training import *

from src.dataset.data_loader import create_dataloaders
from src.models.multimodal_model import MultimodalEQModel
from src.training.losses import SpectralLoss
from src.utils.interpolate_eq import LogFrequencyInterpolator


ABLATION_CONFIGS = [
    ("Audio Only", True, False, False),
    ("Audio + Device", True, True, False),
    ("Audio + Preference", True, False, True),
    ("Device + Preference", False, True, True),
    ("Full Model", True, True, True),
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_model_output(output):
    if isinstance(output, dict):
        return output["eq"]
    return output


def train_one_epoch(
    model,
    train_loader,
    mse_criterion,
    spectral_criterion,
    interpolator,
    optimizer,
    run_device,
    use_audio,
    use_device,
    use_preference,
):
    model.train()

    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_spec = 0.0

    n_batches = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        audio = batch["audio"].to(run_device) if use_audio else None
        device_fr = batch["device"].to(run_device)
        preference = batch["preference"].to(run_device) if use_preference else None

        target_eq = batch["target_eq"].to(run_device)
        personalized_target = batch["personalized_target"].to(run_device)

        model_device_input = device_fr if use_device else None

        optimizer.zero_grad()

        output = model(
            audio=audio,
            device=model_device_input,
            preference=preference
        )

        pred_eq = get_model_output(output)
        mse_loss = mse_criterion(pred_eq, target_eq)
        pred_curve = interpolator(pred_eq)
        predicted_response = (device_fr + pred_curve)
        
        spec_loss = spectral_criterion(predicted_response, personalized_target)
        loss = (MSE_WEIGHT * mse_loss + SPEC_WEIGHT * spec_loss)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        mae = torch.mean(torch.abs(pred_eq - target_eq))

        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_mae += mae.item()
        total_spec += spec_loss.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "mse": total_mse / n_batches,
        "mae": total_mae / n_batches,
        "spec": total_spec / n_batches,
    }


def evaluate_model(
    model,
    test_loader,
    mse_criterion,
    spectral_criterion,
    interpolator,
    run_device,
    use_audio,
    use_device,
    use_preference,
):
    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_spec = 0.0
    total_sc = 0.0
    total_lsd = 0.0

    per_band_mae_list = []

    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            audio = batch["audio"].to(run_device) if use_audio else None
            device_fr = batch["device"].to(run_device)
            preference = batch["preference"].to(run_device) if use_preference else None

            target_eq = batch["target_eq"].to(run_device)
            personalized_target = batch["personalized_target"].to(run_device)

            model_device_input = device_fr if use_device else None

            output = model(
                audio=audio,
                device=model_device_input,
                preference=preference
            )

            pred_eq = get_model_output(output)
            mse_loss = mse_criterion(pred_eq, target_eq)
            mae = torch.mean(torch.abs(pred_eq - target_eq))
            rmse = torch.sqrt(torch.mean((pred_eq - target_eq) ** 2))
            pred_curve = interpolator(pred_eq)
            predicted_response = (device_fr + pred_curve)

            spec_loss = spectral_criterion(predicted_response, personalized_target)
            sc, lsd = spectral_criterion.components(predicted_response, personalized_target)
            loss = (MSE_WEIGHT * mse_loss + SPEC_WEIGHT * spec_loss)

            per_band_mae = torch.mean(torch.abs(pred_eq - target_eq), dim=0)
            per_band_mae_list.append(per_band_mae.detach().cpu())

            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_mae += mae.item()
            total_rmse += rmse.item()
            total_spec += spec_loss.item()
            total_sc += sc.item()
            total_lsd += lsd.item()

            n_batches += 1

    per_band_mae = torch.stack(
        per_band_mae_list
    ).mean(dim=0).numpy()

    return {
        "loss": total_loss / n_batches,
        "mse": total_mse / n_batches,
        "mae": total_mae / n_batches,
        "rmse": total_rmse / n_batches,
        "spec": total_spec / n_batches,
        "sc": total_sc / n_batches,
        "lsd": total_lsd / n_batches,
        "per_band_mae": per_band_mae.tolist(),
    }


def train_ablation_model(
    name,
    use_audio,
    use_device,
    use_preference,
    train_loader,
    val_loader,
    test_loader,
    run_device,
):
    print(f"\n=== Training {name} ===")

    model = MultimodalEQModel(
        use_audio=use_audio,
        use_device=use_device,
        use_preference=use_preference
    ).to(run_device)

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

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    max_epochs = NUM_EPOCHS_ABL

    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs}")

        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            mse_criterion=mse_criterion,
            spectral_criterion=spectral_criterion,
            interpolator=interpolator,
            optimizer=optimizer,
            run_device=run_device,
            use_audio=use_audio,
            use_device=use_device,
            use_preference=use_preference,
        )

        val_metrics = evaluate_model(
            model=model,
            test_loader=val_loader,
            mse_criterion=mse_criterion,
            spectral_criterion=spectral_criterion,
            interpolator=interpolator,
            run_device=run_device,
            use_audio=use_audio,
            use_device=use_device,
            use_preference=use_preference,
        )

        scheduler.step(
            val_metrics["loss"]
        )

        print(
            f"Train | Loss: {train_metrics['loss']:.4f} | "
            f"MSE: {train_metrics['mse']:.4f} | "
            f"MAE: {train_metrics['mae']:.4f}"
        )

        print(
            f"Val   | Loss: {val_metrics['loss']:.4f} | "
            f"MSE: {val_metrics['mse']:.4f} | "
            f"MAE: {val_metrics['mae']:.4f} | "
            f"SC: {val_metrics['sc']:.4f} | "
            f"LSD: {val_metrics['lsd']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_model_state = copy.deepcopy(
                model.state_dict()
            )
            patience_counter = 0
            print("New best ablation model.")
        else:
            patience_counter += 1
            print(
                f"No improvement "
                f"({patience_counter}/{EARLY_STOPPING_ABL})"
            )

        if patience_counter >= EARLY_STOPPING_ABL:
            print("Early stopping.")
            break

    if best_model_state is not None:
        model.load_state_dict(
            best_model_state
        )

    test_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        mse_criterion=mse_criterion,
        spectral_criterion=spectral_criterion,
        interpolator=interpolator,
        run_device=run_device,
        use_audio=use_audio,
        use_device=use_device,
        use_preference=use_preference,
    )

    return test_metrics


def main():
    set_seed(RANDOM_SEED)

    run_device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    train_loader, val_loader, test_loader = create_dataloaders()

    results = {}

    for name, use_audio, use_device, use_preference in ABLATION_CONFIGS:
        metrics = train_ablation_model(
            name=name,
            use_audio=use_audio,
            use_device=use_device,
            use_preference=use_preference,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            run_device=run_device,
        )

        results[name] = metrics

    output_dir = CHECKPOINT_DIR / "ablation"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(
            results,
            f,
            indent=4
        )

    print("\n=== ABLATION SUMMARY ===")
    print(
        f"{'Model':<22} "
        f"{'MSE':>10} "
        f"{'MAE':>10} "
        f"{'RMSE':>10} "
        f"{'SC':>10} "
        f"{'LSD':>10}"
    )

    print("-" * 78)

    for name, metrics in results.items():
        print(
            f"{name:<22} "
            f"{metrics['mse']:>10.4f} "
            f"{metrics['mae']:>10.4f} "
            f"{metrics['rmse']:>10.4f} "
            f"{metrics['sc']:>10.4f} "
            f"{metrics['lsd']:>10.4f}"
        )

    print(f"\nSaved to: {output_dir / 'ablation_results.json'}")


if __name__ == "__main__":
    main()