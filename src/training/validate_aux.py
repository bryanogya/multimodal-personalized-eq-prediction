import json

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from configs.paths import CHECKPOINT_DIR
from configs.eq import EQ_FREQS
from configs.training import EQ_LOSS_WEIGHT, AUX_CURVE_WEIGHT, SPEC_WEIGHT

from src.dataset.data_loader import create_dataloaders
from src.models.multimodal_model import MultimodalEQModel
from src.training.losses import SpectralLoss
from src.utils.interpolate_eq import LogFrequencyInterpolator


def evaluate(
    model,
    test_loader,
    mse_criterion,
    spectral_criterion,
    interpolator,
    run_device,
):
    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_aux = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_spec = 0.0

    total_sc_eq = 0.0
    total_lsd_eq = 0.0
    total_sc_aux = 0.0
    total_lsd_aux = 0.0

    n_batches = 0

    per_band_mae_list = []
    per_band_rmse_list = []

    all_predictions = []
    all_targets = []
    all_aux_curves = []
    all_target_curves = []

    pbar = tqdm(
        test_loader,
        desc="Testing",
        leave=False
    )

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

            loss = (
                EQ_LOSS_WEIGHT * mse_loss
                +
                AUX_CURVE_WEIGHT * aux_curve_loss
                +
                SPEC_WEIGHT * spec_loss
            )

            mae = torch.mean(
                torch.abs(pred_eq - target_eq)
            )

            rmse = torch.sqrt(
                torch.mean((pred_eq - target_eq) ** 2)
            )

            sc_eq, lsd_eq = spectral_criterion.components(
                predicted_response_from_eq,
                personalized_target
            )

            sc_aux, lsd_aux = spectral_criterion.components(
                pred_response_curve,
                personalized_target
            )

            per_band_mae = torch.mean(
                torch.abs(pred_eq - target_eq),
                dim=0
            )

            per_band_rmse = torch.sqrt(
                torch.mean(
                    (pred_eq - target_eq) ** 2,
                    dim=0
                )
            )

            per_band_mae_list.append(
                per_band_mae.detach().cpu()
            )

            per_band_rmse_list.append(
                per_band_rmse.detach().cpu()
            )

            all_predictions.append(
                pred_eq.detach().cpu()
            )

            all_targets.append(
                target_eq.detach().cpu()
            )

            all_aux_curves.append(
                pred_response_curve.detach().cpu()
            )

            all_target_curves.append(
                personalized_target.detach().cpu()
            )

            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_aux += aux_curve_loss.item()
            total_mae += mae.item()
            total_rmse += rmse.item()
            total_spec += spec_loss.item()

            total_sc_eq += sc_eq.item()
            total_lsd_eq += lsd_eq.item()
            total_sc_aux += sc_aux.item()
            total_lsd_aux += lsd_aux.item()

            n_batches += 1

            pbar.set_postfix(
                test_loss=f"{total_loss / n_batches:.4f}",
                test_mae=f"{total_mae / n_batches:.4f}"
            )

    per_band_mae_final = torch.stack(
        per_band_mae_list
    ).mean(dim=0).numpy()

    per_band_rmse_final = torch.stack(
        per_band_rmse_list
    ).mean(dim=0).numpy()

    predictions = torch.cat(
        all_predictions,
        dim=0
    ).numpy()

    targets = torch.cat(
        all_targets,
        dim=0
    ).numpy()

    aux_curves = torch.cat(
        all_aux_curves,
        dim=0
    ).numpy()

    target_curves = torch.cat(
        all_target_curves,
        dim=0
    ).numpy()

    metrics = {
        "test_loss": total_loss / n_batches,
        "test_mse": total_mse / n_batches,
        "test_aux": total_aux / n_batches,
        "test_mae": total_mae / n_batches,
        "test_rmse": total_rmse / n_batches,
        "test_spec": total_spec / n_batches,

        "test_sc_eq": total_sc_eq / n_batches,
        "test_lsd_eq": total_lsd_eq / n_batches,

        "test_sc_aux": total_sc_aux / n_batches,
        "test_lsd_aux": total_lsd_aux / n_batches,

        "per_band_mae": per_band_mae_final.tolist(),
        "per_band_rmse": per_band_rmse_final.tolist(),
    }

    return metrics, predictions, targets, aux_curves, target_curves


def main():
    run_device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("=== AUXILIARY MODEL VALIDATION / TESTING ===")
    print(f"Device: {run_device}")

    train_loader, val_loader, test_loader = create_dataloaders()

    model = MultimodalEQModel().to(run_device)

    checkpoint_path = CHECKPOINT_DIR / "aux_model" / "best_model.pth"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=run_device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Best Val Loss    : {checkpoint['best_val_loss']:.4f}")

    mse_criterion = nn.MSELoss()
    spectral_criterion = SpectralLoss().to(run_device)
    interpolator = LogFrequencyInterpolator().to(run_device)

    metrics, predictions, targets, aux_curves, target_curves = evaluate(
        model=model,
        test_loader=test_loader,
        mse_criterion=mse_criterion,
        spectral_criterion=spectral_criterion,
        interpolator=interpolator,
        run_device=run_device,
    )

    print("\n=== TEST RESULT: 6-BAND EQ OUTPUT ===")
    print(f"Test Loss : {metrics['test_loss']:.4f}")
    print(f"Test MSE  : {metrics['test_mse']:.4f}")
    print(f"Test MAE  : {metrics['test_mae']:.4f}")
    print(f"Test RMSE : {metrics['test_rmse']:.4f}")
    print(f"SC_EQ     : {metrics['test_sc_eq']:.4f}")
    print(f"LSD_EQ    : {metrics['test_lsd_eq']:.4f}")

    print("\n=== TEST RESULT: AUXILIARY CURVE HEAD ===")
    print(f"Aux MSE   : {metrics['test_aux']:.4f}")
    print(f"Aux Spec  : {metrics['test_spec']:.4f}")
    print(f"SC_AUX    : {metrics['test_sc_aux']:.4f}")
    print(f"LSD_AUX   : {metrics['test_lsd_aux']:.4f}")

    print("\n=== PER-BAND MAE ===")
    for freq, err in zip(
        EQ_FREQS,
        metrics["per_band_mae"]
    ):
        print(f"{freq:>7.0f} Hz : {err:.4f} dB")

    print("\n=== PER-BAND RMSE ===")
    for freq, err in zip(
        EQ_FREQS,
        metrics["per_band_rmse"]
    ):
        print(f"{freq:>7.0f} Hz : {err:.4f} dB")

    output_dir = CHECKPOINT_DIR / "aux_model"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(
            metrics,
            f,
            indent=4
        )

    np.save(
        output_dir / "test_predictions.npy",
        predictions
    )

    np.save(
        output_dir / "test_targets.npy",
        targets
    )

    np.save(
        output_dir / "test_aux_curves.npy",
        aux_curves
    )

    np.save(
        output_dir / "test_target_curves.npy",
        target_curves
    )

    print("\nSaved:")
    print(output_dir / "test_metrics.json")
    print(output_dir / "test_predictions.npy")
    print(output_dir / "test_targets.npy")
    print(output_dir / "test_aux_curves.npy")
    print(output_dir / "test_target_curves.npy")


if __name__ == "__main__":
    main()