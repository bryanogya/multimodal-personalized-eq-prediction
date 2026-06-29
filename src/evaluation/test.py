import json

import numpy as np
import torch
from tqdm import tqdm

from configs.paths import CHECKPOINT_DIR
from configs.training import MSE_WEIGHT, SPEC_WEIGHT
from configs.eq import EQ_FREQS

from src.evaluation.metrics import (
    compute_mse,
    compute_mae,
    compute_rmse,
    compute_per_band_mae,
    compute_per_band_rmse
)


def get_model_output(output):
    """
    Extract EQ prediction from the model output.

    Args:
        output: Model output, either a tensor or a dictionary.

    Returns:
        Tensor containing predicted EQ values.
    """
    if isinstance(output, dict):
        return output["eq"]

    return output


def predict_test_set(
    model,
    loader,
    interpolator,
    run_device
):
    """
    Run prediction on the test set and collect output arrays.

    Args:
        model: Trained EQ prediction model.
        loader: DataLoader for the test set.
        interpolator: Module for converting EQ bands into response curves.
        run_device: Device used for computation.

    Returns:
        Dictionary containing predicted EQ, target EQ, predicted response,
        target response, device response, and preference vectors.
    """
    model.eval()

    all_pred_eq = []
    all_target_eq = []

    all_pred_response = []
    all_target_response = []
    all_device_response = []

    all_preference = []

    pbar = tqdm(
        loader,
        desc="Predicting Test Set",
        leave=False
    )

    with torch.no_grad():
        for batch in pbar:
            audio = batch["audio"].to(run_device)
            device_fr = batch["device"].to(run_device)
            preference = batch["preference"].to(run_device)

            target_eq = batch["target_eq"].to(run_device)
            personalized_target = batch["personalized_target"].to(run_device)

            output = model(
                audio=audio,
                device=device_fr,
                preference=preference
            )

            pred_eq = get_model_output(output)

            pred_curve = interpolator(pred_eq)
            pred_response = device_fr + pred_curve

            all_pred_eq.append(
                pred_eq.detach().cpu().numpy()
            )

            all_target_eq.append(
                target_eq.detach().cpu().numpy()
            )

            all_pred_response.append(
                pred_response.detach().cpu().numpy()
            )

            all_target_response.append(
                personalized_target.detach().cpu().numpy()
            )

            all_device_response.append(
                device_fr.detach().cpu().numpy()
            )

            all_preference.append(
                preference.detach().cpu().numpy()
            )

    return {
        "pred_eq": np.concatenate(all_pred_eq, axis=0),
        "target_eq": np.concatenate(all_target_eq, axis=0),

        "pred_response": np.concatenate(all_pred_response, axis=0),
        "target_response": np.concatenate(all_target_response, axis=0),
        "device_response": np.concatenate(all_device_response, axis=0),

        "preference": np.concatenate(all_preference, axis=0),
    }


def compute_test_metrics(
    results,
    spectral_criterion,
    run_device,
):
    """
    Compute test metrics for EQ prediction and response matching.

    Args:
        results: Dictionary returned by predict_test_set.
        spectral_criterion: Loss function for spectral response comparison.
        run_device: Device used for spectral metric computation.

    Returns:
        Dictionary containing test loss, EQ metrics, spectral metrics,
        and per-band errors.
    """
    pred_eq_np = results["pred_eq"]
    target_eq_np = results["target_eq"]

    pred_response = torch.tensor(
        results["pred_response"],
        dtype=torch.float32,
        device=run_device
    )

    target_response = torch.tensor(
        results["target_response"],
        dtype=torch.float32,
        device=run_device
    )

    mse_loss = compute_mse(
        pred_eq_np,
        target_eq_np
    )

    mae = compute_mae(
        pred_eq_np,
        target_eq_np
    )

    rmse = compute_rmse(
        pred_eq_np,
        target_eq_np
    )

    per_band_mae = compute_per_band_mae(
        pred_eq_np,
        target_eq_np
    )

    per_band_rmse = compute_per_band_rmse(
        pred_eq_np,
        target_eq_np
    )

    spec_loss = spectral_criterion(
        pred_response,
        target_response
    )

    sc, lsd = spectral_criterion.components(
        pred_response,
        target_response
    )

    loss = (
        MSE_WEIGHT * mse_loss
        +
        SPEC_WEIGHT * spec_loss.item()
    )

    metrics = {
        "test_loss": float(loss),
        "test_mse": float(mse_loss),
        "test_mae": float(mae),
        "test_rmse": float(rmse),
        "test_spec": float(spec_loss.item()),
        "test_spectral_convergence": float(sc.item()),
        "test_log_spectral_distance": float(lsd.item()),
        "per_band_mae": per_band_mae.tolist(),
        "per_band_rmse": per_band_rmse.tolist(),
    }

    return metrics


def save_test_results(
    results,
    output_dir=None,
):
    """
    Save test prediction results into an NPZ file.

    Args:
        results: Dictionary containing prediction and target arrays.
        output_dir: Directory used to save the result file.
    """
    if output_dir is None:
        output_dir = CHECKPOINT_DIR / "full_model"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez(
        output_dir / "test_results.npz",
        pred_eq=results["pred_eq"],
        target_eq=results["target_eq"],
        pred_response=results["pred_response"],
        target_response=results["target_response"],
        device_response=results["device_response"],
        preference=results["preference"],
    )

    print(f"Saved: {output_dir / 'test_results.npz'}")


def save_test_metrics(
    metrics,
    output_dir=None,
):
    """
    Save test metrics into a JSON file.

    Args:
        metrics: Dictionary containing evaluation metrics.
        output_dir: Directory used to save the metric file.
    """
    if output_dir is None:
        output_dir = CHECKPOINT_DIR / "full_model"

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

    print(f"Saved: {output_dir / 'test_metrics.json'}")
    
def print_metrics(metrics):
    """
    Print test metrics and per-band errors to the terminal.

    Args:
        metrics: Dictionary containing evaluation metrics.
    """
    print("\n=== TEST RESULT ===")
    print(f"Test Loss : {metrics['test_loss']:.4f}")
    print(f"Test MSE  : {metrics['test_mse']:.4f}")
    print(f"Test MAE  : {metrics['test_mae']:.4f}")
    print(f"Test Spec : {metrics['test_spec']:.4f}")
    print(f"SC        : {metrics['test_spectral_convergence']:.4f}")
    print(f"LSD       : {metrics['test_log_spectral_distance']:.4f}")
    
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