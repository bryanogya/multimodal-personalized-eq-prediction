import json
import numpy as np
import torch
import torch.nn as nn

from configs.paths import CHECKPOINT_DIR
from configs.eq import EQ_FREQS

from src.dataset.data_loader import create_dataloaders
from src.training.losses import SpectralLoss
from src.utils.interpolate_eq import LogFrequencyInterpolator

def init_metric_storage():
    """
    Initialize metric storage for baseline evaluation.

    Returns:
        Dictionary used to accumulate batch-level metrics.
    """
    return {
        "mse": 0.0,
        "mae": 0.0,
        "rmse": 0.0,
        "sc": 0.0,
        "lsd": 0.0,
        "per_band_mae": [],
        "per_band_rmse": [],
        "n_batches": 0,
    }

def update_metrics(
    storage,
    pred_eq,
    target_eq,
    device_response,
    personalized_target,
    mse_criterion,
    spectral_criterion,
    interpolator,
):
    """
    Update metric storage using one evaluated batch.

    Args:
        storage: Dictionary used to accumulate metrics.
        pred_eq: Predicted EQ tensor.
        target_eq: Ground-truth EQ tensor.
        device_response: Device frequency response tensor.
        personalized_target: Target personalized response tensor.
        mse_criterion: MSE loss function.
        spectral_criterion: Spectral metric function.
        interpolator: Module for converting EQ bands into response curves.
    """
    mse = mse_criterion(pred_eq, target_eq)
    mae = torch.mean(torch.abs(pred_eq - target_eq))
    rmse = torch.sqrt(torch.mean((pred_eq - target_eq) ** 2))
    pred_curve = interpolator(pred_eq)
    predicted_response = (device_response + pred_curve)
    sc, lsd = spectral_criterion.components(
        predicted_response,
        personalized_target
    )
    per_band_mae = torch.mean(torch.abs(pred_eq - target_eq), dim=0)
    per_band_rmse = torch.sqrt(torch.mean((pred_eq - target_eq) ** 2, dim=0))

    storage["mse"] += mse.item()
    storage["mae"] += mae.item()
    storage["rmse"] += rmse.item()
    storage["sc"] += sc.item()
    storage["lsd"] += lsd.item()

    storage["per_band_mae"].append(per_band_mae.detach().cpu())
    storage["per_band_rmse"].append(per_band_rmse.detach().cpu())

    storage["n_batches"] += 1
    
def finalize_metrics(storage):
    """
    Average accumulated metrics across all evaluated batches.

    Args:
        storage: Dictionary containing accumulated metric values.

    Returns:
        Dictionary containing final averaged metrics.
    """
    n_batches = storage["n_batches"]

    if n_batches == 0:
        raise ValueError("No batches were evaluated.")

    per_band_mae = torch.stack(
        storage["per_band_mae"]
    ).mean(dim=0).numpy()

    per_band_rmse = torch.stack(
        storage["per_band_rmse"]
    ).mean(dim=0).numpy()

    return {
        "mse": storage["mse"] / n_batches,
        "mae": storage["mae"] / n_batches,
        "rmse": storage["rmse"] / n_batches,
        "sc": storage["sc"] / n_batches,
        "lsd": storage["lsd"] / n_batches,
        "per_band_mae": per_band_mae.tolist(),
        "per_band_rmse": per_band_rmse.tolist(),
    }
    
def compute_mean_eq(train_loader):
    """
    Compute the average target EQ from the training set.

    Args:
        train_loader: DataLoader for the training set.

    Returns:
        Mean EQ vector as a float32 NumPy array.
    """
    all_eq = []

    for batch in train_loader:
        target_eq = batch["target_eq"]
        all_eq.append(target_eq.numpy())

    all_eq = np.concatenate(
        all_eq,
        axis=0
    )

    mean_eq = np.mean(
        all_eq,
        axis=0
    )

    return mean_eq.astype(np.float32)    
    

def evaluate_zero_eq(
    test_loader,
    mse_criterion,
    spectral_criterion,
    interpolator,
    run_device,
):
    """
    Evaluate the zero EQ baseline on the test set.

    Returns:
        Dictionary containing baseline metrics.
    """
    storage = init_metric_storage()

    with torch.no_grad():
        for batch in test_loader:
            target_eq = batch["target_eq"].to(run_device)
            device_response = batch["device"].to(run_device)
            personalized_target = batch["personalized_target"].to(run_device)

            pred_eq = torch.zeros_like(
                target_eq
            )

            update_metrics(
                storage=storage,
                pred_eq=pred_eq,
                target_eq=target_eq,
                device_response=device_response,
                personalized_target=personalized_target,
                mse_criterion=mse_criterion,
                spectral_criterion=spectral_criterion,
                interpolator=interpolator,
            )

    return finalize_metrics(storage)


def evaluate_mean_eq(
    test_loader,
    mean_eq,
    mse_criterion,
    spectral_criterion,
    interpolator,
    run_device,
):
    """
    Evaluate the mean EQ baseline on the test set.

    Returns:
        Dictionary containing baseline metrics.
    """
    storage = init_metric_storage()

    mean_eq_tensor = torch.tensor(
        mean_eq,
        dtype=torch.float32,
        device=run_device
    )

    with torch.no_grad():
        for batch in test_loader:
            target_eq = batch["target_eq"].to(run_device)
            device_response = batch["device"].to(run_device)
            personalized_target = batch["personalized_target"].to(run_device)

            pred_eq = mean_eq_tensor.unsqueeze(0).repeat(
                target_eq.shape[0],
                1
            )

            update_metrics(
                storage=storage,
                pred_eq=pred_eq,
                target_eq=target_eq,
                device_response=device_response,
                personalized_target=personalized_target,
                mse_criterion=mse_criterion,
                spectral_criterion=spectral_criterion,
                interpolator=interpolator,
            )

    return finalize_metrics(storage)

def compute_improvement(baseline_mae, model_mae):
    """
    Compute MAE improvement of the model over a baseline.

    Args:
        baseline_mae: Baseline MAE value.
        model_mae: Model MAE value.

    Returns:
        Improvement percentage.
    """
    return (
        (baseline_mae - model_mae)
        /
        baseline_mae
    ) * 100.0
    
    
def print_result_table(results):
    """
    Print baseline comparison metrics as a table.

    Args:
        results: Dictionary containing metrics for each method.
    """
    print("\n=== BASELINE COMPARISON ===")
    print(
        f"{'Model':<18} "
        f"{'MSE':>10} "
        f"{'MAE':>10} "
        f"{'RMSE':>10} "
        f"{'SC':>10} "
        f"{'LSD':>10}"
    )

    print("-" * 72)

    for name, metrics in results.items():
        print(
            f"{name:<18} "
            f"{metrics['mse']:>10.4f} "
            f"{metrics['mae']:>10.4f} "
            f"{metrics['rmse']:>10.4f} "
            f"{metrics['sc']:>10.4f} "
            f"{metrics['lsd']:>10.4f}"
        )
        

def print_per_band_table(results):
    """
    Print per-band MAE comparison for each baseline and model.

    Args:
        results: Dictionary containing per-band metrics.
    """
    print("\n=== PER-BAND MAE ===")
    print(
        f"{'Band':<10} "
        f"{'Zero EQ':>12} "
        f"{'Mean EQ':>12} "
        f"{'Model':>12}"
    )

    print("-" * 50)

    for i, freq in enumerate(EQ_FREQS):
        print(
            f"{freq:>5.0f} Hz   "
            f"{results['Zero EQ']['per_band_mae'][i]:>12.4f} "
            f"{results['Mean EQ']['per_band_mae'][i]:>12.4f} "
            f"{results['Proposed Model']['per_band_mae'][i]:>12.4f}"
        )
        
def load_model_metrics():
    """
    Load saved full model test metrics.

    Returns:
        Dictionary containing model metrics in baseline-compatible format.
    """
    metrics_path = CHECKPOINT_DIR / "full_model" / "test_metrics.json"

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    return {
        "mse": metrics["test_mse"],
        "mae": metrics["test_mae"],
        "rmse": np.sqrt(metrics["test_mse"]),
        "sc": metrics["test_spectral_convergence"],
        "lsd": metrics["test_log_spectral_distance"],
        "per_band_mae": metrics["per_band_mae"],
        "per_band_rmse": metrics["per_band_rmse"],
    }

def main():
    """
    Run baseline evaluation and save comparison results.
    """
    run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== BASELINE EVALUATION ===")
    print(f"Device: {run_device}")

    train_loader, val_loader, test_loader = create_dataloaders()

    mse_criterion = nn.MSELoss()
    spectral_criterion = SpectralLoss().to(run_device)
    interpolator = LogFrequencyInterpolator().to(run_device)

    zero_result = evaluate_zero_eq(
        test_loader=test_loader,
        mse_criterion=mse_criterion,
        spectral_criterion=spectral_criterion,
        interpolator=interpolator,
        run_device=run_device,
    )

    mean_eq = compute_mean_eq(
        train_loader
    )

    mean_result = evaluate_mean_eq(
        test_loader=test_loader,
        mean_eq=mean_eq,
        mse_criterion=mse_criterion,
        spectral_criterion=spectral_criterion,
        interpolator=interpolator,
        run_device=run_device,
    )

    model_result = load_model_metrics()

    results = {
        "Zero EQ": zero_result,
        "Mean EQ": mean_result,
        "Proposed Model": model_result,
    }

    print_result_table(results)
    print_per_band_table(results)

    improvement_zero = compute_improvement(
        zero_result["mae"],
        model_result["mae"]
    )

    improvement_mean = compute_improvement(
        mean_result["mae"],
        model_result["mae"]
    )

    print("\n=== IMPROVEMENT BASED ON MAE ===")
    print(f"vs Zero EQ : {improvement_zero:.2f}%")
    print(f"vs Mean EQ : {improvement_mean:.2f}%")

    print("\n=== MEAN EQ VALUES ===")
    for freq, gain in zip(EQ_FREQS, mean_eq):
        print(f"{freq:>7.0f} Hz : {gain:.4f} dB")

    output_dir = CHECKPOINT_DIR / "baseline"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    save_data = {
        "zero_eq": zero_result,
        "mean_eq": mean_result,
        "proposed_model": model_result,
        "mean_eq_values": mean_eq.tolist(),
        "improvement_vs_zero_eq_mae": improvement_zero,
        "improvement_vs_mean_eq_mae": improvement_mean,
    }

    with open(output_dir / "baseline_metrics.json", "w") as f:
        json.dump(
            save_data,
            f,
            indent=4
        )

    print("\nSaved:")
    print(output_dir / "baseline_metrics.json")


if __name__ == "__main__":
    main()