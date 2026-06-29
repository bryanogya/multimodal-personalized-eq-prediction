import json

import numpy as np

from configs.paths import CHECKPOINT_DIR
from configs.eq import EQ_FREQS


def load_test_results(output_dir=None):
    """
    Load saved test prediction results.

    Args:
        output_dir: Directory containing test_results.npz.

    Returns:
        Dictionary containing test result arrays.
    """
    if output_dir is None:
        output_dir = CHECKPOINT_DIR / "full_model"

    results_path = output_dir / "test_results.npz"

    if not results_path.exists():
        raise FileNotFoundError(f"Test results not found: {results_path}")

    data = np.load(results_path)

    return {
        key: data[key]
        for key in data.files
    }


def compute_sample_mae(pred_eq, target_eq):
    """
    Compute MAE for each test sample.

    Args:
        pred_eq: Predicted EQ array with shape [samples, bands].
        target_eq: Target EQ array with shape [samples, bands].

    Returns:
        MAE value for each sample.
    """
    return np.mean(
        np.abs(pred_eq - target_eq),
        axis=1
    )


def get_worst_predictions(results, top_k=10):
    """
    Get samples with the highest EQ prediction error.

    Args:
        results: Dictionary containing prediction and target arrays.
        top_k: Number of worst samples to return.

    Returns:
        List of dictionaries containing worst prediction details.
    """
    sample_mae = compute_sample_mae(
        results["pred_eq"],
        results["target_eq"]
    )

    worst_indices = np.argsort(sample_mae)[::-1][:top_k]

    worst_samples = []

    for rank, index in enumerate(worst_indices, start=1):
        worst_samples.append({
            "rank": int(rank),
            "sample_index": int(index),
            "sample_mae": float(sample_mae[index]),
            "pred_eq": results["pred_eq"][index].tolist(),
            "target_eq": results["target_eq"][index].tolist(),
            "per_band_abs_error": np.abs(
                results["pred_eq"][index]
                -
                results["target_eq"][index]
            ).tolist(),
            "eq_freqs": EQ_FREQS.tolist()
            if hasattr(EQ_FREQS, "tolist")
            else list(EQ_FREQS),
            "preference": results["preference"][index].tolist(),
        })

    return worst_samples


def save_worst_predictions(worst_samples, output_dir=None):
    """
    Save worst prediction samples into a JSON file.

    Args:
        worst_samples: List of worst prediction dictionaries.
        output_dir: Directory used to save the JSON file.
    """
    if output_dir is None:
        output_dir = CHECKPOINT_DIR / "full_model"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = output_dir / "worst_predictions.json"

    with open(output_path, "w") as f:
        json.dump(
            worst_samples,
            f,
            indent=4
        )

    print(f"Saved: {output_path}")


def print_worst_predictions(worst_samples):
    """
    Print a summary of worst prediction samples.

    Args:
        worst_samples: List of worst prediction dictionaries.
    """
    print("\n=== WORST PREDICTIONS ===")

    for sample in worst_samples:
        print(
            f"Rank {sample['rank']:>2} | "
            f"Index {sample['sample_index']:>4} | "
            f"MAE {sample['sample_mae']:.4f} dB"
        )


def main():
    """
    Run worst prediction analysis from saved test results.
    """
    results = load_test_results()

    worst_samples = get_worst_predictions(
        results,
        top_k=10
    )

    save_worst_predictions(worst_samples)
    print_worst_predictions(worst_samples)


if __name__ == "__main__":
    main()