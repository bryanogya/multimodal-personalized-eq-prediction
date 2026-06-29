import numpy as np


def compute_mse(predictions, targets):
    """
    Compute mean squared error between predictions and targets.

    Args:
        predictions: Predicted values.
        targets: Ground-truth values.

    Returns:
        Mean squared error value.
    """
    return np.mean((predictions - targets) ** 2)


def compute_mae(predictions, targets):
    """
    Compute mean absolute error between predictions and targets.

    Args:
        predictions: Predicted values.
        targets: Ground-truth values.

    Returns:
        Mean absolute error value.
    """
    return np.mean(np.abs(predictions - targets))


def compute_rmse(predictions, targets):
    """
    Compute root mean squared error between predictions and targets.

    Args:
        predictions: Predicted values.
        targets: Ground-truth values.

    Returns:
        Root mean squared error value.
    """
    return np.sqrt(compute_mse(predictions, targets))


def compute_per_band_mae(predictions, targets):
    """
    Compute mean absolute error for each EQ frequency band.

    Args:
        predictions: Predicted EQ values with shape [samples, bands].
        targets: Ground-truth EQ values with shape [samples, bands].

    Returns:
        Per-band MAE values.
    """
    return np.mean(
        np.abs(predictions - targets),
        axis=0
    )


def compute_per_band_rmse(predictions, targets):
    """
    Compute root mean squared error for each EQ frequency band.

    Args:
        predictions: Predicted EQ values with shape [samples, bands].
        targets: Ground-truth EQ values with shape [samples, bands].

    Returns:
        Per-band RMSE values.
    """
    return np.sqrt(
        np.mean(
            (predictions - targets) ** 2,
            axis=0
        )
    )

def compute_error_statistics(predictions, targets):
    errors = predictions - targets

    return {
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "max_abs_error": float(np.max(np.abs(errors))),
        "min_abs_error": float(np.min(np.abs(errors))),
    }


def compute_regression_metrics(predictions, targets):
    return {
        "mse": float(compute_mse(predictions, targets)),
        "mae": float(compute_mae(predictions, targets)),
        "rmse": float(compute_rmse(predictions, targets)),
        "per_band_mae": compute_per_band_mae(predictions, targets).tolist(),
        "per_band_rmse": compute_per_band_rmse(predictions, targets).tolist(),
        **compute_error_statistics(predictions, targets),
    }