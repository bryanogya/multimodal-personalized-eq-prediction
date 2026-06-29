import torch
import torch.nn as nn

from configs.paths import (
    PROJECT_ROOT,
    DATA_DIR,
    PROCESSED_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR
)
from configs.training import MSE_WEIGHT, SPEC_WEIGHT

from src.dataset.data_loader import create_dataloaders
from src.models.multimodal_model import MultimodalEQModel
from src.training.losses import SpectralLoss
from src.utils.interpolate_eq import LogFrequencyInterpolator


def check_path_exists(name, path):
    """
    Check whether a required project path exists.

    Args:
        name: Display name of the path.
        path: Path object to check.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")

    print(f"{name:<18}: OK")


def check_tensor(name, tensor):
    """
    Check tensor shape and detect NaN or Inf values.

    Args:
        name: Display name of the tensor.
        tensor: Tensor to check.

    Raises:
        ValueError: If tensor contains NaN or Inf values.
    """
    if torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN values.")

    if torch.isinf(tensor).any():
        raise ValueError(f"{name} contains Inf values.")

    print(f"{name:<18}: {tuple(tensor.shape)}")


def check_gradient(model):
    """
    Check whether model parameters receive gradients after backward pass.

    Args:
        model: PyTorch model.

    Raises:
        ValueError: If no trainable parameter receives gradient.
    """
    has_gradient = False

    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            has_gradient = True
            break

    if not has_gradient:
        raise ValueError("No gradient found in model parameters.")

    print("Gradient          : OK")


def main():
    """
    Run sanity checks for the project pipeline.

    This function checks important paths, DataLoader output, model forward pass,
    loss computation, and backward propagation.
    """
    run_device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=== PATH CHECK ===")
    check_path_exists("PROJECT_ROOT", PROJECT_ROOT)
    check_path_exists("DATA_DIR", DATA_DIR)
    check_path_exists("PROCESSED_DATA_DIR", PROCESSED_DIR)
    check_path_exists("CHECKPOINT_DIR", CHECKPOINT_DIR)
    check_path_exists("OUTPUT_DIR", OUTPUT_DIR)

    print("\n=== DATALOADER CHECK ===")
    train_loader, val_loader, test_loader = create_dataloaders()

    print(f"Train batches     : {len(train_loader)}")
    print(f"Val batches       : {len(val_loader)}")
    print(f"Test batches      : {len(test_loader)}")

    batch = next(iter(train_loader))

    audio = batch["audio"].to(run_device)
    device_fr = batch["device"].to(run_device)
    preference = batch["preference"].to(run_device)

    target_eq = batch["target_eq"].to(run_device)
    target_curve = batch["target_curve"].to(run_device)

    print("\n=== TENSOR CHECK ===")
    check_tensor("audio", audio)
    check_tensor("device", device_fr)
    check_tensor("preference", preference)
    check_tensor("target_eq", target_eq)
    check_tensor("target_curve", target_curve)

    print("\n=== MODEL CHECK ===")
    model = MultimodalEQModel().to(run_device)

    mse_criterion = nn.MSELoss()
    spectral_criterion = SpectralLoss().to(run_device)
    interpolator = LogFrequencyInterpolator().to(run_device)

    pred_eq = model(
        audio=audio,
        device=device_fr,
        preference=preference
    )

    if isinstance(pred_eq, dict):
        pred_eq = pred_eq["eq"]

    pred_curve = interpolator(pred_eq)

    check_tensor("pred_eq", pred_eq)
    check_tensor("pred_curve", pred_curve)

    if pred_eq.shape != target_eq.shape:
        raise ValueError(
            f"pred_eq shape {pred_eq.shape} does not match target_eq shape {target_eq.shape}"
        )

    if pred_curve.shape != target_curve.shape:
        raise ValueError(
            f"pred_curve shape {pred_curve.shape} does not match target_curve shape {target_curve.shape}"
        )

    print("Shape match       : OK")

    print("\n=== LOSS CHECK ===")
    mse_loss = mse_criterion(
        pred_eq,
        target_eq
    )

    spec_loss = spectral_criterion(
        pred_curve,
        target_curve
    )

    loss = (
        MSE_WEIGHT * mse_loss
        +
        SPEC_WEIGHT * spec_loss
    )

    print(f"MSE Loss          : {mse_loss.item():.6f}")
    print(f"Spec Loss         : {spec_loss.item():.6f}")
    print(f"Total Loss        : {loss.item():.6f}")

    if torch.isnan(loss):
        raise ValueError("Total loss is NaN.")

    if torch.isinf(loss):
        raise ValueError("Total loss is Inf.")

    print("\n=== BACKWARD CHECK ===")
    loss.backward()
    check_gradient(model)

    print("\n=== SANITY CHECK PASSED ===")
    print(f"Device            : {run_device}")


if __name__ == "__main__":
    main()