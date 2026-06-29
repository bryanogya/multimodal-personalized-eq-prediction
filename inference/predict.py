import argparse
from pathlib import Path

import numpy as np
import torch

from configs.paths import CHECKPOINT_DIR, PROJECT_ROOT
from configs.training import DEVICE
from src.models.multimodal_model import MultimodalEQModel


def resolve_path(path):
    """
    Convert relative path into absolute project path.

    Args:
        path: Input file path.

    Returns:
        Absolute Path object.
    """
    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_model(checkpoint_path, run_device):
    """
    Load trained model checkpoint.

    Args:
        checkpoint_path: Path to the saved checkpoint.
        run_device: Device used for inference.

    Returns:
        Model in evaluation mode.
    """
    model = MultimodalEQModel()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=run_device
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(run_device)
    model.eval()

    return model


def load_input(audio_path, device_path, preference_path):
    """
    Load processed audio, device response, and preference vector.

    Args:
        audio_path: Path to processed audio feature file.
        device_path: Path to processed device response file.
        preference_path: Path to processed preference vector file.

    Returns:
        Tuple of input tensors.
    """
    audio_path = resolve_path(audio_path)
    device_path = resolve_path(device_path)
    preference_path = resolve_path(preference_path)

    audio = np.load(audio_path).astype(np.float32)
    device = np.load(device_path).astype(np.float32)
    preference = np.load(preference_path).astype(np.float32)

    audio = torch.tensor(audio).unsqueeze(0)
    device = torch.tensor(device).unsqueeze(0)
    preference = torch.tensor(preference).unsqueeze(0)

    return audio, device, preference


def predict_eq(
    model,
    audio,
    device,
    preference,
    run_device
):
    """
    Predict EQ values for one input sample.

    Args:
        model: Trained EQ prediction model.
        audio: Audio input tensor.
        device: Device response tensor.
        preference: Preference vector tensor.
        run_device: Device used for inference.

    Returns:
        Predicted EQ values as a NumPy array.
    """
    audio = audio.to(run_device)
    device = device.to(run_device)
    preference = preference.to(run_device)

    with torch.no_grad():
        output = model(
            audio=audio,
            device=device,
            preference=preference
        )

        if isinstance(output, dict):
            output = output["eq"]

    return output.detach().cpu().numpy()[0]


def main():
    """
    Run EQ prediction from processed input files.
    """
    parser = argparse.ArgumentParser(
        description="Run EQ prediction for one processed sample."
    )

    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to processed audio .npy file"
    )

    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Path to processed device response .npy file"
    )

    parser.add_argument(
        "--preference",
        type=str,
        required=True,
        help="Path to processed preference .npy file"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_DIR / "full_model" / "best_model.pth"),
        help="Path to trained model checkpoint"
    )

    args = parser.parse_args()

    run_device = torch.device(DEVICE)

    checkpoint_path = resolve_path(args.checkpoint)

    model = load_model(
        checkpoint_path,
        run_device
    )

    audio, device, preference = load_input(
        args.audio,
        args.device,
        args.preference
    )

    pred_eq = predict_eq(
        model,
        audio,
        device,
        preference,
        run_device
    )

    print("\nPredicted EQ:")
    for index, value in enumerate(pred_eq):
        print(f"Band {index + 1}: {value:.4f} dB")


if __name__ == "__main__":
    main()