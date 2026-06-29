import numpy as np
import torch
import matplotlib.pyplot as plt

from configs.paths import CHECKPOINT_DIR, FIGURE_DIR
from configs.eq import EQ_FREQS
from configs.plot_style import *

from src.dataset.data_loader import create_dataloaders
from src.models.multimodal_model import MultimodalEQModel


def get_model_output(output):
    if isinstance(output, dict):
        return output["eq"]

    return output


def load_model(run_device):
    model = MultimodalEQModel().to(run_device)

    checkpoint_path = (
        CHECKPOINT_DIR
        / "full_model"
        / "best_model.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=run_device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Best Val Loss: {checkpoint['best_val_loss']:.4f}")

    return model


def get_reference_batch(test_loader, run_device):
    batch = next(iter(test_loader))

    audio = batch["audio"][0:1].to(run_device)
    device = batch["device"][0:1].to(run_device)
    preference = batch["preference"][0:1].to(run_device)

    return audio, device, preference, batch


def predict_eq(
    model,
    audio,
    device,
    preference,
):
    with torch.no_grad():
        output = model(
            audio=audio,
            device=device,
            preference=preference
        )

        pred_eq = get_model_output(output)

    return pred_eq.detach().cpu().numpy()[0]


def save_preference_sensitivity(
    model,
    audio,
    device,
    base_preference,
    run_device,
):
    apply_plot_style()

    output_dir = FIGURE_DIR / "sensitivity"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    scenarios = {
        "Low Bass": np.array([[0.2, 0.5, 0.5]], dtype=np.float32),
        "Neutral": np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
        "High Bass": np.array([[0.8, 0.5, 0.5]], dtype=np.float32),
    }

    plt.figure(
        figsize=FIG_DEFAULT,
        dpi=DPI
    )

    for label, pref in scenarios.items():
        pref_tensor = torch.tensor(
            pref,
            dtype=torch.float32,
            device=run_device
        )

        pred_eq = predict_eq(
            model=model,
            audio=audio,
            device=device,
            preference=pref_tensor
        )

        plt.plot(
            EQ_FREQS,
            pred_eq,
            marker="o",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label=label
        )

    plt.xscale("log")
    plt.xticks(
        EQ_FREQS,
        [f"{int(freq)}" for freq in EQ_FREQS]
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Predicted EQ Gain (dB)")
    plt.title("Preference Sensitivity - Bass Preference")
    plt.legend()
    plt.tight_layout()

    save_path = output_dir / "preference_sensitivity_bass.png"

    plt.savefig(
        save_path,
        dpi=DPI,
        bbox_inches=SAVE_BBOX
    )

    plt.close()

    print(f"Saved: {save_path}")


def save_preference_sensitivity_all(
    model,
    audio,
    device,
    run_device,
):
    preferences = {
        "Bass Preference": {
            "Low": [0.2, 0.5, 0.5],
            "Neutral": [0.5, 0.5, 0.5],
            "High": [0.8, 0.5, 0.5],
            "filename": "preference_sensitivity_bass.png",
        },
        "Mid Preference": {
            "Low": [0.5, 0.2, 0.5],
            "Neutral": [0.5, 0.5, 0.5],
            "High": [0.5, 0.8, 0.5],
            "filename": "preference_sensitivity_mid.png",
        },
        "Treble Preference": {
            "Low": [0.5, 0.5, 0.2],
            "Neutral": [0.5, 0.5, 0.5],
            "High": [0.5, 0.5, 0.8],
            "filename": "preference_sensitivity_treble.png",
        },
    }

    output_dir = FIGURE_DIR / "sensitivity"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for title, config in preferences.items():
        apply_plot_style()

        plt.figure(
            figsize=FIG_DEFAULT,
            dpi=DPI
        )

        for label in ["Low", "Neutral", "High"]:
            pref = torch.tensor(
                [config[label]],
                dtype=torch.float32,
                device=run_device
            )

            pred_eq = predict_eq(
                model=model,
                audio=audio,
                device=device,
                preference=pref
            )

            plt.plot(
                EQ_FREQS,
                pred_eq,
                marker="o",
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                label=label
            )

        plt.xscale("log")
        plt.xticks(
            EQ_FREQS,
            [f"{int(freq)}" for freq in EQ_FREQS]
        )

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Predicted EQ Gain (dB)")
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        save_path = output_dir / config["filename"]

        plt.savefig(
            save_path,
            dpi=DPI,
            bbox_inches=SAVE_BBOX
        )

        plt.close()

        print(f"Saved: {save_path}")


def save_device_sensitivity(
    model,
    audio,
    preference,
    test_loader,
    run_device,
    n_devices=3,
):
    apply_plot_style()

    output_dir = FIGURE_DIR / "sensitivity"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    device_list = []

    for batch in test_loader:
        devices = batch["device"]

        for i in range(devices.shape[0]):
            device_list.append(
                devices[i:i+1]
            )

            if len(device_list) >= n_devices:
                break

        if len(device_list) >= n_devices:
            break

    plt.figure(
        figsize=FIG_DEFAULT,
        dpi=DPI
    )

    for i, device in enumerate(device_list):
        device = device.to(run_device)

        pred_eq = predict_eq(
            model=model,
            audio=audio,
            device=device,
            preference=preference
        )

        plt.plot(
            EQ_FREQS,
            pred_eq,
            marker="o",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label=f"Device {i+1}"
        )

    plt.xscale("log")
    plt.xticks(
        EQ_FREQS,
        [f"{int(freq)}" for freq in EQ_FREQS]
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Predicted EQ Gain (dB)")
    plt.title("Device Sensitivity")
    plt.legend()
    plt.tight_layout()

    save_path = output_dir / "device_sensitivity.png"

    plt.savefig(
        save_path,
        dpi=DPI,
        bbox_inches=SAVE_BBOX
    )

    plt.close()

    print(f"Saved: {save_path}")


def main():
    run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== SENSITIVITY PLOT ===")
    print(f"Device: {run_device}")

    _, _, test_loader = create_dataloaders()

    model = load_model(run_device=run_device)

    audio, device, preference, batch = get_reference_batch(
        test_loader=test_loader,
        run_device=run_device
    )

    save_preference_sensitivity_all(
        model=model,
        audio=audio,
        device=device,
        run_device=run_device
    )

    save_device_sensitivity(
        model=model,
        audio=audio,
        preference=preference,
        test_loader=test_loader,
        run_device=run_device,
        n_devices=3
    )


if __name__ == "__main__":
    main()