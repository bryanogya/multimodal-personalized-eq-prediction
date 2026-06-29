import random

import matplotlib.pyplot as plt
import torch

from configs.frequencies import TARGET_FREQ
from configs.eq import EQ_FREQS
from configs.paths import FIGURE_DIR

from src.dataset.data_loader import create_dataloaders
from src.utils.interpolate_eq import LogFrequencyInterpolator


def plot_multiple_target_curves(dataset, n_samples=10):
    indices = random.sample(
        range(len(dataset)),
        min(n_samples, len(dataset))
    )

    plt.figure(figsize=(12, 6))

    for idx in indices:
        sample = dataset[idx]
        target_curve = sample["target_curve"].numpy()

        plt.semilogx(
            TARGET_FREQ,
            target_curve,
            alpha=0.6
        )

    plt.title("Target EQ Curves")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("EQ Gain (dB)")
    plt.grid(True, which="both")
    plt.tight_layout()

    save_path = FIGURE_DIR / "dataset" / "target_eq_curves.png"
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved to: {save_path}")


def plot_target_components(dataset, idx=None):
    if idx is None:
        idx = random.randint(
            0,
            len(dataset) - 1
        )

    sample = dataset[idx]

    device_response = sample["device"].numpy()
    target_curve = sample["target_curve"].numpy()
    personalized_target = sample["personalized_target"].numpy()

    plt.figure(figsize=(12, 6))

    plt.semilogx(
        TARGET_FREQ,
        device_response,
        label="Device Response"
    )

    plt.semilogx(
        TARGET_FREQ,
        personalized_target,
        label="Personalized Target"
    )

    plt.semilogx(
        TARGET_FREQ,
        target_curve,
        label="Target EQ Curve"
    )

    plt.title(f"Target Components, Sample {idx}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude / Gain (dB)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    save_path = FIGURE_DIR / "dataset" / f"target_components_sample_{idx}.png"
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved to: {save_path}")


def plot_6band_reconstruction(dataset, idx=None):
    if idx is None:
        idx = random.randint(
            0,
            len(dataset) - 1
        )

    sample = dataset[idx]

    target_curve = sample["target_curve"].numpy()
    target_eq = sample["target_eq"]

    interpolator = LogFrequencyInterpolator()

    with torch.no_grad():
        reconstructed_curve = interpolator(
            target_eq.unsqueeze(0)
        ).squeeze(0).numpy()

    plt.figure(figsize=(12, 6))

    plt.semilogx(
        TARGET_FREQ,
        target_curve,
        label="Original Target Curve"
    )

    plt.semilogx(
        TARGET_FREQ,
        reconstructed_curve,
        label="6-Band Reconstruction"
    )

    plt.scatter(
        EQ_FREQS,
        target_eq.numpy(),
        label="6-Band Target EQ",
        zorder=3
    )

    plt.title(f"6-Band Reconstruction vs Target Curve, Sample {idx}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("EQ Gain (dB)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    save_path = FIGURE_DIR / "dataset" / f"target_reconstruction_sample_{idx}.png"
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved to: {save_path}")


def main():
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_loader, val_loader, test_loader = create_dataloaders()

    dataset = train_loader.dataset

    plot_multiple_target_curves(
        dataset,
        n_samples=10
    )

    plot_target_components(
        dataset
    )

    plot_6band_reconstruction(
        dataset
    )


if __name__ == "__main__":
    main()