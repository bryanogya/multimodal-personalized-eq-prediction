import numpy as np

from src.dataset.data_loader import create_dataloaders


def analyze_curve(name, curve):
    curve = np.concatenate(
        curve,
        axis=0
    )

    print(f"\n=== {name} ===")

    print("Percentiles:")
    print(
        np.percentile(
            curve,
            [0, 1, 5, 25, 50, 75, 95, 99, 100]
        )
    )

    print("Min :", np.min(curve))
    print("Max :", np.max(curve))

    print(
        "Outside ±12 dB:",
        np.mean(np.abs(curve) > 12) * 100,
        "%"
    )


def main():
    train_loader, val_loader, test_loader = create_dataloaders()

    all_target_curve = []
    all_target_curve_raw = []

    for batch in train_loader:
        all_target_curve.append(
            batch["target_curve"].numpy()
        )

        all_target_curve_raw.append(
            batch["target_curve_raw"].numpy()
        )

    analyze_curve(
        "Target Curve Clipped",
        all_target_curve
    )

    analyze_curve(
        "Target Curve Raw",
        all_target_curve_raw
    )


if __name__ == "__main__":
    main()