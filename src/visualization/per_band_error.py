import json

import matplotlib.pyplot as plt

from configs.paths import CHECKPOINT_DIR, FIGURE_DIR
from configs.plot_style import *
from configs.eq import EQ_FREQS


def load_per_band():
    metrics_path = (CHECKPOINT_DIR / "full_model" / "test_metrics.json")

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    return metrics


def save_band_bar(
    values,
    title,
    ylabel,
    filename,
):
    apply_plot_style()

    output_dir = FIGURE_DIR / "full_model"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=FIG_DEFAULT,
        dpi=DPI
    )

    bars = plt.bar(
        [f"{int(freq)}" for freq in EQ_FREQS],
        values,
        color=BAR_COLOR,
        edgecolor="black",
    )

    plt.title(title)
    plt.xlabel("Frequency Band (Hz)")
    plt.ylabel(ylabel)

    plt.ylim(
        0,
        max(values) * 1.15
    )

    for bar, value in zip(
        bars,
        values
    ):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=TICK_SIZE
        )

    plt.tight_layout()

    plt.savefig(
        output_dir / filename,
        dpi=DPI,
        bbox_inches=SAVE_BBOX
    )

    plt.close()

    print(f"Saved: {output_dir / filename}")


def main():

    metrics = load_per_band()

    save_band_bar(
        values=metrics["per_band_mae"],
        title="Per-band Mean Absolute Error",
        ylabel="Mean Absolute Error (dB)",
        filename="per_band_mae.png"
    )

    save_band_bar(
        values=metrics["per_band_rmse"],
        title="Per-band Root Mean Squared Error",
        ylabel="Root Mean Squared Error (dB)",
        filename="per_band_rmse.png"
    )


if __name__ == "__main__":
    main()