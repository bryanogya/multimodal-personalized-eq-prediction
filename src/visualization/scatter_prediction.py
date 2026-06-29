import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from configs.eq import EQ_FREQS
from configs.paths import CHECKPOINT_DIR, FIGURE_DIR
from configs.plot_style import *

def load_results():

    result_path = (
        CHECKPOINT_DIR
        / "full_model"
        / "test_results.npz"
    )

    return np.load(result_path)


def save_scatter(
    target,
    prediction,
    freq,
):
    apply_plot_style()
    
    r2 = r2_score(target, prediction)

    output_dir = FIGURE_DIR / "scatter"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=FIG_SQUARE,
        dpi=DPI
    )

    plt.scatter(
        target,
        prediction,
        s=20,
        alpha=0.6,
    )

    min_value = min(
        target.min(),
        prediction.min()
    )

    max_value = max(
        target.max(),
        prediction.max()
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        "--",
        color="red",
        linewidth=LINE_WIDTH,
        label="Ideal Prediction"
    )
    
    plt.text(
        0.05,
        0.95,
        f"$R^2$ = {r2:.3f}",
        transform=plt.gca().transAxes,
        va="top",
        bbox=dict(facecolor="white", alpha=0.8)
    )

    plt.xlabel("Target Gain (dB)")
    plt.ylabel("Predicted Gain (dB)")
    plt.title(f"{int(freq)} Hz")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir /
        f"scatter_{int(freq)}Hz.png",
        dpi=DPI,
        bbox_inches=SAVE_BBOX
    )

    plt.close()

    print(f"Saved: scatter_{int(freq)}Hz.png")


def main():
    results = load_results()

    pred_eq = results["pred_eq"]
    target_eq = results["target_eq"]

    for i, freq in enumerate(EQ_FREQS):
        save_scatter(
            target=target_eq[:, i],
            prediction=pred_eq[:, i],
            freq=freq
        )


if __name__ == "__main__":
    main()