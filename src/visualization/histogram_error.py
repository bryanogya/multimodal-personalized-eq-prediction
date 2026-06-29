import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import skew

from configs.plot_style import *
from configs.paths import CHECKPOINT_DIR, FIGURE_DIR
from configs.eq import EQ_FREQS

def load_results():
    result_path = CHECKPOINT_DIR / "full_model" / "test_results.npz"

    return np.load(result_path)

def save_per_band_histogram(
    pred_eq,
    target_eq,
):
    apply_plot_style()

    output_dir = FIGURE_DIR / "histogram"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=FIG_EXTRA_LARGE,
        dpi=DPI
    )
    
    fig.suptitle(
    "Per-Band Prediction Error Distribution",
    fontsize=SUPTITLE_SIZE,
    y=0.98
)
    
    legend_elements = [
            Line2D(
                [0], [0],
                color="red",
                linestyle="--",
                linewidth=2,
                label="Zero Error"
            ),
            Line2D(
                [0], [0],
                color="orange",
                linestyle=":",
                linewidth=2,
                label="Mean Error"
            ),
        ]
    
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        fontsize=LEGEND_SIZE,
        frameon=True
    )

    axes = axes.flatten()

    for i, freq in enumerate(EQ_FREQS):
        errors = pred_eq[:, i] - target_eq[:, i]
        mae = np.mean(np.abs(errors))
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        skewness = skew(errors)

        axes[i].hist(
            errors,
            bins=25,
            color=BAR_COLOR,
            edgecolor="black",
            alpha=0.8
        )

        axes[i].axvline(
            0,
            color="red",
            linestyle="--",
            linewidth=LINE_WIDTH,
        )

        axes[i].axvline(
            mean_error,
            color="orange",
            linestyle=":",
            linewidth=LINE_WIDTH,
        )

        axes[i].set_title(f"{int(freq)} Hz")
        axes[i].set_xlabel("Prediction Error (dB)")
        axes[i].set_ylabel("Count")
        axes[i].grid(alpha=GRID_ALPHA)

        axes[i].text(
            0.98,
            0.95,
            f"MAE = {mae:.3f}\nσ = {std_error:.3f}\nSkew = {skewness:.3f}",
            transform=axes[i].transAxes,
            ha="right",
            va="top",
            fontsize=TEXT_SIZE,
            bbox=dict(
                facecolor="white",
                alpha=0.8,
                edgecolor="gray"
            )
        )

    fig.subplots_adjust(
        top=0.88,
        hspace=0.35,
        wspace=0.25
    )
    
    save_path = output_dir / "per_band_error_distribution.png"

    plt.savefig(
        save_path,
        dpi=DPI,
        bbox_inches=SAVE_BBOX
    )

    plt.close()
    
    print(f"Saved: {save_path}")
    
def save_overall_histogram(
    pred_eq,
    target_eq,
):
    apply_plot_style()

    output_dir = FIGURE_DIR / "histogram"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    errors = (pred_eq - target_eq).flatten()
    
    plt.figure(figsize=FIG_DEFAULT, dpi=DPI)
    plt.hist(
        errors,
        bins=40,
        color=BAR_COLOR,
        edgecolor="black"
    )

    plt.axvline(
        0,
        color="red",
        linestyle="--",
        linewidth=LINE_WIDTH
    )

    plt.axvline(
        np.mean(errors),
        color="orange",
        linestyle=":",
        linewidth=LINE_WIDTH
    )

    plt.xlabel("Prediction Error (dB)")
    plt.ylabel("Count")
    plt.title("Overall Prediction Error Distribution")
    plt.grid(alpha=GRID_ALPHA)
    plt.tight_layout()
    
    save_path = output_dir / "overall_error_distribution.png"

    plt.savefig(
        save_path,
        dpi=DPI,
        bbox_inches=SAVE_BBOX
    )

    plt.close()
    
    print(f"Saved: {save_path}")
       
def main():
    results = load_results()
    
    pred_eq = results["pred_eq"]
    target_eq = results["target_eq"]  
    
    save_overall_histogram(
        pred_eq=pred_eq,
        target_eq=target_eq
    )  
    
    save_per_band_histogram(
        pred_eq=pred_eq,
        target_eq=target_eq
    )
        
    
    
if __name__ == "__main__":
    main()