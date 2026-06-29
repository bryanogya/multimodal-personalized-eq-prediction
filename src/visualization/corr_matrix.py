import numpy as np
import matplotlib.pyplot as plt

from configs.paths import CHECKPOINT_DIR, FIGURE_DIR
from configs.eq import EQ_FREQS
from configs.plot_style import *


def load_results():
    result_path = CHECKPOINT_DIR / "full_model" / "test_results.npz"
    
    return np.load(result_path)


def compute_correlation_matrix(
    pred_eq,
    target_eq,
):
    n_bands = pred_eq.shape[1]

    corr_matrix = np.zeros((n_bands, n_bands))

    for i in range(n_bands):
        for j in range(n_bands):
            corr = np.corrcoef(
                pred_eq[:, i],
                target_eq[:, j]
            )[0, 1]

            corr_matrix[i, j] = corr

    return corr_matrix


def save_corr_matrix(corr_matrix):
    apply_plot_style()

    output_dir = FIGURE_DIR / "correlation"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    labels = [f"{int(freq)} Hz" for freq in EQ_FREQS]

    fig, ax = plt.subplots(
        figsize=FIG_LARGE,
        dpi=DPI
    )

    im = ax.imshow(
        corr_matrix,
        vmin=-1,
        vmax=1,
        cmap="coolwarm"
    )

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right"
    )
    ax.set_yticklabels(labels)
    ax.set_xlabel("Target EQ Band")
    ax.set_ylabel("Predicted EQ Band")
    ax.set_title("Correlation Matrix Between Predicted and Target EQ Bands")

    cbar = fig.colorbar(
        im,
        ax=ax
    )

    cbar.set_label(
        "Pearson Correlation"
    )

    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            value = corr_matrix[i, j]

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=10
            )

    plt.tight_layout()

    save_path = output_dir / "correlation_matrix.png"
    plt.savefig(
        save_path,
        dpi=DPI,
        bbox_inches=SAVE_BBOX
    )

    plt.close()

    print(f"Saved: {save_path}")


def save_diagonal_correlation_bar(
    corr_matrix,
):
    apply_plot_style()

    output_dir = FIGURE_DIR / "correlation"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    labels = [f"{int(freq)} Hz" for freq in EQ_FREQS]
    diagonal_corr = np.diag(corr_matrix)

    plt.figure(
        figsize=FIG_DEFAULT,
        dpi=DPI
    )

    bars = plt.bar(
        labels,
        diagonal_corr,
        color=BAR_COLOR,
        edgecolor="black"
    )

    plt.ylim(0.9, 1.01)
    plt.xlabel("Frequency Band")
    plt.ylabel("Pearson Correlation")
    plt.title("Target vs Prediction Correlation per EQ Band", pad=15)

    for bar, value in zip(
        bars,
        diagonal_corr
    ):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.008,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=TICK_SIZE
        )

    plt.tight_layout()

    save_path = output_dir / "diagonal_correlation_bar.png"
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

    corr_matrix = compute_correlation_matrix(
        pred_eq=pred_eq,
        target_eq=target_eq
    )

    save_corr_matrix(
        corr_matrix=corr_matrix
    )

    save_diagonal_correlation_bar(
        corr_matrix=corr_matrix
    )


if __name__ == "__main__":
    main()