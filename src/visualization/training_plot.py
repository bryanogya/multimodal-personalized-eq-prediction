import json

import matplotlib.pyplot as plt

from configs.paths import CHECKPOINT_DIR, FIGURE_DIR
from configs.plot_style import *


def load_history():
    history_path = CHECKPOINT_DIR / "full_model" / "history.json"

    with open(history_path, "r") as f:
        history = json.load(f)

    return history


def save_line_plot(
    epochs,
    train_values,
    val_values,
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

    plt.plot(
        epochs,
        train_values,
        label="Train",
        linewidth=LINE_WIDTH,
        marker="o",
        markersize=MARKER_SIZE,
        color=TRAIN_COLOR
    )

    plt.plot(
        epochs,
        val_values,
        label="Validation",
        linewidth=LINE_WIDTH,
        marker="s",
        markersize=MARKER_SIZE,
        color=VAL_COLOR
    )

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

    save_path = output_dir / filename

    plt.savefig(
        save_path,
        dpi=DPI,
        bbox_inches=SAVE_BBOX
    )

    plt.close()

    print(f"Saved: {save_path}")


def save_single_line_plot(
    epochs,
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

    plt.plot(
        epochs,
        values,
        linewidth=LINE_WIDTH,
        marker="o",
        markersize=MARKER_SIZE
    )

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.tight_layout()

    save_path = output_dir / filename

    plt.savefig(
        save_path,
        dpi=DPI,
        bbox_inches=SAVE_BBOX
    )

    plt.close()

    print(f"Saved: {save_path}")


def main():
    history = load_history()

    epochs = list(
        range(
            1,
            len(history["train_loss"]) + 1
        )
    )

    save_line_plot(
        epochs=epochs,
        train_values=history["train_loss"],
        val_values=history["val_loss"],
        title="Training and Validation Loss",
        ylabel="Loss",
        filename="history_loss.png"
    )

    save_line_plot(
        epochs=epochs,
        train_values=history["train_mse"],
        val_values=history["val_mse"],
        title="Training and Validation MSE",
        ylabel="MSE",
        filename="history_mse.png"
    )

    save_line_plot(
        epochs=epochs,
        train_values=history["train_mae"],
        val_values=history["val_mae"],
        title="Training and Validation MAE",
        ylabel="MAE (dB)",
        filename="history_mae.png"
    )

    save_line_plot(
        epochs=epochs,
        train_values=history["train_spec"],
        val_values=history["val_spec"],
        title="Training and Validation Spectral Loss",
        ylabel="Spectral Loss",
        filename="history_spectral_loss.png"
    )

    save_line_plot(
        epochs=epochs,
        train_values=history["train_sc"],
        val_values=history["val_sc"],
        title="Training and Validation Spectral Convergence",
        ylabel="SC",
        filename="history_sc.png"
    )

    save_line_plot(
        epochs=epochs,
        train_values=history["train_lsd"],
        val_values=history["val_lsd"],
        title="Training and Validation Log Spectral Distance",
        ylabel="LSD (dB)",
        filename="history_lsd.png"
    )

    save_single_line_plot(
        epochs=epochs,
        values=history["learning_rate"],
        title="Learning Rate Schedule",
        ylabel="Learning Rate",
        filename="history_learning_rate.png"
    )


if __name__ == "__main__":
    main()