import numpy as np
import matplotlib.pyplot as plt

from configs.eq import EQ_FREQS
from configs.paths import CHECKPOINT_DIR, FIGURE_DIR
from configs.plot_style import *


N_SAMPLES = 5


def load_test_results():

    results_path = (
        CHECKPOINT_DIR
        / "full_model"
        / "test_results.npz"
    )

    return np.load(results_path)

def select_representative_samples(
    pred_eq,
    target_eq,
    n_random=2,
    seed=42
):
    errors = np.mean(
        np.abs(pred_eq - target_eq),
        axis=1
    )

    best_idx = int(np.argmin(errors))
    worst_idx = int(np.argmax(errors))
    median_idx = int(
        np.argsort(errors)[len(errors) // 2]
    )

    rng = np.random.default_rng(seed)

    excluded = {
        best_idx,
        median_idx,
        worst_idx
    }

    candidates = [
        i for i in range(len(errors))
        if i not in excluded
    ]

    random_indices = rng.choice(
        candidates,
        size=n_random,
        replace=False
    ).tolist()

    selected = [
        ("best", best_idx),
        ("median", median_idx),
        ("worst", worst_idx),
    ]

    for i, idx in enumerate(random_indices):
        selected.append(
            (f"random_{i+1}", int(idx))
        )

    return selected, errors

def save_eq_plot(
    pred_eq,
    target_eq,
    sample_idx,
    label,
):
    apply_plot_style()

    output_dir = FIGURE_DIR / "prediction"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=FIG_DEFAULT,
        dpi=DPI
    )

    plt.plot(
        EQ_FREQS,
        target_eq,
        marker="o",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        color=TARGET_COLOR,
        label="Target EQ"
    )

    plt.plot(
        EQ_FREQS,
        pred_eq,
        marker="s",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        color=PREDICT_COLOR,
        label="Predicted EQ"
    )

    plt.xscale("log")

    plt.xticks(
        EQ_FREQS,
        [f"{int(f)}" for f in EQ_FREQS]
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain (dB)")
    plt.title(f"EQ Prediction - {label.title()} Case")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir /
        f"{label}_sample_{sample_idx+1:03d}_eq.png",
        dpi=DPI,
        bbox_inches=SAVE_BBOX
    )

    plt.close()
    
def save_response_plot(
    device_response,
    pred_response,
    target_response,
    sample_idx,
    label
):
    apply_plot_style()

    output_dir = FIGURE_DIR / "prediction"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    x = np.arange(
        len(device_response)
    )

    plt.figure(
        figsize=FIG_WIDE,
        dpi=DPI
    )

    plt.plot(
        x,
        device_response,
        linewidth=LINE_WIDTH,
        color=DEVICE_COLOR,
        label="Device Response"
    )

    plt.plot(
        x,
        pred_response,
        linewidth=LINE_WIDTH,
        color=PREDICT_COLOR,
        label="Predicted Response"
    )

    plt.plot(
        x,
        target_response,
        linewidth=LINE_WIDTH,
        color=TARGET_COLOR,
        label="Personalized Target"
    )

    plt.xlabel("Frequency Bin")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"Frequency Response - {label.title()} Case")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir /
        f"{label}_sample_{sample_idx+1:03d}_response.png",
        dpi=DPI,
        bbox_inches=SAVE_BBOX
    )

    plt.close()


def main():

    results = load_test_results()

    pred_eq = results["pred_eq"]
    target_eq = results["target_eq"]

    pred_response = results["pred_response"]
    target_response = results["target_response"]
    device_response = results["device_response"]

    selected_samples, errors = select_representative_samples(
        pred_eq=pred_eq,
        target_eq=target_eq,
        n_random=2,
        seed=42
    )

    for label, idx in selected_samples:

        print(
            f"{label} | index={idx} | MAE={errors[idx]:.4f} dB"
        )

        save_eq_plot(
            pred_eq=pred_eq[idx],
            target_eq=target_eq[idx],
            sample_idx=idx,
            label=label
        )

        save_response_plot(
            device_response=device_response[idx],
            pred_response=pred_response[idx],
            target_response=target_response[idx],
            sample_idx=idx,
            label=label
        )

    print("Representative prediction plots saved.")
    

if __name__ == "__main__":
    main()