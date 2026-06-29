import json

import matplotlib.pyplot as plt

from configs.paths import CHECKPOINT_DIR, FIGURE_DIR
from configs.plot_style import *

def load_baseline_comp():
    baseline_path = CHECKPOINT_DIR / "baseline" / "baseline_metrics.json"
    
    with open(baseline_path, "r") as f:
        baseline = json.load(f)
        
    return baseline

def save_bar_chart(
    zero_value,
    mean_value,
    proposed_value,
    title,
    ylabel,
    filename,
):
    apply_plot_style()

    output_dir = FIGURE_DIR / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    models = ["Zero EQ", "Mean EQ", "Proposed Model"]
    values = [zero_value, mean_value, proposed_value]
    colors = [BAR_COLOR, BAR_COLOR, PROPOSED_BAR_COLOR]

    plt.figure(figsize=FIG_DEFAULT, dpi=DPI)

    plt.bar(
        models,
        values,
        color=colors
    )

    plt.title(title)
    plt.xlabel("Model")
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
    baseline = load_baseline_comp()
    
    save_bar_chart(
        zero_value=baseline["zero_eq"]["mse"],
        mean_value=baseline["mean_eq"]["mse"],
        proposed_value=baseline["proposed_model"]["mse"],
        title="Comparison of Mean Squared Error",
        ylabel="Mean Squared Error (MSE)",
        filename="mse_baseline_comparison.png"
    )
    
    save_bar_chart(
        zero_value=baseline["zero_eq"]["mae"],
        mean_value=baseline["mean_eq"]["mae"],
        proposed_value=baseline["proposed_model"]["mae"],
        title="Comparison of Mean Absolute Error",
        ylabel="Mean Absolute Error (dB)",
        filename="mae_baseline_comparison.png"
    )
    
    save_bar_chart(
        zero_value=baseline["zero_eq"]["rmse"],
        mean_value=baseline["mean_eq"]["rmse"],
        proposed_value=baseline["proposed_model"]["rmse"],
        title="Comparison of Root Mean Squared Error",
        ylabel="Root Mean Squared Error (dB)",
        filename="rmse_baseline_comparison.png"
    )
    
    save_bar_chart(
        zero_value=baseline["zero_eq"]["sc"],
        mean_value=baseline["mean_eq"]["sc"],
        proposed_value=baseline["proposed_model"]["sc"],
        title="Comparison of Spectral Convergence",
        ylabel="Spectral Convergence",
        filename="sc_baseline_comparison.png"
    )
    
    save_bar_chart(
        zero_value=baseline["zero_eq"]["lsd"],
        mean_value=baseline["mean_eq"]["lsd"],
        proposed_value=baseline["proposed_model"]["lsd"],
        title="Comparison of Log Spectral Distance",
        ylabel="Log Spectral Distance (dB)",
        filename="lsd_baseline_comparison.png"
    )
    
if __name__ == "__main__":
    main()