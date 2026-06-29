import json

import matplotlib.pyplot as plt

from configs.paths import CHECKPOINT_DIR, FIGURE_DIR
from configs.plot_style import *

def load_ablation():
    ablation_path = CHECKPOINT_DIR / "ablation" / "ablation_results.json"
    
    with open(ablation_path, "r") as f:
        ablation = json.load(f)
        
    return ablation

"""
Audio Only = audio
Audio + Device = AD
Audio + Preference = AP
Device + Preference = DP
Full Model = FM
"""

def save_bar_ablation(
    audio_value,
    AD_value,
    AP_value,
    DP_value,
    FM_value,
    title,
    xlabel,
    filename
):
    apply_plot_style()
    
    output_dir = FIGURE_DIR / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = ["Audio", "Audio + Device",  "Audio + Preference",
             "Device + Preference", "Full Model"]
    values = [audio_value, AD_value, AP_value, 
              DP_value, FM_value,]
    colors = [BAR_COLOR, BAR_COLOR, BAR_COLOR,
              BAR_COLOR, PROPOSED_BAR_COLOR]
    
    plt.figure(figsize=FIG_DEFAULT, dpi=DPI)
    
    plt.barh(
        models,
        values,
        color=colors
    )
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Model")
    
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
    ablation = load_ablation()
    
    save_bar_ablation(
        audio_value=ablation["Audio Only"]["mse"],
        AD_value=ablation["Audio + Device"]["mse"],
        AP_value=ablation["Audio + Preference"]["mse"],
        DP_value=ablation["Device + Preference"]["mse"],
        FM_value=ablation["Full Model"]["mse"],
        title="Ablation Study Comparison - MSE",
        xlabel="Mean Squared Error (MSE)",
        filename="mse_ablation_comparison.png"
    )
    
    save_bar_ablation(
        audio_value=ablation["Audio Only"]["mae"],
        AD_value=ablation["Audio + Device"]["mae"],
        AP_value=ablation["Audio + Preference"]["mae"],
        DP_value=ablation["Device + Preference"]["mae"],
        FM_value=ablation["Full Model"]["mae"],
        title="Ablation Study Comparison - MAE",
        xlabel="Mean Abslute Error (dB)",
        filename="mae_ablation_comparison.png"
    )
    
    save_bar_ablation(
        audio_value=ablation["Audio Only"]["sc"],
        AD_value=ablation["Audio + Device"]["sc"],
        AP_value=ablation["Audio + Preference"]["sc"],
        DP_value=ablation["Device + Preference"]["sc"],
        FM_value=ablation["Full Model"]["sc"],
        title="Ablation Study Comparison - SC",
        xlabel="Spectral Convergence",
        filename="sc_ablation_comparison.png"
    )
    
    save_bar_ablation(
        audio_value=ablation["Audio Only"]["lsd"],
        AD_value=ablation["Audio + Device"]["lsd"],
        AP_value=ablation["Audio + Preference"]["lsd"],
        DP_value=ablation["Device + Preference"]["lsd"],
        FM_value=ablation["Full Model"]["lsd"],
        title="Ablation Study Comparison - LSD",
        xlabel="Log Spectral Distance (dB)",
        filename="lsd_ablation_comparison.png"
    )
    
if __name__ == "__main__":
    main()