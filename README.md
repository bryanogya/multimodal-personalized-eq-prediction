# Multimodal Personalized EQ Prediction 🎧✨

This project leverages deep learning to predict personalized audio equalizer (EQ) settings by utilizing multimodal input data. It combines audio features, device-specific frequency responses, and user preference data to intelligently adjust EQ settings, aiming for an audio experience that better matches individual listening preferences.

## Table of Contents

* [Project Overview](#project-overview)
* [Features](#features)
* [Tech Stack](#tech-stack)
* [Dataset](#dataset)
* [Model Input Configurations](#model-input-configurations)
* [Installation](#installation)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [Evaluation Metrics](#evaluation-metrics)
* [Baseline Comparison](#baseline-comparison)
* [Ablation Study](#ablation-study)
* [Key Findings](#key-findings)
* [License](#license)
* [Contributing](#contributing)
* [Author](#author)

## Project Overview 🚀

Personalized equalizer prediction is a complex task that aims to tailor audio output to individual user tastes. This project moves beyond generic EQ profiles by learning from several distinct data sources:

- **Audio Data**: Extracted Mel-spectrograms to capture the acoustic characteristics of audio content.
- **Device Data**: Frequency response data specific to audio playback devices (e.g., headphones, speakers).
- **User Preference Data**: Learned embeddings representing individual listening preferences.

The ultimate goal is to build a model that fuses these modalities to deliver superior EQ predictions, enhancing the listening experience.

## Features ⭐

* **Multimodal Fusion**: Integrates audio features, device frequency responses, and user preferences.
* **Personalized EQ Prediction**: Tailors EQ settings based on individual user preferences.
* **Comprehensive Evaluation**: Assesses model performance using various metrics including MSE, MAE, RMSE, Spectral Convergence (SC), and Log Spectral Distance (LSD).
* **Ablation Studies**: Analyzes the contribution of each modality (audio, device, preference) to the overall prediction accuracy.
* **Baseline Comparisons**: Benchmarks against simple methods like Zero EQ and Mean EQ to demonstrate the model's effectiveness.
* **Visualization Tools**: Generates plots for training history, correlation matrices, scatter plots of predictions vs. targets, and sensitivity analyses.

## Tech Stack 🛠️

* **Primary Language**: Python 
* **Frameworks**: PyTorch, NumPy, Pandas, Matplotlib, Librosa, Scikit-learn
* **Data Handling**: CSV, NPZ, NPY

## Dataset 📊

This project utilizes three primary datasets:

| Dataset             | Amount     | Description                                     |
| ------------------- | ---------- | ----------------------------------------------- |
| GTZAN Audio         | 300 samples| Audio samples for feature extraction            |
| Squiglink Device    | 30 devices | Device-specific frequency response data         |
| User Preference     | 50 profiles| Learned user preference vectors                 |

## Model Input Configurations ⚙️

The model's performance is evaluated across different input combinations to understand the impact of each modality:

| Model Variant         | Input Used                                 |
| --------------------- | ------------------------------------------ |
| Audio Only            | Audio features only                        |
| Audio + Device        | Audio features and device information      |
| Audio + Preference    | Audio features and user preference         |
| Device + Preference   | Device information and user preference     |
| **Full Model**        | Audio, device, and user preference         |

## Installation 📋

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bryanogya/multimodal-personalized-eq-prediction.git
   cd multimodal-personalized-eq-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage ▶️

This project provides several entry points for training, evaluation, and visualization.

**Core Scripts:**

* **Training**: `python main.py --mode train`
* **Auxiliary Training (for response curve prediction)**: `python main.py --mode train_aux`
* **Evaluation**: `python main.py --mode evaluate`
* **Auxiliary Evaluation**: `python main.py --mode evaluate_aux`
* **Baseline Evaluation**: `python main.py --mode baseline`
* **Ablation Study**: `python main.py --mode ablation`
* **Inference/Prediction**: `python main.py --mode predict --audio <audio.npy> --device <device.npy> --preference <preference.npy>`

**Visualization Scripts:**

* **Training Plots**: `python main.py --mode training_plot`
* **Ablation Comparison Plots**: `python main.py --mode ablation_plot`
* **Baseline Comparison Plots**: `python main.py --mode baseline_plot`
* **Prediction Plots**: `python main.py --mode prediction_plot`

**Example Prediction:**

To predict EQ for a specific audio sample, device response, and preference:

```bash
python main.py --mode predict \
    --audio data/processed/audio/sample_0001.npy \
    --device data/processed/device/7Hz Salnotes Zero.npy \
    --preference data/processed/preferences/user_preferences.npy \
    --preference-index 0
```

## Project Structure 📁

```text
multimodal-personalized-eq-prediction/
├── configs/         # Configuration files (paths, model params, training settings)
│   ├── audio.py
│   ├── dataset.py
│   ├── eq.py
│   ├── frequencies.py
│   ├── model.py
│   ├── paths.py
│   ├── plot_style.py
│   ├── preferences.py
│   └── training.py
│
├── data/
│   ├── raw/         # Raw datasets (audio, device responses)
│   ├── processed/   # Processed data (features, normalized responses)
│   ├── metadata/    # Metadata CSV files
│   └── splits/      # Train/val/test split definitions
│
├── src/
│   ├── dataset/     # Data loading and preprocessing
│   ├── evaluation/  # Evaluation scripts and metrics
│   ├── inference/   # Inference and prediction logic
│   ├── models/      # Neural network model definitions
│   ├── preprocessing/ # Scripts for data preprocessing
│   ├── training/    # Training scripts and loss functions
│   └── visualization/ # Scripts for generating plots and reports
│
├── checkpoints/    # Saved model checkpoints and experiment results
│   ├── ablation/
│   ├── aux_model/
│   └── full_model/
│
├── outputs/
│   ├── figures/     # Generated plots and visualizations
│   ├── baseline/
│   └── ablation/
│
├── requirements.txt # Project dependencies
└── main.py          # Main entry point for running experiments
```

## Evaluation Metrics 🎯

Model performance is assessed using the following metrics:

| Metric | Description                                       |
| ------ | ------------------------------------------------- |
| MSE    | Mean Squared Error (lower is better)              |
| MAE    | Mean Absolute Error (lower is better)             |
| RMSE   | Root Mean Squared Error (lower is better)         |
| SC     | Spectral Convergence (higher is better)           |
| LSD    | Log Spectral Distance (lower is better)           |

## Baseline Comparison ⚖️

This project compares the proposed deep learning model against simple baseline methods:

| Baseline       | Description                                      |
| -------------- | ------------------------------------------------ |
| Zero EQ        | Predicts zero gain for all EQ bands.           |
| Mean EQ        | Predicts the average EQ values from the training set. |
| Proposed Model | The trained multimodal deep learning model.      |

The baseline comparison validates the model's ability to learn meaningful EQ patterns beyond trivial predictions.

## Ablation Study 🔬

An ablation study is conducted to quantify the impact of each input modality on EQ prediction performance. The following variants are evaluated:

*   **Audio Only**
*   **Audio + Device**
*   **Audio + Preference**
*   **Device + Preference**
*   **Full Model (Audio + Device + Preference)**

This analysis highlights the contribution of each feature set to personalized EQ prediction.

## Key Findings 💡

*   **Audio features** serve as the primary signal for EQ prediction.
*   **Device information** aids in understanding playback characteristics, refining predictions.
*   **User preference data** is crucial for personalization, adapting EQ to individual tastes.
*   The **Full Model**, utilizing all modalities, demonstrates the most robust performance, indicating the synergistic benefit of combining diverse data sources.
*   The ablation study confirms the importance of **multimodal integration** for achieving accurate and personalized EQ.

## License 📜

This project is intended for academic and research purposes.

## Contributing 🤝

Contributions are welcome! Please feel free to submit pull requests or open issues for any improvements or bug fixes.

## Author 👨‍💻

*   **Bryan Ogya Kusuma**
*   **Ahmad Ghifari Ramadhani**
*   **Muhammad Alvarezi Usamah**

## Important Links 🔗

*   **Repository**: [multimodal-personalized-eq-prediction](https://github.com/bryanogya/multimodal-personalized-eq-prediction)
*   **GTZAN Dataset**: [GTZAN Dataset-Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
*   **SquigLink**: [Squiglink - IEM frequency response database by Super* Review](https://squig.link/)

---
