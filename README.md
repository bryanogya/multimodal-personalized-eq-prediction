# Multimodal Personalized EQ Prediction

Deep learning project for predicting personalized equalizer settings using multimodal input.

The model uses audio features, device information, and user preference data to predict target EQ values. This project compares several input combinations through baseline evaluation and ablation study.

## Project Overview

Personalized equalizer prediction aims to estimate EQ settings that better match user listening preferences. Instead of using one fixed EQ profile for all users, this project learns from several data sources:

- Audio data
- Squiglink device data
- User preference data

The final model combines all available modalities to produce better EQ prediction.

## Dataset

This project uses three main data sources:

| Dataset | Amount | Description |
|---|---:|---|
| GTZAN Audio | 300 data | Audio samples used as the main audio input |
| Squiglink Device | 30 data | Device-related information |
| User Preference | 50 data | User preference data for personalized EQ learning |

## Model Input

The model is evaluated using several input combinations:

| Model Variant | Input Used |
|---|---|
| Audio Only | Audio feature only |
| Audio + Device | Audio feature and device information |
| Audio + Preference | Audio feature and user preference |
| Device + Preference | Device information and user preference |
| Full Model | Audio, device, and user preference |

## Project Structure

```text
multimodal-personalized-eq-prediction/
├── configs/
│   ├── paths.py
│   ├── plot_style.py
│   └── eq.py
│
├── data/
│   └── README.md
│
├── src/
│   ├── dataset/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── visualization/
│
├── checkpoints/
│   ├── baseline/
│   ├── ablation/
│   └── full_model/
│
├── outputs/
│   ├── baseline/
│   ├── ablation/
│   ├── evaluation/
│   └── figures/
│
├── requirements.txt
└── README.md
````

## Evaluation Metrics

The model performance is evaluated using:

| Metric | Description             |
| ------ | ----------------------- |
| MSE    | Mean Squared Error      |
| MAE    | Mean Absolute Error     |
| RMSE   | Root Mean Squared Error |
| SC     | Spectral Convergence    |
| LSD    | Log Spectral Distance   |

Lower values indicate better performance for MSE, MAE, RMSE, and LSD.

## Baseline Comparison

This project compares the proposed model with simple baseline methods:

| Baseline       | Description                              |
| -------------- | ---------------------------------------- |
| Zero EQ        | Uses zero equalizer values as prediction |
| Mean EQ        | Uses average EQ values as prediction     |
| Proposed Model | Uses the trained deep learning model     |

The baseline comparison is used to verify whether the model learns meaningful EQ patterns beyond simple fixed predictions.

## Ablation Study

An ablation study is performed to measure the contribution of each modality.

The evaluated variants are:

* Audio Only
* Audio + Device
* Audio + Preference
* Device + Preference
* Full Model

This analysis helps identify whether audio, device, and preference features improve personalized EQ prediction.

## Output Examples

The project generates several visual outputs:

* Dataset summary
* EQ target curves
* Prediction vs target comparison
* Baseline comparison bar charts
* Ablation study comparison bar charts
* Histogram per EQ dimension
* Correlation matrix
* 3D preference scatter plot

Example output folders:

```text
outputs/
├── figures/
├── baseline/
├── ablation/
└── evaluation/
```

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python train.py
```

Run evaluation:

```bash
python evaluate.py
```

Generate visualization:

```bash
python visualize.py
```

Adjust the command based on the actual script names in the project.

## Results

The Full Model is expected to perform better than partial input combinations because it uses complete multimodal information.

The main comparison focuses on:

* How much better the proposed model is compared to Zero EQ and Mean EQ
* How much each modality contributes to EQ prediction
* Whether user preference improves personalization
* Whether device information improves EQ estimation

## Key Findings

* Audio features provide the main signal for EQ prediction.
* Device information helps the model understand playback characteristics.
* User preference data supports personalization.
* The Full Model gives the most complete input representation.
* Ablation study confirms the importance of combining multiple modalities.

## Technologies Used

* Python
* NumPy
* Matplotlib
* PyTorch or TensorFlow
* JSON
* NPZ
* Git and GitHub

## Author

Bryan Ogya Kusuma

## License

This project is intended for academic and research purposes.

