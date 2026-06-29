# Final Project Deep Learning

Project ini membangun model deep learning multimodal untuk memprediksi parameter equalizer berdasarkan fitur audio, karakteristik device, dan preferensi pengguna.

## Struktur Folder

```text
Final_Project/
├── configs/                 # Konfigurasi path, model, training, dan plotting
├── data/                    # Dataset raw dan processed
├── checkpoints/             # Checkpoint model dan hasil training
├── notebooks/               # Notebook eksplorasi
├── outputs/                 # Output visualisasi dan hasil eksperimen
├── src/                     # Source code utama
│   ├── data/                # Dataset loader dan data split
│   ├── evaluation/          # Evaluasi, baseline, dan ablation study
│   ├── models/              # Arsitektur model
│   ├── preprocessing/       # Preprocessing audio, device, dan preference
│   ├── training/            # Training pipeline
│   ├── utils/               # Utility function
│   └── visualization/       # Visualisasi hasil eksperimen
├── requirements.txt         # Dependency project
└── README.md                # Dokumentasi project
```

## Instalasi

Buat environment Python, lalu install dependency:

```bash
pip install -r requirements.txt
```

## Menjalankan Project

Project ini dapat dijalankan melalui `main.py` dari root folder project.

Training model utama:

```bash
python main.py --mode train
```

Evaluasi Model:

```bash
python main.py --mode evaluate
```

Inference untuk satu input baru:

```bash
python main.py --mode predict --audio data/processed/audio/sample_0000.npy --device data/processed/device/device_0000.npy --preference data/processed/preference/preference_0000.npy
```

Ablation Study:

```bash
python main.py --mode ablation
```

Visualisasi prediksi:

```bash
python main.py --mode prediction_plot
```

Melihat daftar mode yang tersedia:

```bash
python main.py --help
```

## Preprocessing Data

Jalankan preprocessing audio:

```bash
python -m src.preprocessing.audio_preprocessing
```

Jalankan preprocessing device:

```bash
python -m src.preprocessing.device_preprocessing
```

Jalankan preprocessing preference:

```bash
python -m src.preprocessing.preference_preprocessing
```

## Training Model

Training model utama:

```bash
python -m src.training.train
```

Training auxiliary model:

```bash
python -m src.training.train_aux
```

Output training akan tersimpan di:

```text
checkpoints/
```

## Evaluasi Model

Evaluasi model pada test set:

```bash
python -m src.evaluation.test
```

Evaluasi baseline:

```bash
python -m src.evaluation.baseline
```

Ablation study:

```bash
python -m src.evaluation.ablation
```

Analisis worst prediction:

```bash
python main.py --mode worst_prediction
```

Output evaluasi tersimpan dalam bentuk file JSON, NPY, atau NPZ di folder checkpoint masing-masing eksperimen.

## Visualisasi

Contoh menjalankan visualisasi prediksi:

```bash
python -m src.visualization.prediction_plot
```

Visualisasi lain tersedia di:

```text
src/visualization/
```

Output gambar tersimpan di:

```text
outputs/figures/
```

## Konfigurasi

Seluruh konfigurasi utama disimpan di folder:

```text
configs/
```

File penting:

```text
configs/paths.py       # Konfigurasi path project
configs/model.py       # Konfigurasi arsitektur model
configs/training.py    # Konfigurasi training dan hyperparameter
configs/plot_style.py  # Konfigurasi tampilan plot
configs/eq.py          # Konfigurasi frequency band equalizer
```

## Output Penting

Beberapa output utama project:

```text
checkpoints/full_model/best_model.pth
checkpoints/full_model/test_metrics.json
checkpoints/full_model/test_results.npz
checkpoints/baseline/baseline_metrics.json
checkpoints/ablation/ablation_result.json
checkpoints/full_model/experiment_config.json
checkpoints/full_model/worst_predictions.json
outputs/figures/
```

## Dokumentasi Struktur Project

Struktur lengkap project dapat dilihat pada file:

```text
final_project_structure.txt

## Reproducibility

Project menggunakan konfigurasi terpusat dan random seed untuk menjaga hasil eksperimen tetap konsisten.

Konfigurasi seed dapat dicek di:

```text
configs/training.py
```

## Catatan

Pastikan data sudah diproses sebelum menjalankan training atau evaluasi.
Pastikan semua path memakai konfigurasi dari `configs/paths.py`.
