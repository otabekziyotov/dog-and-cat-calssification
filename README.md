# 🐱 Cat vs Dog Image Classifier 🐶

> PyTorch cat vs dog image classifier with transfer learning (timm), Grad-CAM explainability, and full visual reports.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/timm-transfer%20learning-9cf.svg" alt="timm">
  <img src="https://img.shields.io/badge/Grad--CAM-explainability-success.svg" alt="Grad-CAM">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

A clean, modular PyTorch project that classifies images of **cats** and **dogs**. It fine-tunes a pretrained [`timm`](https://github.com/huggingface/pytorch-image-models) model and produces a full set of visual reports — dataset samples, class-balance analysis, learning curves, and inference results with **Grad-CAM** heatmaps.

---

## ✨ Features

- **Custom `Dataset`** — loads images straight from folders (`train/cat`, `train/dog`, ...) with automatic label assignment.
- **Transfer learning** — fine-tunes `rexnet_150` (or any `timm` model) on the cat/dog data.
- **Full training loop** — accuracy & F1 metrics, learning-rate scheduler, early stopping, best-model checkpointing.
- **Rich visualizations** — dataset samples, class-imbalance bar/pie charts, loss/accuracy/F1 curves.
- **Explainable inference** — Grad-CAM heatmaps + confusion matrix showing where the model "looks".
- **Reproducible** — pinned `requirements.txt`, portable paths that work on any machine (Colab, Windows, Linux, Mac).

---

## 📁 Project Structure

```
dog_cat/
├── data/
│   └── data_downloading.py     # Downloads & extracts the dataset from Google Drive
├── custom_dataset.py           # CustomDataset + get_dls() (train/val/test DataLoaders)
├── transform.py                # get_tfs() — image transforms (resize, normalize)
├── vis.py                      # Visualization — dataset samples & class analysis
├── train.py                    # TrainValidation — training/validation loop
├── plot.py                     # PlotLearningCurves — loss/accuracy/F1 curves
├── infer.py                    # ModelInferenceVisualizer — Grad-CAM + confusion matrix
├── main.py                     # Entry point — runs the whole pipeline
├── requirements.txt            # Pinned dependencies
└── results/                    # All generated reports (created on run)
    ├── samples/                #   dataset sample grids
    ├── analysis/               #   class-balance bar & pie charts
    ├── learning_curves/        #   loss / accuracy / f1 curves
    └── inference/              #   Grad-CAM predictions + confusion matrix
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd dog_cat
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU note:** `requirements.txt` installs the CPU build of PyTorch. If you have an NVIDIA GPU, install the CUDA build from the [official PyTorch index](https://pytorch.org/get-started/locally/) for your CUDA version instead.

### 3. Download the dataset

```bash
python data/data_downloading.py
```

This downloads and extracts the dataset into `datasets/cat_dog/dataset/` (with `train/` and `test/` splits).

### 4. Run the full pipeline

```bash
python main.py
```

This trains the model and writes **all** reports into the `results/` folder.

> **Tip:** For a quick smoke test without full training, set `DEV_MODE = True` in [`main.py`](main.py) — it runs just 1 batch / 1 epoch.

---

## 🎬 Live Demo (Streamlit)

An interactive web demo where you can upload an image — or pick a sample from the test set — and see the prediction, confidence, and a **Grad-CAM** heatmap.

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

> **Note:** Streamlit apps must be launched with `streamlit run` (not `python app.py`).
> Requires a trained model at `saved_models/cat_dog_best_model.pth` — run `python main.py` first.

---

## ⚙️ Configuration

All settings live at the top of [`main.py`](main.py):

| Setting | Default | Description |
|---------|---------|-------------|
| `IM_SIZE` | `224` | Input image size |
| `BS` | `16` | Batch size |
| `MODEL_NAME` | `"rexnet_150"` | Any `timm` model name |
| `EPOCHS` | `10` | Max training epochs |
| `PATIENCE` | `3` | Early-stopping patience |
| `DEV_MODE` | `False` | Quick 1-batch / 1-epoch run |

---

## 📊 Results

After running `main.py`, the `results/` folder contains:

| Report | Location | What it shows |
|--------|----------|---------------|
| **Dataset samples** | `results/samples/` | Random images with their ground-truth labels |
| **Class analysis** | `results/analysis/` | Bar & pie charts of class balance per split |
| **Learning curves** | `results/learning_curves/` | Loss, accuracy and F1 over epochs |
| **Inference + Grad-CAM** | `results/inference/inference_results.png` | Predictions vs ground truth with Grad-CAM heatmaps |
| **Confusion matrix** | `results/inference/confusion_matrix.png` | Per-class prediction performance |

<!-- After your first run you can embed the generated images here, e.g.:
![Learning curves](results/learning_curves/accuracy_curve.png)
![Inference](results/inference/inference_results.png)
![Confusion matrix](results/inference/confusion_matrix.png)
-->

---

## 🛠️ Tech Stack

- **PyTorch** & **torchvision** — deep learning
- **timm** — pretrained models (transfer learning)
- **torchmetrics** — accuracy & F1 metrics
- **grad-cam** — model explainability
- **matplotlib**, **seaborn** — visualization
- **scikit-learn** — confusion matrix
- **gdown** — dataset download
