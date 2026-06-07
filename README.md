# 🐱 Cat vs Dog Image Classifier 🐶

> PyTorch cat/dog classifier — transfer learning (timm), Grad-CAM, Streamlit demo, ONNX export.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/timm-transfer%20learning-9cf.svg" alt="timm">
  <img src="https://img.shields.io/badge/Grad--CAM-explainability-success.svg" alt="Grad-CAM">
  <img src="https://img.shields.io/badge/ONNX-export-orange.svg" alt="ONNX">
</p>

Fine-tunes a pretrained `rexnet_150` model to classify **cats** and **dogs**, with full visual reports, an interactive demo, and ONNX export.

## Features

- **Custom dataset** loader with automatic labels from folder names.
- **Transfer learning** on `rexnet_150` (any `timm` model works).
- **Training loop** with accuracy/F1, LR scheduler, early stopping, best-model saving.
- **Reports**: dataset samples, class-balance charts, learning curves, Grad-CAM, confusion matrix.
- **Streamlit demo** + **ONNX export** for deployment.

## Project Structure

```
dog_cat/
├── data/data_downloading.py   # download & extract dataset
├── custom_dataset.py          # Dataset + get_dls() (train/val/test loaders)
├── transform.py               # image transforms
├── train.py                   # training / validation loop
├── plot.py                    # learning curves
├── vis.py                     # dataset samples & class analysis
├── infer.py                   # Grad-CAM + confusion matrix
├── main.py                    # runs the full pipeline
├── app.py                     # Streamlit demo
├── onnx_converter.py          # export model to ONNX
├── requirements.txt
└── results/                   # generated reports
```

## Setup

```bash
git clone https://github.com/otabekziyotov/dog-and-cat-calssification.git
cd dog-and-cat-calssification
pip install -r requirements.txt
python data/data_downloading.py   # download dataset
```

> CPU build of PyTorch by default. For GPU, install the CUDA build from the [PyTorch site](https://pytorch.org/get-started/locally/).

## Train

```bash
python main.py
```

Trains the model and writes all reports to `results/`.
Set `DEV_MODE = True` in [`main.py`](main.py) for a quick 1-batch test.

## Demo (Streamlit)

```bash
streamlit run app.py
```

Open **http://localhost:8501**. Upload an image or pick a test sample to see the prediction, confidence, and Grad-CAM heatmap. (Train the model first.)

## ONNX Export

Export the trained model to ONNX and verify it matches PyTorch:

```bash
python onnx_converter.py
```

Saves `saved_models/cat_dog_model.onnx`, validates the graph, and compares PyTorch vs ONNX Runtime outputs (difference should be ~1e-5). Use the `.onnx` file for fast, framework-independent deployment.

## Config

Edit the top of [`main.py`](main.py): `IM_SIZE`, `BS`, `MODEL_NAME`, `EPOCHS`, `PATIENCE`, `DEV_MODE`.

## Tech Stack

PyTorch · torchvision · timm · torchmetrics · grad-cam · onnx / onnxruntime · streamlit · matplotlib · seaborn · scikit-learn
