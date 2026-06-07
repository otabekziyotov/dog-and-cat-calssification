import torch
from pathlib import Path
from collections import Counter

from transform import get_tfs
from custom_dataset import CustomDataset
from vis import Visualization
from train import TrainValidation
from plot import PlotLearningCurves
from infer import ModelInferenceVisualizer


# ----------------------- CONFIG -----------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH    = PROJECT_ROOT / "datasets" / "cat_dog" / "dataset"

IM_SIZE     = 224
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]
BS          = 16
MODEL_NAME  = "rexnet_150"
SAVE_PREFIX = "cat_dog"
SAVE_DIR    = "saved_models"
EPOCHS      = 10
PATIENCE    = 3
DEV_MODE    = False   # True -> only 1 batch / 1 epoch (for a quick check)

# All results are saved into this folder
RESULTS_DIR = PROJECT_ROOT / "results"
CURVES_DIR  = RESULTS_DIR / "learning_curves"
INFER_DIR   = RESULTS_DIR / "inference"
# ------------------------------------------------------


def named_counts(dl, classes):
    """Bitta split (dataloader) ichida har sinfdan nechta rasm borligini sanaydi."""
    c = Counter(int(label) for _, label in dl.dataset)
    return {classes[k]: v for k, v in c.items()}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1) Transforms
    tfs = get_tfs(im_size=IM_SIZE, mean=MEAN, std=STD)

    # 2) DataLoaders
    tr_dl, val_dl, ts_dl, cls_names, _ = CustomDataset.get_dls(DATA_PATH, tfs, BS)
    classes = list(cls_names.keys())          # ["cat", "dog"]
    print("Classes:", classes)

    # 3) Dataset analysis and visualization
    cls_counts = [named_counts(tr_dl, classes),
                  named_counts(val_dl, classes),
                  named_counts(ts_dl, classes)]
    vis = Visualization(vis_datas=[tr_dl, val_dl, ts_dl], n_ims=18, rows=6,
                        cmap="rgb", cls_names=classes, cls_counts=cls_counts,
                        save_dir=str(RESULTS_DIR))
    vis.analysis()        # -> results/analysis/*_analysis.png
    vis.pie_chart()       # -> results/analysis/*_pie.png
    vis.visualization()   # -> results/samples/*_samples.png

    # 4) Training
    trainer = TrainValidation(model_name=MODEL_NAME, classes=classes,
                              tr_dl=tr_dl, val_dl=val_dl, device=device,
                              save_dir=SAVE_DIR, save_prefix=SAVE_PREFIX,
                              epochs=EPOCHS, patience=PATIENCE, dev_mode=DEV_MODE)
    trainer.run()

    # 5) Learning curves -> results/learning_curves/
    PlotLearningCurves(trainer.tr_losses, trainer.val_losses,
                       trainer.tr_accs, trainer.val_accs,
                       trainer.tr_f1s, trainer.val_f1s,
                       save_dir=str(CURVES_DIR)).visualize()

    # 6) Inference + confusion matrix -> results/inference/
    inferer = ModelInferenceVisualizer(model=trainer.model, device=device,
                                       class_names=classes, im_size=IM_SIZE,
                                       mean=MEAN, std=STD, save_dir=str(INFER_DIR))
    inferer.infer_and_visualize(ts_dl, num_images=20, rows=4)

    print(f"\nBarcha natijalar saqlandi -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
