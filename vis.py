import os
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms as T
from collections import Counter

class Visualization:

    def __init__(self, vis_datas, n_ims, rows, cmap=None, cls_names=None, cls_counts=None, t_type="rgb", save_dir=None):
        self.n_ims, self.rows = n_ims, rows
        self.t_type, self.cmap = t_type, cmap
        self.cls_names = cls_names
        self.colors = ["darkorange", "seagreen", "salmon"]

        # If save_dir is given -> save images to file, otherwise show them on screen
        self.save_dir = save_dir
        self.samples_dir = self.analysis_dir = None
        if save_dir:
            self.samples_dir = os.path.join(save_dir, "samples")
            self.analysis_dir = os.path.join(save_dir, "analysis")
            os.makedirs(self.samples_dir, exist_ok=True)
            os.makedirs(self.analysis_dir, exist_ok=True)

        data_names = ["train", "val", "test"]
        self.vis_datas = {data_names[i]: vis_datas[i] for i in range(len(vis_datas))}
        if isinstance(cls_counts, list):
            self.analysis_datas = {data_names[i]: cls_counts[i] for i in range(len(cls_counts))}
        else:
            self.analysis_datas = {"all": cls_counts}

    def tn2np(self, t):
        gray_tfs = T.Compose([T.Normalize(mean=[0.], std=[1/0.5]), T.Normalize(mean=[-0.5], std=[1])])
        rgb_tfs = T.Compose([T.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
                             T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])

        invTrans = gray_tfs if self.t_type == "gray" else rgb_tfs

        return (invTrans(t) * 255).detach().squeeze().cpu().permute(1, 2, 0).numpy().astype(np.uint8) if self.t_type == "gray" \
               else (invTrans(t) * 255).detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)

    def _save_or_show(self, out_dir, filename):
        # If out_dir is set -> save to file, otherwise -> show on screen
        if out_dir:
            path = os.path.join(out_dir, filename)
            plt.savefig(path, bbox_inches="tight")
            plt.close()
            print(f"Saqlandi -> {path}")
        else:
            plt.show()

    def plot(self, rows, cols, count, im, title="Original Image"):
        plt.subplot(rows, cols, count)
        plt.imshow(self.tn2np(im))
        plt.axis("off")
        plt.title(title)
        return count + 1

    def vis(self, data, save_name):
        print(f"{save_name.upper()} Data Visualization is in process...\n")
        assert self.cmap in ["rgb", "gray"], "Please choose rgb or gray cmap"
        cmap = "viridis" if self.cmap == "rgb" else None
        cols = self.n_ims // self.rows
        count = 1

        plt.figure(figsize=(25, 20))
        indices = [np.random.randint(low=0, high=len(data) - 1) for _ in range(self.n_ims)]

        for idx, index in enumerate(indices):
            if count == self.n_ims + 1: break
            image, label = data[index]
            plt.subplot(self.rows, self.n_ims // self.rows, idx + 1)

            if cmap:
                plt.imshow(self.tn2np(image), cmap=cmap)
            else:
                plt.imshow(self.tn2np(image))

            plt.axis('off')
            if self.cls_names is not None:
                plt.title(f"GT -> {self.cls_names[int(label)]}")
            else:
                plt.title(f"GT -> {label}")

        self._save_or_show(self.samples_dir, f"{save_name}_samples.png")

    def data_analysis(self, cls_counts, save_name, color):
        print("Data analysis is in process...\n")
        width, text_width, text_height = 0.7, 0.05, 2
        cls_names = list(cls_counts.keys())
        counts = list(cls_counts.values())
        _, ax = plt.subplots(figsize=(20, 10))
        indices = np.arange(len(counts))
        ax.bar(indices, counts, width, color=color)
        ax.set_xlabel("Class Names", color="black")
        ax.set(xticks=indices, xticklabels=cls_names)
        ax.set_ylabel("Data Counts", color="black")
        ax.set_title(f"{save_name.upper()} Dataset Class Imbalance Analysis")
        for i, v in enumerate(counts):
            ax.text(i - text_width, v + text_height, str(v), color="royalblue")
        self._save_or_show(self.analysis_dir, f"{save_name}_analysis.png")

    def plot_pie_chart(self, cls_counts, save_name="all"):
        print("Generating pie chart...\n")
        labels = list(cls_counts.keys())
        sizes = list(cls_counts.values())
        explode = [0.1] * len(labels)  # To highlight all slices equally (optional)

        plt.figure(figsize=(8, 8))
        plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.tab20.colors)
        plt.title(f"{save_name.upper()} Class Distribution")
        plt.axis("equal")  # Equal aspect ratio ensures the pie chart is circular
        self._save_or_show(self.analysis_dir, f"{save_name}_pie.png")

    def visualization(self): [self.vis(data.dataset, save_name) for (save_name, data) in self.vis_datas.items()]

    def analysis(self): [self.data_analysis(data, save_name, color) for (save_name, data), color in zip(self.analysis_datas.items(), self.colors)]

    def pie_chart(self): [self.plot_pie_chart(data, save_name) for save_name, data in self.analysis_datas.items()]


if __name__ == "__main__":
    from pathlib import Path
    from torchvision import transforms as T
    from custom_dataset import CustomDataset

    PROJECT_ROOT = Path(__file__).resolve().parent
    data_turgan_yulak = PROJECT_ROOT / "datasets" / "cat_dog" / "dataset"
    tfs = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    tr_dl, vl_dl, ts_dl, cls_names, _ = CustomDataset.get_dls(data_turgan_yulak, tfs, bs=16)
    classes = list(cls_names.keys())   # ["cat", "dog"] — vis() needs a list for label names

    # Per-split class counts (via dl.dataset — no worker spawn)
    def named_counts(dl):
        c = Counter(int(label) for _, label in dl.dataset)
        return {classes[k]: v for k, v in c.items()}

    all_cls_counts = [named_counts(tr_dl), named_counts(vl_dl), named_counts(ts_dl)]

    vis = Visualization(vis_datas=[tr_dl, vl_dl, ts_dl], n_ims=18, rows=6,
                        cmap="rgb", cls_names=classes, cls_counts=all_cls_counts)
    vis.analysis()
    vis.visualization()