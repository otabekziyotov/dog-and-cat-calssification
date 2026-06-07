from torch.utils.data import random_split, Dataset, DataLoader
from glob import glob
from PIL import Image
from torchvision import transforms as T
from pathlib import Path
import os


class CustomDataset(Dataset):
    def __init__(self, data_turgan_yulak, data_turi, transformations):
        super().__init__()
        self.rasm_yolaklari = glob(f"{str(data_turgan_yulak)}/{data_turi}/*/*")
        self.transformations = transformations

    def __len__(self): return len(self.rasm_yolaklari)

    def get_class(self, data_yolagi):
        # The folder the image is in = class name (e.g. ".../train/cat/img.jpg" -> "cat")
        return os.path.basename(os.path.dirname(data_yolagi))

    def javobni_olib_kelish(self, data_yolagi):
        return 1 if "dog" == self.get_class(data_yolagi) else 0

    def get_info(self):

        self.cls_names, self.cls_counts = {}, {}
        count = 0
        for im_path in self.rasm_yolaklari:
            class_name = self.get_class(im_path)
            if class_name not in self.cls_names:
                self.cls_names[class_name] = count
                self.cls_counts[class_name] = 1
                count += 1
            else: self.cls_counts[class_name] += 1

        return self.cls_names, self.cls_counts



    def __getitem__(self, idx):
        rasm_yulagi = self.rasm_yolaklari[idx]
        rasm = Image.open(rasm_yulagi).convert('RGB')
        if self.transformations: rasm = self.transformations(rasm)

        javob = self.javobni_olib_kelish(rasm_yulagi)
        return rasm, javob
    
    @classmethod
    def get_dls(cls, data_turgan_yulak, tfs, bs):

        tr_ds = CustomDataset(data_turgan_yulak=data_turgan_yulak, data_turi="train", transformations=tfs)
        ts_ds = CustomDataset(data_turgan_yulak=data_turgan_yulak, data_turi="test", transformations=tfs)

        # cls_names/cls_counts attributes appear only after get_info() is called
        tr_ds.get_info()
        ts_ds.get_info()
        cls_names, cls_counts = tr_ds.cls_names, [tr_ds.cls_counts, ts_ds.cls_counts]

        total_len = len(tr_ds)
        tr_len = int(total_len * 0.8)
        vl_len = total_len - tr_len

        tr_ds, vl_ds = random_split(dataset = tr_ds, lengths=[tr_len, vl_len])

        tr_dl = DataLoader(dataset = tr_ds, batch_size = bs, shuffle = True, num_workers = 2)
        vl_dl = DataLoader(dataset = vl_ds, batch_size = bs, shuffle = False, num_workers = 2)
        ts_dl = DataLoader(dataset = ts_ds, batch_size = bs,  shuffle = False, num_workers = 2)


        return tr_dl, vl_dl, ts_dl, cls_names, cls_counts





# # Project root (folder of this file) — works on any machine
# PROJECT_ROOT = Path(__file__).resolve().parent
# data_turgan_yulak = PROJECT_ROOT / "datasets" / "cat_dog" / "dataset"

# tfs = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
# ds = CustomDataset(data_turgan_yulak=data_turgan_yulak, data_turi="train", transformations=tfs)
# ts_ds = CustomDataset(data_turgan_yulak=data_turgan_yulak, data_turi="test", transformations=tfs)

# #ds[1][0].shape