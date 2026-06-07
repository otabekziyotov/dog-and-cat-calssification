#!pip install gdown
import gdown, os, shutil
from pathlib import Path

# Project root (the folder above data/) — works on any machine
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def fayl_yuklab_olish(fayl_id, saqlash_nomi, saqlash_uchun_papka = PROJECT_ROOT / "datasets", ds_nomi = None):

    os.makedirs(saqlash_uchun_papka, exist_ok = True)
    url = f"https://drive.google.com/uc?id={fayl_id}"

    # Check whether the data is already downloaded
    if os.path.exists(f"{saqlash_uchun_papka}/{ds_nomi}"): print(f"{ds_nomi} dataset allaqachon yuklab olingan.")

    # Download the data
    else:
        print(f"Fayl {saqlash_nomi} nomi bilan yuklanmoqda...")
        # 1) link; 2) name to save as
        gdown.download(url, saqlash_nomi, quiet = True)
        print(f"{saqlash_nomi} fayli muvaffaqiyatli yuklab olindi.")
        # 1st arg -> saved file name; 2nd arg -> where to extract
        shutil.unpack_archive(f"{saqlash_nomi}", f"{saqlash_uchun_papka}/{ds_nomi}")
        print("Fayl arxivdan ochilmoqda....")
        os.remove(f"{saqlash_nomi}")
        print(f"Tanlangan dataset {saqlash_uchun_papka}/{ds_nomi} papkasiga yuklab olindi!")

fayl_yuklab_olish(fayl_id = "1HTiUnP2O7ZJL1zND3vN71q7MSJ2w5R6s", saqlash_nomi = "tttt.zip", ds_nomi = "cat_dog")