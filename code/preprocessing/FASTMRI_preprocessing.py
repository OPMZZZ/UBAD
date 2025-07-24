#  Copyright (C) 2022 Canon Medical Systems Corporation. All rights reserved
import os.path

import torch
import random
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch.nn.functional as F

from fast_slic.avx2 import SlicAvx2
from skimage.color import gray2rgb
def normalise_percentile(volume):
    """
    Normalise the intensity values in each modality by scaling by 99 percentile foreground (nonzero) value.
    """
    for mdl in range(volume.shape[1]):
        v_ = volume[:, mdl, :, :].reshape(-1)
        v_ = v_[v_ > 0]  # Use only the brain foreground to calculate the quantile
        p_99 = torch.quantile(v_, 0.99)
        volume[:, mdl, :, :] /= p_99

    return volume

def create_label_volume(T1, label_data):
    label_volume = np.zeros(T1.shape)
    for slice in label_data:
        for item in label_data[slice]:
            x, y, h, w = item
            label_volume[ x:x + h, y:y + w, slice] = 1
    return label_volume
def process_patient(path, target_path):

    T1 = nib.load(path).get_fdata()
    prefix = path.name.split(path.name.split('_')[4])[0] + path.name.split('_')[4]
    label_data = label_dict[prefix] if prefix in label_dict.keys() else []
    labels = create_label_volume(T1, label_data)
    labels = np.flip(labels, axis=1)

    T1 = T1.transpose(1, 0, 2)
    labels = labels.transpose(1, 0, 2)
    #


    # from matplotlib import pyplot as plt
    # for index in range(T1.shape[-1]):
    #     if index in label_data:
    #         plt.imshow(T1[:, :, index])
    #         plt.show()
    #         plt.imshow(labels[:, :, index])
    #         plt.show()
    #         pass

    volume = torch.stack([torch.from_numpy(x) for x in [T1, T1, T1, T1]], dim=0).unsqueeze(dim=0)
    labels = torch.from_numpy(labels > 0.5).float().unsqueeze(dim=0).unsqueeze(dim=0)

    patient_dir = target_path / f"patient_{prefix}"
    patient_dir.mkdir(parents=True, exist_ok=True)

    volume = normalise_percentile(volume)

    sum_dim2 = (volume[0].mean(dim=0).sum(axis=0).sum(axis=0) > 0.5).int()
    fs_dim2 = sum_dim2.argmax()
    ls_dim2 = volume[0].mean(dim=0).shape[2] - sum_dim2.flip(dims=[0]).argmax()

    for slice_idx in range(fs_dim2, ls_dim2):
        low_res_x = F.interpolate(volume[:, :, :, :, slice_idx], mode="bilinear", size=(128, 128))
        flair = low_res_x[0, 0]
        gt_cpu = gray2rgb(flair.numpy() * 255).astype(np.uint8)
        slic = SlicAvx2(num_components=100, compactness=0)
        segments = slic.iterate(gt_cpu)  # Cluster Map
        segments = torch.from_numpy(segments).unsqueeze(0).unsqueeze(0)
        low_res_x = torch.cat((low_res_x, segments), dim=1)
        low_res_y = F.interpolate(labels[:, :, :, :, slice_idx], mode="bilinear", size=(128, 128))

        np.savez_compressed(patient_dir / f"slice_{slice_idx}", x=low_res_x, y=low_res_y)


def preprocess(datapath: Path):
    Paths = list((datapath).iterdir())
    for path in Paths:
        m = path.name
        all_imgs = sorted(list((path).iterdir()))

        splits_path = Path(__file__).parent.parent / "data" / f"NEW_FASTMRI_{m}_preprocessed" / "data_splits"

        if not splits_path.exists():
            indices = list(range(len(all_imgs)))
            random.seed(0)
            random.shuffle(indices)

            n_train = int(len(indices) * 0.75)
            n_val = int(len(indices) * 0.05)
            n_test = len(indices) - n_train - n_val

            split_indices = {}
            split_indices["train"] = indices[:n_train]
            split_indices["val"] = indices[n_train:n_train + n_val]
            split_indices["test"] = indices[n_train + n_val:]

            for split in ["train", "val", "test"]:
                (splits_path / split).mkdir(parents=True, exist_ok=True)
                with open(splits_path / split / "scans.csv", "w") as f:
                    f.write("\n".join([all_imgs[idx].name for idx in split_indices[split]]))

        for split in ["train", "val", "test"]:
            paths = [datapath / m /x.strip() for x in open(splits_path / split / "scans.csv").readlines()]

            print(f"Patients in {split}]: {len(paths)}")

            for source_path in tqdm(paths):
                target_path = Path(__file__).parent.parent / "data" / f"NEW_FASTMRI_{m}_preprocessed" / f"npy_{split}"
                process_patient(source_path, target_path)



if __name__ == "__main__":

    import argparse
    import pandas as pd

    # 读取 CSV 文件
    df = pd.read_csv("E:/1/data/1.csv")

    # 打印基本信息
    print(df.head())
    label_dict = {}
    # 遍历每一行（如果你要生成 mask 就在这里处理每行）
    for index, row in df.iterrows():
        filename = row['file']
        slice_idx = int(row['slice'])
        try:
            x, y, w, h = int(row['x']), int(row['y']), int(row['width']), int(row['height'])
        except:
            continue
        if filename not in label_dict:
            label_dict[filename] = {}
        if slice_idx not in label_dict[filename]:
            label_dict[filename][slice_idx] = []
        label_dict[filename][slice_idx].append([x, y, w, h])


        # 可在此调用生成 mask 的逻辑

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", type=str, default="E:/1/data/nii"
                        ,help="path to Brats2021 Training Data directory")

    args = parser.parse_args()

    datapath = Path(args.source)

    preprocess(datapath)
