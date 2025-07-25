#  Copyright (C) 2022 Canon Medical Systems Corporation. All rights reserved
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


def process_patient(path, target_path):
    t1ce = nib.load(path / f"{path.name}-t1c.nii.gz").get_fdata()
    t1 = nib.load(path / f"{path.name}-t1n.nii.gz").get_fdata()
    flair = nib.load(path / f"{path.name}-t2f.nii.gz").get_fdata()
    t2 = nib.load(path / f"{path.name}-t2w.nii.gz").get_fdata()
    labels = nib.load(path / f"{path.name}-seg.nii.gz").get_fdata()

    volume = torch.stack([torch.from_numpy(x) for x in [flair, t1, t1ce, t2]], dim=0).unsqueeze(dim=0)
    labels = torch.from_numpy(labels > 0.5).float().unsqueeze(dim=0).unsqueeze(dim=0)

    patient_dir = target_path / f"patient_{path.name}"
    patient_dir.mkdir(parents=True, exist_ok=True)

    volume = normalise_percentile(volume)

    sum_dim2 = (volume[0].mean(dim=0).sum(axis=0).sum(axis=0) > 0.5).int()
    fs_dim2 = sum_dim2.argmax()
    ls_dim2 = volume[0].mean(dim=0).shape[2] - sum_dim2.flip(dims=[0]).argmax()

    for slice_idx in range(fs_dim2, ls_dim2):
        low_res_x = F.interpolate(volume[:, :, :, :, slice_idx], mode="bilinear", size=(128, 128))
        t2 = low_res_x[0, -1]
        gt_cpu = gray2rgb(t2.numpy() * 255).astype(np.uint8)
        slic = SlicAvx2(num_components=100, compactness=0)
        segments = slic.iterate(gt_cpu)  # Cluster Map
        segments = torch.from_numpy(segments).unsqueeze(0).unsqueeze(0)
        low_res_x = torch.cat((low_res_x, segments), dim=1)
        low_res_y = F.interpolate(labels[:, :, :, :, slice_idx], mode="bilinear", size=(128, 128))

        np.savez_compressed(patient_dir / f"slice_{slice_idx}", x=low_res_x, y=low_res_y)


def preprocess(datapath: Path):

    all_imgs = sorted(list((datapath).iterdir()))

    splits_path = Path(__file__).parent.parent / "data" / "new_men_preprocessed" / "data_splits"

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
        paths = [datapath / x.strip() for x in open(splits_path / split / "scans.csv").readlines()]

        print(f"Patients in {split}]: {len(paths)}")

        for source_path in tqdm(paths):
            target_path = Path(__file__).parent.parent / "data" / "new_men_preprocessed" / f"npy_{split}"
            process_patient(source_path, target_path)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", type=str, default="../../BraTS-MEN-Train"
                        ,help="path to Brats2021 Training Data directory")

    args = parser.parse_args()

    datapath = Path(args.source)

    preprocess(datapath)
