#  Copyright (C) 2022 Canon Medical Systems Corporation. All rights reserved
from pathlib import Path
import random
import math
import torch
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
import numpy as np
import imgaug.augmenters as iaa
import torch.nn.functional as F
import os
from math import ceil

import nibabel as nib


class PatientDataset(torch.utils.data.Dataset):
    """
    Dataset class representing a collection of slices from a single scan.
    """

    def __init__(self, patient_dir: Path, split=True, modalities=None, dataset=None, **kwargs):
        patient = patient_dir.parts[-1]
        self.patient = patient
        self.modalities = modalities
        self.dataset = dataset
        self.perlin_noise_threshold: float = 0.3

        import nibabel as nib
        import numpy as np
        import torch

        if modalities == 't2':
            t2_path = patient_dir / f'{patient}_t2.nii.gz'
            t2_nifti = nib.load(t2_path)
            t2 = torch.from_numpy(np.array(t2_nifti.get_fdata())).float()  # 转为 float32 Tensor
            input_data = t2.unsqueeze(1)  # 在第 0 维增加 channel，例如变成 (1, H, W, D)
        else:
            raise AssertionError(f"Unsupported modality: {modalities}")

        if dataset == 'IXI':
            self.slices = torch.cat((input_data, input_data), dim=1)
        else:
            seg_dir = patient_dir / f'{patient}_seg.nii.gz'
            seg = torch.from_numpy(np.array(nib.load(seg_dir).get_fdata())).unsqueeze(1)
            self.slices = torch.cat((input_data, seg), dim=1)

        self.slices = self.slices.float()
        self.len = self.slices.shape[0]
        self.idx_map = {x: x for x in range(self.len)}
        self.split = split

    def __getitem__(self, idx):
        idx = self.idx_map[idx]
        data = self.slices[idx]
        if self.dataset == "IXI":
            data = self.generate_anomaly(data)
        data = F.interpolate(data.unsqueeze(0), size=[128, 128]).squeeze()

        return data.float()

    def __len__(self):
        return self.len

    def generate_anomaly(self, imgs):

        target_foreground_mask = imgs[0] > 0.01
        perlin_noise_mask = torch.from_numpy(self.generate_perlin_noise_mask(imgs[0].shape[-2:]))
        ## mask
        anomaly_mask = perlin_noise_mask * target_foreground_mask
        while anomaly_mask.sum() < 5 and target_foreground_mask.sum() > 100 :
            perlin_noise_mask = torch.from_numpy(self.generate_perlin_noise_mask(imgs.shape[-2:]))
            ## mask
            anomaly_mask = perlin_noise_mask * target_foreground_mask


        img = imgs[0]
        label = imgs[-1]
        factor = random.uniform(0.75, 1.5)
        k = random.uniform(0.75, 1)
        ima = torch.ones_like(img) * (anomaly_mask * img).sum() / anomaly_mask.sum()

        anomaly_source_img = factor * (k * ima + (1 - k) * img) * anomaly_mask + img * (
                1 - anomaly_mask)
        return torch.stack(
            (anomaly_source_img, target_foreground_mask, anomaly_mask, label), dim=0)


    def generate_perlin_noise_mask(self, size) -> np.ndarray:

        perlin_scalex = 2 ** 2
        perlin_scaley = 2 ** 2

        # generate perlin noise
        perlin_noise = self.rand_perlin_2d_np((size[0], size[1]), (perlin_scalex, perlin_scaley))

        # apply affine transform
        rot = iaa.Affine(rotate=(-90, 90))
        perlin_noise = rot(image=perlin_noise)

        # make a mask by applying threshold
        mask_noise = np.where(
            perlin_noise > self.perlin_noise_threshold,
            np.ones_like(perlin_noise),
            np.zeros_like(perlin_noise)
        )
        return mask_noise

    def rand_perlin_2d_np(self, shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
        def lerp_np(x, y, w):
            fin_out = (y - x) * w + x
            return fin_out

        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

        angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
        gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        tt = np.repeat(np.repeat(gradients, d[0], axis=0), d[1], axis=1)

        tile_grads = lambda slice1, slice2: np.repeat(
            np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]], d[0], axis=0), d[1], axis=1)
        dot = lambda grad, shift: (
                np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                         axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

        n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
        n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
        n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
        n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
        t = fade(grid[:shape[0], :shape[1]])
        return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])



class BrainDataset(torch.utils.data.Dataset):
    """
    Dataset class representing a collection of slices from scans from a specific dataset split.
    """

    def __init__(self, dataset="brats2021", split="val", modalities=None, seed=0, **kwargs):
        self.rng = random.Random(seed)

        assert split in ["train", "val", "test"]
        assert modalities in ["t2"]

        if "IXI" in dataset:
            if 'fold' in dataset:
                path = Path(__file__).parent.parent.parent / "IXI" / "data" / "IXI" / f"{dataset.split('_')[0]}_{split}_{dataset.split('_')[1]}"

            else:
                path = Path(__file__).parent.parent.parent / "IXI" / "data" / "IXI" / f"{dataset}_{split}"
            dataset = 'IXI'

        else:
            path = Path(__file__).parent.parent.parent / "IXI" / "data" / dataset / f"{dataset}_{split}"

        patient_dirs = sorted(list(path.iterdir()))

        self.rng.shuffle(patient_dirs)
        data_length = len(patient_dirs)

        assert dataset in ["IXI", "MSLUB", "Brats21"]

        self.patient_datasets = [PatientDataset(patient_dirs[i], modalities=modalities, dataset=dataset, **kwargs)
                                 for i in range(data_length)]

        self.dataset = ConcatDataset(self.patient_datasets)

    def __getitem__(self, idx):
        x = self.dataset[idx]
        return x

    def __len__(self):
        return len(self.dataset)
