#  Copyright (C) 2022 Canon Medical Systems Corporation. All rights reserved
from pathlib import Path
import random
import math
import torch
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
import numpy as np
import imgaug.augmenters as iaa


class PatientDataset(torch.utils.data.Dataset):
    """
    Dataset class representing a collection of slices from a single scan.
    """

    def __init__(self, patient_dir: Path, process_fun=None, id=None, skip_condition=None, split=True):

        self.patient_dir = patient_dir
        # Make sure the slices are correctly sorted according to the slice number in case we want to assemble
        # "pseudo"-volumes later.
        self.slice_paths = sorted(list(patient_dir.iterdir()), key=lambda x: int(x.name[6:-4]))
        self.process = process_fun
        self.skip_condition = skip_condition
        self.id = id
        self.len = len(self.slice_paths)
        self.idx_map = {x: x for x in range(self.len)}
        self.split = split
        self.perlin_scale = 2
        self.perlin_noise_threshold: float = 0.3

        # self.perlin_noise_threshold: float = 0.3

        if self.skip_condition is not None:

            # Try and find which slices should be skipped and thus determine the length of the dataset.
            valid_indices = []
            for idx in range(self.len):
                with np.load(self.slice_paths[idx]) as data:
                    if self.process is not None:
                        data = self.process(**data)
                    if not skip_condition(data):
                        valid_indices.append(idx)
            self.len = len(valid_indices)
            self.idx_map = {x: valid_indices[x] for x in range(self.len)}

    def __getitem__(self, idx):
        idx = self.idx_map[idx]
        data = np.load(self.slice_paths[idx])

        if self.process is not None:
            data = self.process(**data)
            data = list(data)
        data[0] = self.generate_anomaly(data[0]).float()

        name = f"{self.patient_dir.name}-{idx}"

        return data

    def __len__(self):
        return self.len

    def generate_anomaly(self, img):
        '''
        step 1. generate mask
            - target foreground mask
            - perlin noise mask

        step 2. generate texture or structure anomaly
            - texture: load DTD
            - structure: we first perform random adjustment of mirror symmetry, rotation, brightness, saturation,
            and hue on the input image  𝐼 . Then the preliminary processed image is uniformly divided into a 4×8 grid
            and randomly arranged to obtain the disordered image  𝐼

        step 3. blending image and anomaly source
        '''

        slic = img[-1:]
        img = img[:4]
        target_foreground_mask = img.sum(dim=0) > 0.01

        img = img * target_foreground_mask
        slic = slic * target_foreground_mask

        if self.split == 'test':
            return torch.cat((img, target_foreground_mask.unsqueeze(0)), dim=0)
        else:
            ## perlin noise mask
            perlin_noise_mask = torch.from_numpy(self.generate_perlin_noise_mask(img.shape[-2:]))
            ## mask
            anomaly_mask = perlin_noise_mask * target_foreground_mask
            iter = 0
            while anomaly_mask.sum() < 5 and iter < 10:
                perlin_noise_mask = torch.from_numpy(self.generate_perlin_noise_mask(img.shape[-2:]))
                ## mask
                anomaly_mask = perlin_noise_mask * target_foreground_mask
                iter += 1
            # step 2. generate texture or structure anomaly
            index_list = list(range(img.shape[0]))
            anomaly_source_img = img.clone()
            factor = [random.uniform(0.75, 1.5), random.uniform(0.25, 1.25), random.uniform(0.25, 1.25),
                      random.uniform(0.75, 1.5)]
            # factor = [random.uniform(0.75, 1.25), random.uniform(0.75, 1.25), random.uniform(0.75, 1.25),
            #           random.uniform(0.75, 1.25)]
            for i in range(img.shape[0]):
                a = i
                k = random.uniform(0.75, 1)
                while a == i:
                    a = random.choice(index_list)
                anomaly_source_img[i] = factor[i] * (k * img[a] + (1 - k) * img[i]) * anomaly_mask + img[i] * (
                            1 - anomaly_mask)


            if self.split == 'val':
                return torch.cat(
                    (anomaly_source_img, target_foreground_mask.unsqueeze(0), slic, img), dim=0)
                # return torch.cat((anomaly_source_img, target_foreground_mask.unsqueeze(0), zero_mask, img), dim=0)
            else:
                # if random.random() > 0.5:
                return torch.cat(
                    (anomaly_source_img, target_foreground_mask.unsqueeze(0), slic, img), dim=0)

    def generate_perlin_noise_mask(self, size) -> np.ndarray:
        perlin_scalex = 2 ** self.perlin_scale
        perlin_scaley = 2 ** self.perlin_scale
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

    def __init__(self, dataset="brats2021", split="val", n_tumour_patients=None, n_healthy_patients=None,
                 skip_healthy_s_in_tumour=False,  # whether to skip healthy slices in "tumour" patients
                 skip_tumour_s_in_healthy=True,  # whether to skip tumour slices in healthy patients
                 seed=0):
        self.rng = random.Random(seed)

        assert split in ["train", "val", "test"]

        if dataset == "brats2021":
            datapath = Path(__file__).parent.parent / "data" / "new_brats2021_preprocessed"

        elif dataset == "brats2020":
            datapath = Path(__file__).parent.parent / "data" / "new_brats2020_preprocessed"

        elif dataset == "MEN":
            datapath = Path(__file__).parent.parent / "data" / "new_men_preprocessed"

        elif dataset == "ISLE":
            datapath = Path(__file__).parent.parent / "data" / "isle_preprocessed"

        elif dataset == "ATLAS":
            datapath = Path(__file__).parent.parent / "data" / "ATLAS_preprocessed"
        else:
            assert False
        # Slice skip conditions:
        path = datapath / f"npy_{split}"
        threshold = 0
        self.skip_tumour = lambda item: item[1].sum() > threshold
        self.skip_healthy = lambda item: item[1].sum() <= threshold

        def process(x, y):
            # treat all tumour classes as one for anomaly detection purposes.
            y = y > 0.5

            return torch.from_numpy(x[0]).float(), torch.from_numpy(y[0]).float()

        patient_dirs = sorted(list(path.iterdir()))
        self.rng.shuffle(patient_dirs)

        assert ((n_tumour_patients is not None) or (n_healthy_patients is not None))
        self.n_tumour_patients = n_tumour_patients if n_tumour_patients is not None else len(patient_dirs)
        self.n_healthy_patients = n_healthy_patients if n_healthy_patients is not None else len(
            patient_dirs) - self.n_tumour_patients
        # self.n_healthy_patients = 1
        # Patients with tumours
        self.patient_datasets = [PatientDataset(patient_dirs[i], process_fun=process, id=i,
                                                skip_condition=self.skip_healthy if skip_healthy_s_in_tumour else None,
                                                split=split)
                                 for i in range(self.n_tumour_patients)]

        # + only healthy slices from "healthy" patients
        self.patient_datasets += [PatientDataset(patient_dirs[i],
                                                 skip_condition=self.skip_tumour if skip_tumour_s_in_healthy else None,
                                                 process_fun=process, id=i, split=split) for i in
                                  range(self.n_tumour_patients, self.n_tumour_patients + self.n_healthy_patients)]

        self.dataset = ConcatDataset(self.patient_datasets)

    def __getitem__(self, idx):
        x, gt = self.dataset[idx]
        return x, gt

    def __len__(self):
        return len(self.dataset)
