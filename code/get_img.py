#  Copyright (C) 2022 Canon Medical Systems Corporation. All rights reserved
from math import ceil
import torch
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from data import BrainDataset
from cc_filter import connected_components_3d
import os
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import trange
from scipy.ndimage import median_filter
import numpy as np
def median_filter_3D(volume, kernelsize=5):
    volume = volume.cpu().numpy()
    pbar = tqdm(range(len(volume)), desc="Median filtering")
    for i in pbar:
        volume[i] = median_filter(volume[i], size=(kernelsize, kernelsize, kernelsize))
    return torch.Tensor(volume)
def eval_anomalies_batched(trainer, dataset, get_scores, batch_size=32, threshold=None, get_y=lambda batch: batch[1],
                           return_dice=False, filter_cc=False):
    def dice(a, b):
        a = a.cuda()
        b = b.cuda()
        num = 2 * (a & b).sum()
        den = a.sum() + b.sum()

        den_float = den.float()
        den_float[den == 0] = float("nan")

        return num.float() / den_float

    dice_thresholds = [x / 1000 for x in range(1000)] if threshold is None else [threshold]

    with torch.no_grad():
        # Now that we have the threshold we can do some filtering and recalculate the Dice
        i = 0
        y_true_ = torch.zeros(128 * 128 * len(dataset), dtype=torch.half)
        y_pred_ = torch.zeros(128 * 128 * len(dataset), dtype=torch.half)
        if threshold is not None:
            y_pred_mask = torch.zeros(128 * 128 * len(dataset), dtype=torch.bool)

        for pd in dataset.patient_datasets:
            batch_items = [pd[x] for x in range(len(pd))]
            collate = torch.utils.data._utils.collate.default_collate
            batch = collate(batch_items)

            batch_y = get_y(batch).cpu()

            with torch.no_grad():
                anomaly_scores = get_scores(trainer, batch).cpu()

            # Do CC filtering:
            anomaly_scores = median_filter_3D(anomaly_scores.permute(1, 2, 3, 0), kernelsize=5).permute(3, 0, 1, 2)
            if threshold is not None:
                anomaly_scores_bin = anomaly_scores > threshold
                anomaly_scores_bin = connected_components_3d(anomaly_scores_bin.squeeze(dim=1)).unsqueeze(dim=1)
                y_m = anomaly_scores_bin.reshape(-1)
                y_pred_mask[i:i + y_m.numel()] = y_m

            y_ = (batch_y.view(-1) > 0.5)
            y_hat = anomaly_scores.reshape(-1)
            y_true_[i:i + y_.numel()] = y_.half()
            y_pred_[i:i + y_hat.numel()] = y_hat.half()
            i += y_.numel()

        # ap = average_precision_score(y_true_, y_pred_)

        with torch.no_grad():
            y_true_ = y_true_
            y_pred_ = y_pred_
            if threshold is not None:
                y_pred_mask = y_pred_mask
                dices = [dice(y_true_ > 0.5, y_pred_mask)]
            else:
                dices = [dice(y_true_ > 0.5, y_pred_ > x).item() for x in tqdm(dice_thresholds)]

            max_dice, threshold = max(zip(dices, dice_thresholds), key=lambda x: x[0])

        return None, max_dice, threshold

from torch.nn import functional as F
def get_scores(trainer, batch):
    x = batch[:, trainer.input_channel]
    trainer.model = trainer.model.eval()
    with torch.no_grad():
        # Assume it's in batch shape
        clean = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1]).clone().to(trainer.device)
        raw_mask = clean.sum(dim=1, keepdim=True) > 0.01

        # Erode the mask a bit to remove some of the reconstruction errors at the edges.
        mask = (F.avg_pool2d(raw_mask.float(), kernel_size=5, stride=1, padding=2) > 0.95)


        res1, res2 = trainer.model(clean, clean)
        res = (res1 + res2) / 2
        res_nomask = res.clone()
        res = res * raw_mask

        err_map3 = ((clean - res1) * mask).abs()
        err_map4 = ((clean - res2) * mask).abs()
        err_map1 = clean * torch.log((clean + 1e-10) / (res1 + 1e-10))
        err_map2 = clean * torch.log((clean + 1e-10) / (res2 + 1e-10))
        err_map1[torch.isnan(err_map1)] = 0
        err_map2[torch.isnan(err_map2)] = 0

        err = (err_map4 + err_map3).mean(dim=1, keepdim=True)


    return err.cpu(), res.cpu(), res_nomask.cpu()


def get_img(trainer, dataset, batch_size=1, get_y=lambda batch: batch[1], threshold=None):

    def save_nii(volume, fname, pname):
        for index in range(volume.shape[1]):
            real_volume = nib.Nifti1Image(volume[:, index], np.eye(4))
            nib.save(real_volume, f'../outputs/{dataset_name}/our/{pname}/{fname}-{index}.nii.gz')

    with torch.no_grad():
        # Now that we have the threshold we can do some filtering and recalculate the Dice

        for pd in dataset.patient_datasets:
            batch_items = [pd[x] for x in range(len(pd))]
            collate = torch.utils.data._utils.collate.default_collate
            batch = collate(batch_items)

            name = pd.patient_dir.name

            batch_y = get_y(batch)
            batch = batch[0]
            anomaly_scores, reconstructions, rnm = get_scores(trainer, batch)
            anomaly_scores = median_filter_3D(anomaly_scores.permute(1, 2, 3, 0), kernelsize=5).permute(3, 0, 1, 2)


            os.makedirs(f'../outputs/{dataset_name}/our/{name}', exist_ok=True)


            anomaly_scores_ = torch.clip(anomaly_scores.cpu(), 0, 2 *threshold) / (2 * threshold)

            batch_y = torch.flip(batch_y, dims=[-2, -1])
            save_nii(batch[:, trainer.input_channel].numpy(), 'real_volume', name)
            save_nii(batch_y.numpy(), 'mask', name)
            save_nii(reconstructions.numpy(), 'reconstructions', name)
            save_nii(rnm.numpy(), 'rnm', name)
            save_nii(anomaly_scores_.numpy(), 'anomaly_scores', name)
            save_nii((anomaly_scores > threshold).float().numpy(), 'seg', name)



def evaluate(split: str = "test", use_cc: bool = True):



    dataset = BrainDataset(dataset=dataset_name, split=split, n_tumour_patients=None, n_healthy_patients=0)
    id = f'OPEN_{dataset_name}'
    trainer = denoising(id, data=None, lr=0.0001, depth=4,
                        wf=6, n_input=len(channels))
    trainer.load(id)
    trainer.input_channel = channels

    results = eval_anomalies_batched(trainer, dataset=dataset, get_scores=trainer.get_scores, return_dice=True,
                                     filter_cc=use_cc)
    threshold = results[2]

    results = get_img(trainer, dataset=dataset, threshold=threshold)

    return results


if __name__ == "__main__":
    import argparse
    from train import denoising

    ds_dict = {'ATLAS': [0]}
    dataset_name = 'ATLAS'
    channels = ds_dict[dataset_name]


    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--split", default="test", type=str, help="'train', 'val' or 'test'")

    args = parser.parse_args()
    print(dataset_name)
    evaluate(
             split=args.split)
