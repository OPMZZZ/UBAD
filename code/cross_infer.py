#  Copyright (C) 2022 Canon Medical Systems Corporation. All rights reserved
import torch
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from cross_data import BrainDataset
from cc_filter import connected_components_3d

import numpy as np

from scipy.ndimage import median_filter
import numpy as np
def median_filter_3D(volume, kernelsize=5):
    volume = volume.cpu().numpy()
    pbar = tqdm(range(len(volume)), desc="Median filtering")
    for i in pbar:
        volume[i] = median_filter(volume[i], size=(kernelsize, kernelsize, kernelsize))
    return torch.Tensor(volume)
def norm_tensor(tensor):
    my_max = torch.max(tensor)
    my_min = torch.min(tensor)
    my_tensor = (tensor - my_min) / (my_max - my_min)
    return my_tensor

def eval_anomalies_batched(trainer, dataset, get_scores, batch_size=32, threshold=None, get_y=lambda batch: batch[:, [-1]],
                           return_dice=False, filter_cc=False):
    def dice(a, b):
        if threshold is not None:
            a = a.cpu()
            b = b.cpu()
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

        ap = average_precision_score(y_true_, y_pred_)

        with torch.no_grad():
            y_true_ = y_true_.to(trainer.device)
            y_pred_ = y_pred_.to(trainer.device)
            if threshold is not None:
                y_pred_mask = y_pred_mask.to(trainer.device)
                dices = [dice(y_true_ > 0.5, y_pred_mask)]
            else:
                dices = [dice(y_true_ > 0.5, y_pred_ > x).cpu().item() for x in tqdm(dice_thresholds)]

            max_dice, threshold = max(zip(dices, dice_thresholds), key=lambda x: x[0])

        return ap, max_dice, threshold


def evaluate(dataset: str = "test", use_cc: bool = False):
    val_dataset = BrainDataset(dataset=dataset, split='val', modalities='t2')
    test_dataset = BrainDataset(dataset=dataset, split='test', modalities='t2')
    APs = []
    Dices = []
    for j in range(5):

        id = f"OPEN_IXI-{j}"
        trainer = denoising(id, data=None, lr=0.0001, depth=4,
                            wf=6)  # Noise parameters don't matter during evaluation.

        trainer.load(id)

        results = eval_anomalies_batched(trainer, dataset=val_dataset, get_scores=trainer.get_scores,
                                         return_dice=True,
                                         filter_cc=use_cc)

        threshold = results[2]

        results = eval_anomalies_batched(trainer, dataset=test_dataset, get_scores=trainer.get_scores,
                                         return_dice=True,
                                         filter_cc=use_cc, threshold=threshold)

        APs.append(results[0])

        Dices.append(results[1])

    print(f"AP: {np.mean(APs), np.std(APs)}")
    print(f"max Dice: {np.mean(Dices), np.std(Dices)}")
    print(f"Optimal threshold: {results[2]}")

    logging.info("=" * 20)
    logging.info(dataset)
    logging.info(f"AP: {np.mean(APs), np.std(APs)}")
    logging.info(f"max Dice: {np.mean(Dices), np.std(Dices)}")
    logging.info(f"Optimal threshold: {results[2]}")


if __name__ == "__main__":
    import logging

    import argparse
    from cross_train import denoising

    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", "--dataset", default="Brats21", type=str, help="'train', 'val' or 'test'")

    args = parser.parse_args()
    logging.basicConfig(filename='../test_saved_models/test_result.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    evaluate(
             dataset=args.dataset,
             )
