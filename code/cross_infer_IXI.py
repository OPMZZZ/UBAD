#  Copyright (C) 2022 Canon Medical Systems Corporation. All rights reserved
from math import ceil
import torch

from cross_data import BrainDataset

from tqdm import trange

import numpy as np


def eval_anomalies_batched(trainer, dataset, batch_size=32):
    n_batches = int(ceil(len(dataset) / batch_size))
    l1 = []
    mae = torch.nn.L1Loss()
    for batch_idx in trange(n_batches):
        batch_items = [dataset[x] for x in
                       range(batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(dataset)))]
        collate = torch.utils.data._utils.collate.default_collate
        batch = collate(batch_items)

        with torch.no_grad():
            ori_imgs = batch[:, -1:].to(trainer.device)
            mask = ori_imgs.sum(dim=1, keepdim=True) > 0.01
            mask = mask.float()
            preds1, preds2 = trainer.model(ori_imgs, ori_imgs)
            pred = (preds1 + preds2) / 2
            l1.append((mae(pred * mask, ori_imgs * mask).item()))

    return np.mean(l1)


def evaluate(dataset: str = "test"):

    l1 = []
    test_dataset = BrainDataset(dataset=dataset, split='test', modalities='t2')
    for j in range(5):
        id = f"OPEN_IXI-{j}"
        trainer = denoising(id, data=None, lr=0.0001, depth=4,
                            wf=6)  # Noise parameters don't matter during evaluation.

        trainer.load(id)

        results = eval_anomalies_batched(trainer, dataset=test_dataset, batch_size=32)
        l1.append(results)

    logging.info("=" * 20)
    logging.info(dataset)
    logging.info(id)
    logging.info(f"l1: {np.mean(l1)}")
    logging.info(f"l1: {np.std(l1)}")


if __name__ == "__main__":
    import logging

    import argparse
    from cross_train import denoising

    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", "--dataset", default="IXI", type=str, help="'train', 'val' or 'test'")
    args = parser.parse_args()

    logging.basicConfig(filename='../test_saved_models/test_result.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    evaluate(
        dataset=args.dataset)
