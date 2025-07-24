#  Copyright (C) 2022 Canon Medical Systems Corporation. All rights reserved
from typing import Optional

import torch

from cross_data import BrainDataset


class DataDescriptor:

    def __init__(self, n_workers=16, batch_size=16, **kwargs):

        self.n_workers = n_workers
        self.batch_size = batch_size
        self.dataset_cache = {}

    def get_dataset(self, split: str):
        raise NotImplemented("get_dataset needs to be overridden in a subclass.")

    def get_dataset_(self, split: str, cache=True, force=False):
        if split not in self.dataset_cache or force:
            dataset = self.get_dataset(split)
            if cache:
                self.dataset_cache[split] = dataset
            return dataset
        else:
            return self.dataset_cache[split]

    def get_dataloader(self, split: str):
        dataset = self.get_dataset_(split, cache=True)

        shuffle = True if split == "train" else False
        drop_last = False if len(dataset) < self.batch_size else True

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=shuffle,
                                                 drop_last=drop_last,
                                                 num_workers=self.n_workers)

        return dataloader


class BrainAEDataDescriptor(DataDescriptor):

    def __init__(self, dataset="brats2021", modalities: Optional[str] = None,
                 seed: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.modalities = modalities
        self.seed = seed
        self.dataset = dataset

    def get_dataset(self, split: str):
        assert split in ["train", "val"]  # "test" should not be used through the DataDescriptor interface in this case.
        seed = 0 if split == "val" else self.seed
        dataset = BrainDataset(split=split, dataset=self.dataset, modalities=self.modalities,
                               seed=seed)

        return dataset
