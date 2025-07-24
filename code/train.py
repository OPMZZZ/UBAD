#  Copyright (C) 2022 Canon Medical Systems Corporation. All rights reserved
from typing import Union, Optional
from collections import defaultdict
from functools import partial
from pathlib import Path
import random

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from training import simple_train_step, simple_val_step
from metrics import Loss
from trainer import Trainer
from data_descriptor import BrainAEDataDescriptor, DataDescriptor
from utilities import ModelSaver, kl_divergence
from unet import Dubble_Unet

def denoising(identifier: str, data: Optional[Union[str, DataDescriptor]] = None, lr=0.001, depth=4, wf=7, n_input=3,
              ):
    device = torch.device("cuda:0")

    def noise(x, noise_std, noise_res, mask=None):
        ns = torch.normal(mean=torch.zeros(x.shape[0], x.shape[1], noise_res, noise_res), std=noise_std).to(x.device)

        ns = F.upsample_bilinear(ns, size=[128, 128])

        # Roll to randomly translate the generated noise.
        roll_x = random.choice(range(128))
        roll_y = random.choice(range(128))
        ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2, -1])
        if mask is None:
            mask = x.sum(dim=1, keepdim=True) > 0.01
        ns *= mask  # Only apply the noise in the foreground.
        res = x + ns
        return res

    def get_scores(trainer, batch):
        x = batch[0][:, trainer.input_channel]
        trainer.model = trainer.model.eval()
        with torch.no_grad():
            # Assume it's in batch shape
            clean = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1]).clone().to(trainer.device)
            mask = clean.sum(dim=1, keepdim=True) > 0.01

            # Erode the mask a bit to remove some of the reconstruction errors at the edges.
            mask = (F.avg_pool2d(mask.float(), kernel_size=5, stride=1, padding=2) > 0.95)

            res1, res2 = trainer.model(clean, clean)
            err_map3 = ((clean - res1) * mask).abs()
            err_map4 = ((clean - res2) * mask).abs()
            err_map1 = clean * torch.log((clean + 1e-10) / (res1 + 1e-10))
            err_map2 = clean * torch.log((clean + 1e-10) / (res2 + 1e-10))
            err_map1[torch.isnan(err_map1)] = 0
            err_map2[torch.isnan(err_map2)] = 0

            err = (err_map3 + err_map4).mean(dim=1, keepdim=True)
            err = torch.flip(err, dims=[-2, -1])
        return err.cpu()

    def SLIC_BSD(gts, fake0s, fake1s, regions):
        B, C, H, W = gts.shape
        device = gts.device

        # 最大区域数（假设所有 batch 中超像素数相同，或统一取最大）
        R = int(regions.max().item() + 1)

        # 扁平化空间和 batch
        N = B * H * W
        # 在 no‐loop 实现里，确保这两行：
        batch_idx = (
            torch.arange(B, device=device, dtype=torch.long)  # 指定 long
            .view(B, 1, 1)
            .expand(B, H, W)
            .reshape(-1)
        )
        reg_flat = regions.reshape(-1).to(torch.long)  # 强制 long

        # 然后 global_reg 也是 long
        global_reg = (batch_idx * R + reg_flat).long()

        # 扁平化像素值 & 特征
        gt_flat = gts.reshape(B, C, -1).permute(0, 2, 1).reshape(N, C)  # (N, C)
        f0_flat = fake0s.reshape(B, C, -1).permute(0, 2, 1).reshape(N, C)  # (N, C)
        f1_flat = fake1s.reshape(B, C, -1).permute(0, 2, 1).reshape(N, C)  # (N, C)

        # 计算每像素的平均通道误差
        err0 = torch.abs(f0_flat - gt_flat).mean(dim=1)  # (N,)
        err1 = torch.abs(f1_flat - gt_flat).mean(dim=1)  # (N,)

        # 在全局区域编号上做累加
        tot_bins = B * R
        sum0 = torch.zeros(tot_bins, device=device).scatter_add_(0, global_reg, err0)
        sum1 = torch.zeros(tot_bins, device=device).scatter_add_(0, global_reg, err1)
        cnt = torch.zeros(tot_bins, device=device).scatter_add_(0, global_reg,
                                                                torch.ones_like(global_reg, dtype=torch.float,
                                                                                device=device))

        # 区域平均误差
        mean0 = (sum0 / cnt).reshape(B, R)  # (B, R)
        mean1 = (sum1 / cnt).reshape(B, R)  # (B, R)

        # 决策掩码
        choose_f1 = mean1 < mean0  # (B, R)
        equal = mean1 == mean0  # (B, R)

        # 每像素对应的决策
        choose_f1_pix = choose_f1[batch_idx, reg_flat]  # (N,)
        equal_pix = equal[batch_idx, reg_flat]  # (N,)

        # 输出扁平化 & 赋值
        out_i_flat = f0_flat.clone()
        out_i_flat[choose_f1_pix, :] = f1_flat[choose_f1_pix, :]
        # 对相等区域取均值
        eq_idx = equal_pix.nonzero(as_tuple=False).squeeze()
        out_i_flat[eq_idx, :] = 0.5 * (
                f0_flat[eq_idx, :] + f1_flat[eq_idx, :]
        )


        # 重塑回 (B, C, H, W) / (B, Cf, H, W)
        target_i = (
            out_i_flat
            .reshape(B, H * W, C)
            .permute(0, 2, 1)
            .reshape(B, C, H, W)
        )
        return target_i

    def loss_f(trainer, batch, batch_results):
        y = batch[1][:, -4:]
        y = y[:, trainer.input_channel]

        mask = batch[1][:, :1]
        regions = batch[1][:, 1:2]
        fake0s, fake1s, _, _ = batch_results
        fake0s = fake0s * mask.float()
        fake1s = fake1s * mask.float()
        target_i = SLIC_BSD(y, fake0s, fake1s, regions)
        rec_loss = ((torch.pow(fake0s - y, 2) * mask.float() + torch.pow(fake1s - y, 2) * mask.float())).mean()
        bsd_i_loss = kl_divergence(target_i, fake0s).mean() + kl_divergence(target_i, fake1s).mean()
        return rec_loss + bsd_i_loss

    def forward(trainer, batch):
        assert batch[0].shape[1] == 10
        batch[1] = batch[0][:, 4:]
        input_img = batch[0][:, trainer.input_channel]
        return trainer.model.forward_with_last(noise(input_img, 0.1, 8), noise(input_img, 0.2, 16))


    model = Dubble_Unet(in_channels=n_input, n_classes=n_input, norm="group", up_mode="upconv", depth=depth, wf=wf,
                 padding=True).to(device)

    train_step = partial(simple_train_step, forward=forward, loss_f=loss_f)
    val_step = partial(simple_val_step, forward=forward, loss_f=loss_f)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True, weight_decay=0.00001)
    callback_dict = defaultdict(list)

    model_saver = ModelSaver(path=Path(__file__).resolve().parent.parent / "saved_models" / f"{identifier}.pt")
    model_saver.register(callback_dict)

    Loss(lambda batch_res, batch: loss_f(trainer, batch, batch_res)).register(callback_dict, log=True, tensorboard=True,
                                                                              train=True, val=True)

    trainer = Trainer(model=model,
                      train_dataloader=None,
                      val_dataloader=None,
                      optimiser=optimiser,
                      train_step=train_step,
                      val_step=val_step,
                      callback_dict=callback_dict,
                      device=device,
                      identifier=identifier)

    trainer.noise = noise
    trainer.get_scores = get_scores
    trainer.set_data(data)
    trainer.reset_state()

    trainer.lr_scheduler = CosineAnnealingLR(optimizer=optimiser, T_max=100)

    def update_learning_rate(trainer):
        trainer.lr_scheduler.step()

    trainer.callback_dict["after_train_epoch"].append(update_learning_rate)

    return trainer


def train(seed: int = 0, batch_size: int = 16):
    ds_dict = {'brats2020':[0, 3],'brats2021':[0, 3],  'ISLE':[0, 1]}

    for key, value in ds_dict.items():
        dd = BrainAEDataDescriptor(dataset=key, n_train_patients=None, n_val_patients=None,
                                   seed=seed, batch_size=batch_size)
        id = f"open_{key}"
        trainer = denoising(id, data=dd, lr=0.0001, depth=4,
                            wf=6, n_input=len(value))
        trainer.input_channel = value
        trainer.train(epoch_len=32, max_epochs=1600, val_epoch_len=32)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=0, help="random seed.")
    parser.add_argument("-bs", "--batch_size", type=int, default=8, help="model training batch size")

    args = parser.parse_args()

    train(
          seed=args.seed,
          batch_size=args.batch_size)
