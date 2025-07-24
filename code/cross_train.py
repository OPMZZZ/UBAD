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
from cross_data_descriptor import BrainAEDataDescriptor, DataDescriptor
from utilities import kl_divergence, ModelSaver
from unet import Dubble_Unet

from fast_slic.avx2 import SlicAvx2
import numpy as np
from skimage.color import gray2rgb


def denoising(identifier: str, data: Optional[Union[str, DataDescriptor]] = None, lr=0.001, depth=4, wf=7, n_input=1):
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

        ns *= mask
        res = x + ns

        return res

    def get_scores(trainer, batch):
        x = batch
        trainer.model = trainer.model.eval()
        with torch.no_grad():
            # Assume it's in batch shape
            clean = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1]).clone().to(trainer.device)[:, [0]]
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

            err = (err_map1 + err_map2 + err_map4 + err_map3).mean(dim=1, keepdim=True)

        return err.cpu()

    def SLIC_BSD(
            gts, fake0s, fake1s, mask_tensor,
            num_components=100, compactness=0
    ):

        B, C, H, W = gts.shape
        device = gts.device

        target_i = torch.zeros_like(fake0s)

        # 只初始化一次 SLIC 实例
        slic = SlicAvx2(num_components=num_components, compactness=compactness)

        for i in range(B):
            # ——1. 运行 SLIC CPU 分割——
            gt = gts[i]  # (C, H, W)
            gt_img = (gt[0].cpu().numpy() * 255).astype(np.uint8)
            seg_cpu = slic.iterate(gray2rgb(gt_img))  # (H, W), np.int32
            seg = (
                    torch.from_numpy(seg_cpu)
                    .to(device)
                    .unsqueeze(0)  # (1, H, W)
                    * mask_tensor[i]
            )

            # ——2. 扁平化像素和分割标签——
            seg_flat = seg.reshape(-1).long()  # (N,), N = H*W

            # 扁平化像素值和特征
            def flatten(x):
                # x: (C?, H, W) -> (N, C?)
                return x.reshape(x.shape[0], -1).permute(1, 0)

            gt_flat = flatten(gt)  # (N, C)
            f0_flat = flatten(fake0s[i])
            f1_flat = flatten(fake1s[i])
            # ——3. 计算每像素误差——
            err0 = torch.abs(f0_flat - gt_flat).mean(dim=1)  # (N,)
            err1 = torch.abs(f1_flat - gt_flat).mean(dim=1)  # (N,)

            # ——4. 区域级聚合 & 平均误差——
            n_regions = int(seg_flat.max().item() + 1)
            sum0 = torch.zeros(n_regions, device=device).scatter_add_(0, seg_flat, err0)
            sum1 = torch.zeros(n_regions, device=device).scatter_add_(0, seg_flat, err1)
            count = torch.zeros(n_regions, device=device).scatter_add_(
                0, seg_flat,
                torch.ones_like(seg_flat, dtype=torch.float, device=device)
            )
            mean0 = sum0 / count  # (R,)
            mean1 = sum1 / count  # (R,)

            # ——5. 生成像素级决策掩码——
            choose_f1 = mean1 < mean0  # (R,) bool
            equal = mean0 == mean1  # (R,) bool
            choose_f1_pix = choose_f1[seg_flat]  # (N,)
            equal_pix = equal[seg_flat]  # (N,)

            # ——6. 向量化合成输出——
            # 原始像素值部分
            out_i_flat = torch.where(
                choose_f1_pix.unsqueeze(1), f1_flat, f0_flat
            )
            # 相等区域取均值
            eq_idx = equal_pix.nonzero(as_tuple=True)[0]
            out_i_flat[eq_idx] = 0.5 * (f0_flat[eq_idx] + f1_flat[eq_idx])

            # ——7. 重塑回原始形状——
            target_i[i] = out_i_flat.permute(1, 0).reshape(C, H, W)

        return target_i

    def loss_f(trainer, batch, batch_results):
        batch = torch.stack(batch)
        y = batch[:, -1:]
        # y = median_pool(y, kernel_size=3, stride=1, padding=1).detach()
        mask = batch[:, 1:2]
        fake0s, fake1s, _, _ = batch_results
        fake0s = fake0s * mask.float()
        fake1s = fake1s * mask.float()
        target_i = SLIC_BSD(y, fake0s, fake1s, mask)
        rec_loss = ((torch.pow(fake0s - y, 2) * mask.float() + torch.pow(fake1s - y, 2) * mask.float())).mean()
        bsd_i_loss = kl_divergence(target_i, fake0s).mean() + kl_divergence(target_i, fake1s).mean()

        return rec_loss + bsd_i_loss

    def forward(trainer, batch):
        batch = torch.stack(batch)
        assert batch.shape[1] == 4
        input_img = batch[:, [0]]

        return trainer.model.forward_with_last(noise(input_img, 0.2, 16), noise(input_img, 0.1, 8))

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
    for j in range(5):
        id = f"OPEN_IXI-{j}"
        # 初始化数据描述器
        dd = BrainAEDataDescriptor(dataset=f"IXI_fold{j}", modalities='t2',
                                   seed=seed, batch_size=batch_size)
        # 初始化训练器并训练
        trainer = denoising(id, data=dd, lr=0.0001, depth=4, wf=6)
        trainer.train(epoch_len=32, max_epochs=200, val_epoch_len=32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=0, help="random seed.")
    parser.add_argument("-bs", "--batch_size", type=int, default=8, help="model training batch size")

    args = parser.parse_args()

    train(
        seed=args.seed,
        batch_size=args.batch_size)
