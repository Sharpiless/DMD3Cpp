#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    criteria.py
# @Project:     BP-Net
# @Author:      jie
# @Time:        2021/3/14 7:51 PM

import torch
import torch.nn as nn
import torch.nn.functional as F
from disp_loss import ScaleAndShiftInvariantLoss, PatchScaleAndShiftInvariantLoss
from disp_loss import MSMSE_ranking_fast

__all__ = [
    'RMSE',
    'MSMSE',
    'MSMSE_mono',
    'MSMSE_pssil',
    'MSMSE_ssil',
    'MSMSE_ranking_fast',
    'MSMSEV2',
    'MetricALL',
    'MSMSEMWasserstein'
]

class MSMSE_pssil(nn.Module):
    def __init__(
        self,
        deltas=(2 ** (-5 * 2), 2 ** (-4 * 2), 2 ** (-3 * 2), 2 ** (-2 * 2), 2 ** (-1 * 2), 1),
        deltas_ssil=(2 ** (-5 * 2), 2 ** (-4 * 2), 2 ** (-3 * 2), 2 ** (-2 * 2), 2 ** (-1 * 2), 0.05),
        lambda_gt=1.0,
        lambda_metric=0.1,
        min_valid_for_scale_shift=24,
        detach_scale_shift=True,

        # new
        patch_size=96,         # int or list[int]
        patch_stride=None,     # None -> patch_size//2
        patch_alpha=0.2,
    ):
        super().__init__()

        self.ssil_loss = PatchScaleAndShiftInvariantLoss(
            patch_size=patch_size,
            stride=patch_stride,
            alpha=patch_alpha,
            min_valid_pixels=min_valid_for_scale_shift,
            detach_scale_shift=detach_scale_shift,
        )

        self.deltas = deltas
        self.deltas_ssil = deltas_ssil
        self.lambda_gt = lambda_gt
        self.lambda_metric = lambda_metric

    def masked_mse(self, pred, target, mask):
        mask = mask.float()
        denom = mask.sum().clamp(min=1.0)
        return ((pred - target) ** 2 * mask).sum() / denom

    def single_scale_loss(self, est, gt, metric_depth):
        """
        est, gt, metric_depth: [B, 1, H, W]
        """
        gt_valid = gt > 1e-3
        metric_valid = metric_depth[:, 0] > 1e-3

        # sparse GT absolute supervision
        loss_gt = self.masked_mse(est, gt, gt_valid)

        # patch-based local SSI on dense prior
        loss_metric = self.ssil_loss(
            est[:, 0],               # [B,H,W]
            metric_depth[:, 0],      # [B,H,W]
            metric_valid.float(),    # [B,H,W]
        )

        stats = {
            "loss_gt": loss_gt.detach(),
            "loss_metric": loss_metric.detach(),
            "gt_pixels": gt_valid.float().sum().detach(),
            "metric_pixels": metric_valid.float().sum().detach(),
        }

        return self.lambda_gt * loss_gt, self.lambda_metric * loss_metric, stats

    def forward(self, outputs, target, metric_depth, return_dis=True):
        """
        outputs: list of multi-scale predictions, each [B, 1, H_i, W_i]
        target: sparse GT, [B, 1, H, W]
        metric_depth: dense / semi-dense prior, [B, 1, H, W]
        """
        total_loss = 0.0
        all_stats = []

        for est, delta, delta_ssil in zip(outputs, self.deltas, self.deltas_ssil):
            size = est.shape[-2:]

            gt_scaled = F.interpolate(target, size=size, mode="nearest")
            metric_scaled = F.interpolate(metric_depth, size=size, mode="bilinear", align_corners=False)

            loss_gt, loss_metric, stats_i = self.single_scale_loss(est, gt_scaled, metric_scaled)
            if return_dis:
                total_loss = total_loss + delta * loss_gt + delta_ssil * loss_metric
            else:
                total_loss = total_loss + delta * loss_gt
            all_stats.append(stats_i)

        return total_loss, all_stats
    

class MSMSE_pssilv2(nn.Module):
    def __init__(
        self,
        deltas=(2 ** (-5 * 2), 2 ** (-4 * 2), 2 ** (-3 * 2), 2 ** (-2 * 2), 2 ** (-1 * 2), 1),
        deltas_ssil=(2 ** (-5 * 2), 2 ** (-4 * 2), 2 ** (-3 * 2), 2 ** (-2 * 2), 2 ** (-1 * 2), 0.05),
        lambda_gt=1.0,
        lambda_metric=0.1,
        min_valid_for_scale_shift=24,
        detach_scale_shift=True,

        # new
        patch_size=64,         # int or list[int]
        patch_stride=None,     # None -> patch_size//2
        patch_alpha=0.2,
    ):
        super().__init__()

        self.ssil_loss = PatchScaleAndShiftInvariantLoss(
            patch_size=patch_size,
            stride=patch_stride,
            alpha=patch_alpha,
            min_valid_pixels=min_valid_for_scale_shift,
            detach_scale_shift=detach_scale_shift,
        )

        self.deltas = deltas
        self.deltas_ssil = deltas_ssil
        self.lambda_gt = lambda_gt
        self.lambda_metric = lambda_metric

    def masked_mse(self, pred, target, mask):
        mask = mask.float()
        denom = mask.sum().clamp(min=1.0)
        return ((pred - target) ** 2 * mask).sum() / denom

    def single_scale_loss(self, est, gt, metric_depth):
        """
        est, gt, metric_depth: [B, 1, H, W]
        """
        gt_valid = gt > 1e-3
        metric_valid = metric_depth[:, 0] > 1e-3

        # sparse GT absolute supervision
        loss_gt = self.masked_mse(est, gt, gt_valid)

        # patch-based local SSI on dense prior
        loss_metric = self.ssil_loss(
            est[:, 0],               # [B,H,W]
            metric_depth[:, 0],      # [B,H,W]
            metric_valid.float(),    # [B,H,W]
        )

        stats = {
            "loss_gt": loss_gt.detach(),
            "loss_metric": loss_metric.detach(),
            "gt_pixels": gt_valid.float().sum().detach(),
            "metric_pixels": metric_valid.float().sum().detach(),
        }

        return self.lambda_gt * loss_gt, self.lambda_metric * loss_metric, stats

    def forward(self, outputs, target, metric_depth):
        """
        outputs: list of multi-scale predictions, each [B, 1, H_i, W_i]
        target: sparse GT, [B, 1, H, W]
        metric_depth: dense / semi-dense prior, [B, 1, H, W]
        """
        total_loss = 0.0
        all_stats = []

        for est, delta, delta_ssil in zip(outputs, self.deltas, self.deltas_ssil):
            size = est.shape[-2:]

            gt_scaled = F.interpolate(target, size=size, mode="nearest")
            metric_scaled = F.interpolate(metric_depth, size=size, mode="bilinear", align_corners=False)

            loss_gt, loss_metric, stats_i = self.single_scale_loss(est, gt_scaled, metric_scaled)
            total_loss = total_loss + delta * loss_gt + delta_ssil * loss_metric
            all_stats.append(stats_i)

        return total_loss, all_stats

class MSMSE_ssil(nn.Module):

    def __init__(
        self,
        deltas=(2 ** (-5 * 2), 2 ** (-4 * 2), 2 ** (-3 * 2), 2 ** (-2 * 2), 2 ** (-1 * 2), 1),
        deltas_ssil=(2 ** (-5 * 2), 2 ** (-4 * 2), 2 ** (-3 * 2), 2 ** (-2 * 2), 2 ** (-1 * 2), 0.05),
        lambda_gt=1.0,
        lambda_metric=0.2,
        min_valid_for_scale_shift=32,
        detach_scale_shift=True,
    ):
        super().__init__()
        self.ssil_loss = ScaleAndShiftInvariantLoss()
        self.deltas = deltas
        self.lambda_gt = lambda_gt
        self.lambda_metric = lambda_metric
        self.min_valid_for_scale_shift = min_valid_for_scale_shift
        self.detach_scale_shift = detach_scale_shift
        self.deltas_ssil = deltas_ssil

    def masked_mse(self, pred, target, mask):
        mask = mask.float()
        denom = mask.sum().clamp(min=1.0)
        return ((pred - target) ** 2 * mask).sum() / denom

    def single_scale_loss(self, est, gt, metric_depth):
        """
        est, gt, metric_depth: [B, 1, H, W]
        """
        gt_valid = gt > 1e-3
        metric_valid = metric_depth[:, 0] > 1e-3

        # 1) sparse GT supervision
        loss_gt = self.masked_mse(est, gt, gt_valid)

        loss_metric = self.ssil_loss(est[:, 0], metric_depth[:, 0], metric_valid.float())

        stats = {
            "loss_gt": loss_gt.detach(),
            "loss_metric": loss_metric.detach(),
            "gt_pixels": gt_valid.float().sum().detach(),
        }
        return self.lambda_gt * loss_gt, self.lambda_metric * loss_metric, stats

    def forward(self, outputs, target, metric_depth, return_dis=True):
        """
        outputs: list of multi-scale predictions, each [B, 1, H_i, W_i]
        target: sparse GT, [B, 1, H, W]
        metric_depth: dense/semi-dense prior, [B, 1, H, W]
        """
        total_loss = 0.0
        all_stats = []

        for est, delta, delta_ssil in zip(outputs, self.deltas, self.deltas_ssil):
            size = est.shape[-2:]

            # sparse GT 下采样建议用 nearest，避免把 0 和有效值混起来
            gt_scaled = F.interpolate(target, size=size, mode="nearest")
            metric_scaled = F.interpolate(metric_depth, size=size, mode="bilinear", align_corners=False)

            loss_gt, loss_metric, stats_i = self.single_scale_loss(est, gt_scaled, metric_scaled)
            total_loss = total_loss + delta * loss_gt + delta_ssil * loss_metric
            all_stats.append(stats_i)

        return total_loss , all_stats

class MSMSE_mono(nn.Module):
    """
    Multi-Scale loss with sparse GT + aligned metric depth prior
    gt: sparse depth, 0 means invalid
    metric_depth: dense or semi-dense metric depth prior
    """

    def __init__(
        self,
        deltas=(2 ** (-5 * 2), 2 ** (-4 * 2), 2 ** (-3 * 2), 2 ** (-2 * 2), 2 ** (-1 * 2), 1),
        lambda_gt=1.0,
        lambda_metric=0.2,
        min_valid_for_scale_shift=32,
        detach_scale_shift=True,
        supervise_metric_only_where_gt_missing=True,
    ):
        super().__init__()
        self.deltas = deltas
        self.lambda_gt = lambda_gt
        self.lambda_metric = lambda_metric
        self.min_valid_for_scale_shift = min_valid_for_scale_shift
        self.detach_scale_shift = detach_scale_shift
        self.supervise_metric_only_where_gt_missing = supervise_metric_only_where_gt_missing

    def masked_mse(self, pred, target, mask):
        mask = mask.float()
        denom = mask.sum().clamp(min=1.0)
        return ((pred - target) ** 2 * mask).sum() / denom
    
    def masked_charbonnier(self, pred, target, mask, eps=1e-3):
        mask = mask.float()
        diff = pred - target
        loss = torch.sqrt(diff * diff + eps * eps)
        denom = mask.sum().clamp(min=1.0)
        return (loss * mask).sum() / denom
    
    def compute_scale_shift_per_sample(self, metric_depth, gt, mask):
        valid = mask.bool()
        num_valid = valid.sum()

        if num_valid < self.min_valid_for_scale_shift:
            scale = metric_depth.new_tensor(1.0)
            shift = metric_depth.new_tensor(0.0)
            return scale, shift

        x = metric_depth[valid].float()
        y = gt[valid].float()

        mean_x = x.mean()
        mean_y = y.mean()

        var_x = ((x - mean_x) ** 2).mean()
        cov_xy = ((x - mean_x) * (y - mean_y)).mean()

        eps = 1e-6
        if var_x < eps:
            scale = metric_depth.new_tensor(1.0)
            shift = mean_y - scale * mean_x
        else:
            scale = cov_xy / (var_x + eps)
            scale = torch.clamp(scale, min=0.01, max=100.0)
            shift = mean_y - scale * mean_x

        scale = scale.to(metric_depth.dtype)
        shift = shift.to(metric_depth.dtype)

        if self.detach_scale_shift:
            scale = scale.detach()
            shift = shift.detach()

        return scale, shift

    def align_metric_depth(self, metric_depth, gt):
        """
        Align metric_depth to sparse GT per sample.
        metric_depth: [B, 1, H, W]
        gt:           [B, 1, H, W]
        """
        assert metric_depth.shape == gt.shape
        B = metric_depth.shape[0]
        aligned = torch.zeros_like(metric_depth)

        for i in range(B):
            md = metric_depth[i, 0]
            g = gt[i, 0]

            overlap_mask = (g > 1e-3) & (md > 1e-3)
            scale, shift = self.compute_scale_shift_per_sample(md, g, overlap_mask)

            aligned_i = scale * md + shift
            aligned_i = torch.clamp(aligned_i, min=0.0)  # depth should be non-negative
            aligned[i, 0] = aligned_i

        return aligned

    def single_scale_loss(self, est, gt, metric_depth):
        """
        est, gt, metric_depth: [B, 1, H, W]
        """
        gt_valid = gt > 1e-3
        metric_valid = metric_depth > 1e-3

        # 1) sparse GT supervision
        loss_gt = self.masked_mse(est, gt, gt_valid)

        # 2) align metric depth to sparse GT
        aligned_metric = self.align_metric_depth(metric_depth, gt)

        # 3) auxiliary metric supervision
        if self.supervise_metric_only_where_gt_missing:
            metric_mask = metric_valid & (~gt_valid)
        else:
            metric_mask = metric_valid

        loss_metric = self.masked_mse(est, aligned_metric, metric_mask)
        # loss_metric = self.masked_charbonnier(est, aligned_metric, metric_mask)

        total = self.lambda_gt * loss_gt + self.lambda_metric * loss_metric

        stats = {
            "loss_gt": loss_gt.detach(),
            "loss_metric": loss_metric.detach(),
            "metric_pixels": metric_mask.float().sum().detach(),
            "gt_pixels": gt_valid.float().sum().detach(),
        }
        return total, stats

    def forward(self, outputs, target, metric_depth):
        """
        outputs: list of multi-scale predictions, each [B, 1, H_i, W_i]
        target: sparse GT, [B, 1, H, W]
        metric_depth: dense/semi-dense prior, [B, 1, H, W]
        """
        total_loss = 0.0
        all_stats = []

        for est, delta in zip(outputs, self.deltas):
            size = est.shape[-2:]

            # sparse GT 下采样建议用 nearest，避免把 0 和有效值混起来
            gt_scaled = F.interpolate(target, size=size, mode="nearest")
            metric_scaled = F.interpolate(metric_depth, size=size, mode="bilinear", align_corners=False)

            loss_i, stats_i = self.single_scale_loss(est, gt_scaled, metric_scaled)
            total_loss = total_loss + delta * loss_i
            all_stats.append(stats_i)

        return total_loss , all_stats
    


class MSMSEMWasserstein(nn.Module):
    """
    Multi-scale sparse depth supervision + aligned metric prior Wasserstein loss.

    设计目标：
    1) sparse LiDAR GT 是主监督
    2) metric depth 仅作为“全局分布先验”，避免 dense pixel-wise teacher 过强
    3) 支持可选局部结构约束（gradient loss）

    Args:
        deltas:
            多尺度权重
        lambda_gt:
            稀疏 GT 主监督权重
        lambda_metric:
            metric depth 分布监督权重（Wasserstein）
        lambda_metric_grad:
            可选：metric depth 局部梯度结构约束权重
        min_valid_for_scale_shift:
            做 scale-shift 对齐所需的最少重叠有效像素数
        detach_scale_shift:
            是否对 scale/shift detach
        supervise_metric_only_where_gt_missing:
            metric 分支是否只监督 GT 缺失区域
        metric_dilate_ks:
            若 > 0，则只在 GT 邻域扩张区域内使用 metric 监督，避免全图 teacher
        use_log_for_metric:
            是否在 metric 分支中先对 depth 取 log，再做 Wasserstein / gradient
            更适合弱化绝对尺度影响
        wasserstein_num_samples:
            每张图用于近似 1D Wasserstein 的采样点数
        wasserstein_trim_quantile:
            为了鲁棒性，对分布两端做裁剪（例如 0.02 表示裁掉头尾各 2%）
        huber_delta:
            GT Huber loss 的 delta
    """

    def __init__(
        self,
        deltas=(2 ** (-5 * 2), 2 ** (-4 * 2), 2 ** (-3 * 2), 2 ** (-2 * 2), 2 ** (-1 * 2), 1.0),
        lambda_gt=1.0,
        lambda_metric=0.03,
        lambda_metric_grad=0.05,
        min_valid_for_scale_shift=32,
        detach_scale_shift=True,
        supervise_metric_only_where_gt_missing=True,
        metric_dilate_ks=None,
        use_log_for_metric=False,
        wasserstein_num_samples=256,
        wasserstein_trim_quantile=0.02,
        huber_delta=1.0,
    ):
        super().__init__()
        self.deltas = deltas
        self.lambda_gt = lambda_gt
        self.lambda_metric = lambda_metric
        self.lambda_metric_grad = lambda_metric_grad

        self.min_valid_for_scale_shift = min_valid_for_scale_shift
        self.detach_scale_shift = detach_scale_shift
        self.supervise_metric_only_where_gt_missing = supervise_metric_only_where_gt_missing

        self.metric_dilate_ks = metric_dilate_ks
        self.use_log_for_metric = use_log_for_metric
        self.wasserstein_num_samples = wasserstein_num_samples
        self.wasserstein_trim_quantile = wasserstein_trim_quantile
        self.huber_delta = huber_delta

    # -----------------------------
    # basic masked losses
    # -----------------------------
    def masked_mse(self, pred, target, mask):
        mask = mask.float()
        denom = mask.sum().clamp(min=1.0)
        return ((pred - target) ** 2 * mask).sum() / denom

    def masked_huber(self, pred, target, mask, delta=None):
        if delta is None:
            delta = self.huber_delta
        mask = mask.float()
        diff = pred - target
        abs_diff = diff.abs()
        loss = torch.where(abs_diff < delta, 0.5 * diff ** 2 / delta, abs_diff - 0.5 * delta)
        denom = mask.sum().clamp(min=1.0)
        return (loss * mask).sum() / denom

    def masked_charbonnier(self, pred, target, mask, eps=1e-3):
        mask = mask.float()
        diff = pred - target
        loss = torch.sqrt(diff * diff + eps * eps)
        denom = mask.sum().clamp(min=1.0)
        return (loss * mask).sum() / denom

    # -----------------------------
    # mask utils
    # -----------------------------
    def dilate_mask(self, mask, kernel_size):
        if kernel_size is None or kernel_size <= 0:
            return mask.bool()
        pad = kernel_size // 2
        out = F.max_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=pad)
        return out > 0

    # -----------------------------
    # scale-shift alignment
    # -----------------------------
    def compute_scale_shift_per_sample(self, metric_depth, gt, mask):
        valid = mask.bool()
        num_valid = valid.sum()

        if num_valid < self.min_valid_for_scale_shift:
            scale = metric_depth.new_tensor(1.0)
            shift = metric_depth.new_tensor(0.0)
            return scale, shift

        x = metric_depth[valid].float()
        y = gt[valid].float()

        mean_x = x.mean()
        mean_y = y.mean()

        var_x = ((x - mean_x) ** 2).mean()
        cov_xy = ((x - mean_x) * (y - mean_y)).mean()

        eps = 1e-6
        if var_x < eps:
            scale = metric_depth.new_tensor(1.0)
            shift = mean_y - scale * mean_x
        else:
            scale = cov_xy / (var_x + eps)
            scale = torch.clamp(scale, min=0.01, max=100.0)
            shift = mean_y - scale * mean_x

        scale = scale.to(metric_depth.dtype)
        shift = shift.to(metric_depth.dtype)

        if self.detach_scale_shift:
            scale = scale.detach()
            shift = shift.detach()

        return scale, shift

    def align_metric_depth(self, metric_depth, gt):
        """
        Align metric_depth to sparse GT per sample.
        metric_depth: [B, 1, H, W]
        gt:           [B, 1, H, W]
        """
        assert metric_depth.shape == gt.shape
        B = metric_depth.shape[0]
        aligned = torch.zeros_like(metric_depth)
        scales = []
        shifts = []

        for i in range(B):
            md = metric_depth[i, 0]
            g = gt[i, 0]

            overlap_mask = (g > 1e-3) & (md > 1e-3)
            scale, shift = self.compute_scale_shift_per_sample(md, g, overlap_mask)

            aligned_i = scale * md + shift
            aligned_i = torch.clamp(aligned_i, min=1e-3)
            aligned[i, 0] = aligned_i

            scales.append(scale.detach())
            shifts.append(shift.detach())

        scales = torch.stack(scales) if len(scales) > 0 else metric_depth.new_zeros((0,))
        shifts = torch.stack(shifts) if len(shifts) > 0 else metric_depth.new_zeros((0,))
        return aligned, scales, shifts

    # -----------------------------
    # transform for metric branch
    # -----------------------------
    def metric_transform(self, x):
        x = torch.clamp(x, min=1e-3)
        if self.use_log_for_metric:
            x = torch.log(x)
        return x

    # -----------------------------
    # gradient structure loss (optional)
    # -----------------------------
    def gradient_loss(self, pred, target, mask):
        """
        pred, target: [B,1,H,W]
        mask:         [B,1,H,W] bool
        """
        pred_t = self.metric_transform(pred)
        target_t = self.metric_transform(target)

        pred_dx = pred_t[..., :, 1:] - pred_t[..., :, :-1]
        pred_dy = pred_t[..., 1:, :] - pred_t[..., :-1, :]
        tgt_dx = target_t[..., :, 1:] - target_t[..., :, :-1]
        tgt_dy = target_t[..., 1:, :] - target_t[..., :-1, :]

        mask_dx = mask[..., :, 1:] & mask[..., :, :-1]
        mask_dy = mask[..., 1:, :] & mask[..., :-1, :]

        loss_dx = self.masked_charbonnier(pred_dx, tgt_dx, mask_dx)
        loss_dy = self.masked_charbonnier(pred_dy, tgt_dy, mask_dy)
        return loss_dx + loss_dy

    # -----------------------------
    # 1D Wasserstein distance
    # -----------------------------
    def wasserstein_1d_from_samples(self, x, y):
        """
        x, y: 1D tensors
        通过分位点采样近似 1D Wasserstein-1:
            W1 = mean_q | Q_x(q) - Q_y(q) |
        """
        device = x.device
        dtype = x.dtype

        if x.numel() < 2 or y.numel() < 2:
            return x.new_tensor(0.0)

        x = x.float()
        y = y.float()

        # optional trimming to reduce heavy-tail / outlier influence
        tq = self.wasserstein_trim_quantile
        if tq is not None and tq > 0.0 and tq < 0.5:
            lx = torch.quantile(x, tq)
            hx = torch.quantile(x, 1.0 - tq)
            ly = torch.quantile(y, tq)
            hy = torch.quantile(y, 1.0 - tq)
            x = x[(x >= lx) & (x <= hx)]
            y = y[(y >= ly) & (y <= hy)]

            if x.numel() < 2 or y.numel() < 2:
                return torch.tensor(0.0, device=device, dtype=dtype)

        n = min(self.wasserstein_num_samples, x.numel(), y.numel())
        if n < 2:
            return torch.tensor(0.0, device=device, dtype=dtype)

        q = torch.linspace(0.0, 1.0, steps=n, device=x.device, dtype=x.dtype)
        qx = torch.quantile(x, q)
        qy = torch.quantile(y, q)
        w1 = torch.mean(torch.abs(qx - qy))
        return w1.to(dtype)

    def distribution_wasserstein_loss(self, pred, target, mask):
        """
        pred, target: [B,1,H,W]
        mask:         [B,1,H,W] bool

        逐样本在 mask 区域比较深度分布，不关心像素位置对应关系，
        只比较“整体深度分布是否接近”。
        """
        B = pred.shape[0]
        total = pred.new_tensor(0.0)
        valid_count = 0

        pred_t = self.metric_transform(pred)
        target_t = self.metric_transform(target)

        for i in range(B):
            m = mask[i, 0]
            if m.sum() < 8:
                continue

            x = pred_t[i, 0][m]
            y = target_t[i, 0][m]

            if x.numel() < 2 or y.numel() < 2:
                continue

            loss_i = self.wasserstein_1d_from_samples(x, y)
            total = total + loss_i
            valid_count += 1

        if valid_count == 0:
            return pred.new_tensor(0.0)

        return total / valid_count

    # -----------------------------
    # single scale
    # -----------------------------
    def single_scale_loss(self, est, gt, metric_depth):
        """
        est, gt, metric_depth: [B,1,H,W]
        """
        gt_valid = gt > 1e-3
        metric_valid = metric_depth > 1e-3

        # 1) sparse GT 主监督（建议更稳的 Huber）
        loss_gt = self.masked_huber(est, gt, gt_valid, delta=self.huber_delta)

        # 2) metric depth 对齐到 sparse GT 坐标系
        aligned_metric, scales, shifts = self.align_metric_depth(metric_depth, gt)

        # 3) metric supervision mask
        if self.supervise_metric_only_where_gt_missing:
            metric_mask = metric_valid & (~gt_valid)
        else:
            metric_mask = metric_valid

        # 限制 metric 只在 GT 邻域附近起作用，减少远离 GT 的 teacher 误导
        if self.metric_dilate_ks is not None and self.metric_dilate_ks > 0:
            support = self.dilate_mask(gt_valid, self.metric_dilate_ks)
            metric_mask = metric_mask & support

        # 4) Wasserstein 分布监督
        loss_metric = self.distribution_wasserstein_loss(est, aligned_metric, metric_mask)

        # 5) 可选：局部梯度结构约束
        if self.lambda_metric_grad > 0:
            loss_metric_grad = self.gradient_loss(est, aligned_metric, metric_mask)
        else:
            loss_metric_grad = est.new_tensor(0.0)

        total = (
            self.lambda_gt * loss_gt
            + self.lambda_metric * loss_metric
            + self.lambda_metric_grad * loss_metric_grad
        )

        stats = {
            "loss_gt": loss_gt.detach(),
            "loss_metric_w1": loss_metric.detach(),
            "loss_metric_grad": loss_metric_grad.detach(),
            "metric_pixels": metric_mask.float().sum().detach(),
            "gt_pixels": gt_valid.float().sum().detach(),
            "scale_mean": scales.mean().detach() if scales.numel() > 0 else est.new_tensor(0.0),
            "shift_mean": shifts.mean().detach() if shifts.numel() > 0 else est.new_tensor(0.0),
        }
        return total, stats

    # -----------------------------
    # forward
    # -----------------------------
    def forward(self, outputs, target, metric_depth):
        """
        outputs: list of multi-scale predictions, each [B,1,H_i,W_i]
        target: sparse GT, [B,1,H,W]
        metric_depth: dense/semi-dense prior, [B,1,H,W]
        """
        total_loss = 0.0
        all_stats = []

        for est, delta in zip(outputs, self.deltas):
            size = est.shape[-2:]

            # sparse GT 下采样用 nearest，避免 0 和有效值混合
            gt_scaled = F.interpolate(target, size=size, mode="nearest")

            # metric prior 下采样用 bilinear 更平滑
            metric_scaled = F.interpolate(
                metric_depth, size=size, mode="bilinear", align_corners=False
            )

            loss_i, stats_i = self.single_scale_loss(est, gt_scaled, metric_scaled)
            total_loss = total_loss + delta * loss_i
            all_stats.append(stats_i)

        return total_loss, all_stats


class RMSE(nn.Module):

    def __init__(self, mul_factor=1.):
        super().__init__()
        self.mul_factor = mul_factor
        self.metric_name = [
            'RMSE',
        ]

    def forward(self, outputs, target):
        outputs = outputs / self.mul_factor
        target = target / self.mul_factor
        val_pixels = (target > 1e-3).float()
        err = (target * val_pixels - outputs * val_pixels) ** 2
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        return torch.sqrt(loss / cnt).mean(),


class MSMSE(nn.Module):
    """
    Multi-Scale MSE
    """

    def __init__(self, deltas=(2 ** (-5 * 2), 2 ** (-4 * 2), 2 ** (-3 * 2), 2 ** (-2 * 2), 2 ** (-1 * 2), 1)):
        super().__init__()
        self.deltas = deltas

    def mse(self, est, gt):
        valid = (gt > 1e-3).float()
        loss = est * valid - gt * valid
        return (loss ** 2).mean()

    def forward(self, outputs, target):
        loss = [delta * self.mse(ests, target) for ests, delta in zip(outputs, self.deltas)]
        return loss
    
class MSMSEV2(nn.Module):
    """
    Multi-Scale MSE
    """

    def __init__(self, deltas=(2 ** (-5 * 2), 2 ** (-4 * 2), 2 ** (-3 * 2), 2 ** (-2 * 2), 2 ** (-1 * 2), 1)):
        super().__init__()
        self.deltas = deltas

    def mse(self, est, gt):
        valid = (gt > 1e-3).float()
        loss = est * valid - gt * valid
        return (loss ** 2).sum() / (valid.sum() + 1e-8)

    def forward(self, outputs, target):
        loss = [delta * self.mse(ests, target) for ests, delta in zip(outputs, self.deltas)]
        return loss


class MetricALL(nn.Module):
    def __init__(self, mul_factor):
        super().__init__()
        self.t_valid = 0.0001
        self.mul_factor = mul_factor
        self.metric_name = [
            'RMSE', 'MAE', 'iRMSE', 'iMAE', 'REL', 'D^1', 'D^2', 'D^3', 'D102', 'D105', 'D110'
        ]

    def forward(self, pred, gt):
        with torch.no_grad():
            pred = pred.detach() / self.mul_factor
            gt = gt.detach() / self.mul_factor
            pred_inv = 1.0 / (pred + 1e-8)
            gt_inv = 1.0 / (gt + 1e-8)

            # For numerical stability
            mask = gt > self.t_valid
            # num_valid = mask.sum()
            B = mask.size(0)
            num_valid = torch.sum(mask.view(B, -1), -1, keepdim=True)

            # pred = pred[mask]
            # gt = gt[mask]
            pred = pred * mask
            gt = gt * mask

            # pred_inv = pred_inv[mask]
            # gt_inv = gt_inv[mask]
            pred_inv = pred_inv * mask
            gt_inv = gt_inv * mask

            # pred_inv[pred <= self.t_valid] = 0.0
            # gt_inv[gt <= self.t_valid] = 0.0

            # RMSE / MAE
            diff = pred - gt
            diff_abs = torch.abs(diff)
            diff_sqr = torch.pow(diff, 2)

            rmse = torch.sum(diff_sqr.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            rmse = torch.sqrt(rmse)

            mae = torch.sum(diff_abs.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)

            # iRMSE / iMAE
            diff_inv = pred_inv - gt_inv
            diff_inv_abs = torch.abs(diff_inv)
            diff_inv_sqr = torch.pow(diff_inv, 2)

            irmse = torch.sum(diff_inv_sqr.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            irmse = torch.sqrt(irmse)

            imae = torch.sum(diff_inv_abs.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)

            # Rel
            rel = diff_abs / (gt + 1e-8)
            rel = torch.sum(rel.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)

            # delta
            r1 = gt / (pred + 1e-8)
            r2 = pred / (gt + 1e-8)
            ratio = torch.max(r1, r2)

            ratio = torch.max(ratio, 10000 * (1 - mask.float()))

            del_1 = (ratio < 1.25).type_as(ratio)
            del_2 = (ratio < 1.25 ** 2).type_as(ratio)
            del_3 = (ratio < 1.25 ** 3).type_as(ratio)
            del_102 = (ratio < 1.02).type_as(ratio)
            del_105 = (ratio < 1.05).type_as(ratio)
            del_110 = (ratio < 1.10).type_as(ratio)

            del_1 = torch.sum(del_1.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            del_2 = torch.sum(del_2.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            del_3 = torch.sum(del_3.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            del_102 = torch.sum(del_102.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            del_105 = torch.sum(del_105.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            del_110 = torch.sum(del_110.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)

            result = [rmse, mae, irmse, imae, rel, del_1, del_2, del_3, del_102, del_105, del_110]

        return result
