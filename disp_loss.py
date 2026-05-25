import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_scale_and_shift_per_patch(pred_patch, tgt_patch, mask_patch, eps=1e-6):
    """
    pred_patch, tgt_patch, mask_patch: [N, P, P]
    mask_patch: float/bool
    return:
        scale: [N]
        shift: [N]
        valid: [N]  # determinant valid
    """
    mask_patch = mask_patch.float()

    a_00 = torch.sum(mask_patch * pred_patch * pred_patch, dim=(1, 2))
    a_01 = torch.sum(mask_patch * pred_patch, dim=(1, 2))
    a_11 = torch.sum(mask_patch, dim=(1, 2))

    b_0 = torch.sum(mask_patch * pred_patch * tgt_patch, dim=(1, 2))
    b_1 = torch.sum(mask_patch * tgt_patch, dim=(1, 2))

    det = a_00 * a_11 - a_01 * a_01
    valid = det.abs() > eps

    scale = torch.zeros_like(b_0)
    shift = torch.zeros_like(b_1)

    scale[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    shift[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return scale, shift, valid


def masked_patch_mse(pred_patch, tgt_patch, mask_patch, eps=1e-6):
    """
    pred_patch, tgt_patch, mask_patch: [N, P, P]
    return: [N]
    """
    mask_patch = mask_patch.float()
    denom = mask_patch.sum(dim=(1, 2)).clamp(min=eps)
    loss = ((pred_patch - tgt_patch) ** 2 * mask_patch).sum(dim=(1, 2)) / denom
    return loss


def masked_patch_gradient_loss(pred_patch, tgt_patch, mask_patch, eps=1e-6):
    """
    pred_patch, tgt_patch, mask_patch: [N, P, P]
    return: [N]
    """
    mask_patch = mask_patch.float()
    diff = (pred_patch - tgt_patch) * mask_patch

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = mask_patch[:, :, 1:] * mask_patch[:, :, :-1]
    grad_x = grad_x * mask_x

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = mask_patch[:, 1:, :] * mask_patch[:, :-1, :]
    grad_y = grad_y * mask_y

    denom_x = mask_x.sum(dim=(1, 2))
    denom_y = mask_y.sum(dim=(1, 2))
    denom = (denom_x + denom_y).clamp(min=eps)

    loss = (grad_x.sum(dim=(1, 2)) + grad_y.sum(dim=(1, 2))) / denom
    return loss


class PatchScaleAndShiftInvariantLoss(nn.Module):
    """
    Patch-based scale-and-shift invariant loss.

    Args:
        patch_size: int or list/tuple[int]
            单个整数表示只用一种 patch 大小；
            list/tuple 表示多种 patch 大小一起算，再平均。
        stride: int or None
            如果是 None，则默认 stride = patch_size // 2（即 50% overlap）
        alpha: gradient loss 权重
        min_valid_pixels: patch 内至少多少个有效像素才参与
        detach_scale_shift: 是否对 scale/shift detach
        eps: 数值稳定项
    """
    def __init__(
        self,
        patch_size=32,
        stride=None,
        alpha=0.5,
        min_valid_pixels=32,
        detach_scale_shift=True,
        eps=1e-6,
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = [patch_size]
        self.patch_sizes = list(patch_size)
        self.stride = stride
        self.alpha = alpha
        self.min_valid_pixels = min_valid_pixels
        self.detach_scale_shift = detach_scale_shift
        self.eps = eps

        self._prediction_ssi = None  # optional debug

    def _extract_patches(self, x, patch_size, stride):
        """
        x: [B, H, W]
        return: [B * L, P, P], L = num_patches_per_image
        """
        patches = F.unfold(
            x.unsqueeze(1),  # [B,1,H,W]
            kernel_size=patch_size,
            stride=stride,
        )  # [B, P*P, L]
        B, PP, L = patches.shape
        patches = patches.transpose(1, 2).reshape(B * L, patch_size, patch_size)
        return patches, L

    def _single_patch_size_loss(self, prediction, target, mask, patch_size):
        """
        prediction, target, mask: [B, H, W]
        """
        stride = self.stride if self.stride is not None else max(1, patch_size // 2)

        pred_patches, num_patches_per_image = self._extract_patches(prediction, patch_size, stride)
        tgt_patches, _ = self._extract_patches(target, patch_size, stride)
        mask_patches, _ = self._extract_patches(mask.float(), patch_size, stride)

        valid_pixels = mask_patches.sum(dim=(1, 2))
        enough_valid = valid_pixels >= self.min_valid_pixels

        scale, shift, det_valid = compute_scale_and_shift_per_patch(
            pred_patches, tgt_patches, mask_patches, eps=self.eps
        )

        patch_valid = enough_valid & det_valid
        if patch_valid.sum() == 0:
            return prediction.new_tensor(0.0)

        if self.detach_scale_shift:
            scale = scale.detach()
            shift = shift.detach()

        pred_aligned = scale[:, None, None] * pred_patches + shift[:, None, None]

        data_loss = masked_patch_mse(pred_aligned, tgt_patches, mask_patches, eps=self.eps)

        total_patch_loss = data_loss
        if self.alpha > 0:
            reg_loss = masked_patch_gradient_loss(pred_aligned, tgt_patches, mask_patches, eps=self.eps)
            total_patch_loss = total_patch_loss + self.alpha * reg_loss

        total_patch_loss = total_patch_loss[patch_valid].mean()
        return total_patch_loss

    def forward(self, prediction, target, mask):
        """
        prediction, target, mask: [B, H, W]
        """
        losses = []
        for p in self.patch_sizes:
            losses.append(self._single_patch_size_loss(prediction, target, mask, p))

        total = torch.stack(losses).mean()

        # 这里只保留原图预测，方便你 debug；真正 patch 对齐后的整图不好唯一重建
        self._prediction_ssi = prediction
        return total

    @property
    def prediction_ssi(self):
        return self._prediction_ssi

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))

    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_relative_inverse_depth(depth, eps=1e-6, clamp_max=None):
    """
    对 monocular prior 用 relative inverse depth，更贴近 SparseNeRF 的做法。
    depth: [B,H,W]
    """
    inv = 1.0 / depth.clamp(min=eps)
    if clamp_max is not None:
        inv = inv.clamp(max=clamp_max)

    # per-image normalize to [0, 1] on valid area later by caller if needed
    B = inv.shape[0]
    inv_flat = inv.view(B, -1)
    inv_min = inv_flat.min(dim=1)[0].view(B, 1, 1)
    inv_max = inv_flat.max(dim=1)[0].view(B, 1, 1)
    inv = (inv - inv_min) / (inv_max - inv_min + eps)
    return inv


class PatchLocalRankingLoss_fast(nn.Module):
    """
    SparseNeRF-style local depth ranking:
      R_rank = sum max(d_pred(k1) - d_pred(k2) + m, 0)
      for pairs where d_prior(k1) <= d_prior(k2)

    prior 默认使用 relative inverse depth 排序。
    与原版接口保持一致，但实现做了向量化加速。
    """

    def __init__(
        self,
        patch_size=32,
        stride=16,
        samples_per_patch=64,
        min_valid_pixels=24,
        margin=1e-4,
        min_prior_diff=0.02,
        use_inverse_depth=True,
        only_missing_gt=True,
        top_mask_ratio=0.18,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.samples_per_patch = samples_per_patch
        self.min_valid_pixels = min_valid_pixels
        self.margin = margin
        self.min_prior_diff = min_prior_diff
        self.use_inverse_depth = use_inverse_depth
        self.only_missing_gt = only_missing_gt
        self.top_mask_ratio = top_mask_ratio

    def forward(self, pred, prior, gt=None):
        """
        pred:  [B,H,W]  网络输出 depth
        prior: [B,H,W]  Depth Anything v3 depth
        gt:    [B,H,W] or None
        """
        assert pred.ndim == 3 and prior.ndim == 3
        assert pred.shape == prior.shape

        B, H, W = pred.shape
        device = pred.device
        P = self.patch_size
        S = self.stride

        if H < P or W < P:
            return pred.new_tensor(0.0)

        # 1) prior 排序值
        prior_rank = prior
        if self.use_inverse_depth:
            prior_rank = _to_relative_inverse_depth(prior)

        # 2) valid mask
        valid = prior > 1e-3

        # 去掉顶部天空区域
        top = int(self.top_mask_ratio * H)
        if top > 0:
            valid = valid.clone()
            valid[:, :top, :] = False

        # 只在 GT 缺失区域使用 prior
        if gt is not None and self.only_missing_gt:
            valid = valid & (gt <= 1e-3)

        # ------------------------------------------------------------
        # 3) unfold 提取所有 patch
        #    pred_p/prior_p/valid_p: [B, nH, nW, P, P]
        # ------------------------------------------------------------
        pred_p = pred.unfold(1, P, S).unfold(2, P, S)
        prior_p = prior_rank.unfold(1, P, S).unfold(2, P, S)
        valid_p = valid.unfold(1, P, S).unfold(2, P, S)

        nH = pred_p.shape[1]
        nW = pred_p.shape[2]
        num_patches = B * nH * nW
        patch_area = P * P

        # [num_patches, patch_area]
        pred_p = pred_p.contiguous().view(num_patches, patch_area)
        prior_p = prior_p.contiguous().view(num_patches, patch_area)
        valid_p = valid_p.contiguous().view(num_patches, patch_area)

        # 4) 过滤有效像素太少的 patch
        valid_counts = valid_p.sum(dim=1)
        patch_keep = valid_counts >= self.min_valid_pixels
        if not patch_keep.any():
            return pred.new_tensor(0.0)

        pred_p = pred_p[patch_keep]     # [M, A]
        prior_p = prior_p[patch_keep]   # [M, A]
        valid_p = valid_p[patch_keep]   # [M, A]
        M, A = pred_p.shape

        # 5) 每个 patch 随机采样 K 对位置
        #    与原版一样是随机 pair，只是现在批量做
        K = self.samples_per_patch
        if K <= 0:
            return pred.new_tensor(0.0)

        # [M, K]
        id1 = torch.randint(0, A, (M, K), device=device)
        id2 = torch.randint(0, A, (M, K), device=device)

        # 去掉相同位置 pair
        keep = id1 != id2

        # 两端都必须 valid
        v1 = torch.gather(valid_p, 1, id1)
        v2 = torch.gather(valid_p, 1, id2)
        keep = keep & v1 & v2

        # prior 值
        prior_1 = torch.gather(prior_p, 1, id1)
        prior_2 = torch.gather(prior_p, 1, id2)

        # 只保留 prior 顺序差得比较明显的 pair
        keep = keep & ((prior_2 - prior_1).abs() >= self.min_prior_diff)

        if not keep.any():
            return pred.new_tensor(0.0)

        # 统一成 prior_1 <= prior_2
        swap = prior_1 > prior_2
        idx_small = torch.where(swap, id2, id1)
        idx_large = torch.where(swap, id1, id2)

        pred_1 = torch.gather(pred_p, 1, idx_small)
        pred_2 = torch.gather(pred_p, 1, idx_large)

        # SparseNeRF hinge ranking
        loss = F.relu(pred_1 - pred_2 + self.margin)

        # 只统计有效 pair
        keep_f = keep.to(loss.dtype)
        loss_sum = (loss * keep_f).sum()
        pair_count = keep_f.sum()

        if pair_count <= 0:
            return pred.new_tensor(0.0)

        return loss_sum / pair_count
    


class PatchContinuityLoss(nn.Module):
    """
    SparseNeRF-style spatial continuity distillation:
      如果 prior 中相近深度像素是连续的，则预测也应连续。

    实现上：每个 patch 内，对每个有效点按 prior 深度差找 k 个邻居，
    然后约束 |pred_i - pred_j| 不要超过 m_conti。
    """
    def __init__(
        self,
        patch_size=32,
        stride=16,
        min_valid_pixels=24,
        k_neighbors=4,
        neighbor_pool=6,
        margin_conti=1e-4,
        prior_neighbor_depth_diff=0.03,
        use_inverse_depth=True,
        only_missing_gt=True,
        top_mask_ratio=0.18,
        max_points_per_patch=96,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.min_valid_pixels = min_valid_pixels
        self.k_neighbors = k_neighbors
        self.neighbor_pool = neighbor_pool
        self.margin_conti = margin_conti
        self.prior_neighbor_depth_diff = prior_neighbor_depth_diff
        self.use_inverse_depth = use_inverse_depth
        self.only_missing_gt = only_missing_gt
        self.top_mask_ratio = top_mask_ratio
        self.max_points_per_patch = max_points_per_patch

    def forward(self, pred, prior, gt=None):
        """
        pred:  [B,H,W]
        prior: [B,H,W]
        gt:    [B,H,W] or None
        """
        B, H, W = pred.shape
        device = pred.device

        prior_rank = prior
        if self.use_inverse_depth:
            prior_rank = _to_relative_inverse_depth(prior)

        losses = []

        for b in range(B):
            valid = prior[b] > 1e-3
            top = int(self.top_mask_ratio * H)
            if top > 0:
                valid[:top, :] = False

            if gt is not None and self.only_missing_gt:
                valid = valid & (gt[b] <= 1e-3)

            for y in range(0, H - self.patch_size + 1, self.stride):
                for x in range(0, W - self.patch_size + 1, self.stride):
                    v_patch = valid[y:y+self.patch_size, x:x+self.patch_size]
                    if int(v_patch.sum()) < self.min_valid_pixels:
                        continue

                    pred_patch = pred[b, y:y+self.patch_size, x:x+self.patch_size]
                    prior_patch = prior_rank[b, y:y+self.patch_size, x:x+self.patch_size]

                    idx = torch.nonzero(v_patch, as_tuple=False)  # [N,2]
                    N = idx.shape[0]
                    if N < 2:
                        continue

                    # 为了省算力，随机下采样一些点
                    if N > self.max_points_per_patch:
                        perm = torch.randperm(N, device=device)[:self.max_points_per_patch]
                        idx = idx[perm]
                        N = idx.shape[0]

                    rows = idx[:, 0]
                    cols = idx[:, 1]

                    prior_vals = prior_patch[rows, cols]  # [N]
                    pred_vals = pred_patch[rows, cols]    # [N]

                    # 只在一个更小的局部邻域里找邻居，更接近 SparseNeRF 小区域 KNN 的意思
                    coord = torch.stack([rows.float(), cols.float()], dim=1)  # [N,2]
                    spatial_dist = torch.cdist(coord, coord, p=2)             # [N,N]

                    # 限制在局部邻域
                    local_mask = spatial_dist <= float(self.neighbor_pool)

                    depth_dist = (prior_vals[:, None] - prior_vals[None, :]).abs()  # [N,N]
                    depth_dist = depth_dist.masked_fill(~local_mask, 1e6)
                    depth_dist.fill_diagonal_(1e6)

                    knn_dist, knn_idx = torch.topk(
                        depth_dist, k=min(self.k_neighbors, N - 1), dim=1, largest=False
                    )

                    # 只保留 prior 上足够连续的邻居
                    valid_pair = knn_dist <= self.prior_neighbor_depth_diff
                    if valid_pair.sum() == 0:
                        continue

                    nbr_pred = pred_vals[knn_idx]  # [N,K]
                    center_pred = pred_vals[:, None].expand_as(nbr_pred)

                    conti = F.relu((center_pred - nbr_pred).abs() - self.margin_conti)
                    conti = conti[valid_pair]
                    if conti.numel() == 0:
                        continue

                    losses.append(conti.mean())

        if len(losses) == 0:
            return pred.new_tensor(0.0)

        return torch.stack(losses).mean()


class MSMSE_ranking_fast(nn.Module):
    """
    用于 KITTI depth completion 的 SparseNeRF 风格损失：
      total = multi-scale sparse GT MSE
            + multi-scale local ranking
            + multi-scale continuity
    """
    def __init__(
        self,
        deltas=(2 ** (-5 * 2), 2 ** (-4 * 2), 2 ** (-3 * 2), 2 ** (-2 * 2), 2 ** (-1 * 2), 1.0),

        lambda_gt=1.0,
        lambda_rank=0.05,
        lambda_conti=0.00,

        patch_size=96,
        patch_stride=48,

        rank_samples_per_patch=64,
        rank_min_valid_pixels=32,
        rank_margin=1e-4,
        rank_min_prior_diff=0.02,

        conti_min_valid_pixels=48,
        conti_k_neighbors=6,
        conti_neighbor_pool=10,
        conti_margin=1e-4,
        conti_prior_neighbor_depth_diff=0.03,
        conti_max_points_per_patch=128,

        top_mask_ratio=0.0,
        only_missing_gt_for_prior=True,
        use_inverse_depth_for_prior=True,
        
    ):
        super().__init__()
        self.deltas = deltas

        self.lambda_gt = lambda_gt
        self.lambda_rank = lambda_rank
        self.lambda_conti = lambda_conti

        self.rank_loss = PatchLocalRankingLoss_fast(
            patch_size=patch_size,
            stride=patch_stride,
            samples_per_patch=rank_samples_per_patch,
            min_valid_pixels=rank_min_valid_pixels,
            margin=rank_margin,
            min_prior_diff=rank_min_prior_diff,
            use_inverse_depth=use_inverse_depth_for_prior,
            only_missing_gt=only_missing_gt_for_prior,
            top_mask_ratio=top_mask_ratio,
        )

        self.conti_loss = PatchContinuityLoss(
            patch_size=patch_size,
            stride=patch_stride,
            min_valid_pixels=conti_min_valid_pixels,
            k_neighbors=conti_k_neighbors,
            neighbor_pool=conti_neighbor_pool,
            margin_conti=conti_margin,
            prior_neighbor_depth_diff=conti_prior_neighbor_depth_diff,
            use_inverse_depth=use_inverse_depth_for_prior,
            only_missing_gt=only_missing_gt_for_prior,
            top_mask_ratio=top_mask_ratio,
            max_points_per_patch=conti_max_points_per_patch,
        )

    def masked_mse(self, pred, target, mask):
        mask = mask.float()
        denom = mask.sum().clamp(min=1.0)
        return ((pred - target) ** 2 * mask).sum() / denom

    def single_scale_loss(self, est, gt, metric_depth):
        """
        est, gt, metric_depth: [B,1,H,W]
        """
        gt_valid = gt > 1e-3

        loss_gt = self.masked_mse(est, gt, gt_valid)
        loss_rank = self.rank_loss(est[:, 0], metric_depth[:, 0], gt[:, 0])
        if self.lambda_conti > 0:
            loss_conti = self.conti_loss(est[:, 0], metric_depth[:, 0], gt[:, 0])
        else:
            loss_conti = torch.tensor(0.0).to(gt.device)

        stats = {
            "loss_gt": loss_gt.detach(),
            "loss_rank": loss_rank.detach(),
            "loss_conti": loss_conti.detach(),
            "gt_pixels": gt_valid.float().sum().detach(),
        }

        return (
                self.lambda_gt * loss_gt,
                self.lambda_rank * loss_rank,
                self.lambda_conti * loss_conti,
                stats,
            )

    def forward(self, outputs, target, metric_depth, return_dis=True):
        total_loss = 0.0
        all_stats = []

        for est, delta in zip(outputs, self.deltas):
            size = est.shape[-2:]

            gt_scaled = F.interpolate(target, size=size, mode="nearest")
            metric_scaled = F.interpolate(metric_depth, size=size, mode="bilinear", align_corners=False)

            loss_gt, loss_rank, loss_conti, stats_i = self.single_scale_loss(
                est, gt_scaled, metric_scaled
            )
            if return_dis:
                total_loss = total_loss + delta * (loss_gt + loss_rank + loss_conti)
            else:
                total_loss = total_loss + delta * loss_gt
            all_stats.append(stats_i)

        return total_loss, all_stats