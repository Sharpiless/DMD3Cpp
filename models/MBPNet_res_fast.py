# -*- coding: utf-8 -*-
# @File : BPNet.py
# @Project: BP-Net
# @Author : jie
# @Time : 4/8/23 12:43 PM
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from .utils import Conv1x1, Basic2d, BasicBlock, weights_init, inplace_relu, GenKernel, PMP_residual
from .input_processor import InputProcessor
from .output_processor import OutputProcessor

__all__ = [
    'Pre_MF_Post_Residual_Norm_fast'
]


def weighted_affine_fit_1d(x, y, w=None, eps=1e-6):
    """
    解 y ≈ a*x + b 的加权最小二乘闭式解
    x, y, w: shape [N]
    return: a, b
    """
    if w is None:
        w = torch.ones_like(x)

    w_sum = w.sum().clamp_min(eps)

    x_bar = (w * x).sum() / w_sum
    y_bar = (w * y).sum() / w_sum

    xc = x - x_bar
    yc = y - y_bar

    var_x = (w * xc * xc).sum()
    cov_xy = (w * xc * yc).sum()

    if var_x.abs() < eps or not torch.isfinite(var_x):
        a = x.new_tensor(1.0)
        b = x.new_tensor(0.0)
    else:
        a = cov_xy / (var_x + eps)
        b = y_bar - a * x_bar

    if not torch.isfinite(a):
        a = x.new_tensor(1.0)
    if not torch.isfinite(b):
        b = x.new_tensor(0.0)

    return a, b


def robust_align_metric_to_sparse(
    metric_depth,
    sparse_depth,
    valid_thresh=1e-3,
    min_valid=32,
    num_iters=5,
    huber_delta=1.0,
    clamp_min=0.0,
    clamp_max=None,
):
    """
    metric_depth: Bx1xHxW
    sparse_depth: Bx1xHxW

    return:
        aligned: Bx1xHxW
        scales: B
        shifts: B
        valid_mask: Bx1xHxW
    """
    B = metric_depth.shape[0]
    valid_mask = sparse_depth > valid_thresh

    aligned_list = []
    scale_list = []
    shift_list = []

    for b in range(B):
        mask = valid_mask[b, 0]   # HxW
        pred = metric_depth[b, 0][mask]   # N
        target = sparse_depth[b, 0][mask] # N

        if pred.numel() < min_valid:
            a = metric_depth.new_tensor(1.0)
            c = metric_depth.new_tensor(0.0)
        else:
            # 初始：普通最小二乘
            a, c = weighted_affine_fit_1d(pred, target)

            # IRLS: Huber-like
            for _ in range(num_iters):
                residual = a * pred + c - target
                abs_r = residual.abs()

                weights = torch.ones_like(abs_r)
                large = abs_r > huber_delta
                weights[large] = huber_delta / (abs_r[large] + 1e-6)

                a, c = weighted_affine_fit_1d(pred, target, weights)

        aligned_b = a * metric_depth[b:b+1] + c

        if clamp_max is None:
            aligned_b = torch.clamp(aligned_b, min=clamp_min)
        else:
            aligned_b = torch.clamp(aligned_b, min=clamp_min, max=clamp_max)

        aligned_list.append(aligned_b)
        scale_list.append(a)
        shift_list.append(c)

    aligned = torch.cat(aligned_list, dim=0)
    scales = torch.stack(scale_list, dim=0)
    shifts = torch.stack(shift_list, dim=0)

    return aligned, scales, shifts, valid_mask


class Net(nn.Module):
    """
    network
    """

    def __init__(self, block=BasicBlock, bc=16, img_layers=[2, 2, 2, 2, 2, 2],
                 drop_path=0.1, norm_layer=nn.BatchNorm2d, padding_mode='zeros', drift=1e6, normalize_mono=False):
        super().__init__()
        self.drift = drift
        self._norm_layer = norm_layer
        self._padding_mode = padding_mode
        self._img_dpc = 0
        self._img_dprs = np.linspace(0, drop_path, sum(img_layers))

        self.inplanes = bc * 2
        self.conv_img = nn.Sequential(
            Basic2d(3, bc * 2, norm_layer=norm_layer, kernel_size=3, padding=1),
            self._make_layer(block, bc * 2, 2, stride=1)
        )

        self.layer1_img = self._make_layer(block, bc * 4, img_layers[1], stride=2)

        self.inplanes = bc * 4
        self.layer2_img = self._make_layer(block, bc * 8, img_layers[2], stride=2)

        self.inplanes = bc * 8
        self.layer3_img = self._make_layer(block, bc * 16, img_layers[3], stride=2)

        self.inplanes = bc * 16
        self.layer4_img = self._make_layer(block, bc * 16, img_layers[4], stride=2)

        self.inplanes = bc * 16
        self.layer5_img = self._make_layer(block, bc * 16, img_layers[5], stride=2)

        self.pred0 = PMP_residual(level=0, in_ch=bc * 4, out_ch=bc * 2, drop_path=drop_path, pool=False)
        self.pred1 = PMP_residual(level=1, in_ch=bc * 8, out_ch=bc * 4, drop_path=drop_path)
        self.pred2 = PMP_residual(level=2, in_ch=bc * 16, out_ch=bc * 8, drop_path=drop_path)
        self.pred3 = PMP_residual(level=3, in_ch=bc * 16, out_ch=bc * 16, drop_path=drop_path)
        self.pred4 = PMP_residual(level=4, in_ch=bc * 16, out_ch=bc * 16, drop_path=drop_path)
        self.pred5 = PMP_residual(level=5, in_ch=bc * 16, out_ch=bc * 16, drop_path=drop_path, up=False)

        from depth_anything_3.api import DepthAnything3
        print("[MONO]:", "depth-anything/DA3METRIC-LARGE")
        depth_anything = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
        # depth_anything = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE-1.1")
        # depth_anything = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
        depth_anything.requires_grad_(False)
        self.depth_anything = [depth_anything]
        self.da3_input_processor = InputProcessor()
        self.da3_output_processor = OutputProcessor()
        self.normalize_mono = normalize_mono

    def _convert_to_prediction(self, raw_output):
        """Convert raw model output to Prediction object."""
        output = self.output_processor(raw_output)
        return output
        
    def forward(self, I, S, K,
        process_res: int = 1344,
        # process_res: int = 504*2,
        process_res_method: str = "upper_bound_resize",
        export_feat_layers = [],
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        return_da3 = False):
        """
        I: Bx3xHxW
        S: Bx1xHxW
        K: Bx3x3
        """
        imgs_da3_cpu, _, in_t = self.da3_input_processor(
            I,
            None,
            K.clone() if K is not None else None,
            process_res,
            process_res_method,
        )
        imgs = imgs_da3_cpu[:, None]
        in_t = in_t[:, None, None]

        with torch.no_grad():
            if self.depth_anything[0].device != I.device:
                self.depth_anything[0] = self.depth_anything[0].to(I.device)
            raw_output = self.depth_anything[0]._run_model_forward(
                    imgs.float(), None, in_t.float(),
                    export_feat_layers, infer_gs, 
                    use_ray_pose, ref_view_strategy
                )

        da3_output = self.da3_output_processor(raw_output)
        ####################################################################3
        layered_feats = [feat[:, 0].detach().clone() for feat in raw_output["feats_reshaped"]]
        assert len(layered_feats) == 4, f"Expected 4 layered_feats, got {len(layered_feats)}"
        # fused_feats = raw_output["fused"][:, 0].detach().clone()
        fused_feats = raw_output["fused"].detach().clone() # torch.Size([1, 128, 210, 1008])
        # depth head 特征 torch.Size([B, 128, 112, 504])
        del raw_output

        size = S.shape[-2:]
        metric_depth = torch.from_numpy(da3_output.depth[:, None]).to(S.device)
        metric_scaled = F.interpolate(metric_depth, size=size, mode="bilinear", align_corners=False)

        metric_aligned, metric_scale, metric_shift, sparse_valid = robust_align_metric_to_sparse(
            metric_scaled, S, valid_thresh=1e-3, min_valid=32
        )
        
        output = []
        XI0 = self.conv_img(I)
        XI1 = self.layer1_img(XI0) # torch.Size([B, 64, 128, 608])
        XI2 = self.layer2_img(XI1) # torch.Size([B, 128, 64, 304])
        XI3 = self.layer3_img(XI2) # torch.Size([B, 256, 32, 152])
        XI4 = self.layer4_img(XI3) # torch.Size([B, 256, 16, 76])
        XI5 = self.layer5_img(XI4) # torch.Size([B, 256, 8, 38])

        fout, dout = self.pred5(fout=None, dout=None, XI=XI5, S=S, K=K, metric_aligned=metric_aligned, normalize_mono=self.normalize_mono)
        # torch.Size([2, 256, 8, 38]), torch.Size([2, 1, 8, 38])
        output.append(F.interpolate(dout, scale_factor=2 ** 5, mode='bilinear', align_corners=True))

        fout, dout = self.pred4(fout=fout, dout=dout, XI=XI4, S=S, K=K, metric_aligned=metric_aligned, normalize_mono=self.normalize_mono)
        # torch.Size([2, 256, 16, 76]), torch.Size([2, 1, 16, 76])
        output.append(F.interpolate(dout, scale_factor=2 ** 4, mode='bilinear', align_corners=True))

        fout, dout = self.pred3(fout=fout, dout=dout, XI=XI3, S=S, K=K, metric_aligned=metric_aligned, normalize_mono=self.normalize_mono)
        # torch.Size([2, 256, 32, 152]), torch.Size([2, 1, 32, 152])
        output.append(F.interpolate(dout, scale_factor=2 ** 3, mode='bilinear', align_corners=True))

        fout, dout = self.pred2(fout=fout, dout=dout, XI=XI2, S=S, K=K, metric_aligned=metric_aligned, normalize_mono=self.normalize_mono)
        # torch.Size([2, 128, 64, 304]), torch.Size([2, 1, 64, 304])
        output.append(F.interpolate(dout, scale_factor=2 ** 2, mode='bilinear', align_corners=True))

        fout, dout = self.pred1(fout=fout, dout=dout, XI=XI1, S=S, K=K, metric_aligned=metric_aligned, normalize_mono=self.normalize_mono)
        # torch.Size([2, 64, 128, 608]), torch.Size([2, 1, 128, 608])
        output.append(F.interpolate(dout, scale_factor=2 ** 1, mode='bilinear', align_corners=True))

        fout, dout = self.pred0(fout=fout, dout=dout, XI=XI0, S=S, K=K, metric_aligned=metric_aligned, normalize_mono=self.normalize_mono)
        # torch.Size([2, 32, 256, 1216]), torch.Size([2, 1, 256, 1216])
        output.append(dout)

        # import IPython
        # IPython.embed()
        # exit()
        if return_da3:
            return output, da3_output
        return output

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        padding_mode = self._padding_mode
        downsample = None
        if norm_layer is None:
            bias = True
            norm_layer = nn.Identity
        else:
            bias = False
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv1x1(self.inplanes, planes * block.expansion, stride, bias=bias),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, norm_layer=norm_layer, padding_mode=padding_mode,
                  drop_path=self._img_dprs[self._img_dpc]))
        self._img_dpc += 1
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, padding_mode=padding_mode,
                                drop_path=self._img_dprs[self._img_dpc]))
            self._img_dpc += 1

        return nn.Sequential(*layers)


def Pre_MF_Post_Residual_Norm_fast():
    """
    Pre.+MF.+Post.
    """
    net = Net(normalize_mono=True)
    net.apply(functools.partial(weights_init, mode='trunc'))
    for m in net.modules():
        if isinstance(m, GenKernel) and m.conv[1].conv.bn.weight is not None:
            nn.init.constant_(m.conv[1].conv.bn.weight, 0)
    net.apply(inplace_relu)
    return net
