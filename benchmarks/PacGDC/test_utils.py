import torch
from src.data.data_tools import rgb_read, depth_read
import numpy as np


class DataReader_npy(object):
    def __init__(self, device, max_depth):
        super(DataReader_npy, self).__init__()
        self.device = device
        self.max_depth = max_depth

    def read_data(self, npy_path):
        data_dict = dict(np.load(npy_path, allow_pickle=True).item())

        rgb = torch.from_numpy(data_dict["rgb"]).permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255
        raw = torch.from_numpy(data_dict["dep"]).unsqueeze(0).unsqueeze(0).to(torch.float32)
        gt = torch.from_numpy(data_dict["gt"]).unsqueeze(0).unsqueeze(0).to(torch.float32)

        # original gt size
        _, _, H, W = gt.shape

        # Resize input to nearest 64 multiple
        new_H = (H + 63) // 64 * 64
        new_W = (W + 63) // 64 * 64

        rgb = torch.nn.functional.interpolate(
            rgb, size=(new_H, new_W), mode="bilinear", align_corners=True
        )
        raw = torch.nn.functional.interpolate(
            raw, size=(new_H, new_W), mode="nearest"
        ) / self.max_depth

        # gt 不 resize，只 normalize
        gt = gt / self.max_depth

        hole_raw = (raw > 0).float()

        return (
            rgb.to(self.device),
            raw.to(self.device),
            gt.to(self.device),
            hole_raw.to(self.device),
            (H, W),
        )

    def toint32(self, pred):
        pred = pred.squeeze().to("cpu").numpy()
        pred = np.clip(pred * 65535.0, 0, 65535).astype(np.int32)
        return pred

class DataReader(object):
    def __init__(self, device):
        super(DataReader, self).__init__()
        self.device = device

    def read_data(self, rgb_path, raw_path, gt_path):
        rgb = rgb_read(rgb_path).unsqueeze(0).to(torch.float32)
        raw = depth_read(raw_path).unsqueeze(0).to(torch.float32)
        gt = depth_read(gt_path).unsqueeze(0).to(torch.float32)

        # Resize to the nearest 64 multiple
        _, _, H, W = rgb.shape
        new_H = (H + 63) // 64 * 64
        new_W = (W + 63) // 64 * 64
        rgb = torch.nn.functional.interpolate(
            rgb, size=(new_H, new_W), mode="bilinear", align_corners=True
        )
        raw = torch.nn.functional.interpolate(raw, size=(new_H, new_W), mode="nearest")
        gt = torch.nn.functional.interpolate(gt, size=(new_H, new_W), mode="nearest")

        hole_raw = (raw > 0).float()
        return (
            rgb.to(self.device),
            raw.to(self.device),
            gt.to(self.device),
            hole_raw.to(self.device),
        )


def metrics(pred, gt):
    # valid pixels
    mask = gt > 0.0
    valid_nums = torch.sum(mask) + 1e-8
    pred = pred[mask]
    gt = gt[mask]

    diff = pred - gt
    # rmse
    rmse = torch.sqrt(torch.sum(diff**2) / valid_nums)
    # mae
    mae = torch.sum(torch.abs(diff)) / valid_nums
    return rmse, mae

def metricsv2(pred, gt):
    # valid pixels
    mask = gt > 0.0
    valid_nums = torch.sum(mask) + 1e-8

    pred = pred[mask]
    gt = gt[mask]

    diff = pred - gt

    # RMSE
    rmse = torch.sqrt(torch.sum(diff ** 2) / valid_nums)

    # MAE
    mae = torch.sum(torch.abs(diff)) / valid_nums

    # REL (mean absolute relative error)
    rel = torch.sum(torch.abs(diff) / (gt + 1e-8)) / valid_nums

    return rmse, mae, rel