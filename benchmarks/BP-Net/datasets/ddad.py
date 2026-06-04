import random
import os
import numpy as np
from glob import glob
from PIL import Image
import torch
import torch.utils.data as data
import os.path as osp
import cv2
import augs

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def read_depth_png(path):
    """
    和你现有 MYDDADDataset / KITTI 风格一致:
    16-bit png 深度图 -> float32 深度, 单位通常是米
    """
    depth_png = np.array(Image.open(path), dtype=np.int32)
    assert np.max(depth_png) > 255, f"Depth png seems not 16-bit: {path}"
    depth = (depth_png / 256.).astype(np.float32)
    return depth


class DDAD(torch.utils.data.Dataset):
    """
    按 KITTI2 的返回/预处理风格包装 MYDDAD 数据

    返回:
        rgb   : float32, [3, H, W]
        lidar : float32, [1, H, W]
        K_cam : float32, [3, 3]
        depth : float32, [1, H, W]
    """

    def __init__(
        self,
        path='datas/ddad',
        mode='train',
        height=1216,
        width=1920,
        mean=(90.9950, 96.2278, 94.3213),
        std=(79.2382, 80.5267, 82.1483), 
        RandCrop=False,
        tp_min=50,
        train_sample_size=-1,

        lidar_noise_prob=0.35,
        lidar_noise_std=0.1,
        lidar_noise_clip=(0.0, 200.0),
        lidar_dropout_prob=0.3,
        lidar_dropout_ratio=0.3,
        *args, **kwargs
    ):
        self.base_dir = os.path.join(path, mode)
        self.mode = mode
        self.height = height
        self.width = width

        self.lidar_noise_prob = lidar_noise_prob
        self.lidar_noise_std = lidar_noise_std
        self.lidar_noise_clip = lidar_noise_clip
        self.lidar_dropout_prob = lidar_dropout_prob
        self.lidar_dropout_ratio = lidar_dropout_ratio

        self.mean = mean
        self.std = std

        # 尽量复用 KITTI2 的 transform 逻辑
        if mode == 'train':
            self.transform = augs.Compose([
                augs.Norm(mean=mean, std=std),
            ])
        else:
            self.transform = augs.Compose([
                augs.Norm(mean=mean, std=std),
            ])

        self.RandCrop = RandCrop and mode == 'train'
        self.tp_min = tp_min

        # 数据组织方式参考你原来的 MYDDADDataset
        self.image_path = os.path.join(self.base_dir, 'rgb')
        self.depth_path = sorted(glob(osp.join(self.base_dir, '*gt', '*.png')))
        self.lidar_path = sorted(glob(osp.join(self.base_dir, 'hints', '*.png')))
        self.images = sorted(glob(osp.join(self.base_dir, 'rgb', '*.png')))
        self.calibs = sorted(glob(osp.join(self.base_dir, 'intrinsics', '*.txt')))

        assert len(self.images) == len(self.depth_path) == len(self.lidar_path) == len(self.calibs), \
            f"Mismatch: rgb={len(self.images)}, gt={len(self.depth_path)}, hints={len(self.lidar_path)}, intrinsics={len(self.calibs)}"

        if mode == 'train' and train_sample_size > 0:
            self.images = self.images[:train_sample_size]
            self.depth_path = self.depth_path[:train_sample_size]
            self.lidar_path = self.lidar_path[:train_sample_size]
            self.calibs = self.calibs[:train_sample_size]

        self.names = [os.path.split(path)[-1] for path in self.images]

        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)
        self.xy = np.stack((xx, yy), axis=-1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.get_item(index)


    def get_item(self, index):
        # 1) 读取深度 / sparse depth / K / RGB
        depth = self.pull_DEPTH(self.depth_path[index])     # H x W
        lidar = self.pull_DEPTH(self.lidar_path[index])     # H x W
        K_cam = self.pull_K_cam(index).astype(np.float32)   # 3 x 3
        rgb = self.pull_RGB(self.images[index])             # H x W x 3

        # 3) 扩 channel，和 KITTI2 保持一致
        depth = np.expand_dims(depth, axis=2)   # H x W x 1
        lidar = np.expand_dims(lidar, axis=2)   # H x W x 1

        rgb = rgb.astype(np.float32)
        lidar = lidar.astype(np.float32)
        depth = depth.astype(np.float32)
        # import IPython
        # IPython.embed()
        # exit()

        # 4) 复用 KITTI2 的 transform 流程
        if self.transform:
            rgb, lidar, depth, K_cam = self.transform(rgb, lidar, depth, K_cam)


        # 6) HWC -> CHW
        rgb = rgb.transpose(2, 0, 1).astype(np.float32)
        lidar = lidar.transpose(2, 0, 1).astype(np.float32)
        depth = depth.transpose(2, 0, 1).astype(np.float32)

        # # 7) 复用 KITTI2 的 crop 逻辑
        # tp = rgb.shape[1] - self.height
        # lp = (rgb.shape[2] - self.width) // 2

        # if tp < 0 or lp < 0:
        #     raise ValueError(
        #         f"Input image too small for crop: rgb.shape={rgb.shape}, target=({self.height}, {self.width})"
        #     )

        # if self.RandCrop:
        #     tp = random.randint(self.tp_min, tp)
        #     lp = random.randint(0, rgb.shape[2] - self.width)

        # rgb = rgb[:, tp:tp + self.height, lp:lp + self.width]
        # lidar = lidar[:, tp:tp + self.height, lp:lp + self.width]
        # depth = depth[:, tp:tp + self.height, lp:lp + self.width]

        # # 8) 裁剪后修正主点
        # K_cam[0, 2] -= lp
        # K_cam[1, 2] -= tp
        # 7) 基于 GT depth 有效区域 crop
        _, H, W = rgb.shape
        crop_h, crop_w = self.height, self.width

        if H < crop_h or W < crop_w:
            raise ValueError(
                f"Input image too small for crop: rgb.shape={rgb.shape}, target=({crop_h}, {crop_w})"
            )

        valid = depth[0] > 0  # H x W

        if np.any(valid):
            ys, xs = np.where(valid)
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()

            box_h = y_max - y_min + 1
            box_w = x_max - x_min + 1

            # ---- vertical crop ----
            if box_h >= crop_h:
                # bbox 比目标高：在能覆盖/穿过 bbox 的范围内采样
                y_low = max(0, y_max - crop_h + 1)
                y_high = min(y_min, H - crop_h)

                if y_low <= y_high:
                    tp = random.randint(y_low, y_high) if self.RandCrop else (y_low + y_high) // 2
                else:
                    # bbox 太大，无法完整覆盖：围绕 bbox 中心裁
                    center_y = (y_min + y_max) // 2
                    tp = center_y - crop_h // 2
            else:
                # bbox 比目标小：扩展到 crop_h
                extra = crop_h - box_h
                if self.RandCrop:
                    top_extra = random.randint(0, extra)
                else:
                    top_extra = extra // 2
                tp = y_min - top_extra

            # ---- horizontal crop ----
            if box_w >= crop_w:
                x_low = max(0, x_max - crop_w + 1)
                x_high = min(x_min, W - crop_w)

                if x_low <= x_high:
                    lp = random.randint(x_low, x_high) if self.RandCrop else (x_low + x_high) // 2
                else:
                    center_x = (x_min + x_max) // 2
                    lp = center_x - crop_w // 2
            else:
                extra = crop_w - box_w
                if self.RandCrop:
                    left_extra = random.randint(0, extra)
                else:
                    left_extra = extra // 2
                lp = x_min - left_extra

            # clamp 到图像范围内
            tp = int(np.clip(tp, 0, H - crop_h))
            lp = int(np.clip(lp, 0, W - crop_w))

        else:
            # 没有 GT depth 有效点时，退回原来的 crop 逻辑
            tp = H - crop_h
            lp = (W - crop_w) // 2

            if self.RandCrop:
                tp_max = H - crop_h
                tp_min = min(self.tp_min, tp_max)
                tp = random.randint(tp_min, tp_max)
                lp = random.randint(0, W - crop_w)

        rgb = rgb[:, tp:tp + crop_h, lp:lp + crop_w]
        lidar = lidar[:, tp:tp + crop_h, lp:lp + crop_w]
        depth = depth[:, tp:tp + crop_h, lp:lp + crop_w]

        # 8) 裁剪后修正主点
        K_cam[0, 2] -= lp
        K_cam[1, 2] -= tp
        return rgb, lidar, K_cam, depth

    def pull_RGB(self, path):
        img = np.array(Image.open(path).convert('RGB'), dtype=np.uint8)
        return img

    def pull_DEPTH(self, path):
        return read_depth_png(path)

    def pull_K_cam(self, index):
        calib_path = self.calibs[index]
        K_cam = np.loadtxt(calib_path).astype(np.float32).reshape(3, 3)
        return K_cam