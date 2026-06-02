import os
import glob
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

from dataloaders.paths_and_transform import *
from dataloaders.NNfill import *


class KittiDepth(data.Dataset):
    """Load data from our npy files, while keeping the original pipeline unchanged."""

    def __init__(self, split, args, data_folder):
        self.args = args
        self.split = split

        self.files = sorted(glob.glob(os.path.join(data_folder, '*.npy')))
        assert len(self.files) > 0, f'No npy files found in {data_folder}'

        self.transforms = kittitransforms
        self.ipfill = fill_in_fast

    def __getraw__(self, index):
        filedir = self.files[index]
        data_dict = np.load(filedir, allow_pickle=True).item()

        rgb = data_dict['rgb']
        dep = data_dict['dep'] if 'dep' in data_dict else data_dict['lidar']
        gt = data_dict['gt'] if 'gt' in data_dict else data_dict['depth']

        rgb = rgb.astype(np.uint8)
        dep = dep.astype(np.float32)
        gt = gt.astype(np.float32)

        if dep.ndim == 3:
            dep = dep[:, :, 0]
        if gt.ndim == 3:
            gt = gt[:, :, 0]

        rgb = Image.fromarray(rgb, mode='RGB')

        if 'ETH3D' in filedir:
            K = [425, 425, 320, 240]

        elif 'iBims' in filedir:
            K = [490, 490, 320, 240]

        elif 'KITTI' in filedir:
            K_raw = np.array(data_dict['K'], dtype=np.float32)

            # KITTI npy 中 K 是 [fx, fy, cx, cy]
            K = [
                float(K_raw[0]),
                float(K_raw[1]),
                float(K_raw[2]),
                float(K_raw[3]),
            ]

        else:
            K_raw = np.array(data_dict['K'], dtype=np.float32)

            # 其他数据集默认 K 是 3x3
            K = [
                float(K_raw[0, 0]),
                float(K_raw[1, 1]),
                float(K_raw[0, 2]),
                float(K_raw[1, 2]),
            ]

        return dep, gt, K, rgb, filedir

    def __getitem__(self, index):
        dep, gt, K, rgb, paths = self.__getraw__(index)
        dep = Image.fromarray(dep.astype('float32'), mode='F')
        gt = Image.fromarray(gt.astype('float32'), mode='F')
        K = None

        dep, gt, K, rgb = self.transforms(
            self.split,
            self.args,
            dep,
            gt,
            K,
            rgb
        )
        a = 1

        dep_np = dep.numpy().squeeze(0)
        dep_clear, _ = outlier_removal(dep_np)
        dep_clear = np.expand_dims(dep_clear, 0)
        dep_clear_torch = torch.from_numpy(dep_clear)

        dep_np = dep.numpy().squeeze(0)
        dep_np_ip = np.copy(dep_np)

        dep_ip = self.ipfill(
            dep_np_ip,
            max_depth=100.0,
            extrapolate=True,
            blur_type='gaussian'
        )

        dep_ip_torch = torch.from_numpy(dep_ip)
        dep_ip_torch = dep_ip_torch.to(dtype=torch.float32)

        candidates = {
            'dep': dep,
            'dep_clear': dep_clear_torch,
            'gt': gt,
            'rgb': rgb,
            'ip': dep_ip_torch
        }

        items = {
            key: val
            for key, val in candidates.items()
            if val is not None
        }

        if self.args.debug_dp or self.args.test:
            items['d_path'] = paths

        return items

    def __len__(self):
        if self.args.toy_test:
            return self.args.toy_test_number
        else:
            return len(self.files)