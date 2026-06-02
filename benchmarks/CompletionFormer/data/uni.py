import os
import glob
import numpy as np
import torch
from PIL import Image

import torchvision.transforms.functional as TF
from . import BaseDataset


__all__ = ['UNI']


class UNI(BaseDataset):
    def __init__(self, args, mode='val'):
        super(UNI, self).__init__(args, mode)

        self.args = args
        self.mode = 'val'

        self.files = sorted(glob.glob(os.path.join(args.dir_data, '*.npy')))
        assert len(self.files) > 0, f'No npy files found in {args.dir_data}'

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        rgb, depth, gt, K = self._load_data(idx)

        # import IPython
        # IPython.embed()
        # exit()

        rgb = TF.to_tensor(rgb)
        rgb = TF.normalize(
            rgb,
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            inplace=True
        )

        depth = TF.to_tensor(np.array(depth))
        gt = TF.to_tensor(np.array(gt))

        return {
            'rgb': rgb,
            'dep': depth,
            'gt': gt,
            'K': torch.Tensor(K)
        }

    def _load_data(self, idx):
        filedir = self.files[idx]
        data = np.load(filedir, allow_pickle=True).item()

        rgb = data['rgb']

        depth = data['dep'] if 'dep' in data else data['lidar']
        gt = data['gt'] if 'gt' in data else data['depth']

        rgb = rgb.astype(np.uint8)
        depth = depth.astype(np.float32)
        gt = gt.astype(np.float32)

        if depth.ndim == 3:
            depth = depth[:, :, 0]
        if gt.ndim == 3:
            gt = gt[:, :, 0]

        if 'ETH3D' in filedir:
            K = [425, 425, 320, 240]

        elif 'iBims' in filedir:
            K = [490, 490, 320, 240]

        elif 'KITTI' in filedir:
            K_raw = np.array(data['K'], dtype=np.float32)

            # KITTI 的 npy 中 K 是 [fx, fy, cx, cy]
            K = [
                K_raw[0],
                K_raw[1],
                K_raw[2],
                K_raw[3],
            ]

        else:
            K_raw = np.array(data['K'], dtype=np.float32)

            # 其他数据集默认 K 是 3x3
            K = [
                K_raw[0, 0],
                K_raw[1, 1],
                K_raw[0, 2],
                K_raw[1, 2],
            ]

        K = np.array(K, dtype=np.float32)

        rgb = Image.fromarray(rgb, mode='RGB')
        depth = Image.fromarray(depth, mode='F')
        gt = Image.fromarray(gt, mode='F')

        assert rgb.size == depth.size == gt.size, \
            f'Size mismatch: rgb={rgb.size}, dep={depth.size}, gt={gt.size}'

        return rgb, depth, gt, K