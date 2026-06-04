import os
import numpy as np
import json
import random
from . import BaseDataset
import glob
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import cv2



class Uniformat(BaseDataset):
    def __init__(self, args, mode):
        super(Uniformat, self).__init__(args, mode)

        self.args = args
        mode = 'test' if mode == 'val' else mode
        self.mode = mode
        self.lidar_lines = args.lidar_lines
        self.files = sorted(glob.glob(os.path.join(args.dir_data, '*.npy')))

        if mode != 'train' and mode != 'val' and mode != 'test':
            raise NotImplementedError

        self.height = args.patch_height
        self.width = args.patch_width

        self.augment = self.args.augment

    def __len__(self):
        return len(self.files)
        # return 10

    def _load_data(self, idx):
        filedir = self.files[idx]
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
            
        rgb = Image.fromarray(rgb, mode='RGB')
        depth = Image.fromarray(dep.astype('float32'), mode='F')
        gt = Image.fromarray(gt.astype('float32'), mode='F')

        w1, h1 = rgb.size
        w2, h2 = depth.size
        w3, h3 = gt.size

        assert w1 == w2 and w1 == w3 and h1 == h2 and h1 == h3

        return rgb, depth, gt, K

    def __getitem__(self, idx):
        rgb, depth, gt, K = self._load_data(idx)
        
        if True: # test
            rgb = TF.to_tensor(rgb)
            rgb = TF.normalize(rgb, (0.485, 0.456, 0.406),
                               (0.229, 0.224, 0.225), inplace=True)

            depth = TF.to_tensor(np.array(depth))

            gt = TF.to_tensor(np.array(gt))

        # print('dataloader', torch.max(gt[gt>0.0]), torch.min(gt[gt>0.0]))

        output = {'rgb': rgb, 'dep': depth, 'gt': gt, 'K': torch.Tensor(K)}

        return output