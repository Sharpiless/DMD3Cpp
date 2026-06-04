import os
import glob
import numpy as np
import torch
import augs
import cv2

__all__ = [
    'UNI',
]


class UNI(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        mode='test',
        height=480,
        width=640,
        mean=(90.9950, 96.2278, 94.3213),
        std=(79.2382, 80.5267, 82.1483),
        *args,
        **kwargs
    ):
        if mode != 'test':
            raise ValueError("KITTI_NPY_Test only supports mode='test'")

        self.base_dir = path
        self.mode = mode

        # 与原 test 版本一致：只做 Norm
        self.transform = augs.Compose([
            augs.Norm(mean=mean, std=std),
        ])

        self.files = sorted(glob.glob(os.path.join(self.base_dir, '*.npy')))
        self.names = [os.path.basename(f) for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.get_item(index)
    
    def resize_to_multiple_of_32(self, rgb, lidar, depth, K_cam):
        h, w = rgb.shape[:2]

        new_h = int(np.ceil(h / 32) * 32)
        new_w = int(np.ceil(w / 32) * 32)

        if new_h == h and new_w == w:
            return rgb, lidar, depth, K_cam

        scale_x = new_w / w
        scale_y = new_h / h

        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        lidar = cv2.resize(
            lidar.squeeze(-1),
            (new_w, new_h),
            interpolation=cv2.INTER_NEAREST
        )[:, :, None]

        depth = cv2.resize(
            depth.squeeze(-1),
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR
        )[:, :, None]

        K_cam = K_cam.copy()
        K_cam[0, 0] *= scale_x
        K_cam[0, 2] *= scale_x
        K_cam[1, 1] *= scale_y
        K_cam[1, 2] *= scale_y

        return rgb, lidar, depth, K_cam

    def get_item(self, index):
        data = np.load(self.files[index], allow_pickle=True).item()

        # 约定 npy 中包含:
        # rgb: H x W x 3
        # dep / lidar: H x W 或 H x W x 1
        # gt / depth: H x W 或 H x W x 1
        # K: 3 x 3
        rgb = data['rgb'].astype(np.float32)

        lidar = data['dep'] if 'dep' in data else data['lidar']
        depth = data['gt'] if 'gt' in data else data['depth']

        if 'ETH3D' in self.base_dir:
            K_cam = np.array(
                [[425, 0, 320],
                 [0, 425, 240],
                 [0, 0, 1.0]]
            )
        elif 'iBims' in self.base_dir:
            K_cam = np.array(
                [[490, 0, 320],
                 [0, 490, 240],
                 [0, 0, 1.0]]
            )
        elif 'KITTI' in self.base_dir:
            K_cam = np.array(
                [[data['K'][0], 0, data['K'][2]],
                 [0, data['K'][1], data['K'][3]],
                 [0, 0, 1.0]]
            ).reshape(3, 3)
        else:
            K_cam = np.array(data['K'])

        if lidar.ndim == 2:
            lidar = np.expand_dims(lidar, axis=2)
        if depth.ndim == 2:
            depth = np.expand_dims(depth, axis=2)

        lidar = lidar.astype(np.float32)
        depth = depth.astype(np.float32)
        K_cam = K_cam.astype(np.float32)
        rgb, lidar, depth, K_cam = self.resize_to_multiple_of_32(
            rgb, lidar, depth, K_cam
        )
        # 与原代码保持一致
        if self.transform:
            rgb, lidar, depth, K_cam = self.transform(rgb, lidar, depth, K_cam)

        rgb = rgb.transpose(2, 0, 1).astype(np.float32)
        lidar = lidar.transpose(2, 0, 1).astype(np.float32)
        depth = depth.transpose(2, 0, 1).astype(np.float32)

        return rgb, lidar, K_cam, depth