import random
import os
import numpy as np
import glob
from PIL import Image
import torch
import augs

__all__ = [
    'KITTI',
]


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


class KITTI(torch.utils.data.Dataset):
    """
    kitti depth completion dataset: http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion
    """

    def __init__(
        self, 
        path='datas/kitti', 
        mode='train', 
        height=256, 
        width=1216, 
        mean=(90.9950, 96.2278, 94.3213),
        std=(79.2382, 80.5267, 82.1483), 
        RandCrop=False, tp_min=0,
        train_sample_size=-1,

        lidar_noise_prob=0.35,      # 每个样本是否加噪声的概率
        lidar_noise_std=0.1,       # 高斯噪声标准差，单位与深度一致（米）
        lidar_noise_clip=(0.0, 200.0),  # 合法深度范围
        lidar_dropout_prob=0.3,      # 每个样本是否启用 dropout
        lidar_dropout_ratio=0.3,     # 启用后，随机丢弃多少比例的有效点
        *args, **kwargs
    ):
        self.base_dir = path
        self.height = height
        self.width = width
        self.mode = mode

        self.lidar_noise_prob = lidar_noise_prob
        self.lidar_noise_std = lidar_noise_std
        self.lidar_noise_clip = lidar_noise_clip
        self.lidar_dropout_prob = lidar_dropout_prob
        self.lidar_dropout_ratio = lidar_dropout_ratio
        
        if mode == 'train':
            self.transform = augs.Compose([
                augs.Jitter(),
                augs.RandomGray(prob=0.2),
                augs.RandomBrightnessContrast(prob=0.3),
                augs.RandomGamma(prob=0.3),
                augs.Flip(),
                augs.Norm(mean=mean, std=std),
            ])
        else:
            self.transform = augs.Compose([
                augs.Norm(mean=mean, std=std),
            ])
        self.RandCrop = RandCrop and mode == 'train'
        self.tp_min = tp_min
        if mode in ['train', 'val']:
            self.depth_path = os.path.join(self.base_dir, 'data_depth_annotated', mode)
            self.lidar_path = os.path.join(self.base_dir, 'data_depth_velodyne', mode)
            self.depths = list(sorted(glob.iglob(self.depth_path + "/**/*.png", recursive=True)))
            self.lidars = list(sorted(glob.iglob(self.lidar_path + "/**/*.png", recursive=True)))
        elif mode == 'selval':
            self.depth_path = os.path.join(self.base_dir, 'val_selection_cropped', 'groundtruth_depth')
            self.lidar_path = os.path.join(self.base_dir, 'val_selection_cropped', 'velodyne_raw')
            self.image_path = os.path.join(self.base_dir, 'val_selection_cropped', 'image')
            self.depths = list(sorted(glob.iglob(self.depth_path + "/*.png", recursive=True)))
            self.lidars = list(sorted(glob.iglob(self.lidar_path + "/*.png", recursive=True)))
            self.images = list(sorted(glob.iglob(self.image_path + "/*.png", recursive=True)))
        elif mode == 'test':
            self.lidar_path = os.path.join(self.base_dir, 'test_depth_completion_anonymous', 'velodyne_raw')
            self.image_path = os.path.join(self.base_dir, 'test_depth_completion_anonymous', 'image')
            self.lidars = list(sorted(glob.iglob(self.lidar_path + "/*.png", recursive=True)))
            self.images = list(sorted(glob.iglob(self.image_path + "/*.png", recursive=True)))
            self.depths = self.lidars
        else:
            raise ValueError("Unknown mode: {}".format(mode))
        assert (len(self.depths) == len(self.lidars))
        self.names = [os.path.split(path)[-1] for path in self.depths]
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)
        xy = np.stack((xx, yy), axis=-1)
        self.xy = xy

    def __len__(self):
        return len(self.depths)

    def __getitem__(self, index):
        return self.get_item(index)

    def add_lidar_noise(self, lidar):
        """
        lidar: H x W x 1, float32
        只对非零深度点加噪声，且每个点有概率保持不变
        """
        if self.mode != 'train':
            return lidar

        # 控制“这一帧是否加噪声”
        if random.random() > self.lidar_noise_prob:
            return lidar

        valid_mask = (lidar > 0)

        if not np.any(valid_mask):
            return lidar

        lidar = lidar.copy()

        # === 新增：每个点是否加噪声 ===
        point_mask = (np.random.rand(*lidar.shape) < self.lidar_noise_prob)  # 50% 点加噪声（可调）

        # 最终生效 mask：既是有效点，又被选中加噪声
        noise_mask = valid_mask & point_mask

        if not np.any(noise_mask):
            return lidar

        noise = np.random.normal(
            loc=0.0,
            scale=self.lidar_noise_std,
            size=lidar.shape
        ).astype(np.float32)

        lidar[noise_mask] = lidar[noise_mask] + noise[noise_mask]

        # clip 只作用在被改动的点
        min_d, max_d = self.lidar_noise_clip
        lidar[noise_mask] = np.clip(lidar[noise_mask], min_d, max_d)

        # 确保原始 0 区域不变
        lidar[~valid_mask] = 0.0

        return lidar
    
    def dropout_lidar_points(self, lidar):
        """
        lidar: H x W x 1, float32
        只对非零有效点随机 dropout，一部分点置 0
        """
        if self.mode != 'train':
            return lidar

        if random.random() > self.lidar_dropout_prob:
            return lidar

        valid_mask = (lidar > 0)

        if not np.any(valid_mask):
            return lidar

        lidar = lidar.copy()

        lidar_dropout_ratio = np.random.random() * self.lidar_dropout_ratio
        # 对每个有效点独立决定是否 dropout
        drop_mask = (np.random.rand(*lidar.shape) < lidar_dropout_ratio) & valid_mask

        lidar[drop_mask] = 0.0
        return lidar

    def get_item(self, index):
        depth = self.pull_DEPTH(self.depths[index])
        depth = np.expand_dims(depth, axis=2)
        lidar = self.pull_DEPTH(self.lidars[index])
        lidar = np.expand_dims(lidar, axis=2)
        K_cam = self.pull_K_cam(index).astype(np.float32)
        file_names = self.depths[index].split('/')
        if self.mode in ['train', 'val']:
            rgb_path = os.path.join(*file_names[:-7], 'raw', file_names[-5].split('_drive')[0], file_names[-5],
                                    file_names[-2], 'data', file_names[-1])
        elif self.mode in ['selval', 'test']:
            rgb_path = self.images[index]
            if rgb_path is None:
                rgb_path = os.path.join(*file_names[:-7], 'raw', file_names[-5].split('_drive')[0], file_names[-5],
                                        file_names[-2], 'data', file_names[-1])
        else:
            raise ValueError("Unknown mode: {}".format(self.mode))
        rgb = self.pull_RGB(rgb_path)
        rgb = rgb.astype(np.float32)
        lidar = lidar.astype(np.float32)
        depth = depth.astype(np.float32)
        if self.transform:
            rgb, lidar, depth, K_cam = self.transform(rgb, lidar, depth, K_cam)

        # 先随机丢弃一部分有效 lidar 点
        lidar = self.dropout_lidar_points(lidar)

        # 只在 train 时，以一定概率给稀疏 lidar 非零位置加噪声
        lidar = self.add_lidar_noise(lidar)

        rgb = rgb.transpose(2, 0, 1).astype(np.float32)
        lidar = lidar.transpose(2, 0, 1).astype(np.float32)
        depth = depth.transpose(2, 0, 1).astype(np.float32)
        tp = rgb.shape[1] - self.height
        lp = (rgb.shape[2] - self.width) // 2
        if self.RandCrop:
            tp = random.randint(self.tp_min, tp)
            lp = random.randint(0, rgb.shape[2] - self.width)
        rgb = rgb[:, tp:tp + self.height, lp:lp + self.width]
        lidar = lidar[:, tp:tp + self.height, lp:lp + self.width]
        depth = depth[:, tp:tp + self.height, lp:lp + self.width]
        K_cam[0, 2] -= lp
        K_cam[1, 2] -= tp
        return rgb, lidar, K_cam, depth

    def pull_RGB(self, path):
        img = np.array(Image.open(path).convert('RGB'), dtype=np.uint8)
        return img

    def pull_DEPTH(self, path):
        depth_png = np.array(Image.open(path), dtype=int)
        assert (np.max(depth_png) > 255)
        depth_image = (depth_png / 256.).astype(np.float32)
        return depth_image

    def pull_K_cam(self, index):
        file_names = self.depths[index].split('/')
        if self.mode in ['train', 'val', 'trainval']:
            calib_path = os.path.join(*file_names[:-7], 'raw', file_names[-5].split('_drive')[0],
                                      'calib_cam_to_cam.txt')
            filedata = read_calib_file(calib_path)
            P_rect_20 = np.reshape(filedata['P_rect_02'], (3, 4))
            P_rect_30 = np.reshape(filedata['P_rect_03'], (3, 4))
            if file_names[-2] == 'image_02':
                K_cam = P_rect_20[0:3, 0:3]
            elif file_names[-2] == 'image_03':
                K_cam = P_rect_30[0:3, 0:3]
            else:
                raise ValueError("Unknown mode: {}".format(file_names[-2]))

        elif self.mode in ['selval', 'test']:
            fns = self.images[index].split('/')
            calib_path = os.path.join(*fns[:-2], 'intrinsics', fns[-1][:-3] + 'txt')
            with open(calib_path, 'r') as f:
                K_cam = f.read().split()
            K_cam = np.array(K_cam, dtype=np.float32).reshape(3, 3)
        else:
            raise ValueError("Unknown mode: {}".format(self.mode))
        return K_cam
