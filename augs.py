#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    augs.py
# @Project:     BP-Net
# @Author:      jie
# @Time:        2021/3/14 8:27 PM


import numpy as np

__all__ = [
    'Compose',
    'Norm',
    'Jitter',
    'Flip',
    'RandomGray',
    'RandomBrightnessContrast',
    'RandomGamma',
]


class Compose(object):
    """
    Sequential operations on input images, (i.e. rgb, lidar and depth).
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, rgb, lidar, depth, K_cam):
        for t in self.transforms:
            rgb, lidar, depth, K_cam = t(rgb, lidar, depth, K_cam)
        return rgb, lidar, depth, K_cam


class Norm(object):
    """
    normalize rgb image.
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, rgb, lidar, depth, K_cam):
        rgb = (rgb - self.mean) / self.std
        return rgb, lidar, depth, K_cam

class Jitter(object):
    """
    borrow from https://github.com/kujason/avod/blob/master/avod/datasets/kitti/kitti_aug.py
    """

    def __call__(self, rgb, lidar, depth, K_cam):
        pca = compute_pca(rgb)
        rgb = add_pca_jitter(rgb, pca)
        return rgb, lidar, depth, K_cam



class Flip(object):
    """
    random horizontal flip of images.
    """

    def __call__(self, rgb, lidar, depth, CamK):
        width = rgb.shape[1]
        flip = bool(np.random.randint(2))
        if flip:
            rgb = rgb[:, ::-1, :]
            lidar = lidar[:, ::-1, :]
            depth = depth[:, ::-1, :]
            CamK[0, 2] = (width - 1) - CamK[0, 2]
        return rgb, lidar, depth, CamK


def compute_pca(image):
    """
    calculate PCA of image
    """

    reshaped_data = image.reshape(-1, 3)
    reshaped_data = (reshaped_data / 255.0).astype(np.float32)
    covariance = np.cov(reshaped_data.T)
    e_vals, e_vecs = np.linalg.eigh(covariance)
    pca = np.sqrt(e_vals) * e_vecs
    return pca


def add_pca_jitter(img_data, pca):
    """
    add a multiple of principle components with Gaussian noise
    """
    new_img_data = np.copy(img_data).astype(np.float32) / 255.0
    magnitude = np.random.randn(3) * 0.1
    noise = (pca * magnitude).sum(axis=1)

    new_img_data = new_img_data + noise
    np.clip(new_img_data, 0.0, 1.0, out=new_img_data)
    new_img_data = (new_img_data * 255).astype(np.uint8)

    return new_img_data

class RandomGray(object):
    """
    random convert rgb image to grayscale, but keep 3 channels.
    """

    def __init__(self, prob=0.2):
        self.prob = prob

    def __call__(self, rgb, lidar, depth, K_cam):
        if np.random.rand() < self.prob:
            # rgb: H x W x 3
            gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
            gray = gray.astype(rgb.dtype)

            # 保持三通道
            rgb = np.stack([gray, gray, gray], axis=2)

        return rgb, lidar, depth, K_cam
    
class RandomBrightnessContrast(object):
    """
    Random brightness and contrast adjustment
    """

    def __init__(self, prob=0.3, brightness=0.2, contrast=0.2):
        """
        brightness: 亮度变化比例，例如 0.2 表示 ±20%
        contrast: 对比度变化比例
        """
        self.prob = prob
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, rgb, lidar, depth, K_cam):
        if np.random.rand() < self.prob:
            rgb = rgb.astype(np.float32)

            # === brightness ===
            b = 1.0 + np.random.uniform(-self.brightness, self.brightness)
            rgb = rgb * b

            # === contrast ===
            c = 1.0 + np.random.uniform(-self.contrast, self.contrast)
            mean = np.mean(rgb, axis=(0, 1), keepdims=True)
            rgb = (rgb - mean) * c + mean

            # clip 回合法范围
            np.clip(rgb, 0, 255, out=rgb)

        return rgb, lidar, depth, K_cam
    
class RandomGamma(object):
    """
    Random gamma correction
    """

    def __init__(self, prob=0.3, gamma_range=(0.8, 1.2)):
        """
        gamma_range: gamma 取值范围
        <1 更亮，>1 更暗
        """
        self.prob = prob
        self.gamma_range = gamma_range

    def __call__(self, rgb, lidar, depth, K_cam):
        if np.random.rand() < self.prob:
            rgb = rgb.astype(np.float32) / 255.0

            gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])

            rgb = np.power(rgb, gamma)

            rgb = (rgb * 255.0).astype(np.float32)
            np.clip(rgb, 0, 255, out=rgb)

        return rgb, lidar, depth, K_cam