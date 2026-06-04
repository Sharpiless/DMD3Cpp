# convert_ddad_to_uni.py

import os
import numpy as np
from tqdm import tqdm
from PIL import Image

from datasets.ddad import DDAD   # 改成你的实际文件名


def save_ddad_as_uni(
        src_root='datas/ddad',
        mode='val',
        out_dir='datasets/uniformat_release/DDAD_val'
):

    os.makedirs(out_dir, exist_ok=True)

    dataset = DDAD(
        path=src_root,
        mode=mode,
        RandCrop=False
    )

    print("Dataset size:", len(dataset))

    for idx in tqdm(range(len(dataset))):

        # --------直接读取原始数据---------
        rgb = dataset.pull_RGB(
            dataset.images[idx]
        )

        lidar = dataset.pull_DEPTH(
            dataset.lidar_path[idx]
        )

        depth = dataset.pull_DEPTH(
            dataset.depth_path[idx]
        )

        K = dataset.pull_K_cam(idx)

        # 确保格式
        rgb = rgb.astype(np.uint8)

        lidar = lidar.astype(np.float32)

        depth = depth.astype(np.float32)

        K = K.astype(np.float32)

        save_dict = {
            'rgb': rgb,      # H×W×3 uint8
            'dep': lidar,    # H×W
            'gt': depth,     # H×W
            'K': K           # 3×3
        }

        name = os.path.splitext(
            os.path.basename(dataset.images[idx])
        )[0]

        np.save(
            os.path.join(
                out_dir,
                f"{name}.npy"
            ),
            save_dict
        )

    print("Done.")
    print("Saved to:", out_dir)


if __name__=="__main__":

    save_ddad_as_uni(
        src_root='datas/ddad',
        mode='val',
        out_dir='datasets/uniformat_release/DDAD_val'
    )