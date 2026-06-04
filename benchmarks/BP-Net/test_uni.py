import os
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
torch.backends.cudnn.enabled = False
from tqdm import tqdm
from omegaconf import OmegaConf
from torchvision.utils import save_image
from utils_infer import Trainer


def metricsv2(pred, gt):
    pred = torch.as_tensor(pred).float()
    gt = torch.as_tensor(gt).float()

    mask = gt > 0.0
    valid_nums = torch.sum(mask) + 1e-8

    pred = pred[mask]
    gt = gt[mask]

    diff = pred - gt

    rmse = torch.sqrt(torch.sum(diff ** 2) / valid_nums)
    mae = torch.sum(torch.abs(diff)) / valid_nums
    rel = torch.sum(torch.abs(diff) / (gt + 1e-8)) / valid_nums

    return rmse, mae, rel


def get_K(data_dict, filedir):
    if "ETH3D" in filedir:
        K = np.array(
            [[700, 0, 320],
             [0, 700, 240],
             [0, 0, 1.0]],
            dtype=np.float32,
        )
    elif "iBims" in filedir:
        K = np.array(
            [[490, 0, 320],
             [0, 490, 240],
             [0, 0, 1.0]],
            dtype=np.float32,
        )
    elif "KITTI" in filedir:
        K_raw = data_dict["K"]
        K = np.array(
            [[K_raw[0], 0, K_raw[2]],
             [0, K_raw[1], K_raw[3]],
             [0, 0, 1.0]],
            dtype=np.float32,
        )
    else:
        K = np.array(data_dict["K"], dtype=np.float32)

    return K


class DataReaderBPNet(object):
    def __init__(self, device):
        super(DataReaderBPNet, self).__init__()
        self.device = device

        # from BP-Net inference script
        self.image_mean = np.array([90.9950, 96.2278, 94.3213], dtype=np.float32)
        self.image_std = np.array([79.2382, 80.5267, 82.1483], dtype=np.float32)

    def read_data(self, npy_path):
        data_dict = dict(np.load(npy_path, allow_pickle=True).item())

        image = data_dict["rgb"].astype(np.float32)[:, :, ::-1]
        lidar = data_dict["dep"].astype(np.float32)
        gt = data_dict["gt"].astype(np.float32)
        K = get_K(data_dict, npy_path)
        # import IPython
        # IPython.embed()
        # exit()

        image = (image - self.image_mean) / self.image_std

        image_tensor = image.transpose(2, 0, 1).astype(np.float32)[None]
        lidar_tensor = lidar[None, None].astype(np.float32)
        gt_tensor = gt[None, None].astype(np.float32)

        image_tensor = torch.from_numpy(image_tensor).to(self.device)
        lidar_tensor = torch.from_numpy(lidar_tensor).to(self.device)
        gt_tensor = torch.from_numpy(gt_tensor).to(self.device)
        K_tensor = torch.from_numpy(K).unsqueeze(0).to(self.device)

        return image_tensor, lidar_tensor, gt_tensor, K_tensor


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg):
    data_dir = OmegaConf.select(cfg, "data_dir")
    output_dir = OmegaConf.select(cfg, "output_dir")
    save_results = OmegaConf.select(cfg, "save_results")

    if data_dir is None:
        raise ValueError(
            "Please provide data_dir, e.g. "
            "python test_bpnet_uni.py data_dir=../datasets/uniformat_release/ETH3D_SfM_Indoor_test"
        )

    data_dir = Path(data_dir)
    output_dir = Path(output_dir) if output_dir is not None else Path("bpnet_results")
    save_results = bool(save_results) if save_results is not None else False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reader = DataReaderBPNet(device)

    with Trainer(cfg) as run:
        net = run.net_ema.module.to(device)
        net.eval()

        npy_files = sorted(data_dir.rglob("*.npy"))
        print(f"Found {len(npy_files)} npy files.")
        print("----------- inferring BP-Net ---------------")

        if save_results:
            pred_dir = output_dir / "prediction"
            vis_dir = output_dir / "visuals"
            pred_dir.mkdir(parents=True, exist_ok=True)
            vis_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        rmse_sum = 0.0
        mae_sum = 0.0
        rel_sum = 0.0

        with torch.no_grad():
            for npy_path in tqdm(npy_files):
                image, lidar, gt, K = reader.read_data(str(npy_path))

                output = net(image, lidar, K)

                if isinstance(output, (list, tuple)):
                    output = output[-1]

                pred = output.squeeze()
                gt_eval = gt.squeeze()

                if pred.shape != gt_eval.shape:
                    pred = torch.nn.functional.interpolate(
                        pred[None, None],
                        size=gt_eval.shape,
                        mode="nearest",
                    )[0, 0]

                rmse, mae, rel = metricsv2(pred, gt_eval)

                count += 1
                rmse_sum += rmse.item()
                mae_sum += mae.item()
                rel_sum += rel.item()

                if save_results:
                    pred_np = pred.detach().cpu().numpy().astype(np.float32)
                    np.save(pred_dir / f"{npy_path.stem}.npy", pred_np)

                    pred_min, pred_max = pred_np.min(), pred_np.max()
                    pred_vis = (pred_np - pred_min) / (pred_max - pred_min + 1e-8) * 255
                    pred_vis = pred_vis.astype(np.uint8)
                    pred_vis = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)
                    cv2.imwrite(str(vis_dir / f"{npy_path.stem}_depth.png"), pred_vis)

        avg_rmse = rmse_sum / max(count, 1)
        avg_mae = mae_sum / max(count, 1)
        avg_rel = rel_sum / max(count, 1)

        print(
            "Average: "
            f"RMSE= {avg_rmse:.6f} "
            f"MAE= {avg_mae:.6f} "
            f"REL= {avg_rel:.6f}"
        )


if __name__ == "__main__":
    main()