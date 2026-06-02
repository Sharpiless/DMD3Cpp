import os
import argparse
import torch
torch.backends.cudnn.enabled = False
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from PIL import Image
from pathlib import Path
from src.SPNet.networks import CompletionNet
from test_utils import DataReader_npy, metricsv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2

def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("true", "1", "yes", "y")


def save_depth_heatmap(pred, save_path, cmap="Spectral_r"):
    """
    pred: torch.Tensor, shape [1, 1, H, W] or [1, H, W] or [H, W], depth in meters
    """
    depth = pred.detach().squeeze().cpu().numpy()

    valid = np.isfinite(depth)
    if valid.sum() > 0:
        d_min = depth[valid].min()
        d_max = depth[valid].max()
        depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
    else:
        depth_norm = np.zeros_like(depth, dtype=np.float32)

    heatmap = plt.get_cmap(cmap)(depth_norm)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    Image.fromarray(heatmap).save(save_path)


def parse_arguments():
    parser = argparse.ArgumentParser(
        "options for PacGDC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=lambda x: Path(x),
        default="Datasets/Data_Test/Ibims",
        help="Path to test folder",
    )
    parser.add_argument(
        "--ckpt_path",
        type=lambda x: Path(x),
        default="Pretrained/L_DA_DepthPro.pth",
        help="Path to load models",
    )
    parser.add_argument(
        "--save_results",
        type=str2bool,
        default=False,
        help="save visualization results",
    )
    parser.add_argument(
        "--save_dir",
        type=lambda x: Path(x),
        default="results",
        help="folder to save prediction visualizations",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="Spectral",
        help="matplotlib colormap, e.g. Spectral, Spectral_r, turbo, jet",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=30.0,
        help="max_depth to normalize depth values [indoor: 30 m, outdoor: 150m]",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    args = parser.parse_args()
    return args


@torch.no_grad()
def demo(args):
    print("-----------building model-------------")

    net = CompletionNet(str(args.ckpt_path.name)[0]).to(args.device).eval()
    net.load_state_dict(torch.load(args.ckpt_path, map_location=args.device)["network"])

    if args.save_results:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving visualizations to: {args.save_dir}")

    avg_rmse, avg_mae, avg_rel = 0.0, 0.0, 0.0
    print("-----------inferring---------------")
    count, rmse, mae, rel = 0.0, 0.0, 0.0, 0.0

    data_reader = DataReader_npy(args.device, args.max_depth)

    for file in tqdm(sorted(args.data_dir.rglob("*.npy")), desc="test", dynamic_ncols=True):
        npy_path = str(file)

        rgb, raw, gt, hole_raw, ori_size = data_reader.read_data(npy_path)

        pred = net(rgb, raw, hole_raw)
        pred = torch.nn.functional.interpolate(
            pred, size=ori_size, mode="bilinear", align_corners=True
        )

        pred = pred * args.max_depth
        gt = gt * args.max_depth

        count += 1.0
        rmse_temp, mae_temp, rel_temp = metricsv2(pred, gt)
        rmse += rmse_temp
        mae += mae_temp
        rel += rel_temp

        if args.save_results:
            rel_path = file.relative_to(args.data_dir)
            save_path = args.save_dir / rel_path.with_suffix(".png")
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # save_depth_heatmap(pred, save_path, cmap=args.cmap)
            
            if True:
                pred_np = pred.squeeze().detach().cpu().numpy().astype(np.float32)

                pred_min, pred_max = pred_np.min(), pred_np.max()
                pred_vis = (pred_np - pred_min) / (pred_max - pred_min + 1e-8) * 255
                pred_vis = pred_vis.astype(np.uint8)
                pred_vis = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)
                cv2.imwrite(save_path, pred_vis)


    rmse /= count
    mae /= count
    rel /= count

    avg_rmse += rmse
    avg_mae += mae
    avg_rel += rel

    print("Average: RMSE=", str(avg_rmse), " MAE=", str(avg_mae), " REL=", str(avg_rel))


if __name__ == "__main__":
    args = parse_arguments()
    demo(args)