import os
import argparse
import torch
torch.backends.cudnn.enabled = False
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # gpus
from torchvision.utils import save_image

from PIL import Image
from pathlib import Path
from src.networks import V2Net
from test_utils import DataReader_npy, metricsv2
import torch
from tqdm import tqdm
import cv2
import numpy as np
# turn fast mode on
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True


def model_config(model_type):
    if model_type == "Tiny":
        dims = [96, 192, 384, 768]  # dimensions
        depths = [3, 3, 9, 3]  # block number
        dp_rate = 0.0
    if model_type == "Small":
        dims = [96, 192, 384, 768]  # dimensions
        depths = [3, 3, 27, 3]  # block number
        dp_rate = 0.1
    elif model_type == "Base":
        dims = [128, 256, 512, 1024]  # dimensions
        depths = [3, 3, 27, 3]  # block number
        dp_rate = 0.1
    elif model_type == "Large":
        dims = [192, 384, 768, 1536]  # dimensions
        depths = [3, 3, 27, 3]  # block number
        dp_rate = 0.2
    return dims, depths, dp_rate


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
        "--save_dir",
        type=str,
        help="Path to test folder",
    )
    parser.add_argument(
        "--model_dir",
        type=lambda x: Path(x),
        default="checkpoints/models/Large_300.pth",
        help="Path to load models",
    )
    parser.add_argument(
        "--norm_type",
        default="CNX",
        help="adopting SP-Normalization",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="Large",
        help="Path to load models",
    )
    parser.add_argument(
        "--save_results",
        type=bool,
        default=False,
        help="save results",
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
    # data_reader = DataReader_npy(args.device, args.max_depth)
    print("-----------building model-------------")
    # hugging face loading
    # net = CompletionNet.from_pretrained("Haotian-sx/PacGDC_large").to(args.device).eval()
    # locally loading
    dims, depths, dp_rate = model_config(args.model_type)
    net = V2Net(dims, depths, dp_rate, args.norm_type).cuda().eval()
    net.load_state_dict(torch.load(args.model_dir)["network"])
    avg_rmse, avg_mae, avg_rel = 0.0, 0.0, 0.0
    print("-----------inferring---------------")
    count, rmse, mae, rel = 0.0, 0.0, 0.0, 0.0
    file_list = list(sorted(args.data_dir.rglob("*.npy")))
    os.makedirs(args.save_dir, exist_ok=True)
    for file in tqdm(file_list):
            npy_path = str(file)
            data_reader = DataReader_npy("cuda", args.max_depth)

            rgb, raw, gt, hole_raw = data_reader.read_data(npy_path)

            # forward
            pred = net(rgb, raw, hole_raw)

            # denormalize
            pred = pred * args.max_depth
            gt = gt * args.max_depth

            # metrics
            rmse_temp, mae_temp, rel_temp = metricsv2(pred, gt)
            rmse += rmse_temp
            mae += mae_temp
            rel += rel_temp
            
            if True:
                pred_np = pred.squeeze().detach().cpu().numpy().astype(np.float32)

                pred_min, pred_max = pred_np.min(), pred_np.max()
                pred_vis = (pred_np - pred_min) / (pred_max - pred_min + 1e-8) * 255
                pred_vis = pred_vis.astype(np.uint8)
                pred_vis = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(args.save_dir ,f"{int(count):08d}_depth.png"), pred_vis)

            count += 1.0
        # metric each raw_dir
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

