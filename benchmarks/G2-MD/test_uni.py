import os
import argparse
import numpy as np
import torch
torch.backends.cudnn.enabled = False

from PIL import Image
from pathlib import Path
from src.networks import UNet
from test_utils import NPYReader, DepthEvaluation

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # gpus
from tqdm import tqdm
# turn fast mode on
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True


def parse_arguments():
    parser = argparse.ArgumentParser(
        "options for G2-MonoDepth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rgbd_dir",
        type=lambda x: Path(x),
        default="Test_Datasets/Ibims",
        help="Path to RGBD folder",
    )
    parser.add_argument(
        "--model_dir",
        type=lambda x: Path(x),
        default="checkpoints/models/epoch_100.pth",
        help="Path to load models",
    )
    args = parser.parse_args()
    return args

def metricsv2(pred, gt):
    # valid pixels
    mask = gt > 0.0
    valid_nums = torch.sum(mask) + 1e-8

    pred = pred[mask]
    gt = gt[mask]

    diff = pred - gt

    # RMSE
    rmse = torch.sqrt(torch.sum(diff ** 2) / valid_nums)

    # MAE
    mae = torch.sum(torch.abs(diff)) / valid_nums

    # REL (mean absolute relative error)
    rel = torch.sum(torch.abs(diff) / (gt + 1e-8)) / valid_nums

    return rmse, mae, rel

def demo_save(args):
    print("-----------building model-------------")
    network = UNet(rezero=True).cuda().eval()
    network.load_state_dict(torch.load(args.model_dir)["network"])

    print("-----------inferring---------------")

    total_rmse = 0.0
    total_mae = 0.0
    total_rel = 0.0
    total_num = 0

    with torch.no_grad():
        file_list = list(args.rgbd_dir.rglob("*.npy"))

        for file in tqdm(file_list, total=len(file_list)):
            str_file = str(file)
            rgbd_reader = NPYReader()

            # read data
            rgb, gt, raw = rgbd_reader.read_data(str_file)
            hole_raw = torch.ones_like(raw)
            hole_raw[raw == 0] = 0

            # inference
            pred = network(rgb.cuda(), raw.cuda(), hole_raw.cuda())
            # pred = rgbd_reader.adjust_domain(pred)
            # import IPython
            # IPython.embed()
            # exit()

            rmse, mae, rel = metricsv2(pred.cpu(), gt.cpu())

            total_rmse += rmse.item()
            total_mae += mae.item()
            total_rel += rel.item()
            total_num += 1

            # print(
            #     f"{file.name} | "
            #     f"RMSE: {rmse.item():.4f}, "
            #     f"MAE: {mae.item():.4f}, "
            #     f"REL: {rel.item():.4f}"
            # )

    print("-----------final metrics---------------")
    print(f"Total samples: {total_num}")
    print(f"Average RMSE: {total_rmse / total_num:.4f}")
    print(f"Average MAE : {total_mae / total_num:.4f}")
    print(f"Average REL : {total_rel / total_num:.4f}")


if __name__ == "__main__":
    args = parse_arguments()
    demo_save(args)