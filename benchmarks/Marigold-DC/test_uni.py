import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import diffusers
import numpy as np
import torch
torch.backends.cudnn.enabled = False
from diffusers import DDIMScheduler
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from marigold_dc import MarigoldDepthCompletionPipeline


warnings.simplefilter(action="ignore", category=FutureWarning)
diffusers.utils.logging.disable_progress_bar()


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


def load_uniformat_npy(npy_path):
    data_dict = dict(np.load(npy_path, allow_pickle=True).item())

    rgb = data_dict["rgb"]
    sparse_depth = data_dict["dep"]
    gt = data_dict["gt"]

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    rgb_img = Image.fromarray(rgb, mode="RGB")

    sparse_depth = sparse_depth.astype(np.float32)
    gt = gt.astype(np.float32)

    return rgb_img, sparse_depth, gt


def parse_arguments():
    parser = argparse.ArgumentParser(
        "Marigold-DC test on Uniformat npy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("marigold_dc_results"))

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="prs-eth/marigold-depth-v1-0",
    )
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--ensemble_size", type=int, default=10)
    parser.add_argument("--processing_resolution", type=int, default=0)
    parser.add_argument("--use_full_precision", action="store_true")
    parser.add_argument("--use_tiny_vae", action="store_true")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--save_results", action="store_true")

    return parser.parse_args()


# @torch.no_grad()
def demo(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_inference_steps = args.num_inference_steps
    ensemble_size = args.ensemble_size
    processing_resolution = args.processing_resolution

    if not torch.cuda.is_available():
        processing_resolution = min(processing_resolution, 512)
        num_inference_steps = min(num_inference_steps, 10)
        ensemble_size = 1

    torch_dtype = torch.float32 if args.use_full_precision else torch.bfloat16

    logging.info("----------- building Marigold-DC pipeline -------------")
    pipe = MarigoldDepthCompletionPipeline.from_pretrained(
        args.checkpoint,
        prediction_type="depth",
    ).to(device, dtype=torch_dtype)

    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
    )

    if args.use_tiny_vae:
        del pipe.vae
        pipe.vae = diffusers.AutoencoderTiny.from_pretrained(
            "madebyollin/taesd"
        ).to(device, dtype=torch_dtype)

    npy_files = sorted(args.data_dir.rglob("*.npy"))
    logging.info(f"Found {len(npy_files)} npy files.")

    prediction_dir = args.output_dir / "prediction"
    visual_dir = args.output_dir / "visuals"

    if args.save_results:
        prediction_dir.mkdir(parents=True, exist_ok=True)
        visual_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    rmse_sum = 0.0
    mae_sum = 0.0
    rel_sum = 0.0

    logging.info("----------- inferring ---------------")

    for npy_path in tqdm(npy_files):
        rgb_img, sparse_depth, gt = load_uniformat_npy(npy_path)

        pred = pipe(
            image=rgb_img,
            sparse_depth=sparse_depth,
            num_inference_steps=num_inference_steps,
            ensemble_size=ensemble_size,
            processing_resolution=processing_resolution,
            seed=args.seed,
        )

        pred = np.asarray(pred, dtype=np.float32)

        if pred.shape != gt.shape:
            pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
            pred_t = torch.nn.functional.interpolate(
                pred_t,
                size=gt.shape,
                mode="nearest",
            )
            pred = pred_t.squeeze().numpy()

        rmse, mae, rel = metricsv2(pred, gt)

        count += 1
        rmse_sum += rmse.item()
        mae_sum += mae.item()
        rel_sum += rel.item()

        if args.save_results:
            stem = npy_path.stem

            np.save(prediction_dir / f"{stem}.npy", pred)

            vis = pipe.image_processor.visualize_depth(
                pred,
                val_min=pred.min(),
                val_max=pred.max(),
            )[0]
            vis.save(visual_dir / f"{stem}_vis.jpg")

        if True:
            pred_np = pred.astype(np.float32).squeeze()

            pred_min, pred_max = pred_np.min(), pred_np.max()
            pred_vis = (pred_np - pred_min) / (pred_max - pred_min + 1e-8) * 255
            pred_vis = pred_vis.astype(np.uint8)
            pred_vis = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(visual_dir, f"{batch:06d}_depth.png"), pred_vis)

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
    args = parse_arguments()
    demo(args)