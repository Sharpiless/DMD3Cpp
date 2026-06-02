"""
    CompletionFormer
    ======================================================================

    main script for training and testing.
"""


from config import args as args_config
import time
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args_config.gpus
os.environ["MASTER_ADDR"] = args_config.address
os.environ["MASTER_PORT"] = args_config.port

import json
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
import cv2
import torch
torch.backends.cudnn.enabled = False
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
torch.autograd.set_detect_anomaly(True)

import utility
from model.completionformer import CompletionFormer
from summary.cfsummary import CompletionFormerSummary
from metric.cfmetric import CompletionFormerMetric
from data import get as get_data
from loss.l1l2loss import L1L2Loss

# Multi-GPU and Mixed precision supports
# NOTE : Only 1 process per GPU is supported now
import torch.multiprocessing as mp
import torch.distributed as dist
import apex
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Minimize randomness
def init_seed(seed=None):
    if seed is None:
        seed = args_config.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_args(args):
    new_args = args
    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        if args.resume:
            checkpoint = torch.load(args.pretrain)

            new_args = checkpoint['args']
            new_args.test_only = args.test_only
            new_args.pretrain = args.pretrain
            new_args.dir_data = args.dir_data
            new_args.resume = args.resume

    return new_args

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

def test(args):
    # Prepare dataset
    data = get_data(args)

    data_test = data(args, 'test')

    loader_test = DataLoader(dataset=data_test, batch_size=1,
                             shuffle=False, num_workers=0)

    # Network
    if args.model == 'CompletionFormer':
        net = CompletionFormer(args).half()
    else:
        raise TypeError(args.model, ['CompletionFormer',])
    net.cuda()

    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        checkpoint = torch.load(args.pretrain)
        key_m, key_u = net.load_state_dict(checkpoint['net'], strict=True)
        print("[debug]")
        if key_u:
            print('Unexpected keys :')
            print(key_u)

        if key_m:
            print('Missing keys :')
            print(key_m)
            raise KeyError
        print('Checkpoint loaded from {}!'.format(args.pretrain))

    net = nn.DataParallel(net)

    metric = CompletionFormerMetric(args)

    try:
        os.makedirs(args.save, exist_ok=True)
    except OSError:
        pass


    net.eval()

    num_sample = len(loader_test)*loader_test.batch_size

    pbar = tqdm(total=num_sample)

    t_total = 0

    init_seed()
    rmse_list = []
    mae_list = []
    rel_list = []
    
    for batch, sample in enumerate(loader_test):
        # sample = {key: val.cuda() for key, val in sample.items()
        #           if val is not None}
        sample = {
            key: val.cuda().half()
            for key, val in sample.items()
            if val is not None
        }
        t0 = time.time()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                    output = net(sample)
        t1 = time.time()

        t_total += (t1 - t0)

        metric_val = metric.evaluate(sample, output, 'test')
        rmse, mae, rel = metricsv2(output['pred'].detach(), sample['gt'].detach())
        # pred = output['pred']
        # save_image(pred/pred.max(), 'pred.png')
        # import IPython
        # IPython.embed()
        # exit()
        rmse_list.append(rmse.cpu().numpy())
        mae_list.append(mae.cpu().numpy())
        rel_list.append(rel.cpu().numpy())

        current_time = time.strftime('%y%m%d@%H:%M:%S')
        error_str = '{} | Test | RMSE:{:.4f}, MAE:{:.4f}, REL:{:.4f}'.format(
            current_time,
            np.mean(rmse_list),
            np.mean(mae_list),
            np.mean(rel_list)
        )
        pbar.set_description(error_str)
        pbar.update(loader_test.batch_size)

        if True:
            pred_np = output['pred'].detach().cpu().numpy().astype(np.float32).squeeze()

            pred_min, pred_max = pred_np.min(), pred_np.max()
            pred_vis = (pred_np - pred_min) / (pred_max - pred_min + 1e-8) * 255
            pred_vis = pred_vis.astype(np.uint8)
            pred_vis = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(args.save, f"{batch:06d}_depth.png"), pred_vis)


    pbar.close()

    t_avg = t_total / num_sample
    print('Elapsed time : {} sec, '
          'Average processing time : {} sec'.format(t_total, t_avg))
    error_str = '{} | Test | RMSE:{:.4f}, MAE:{:.4f}, REL:{:.4f}'.format(
            current_time,
            np.mean(rmse_list),
            np.mean(mae_list),
            np.mean(rel_list)
        )
    print(error_str)


def main(args):
    init_seed()
    if not args.test_only:
        if args.no_multiprocessing:
            train(0, args)
        else:
            assert args.num_gpus > 0

            spawn_context = mp.spawn(train, nprocs=args.num_gpus, args=(args,),
                                     join=False)

            while not spawn_context.join():
                pass

            for process in spawn_context.processes:
                if process.is_alive():
                    process.terminate()
                process.join()

        args.pretrain = '{}/model_{:05d}.pt'.format(args.save_dir, args.epochs)

    test(args)


if __name__ == '__main__':
    args_main = check_args(args_config)

    print('\n\n=== Arguments ===')
    cnt = 0
    for key in sorted(vars(args_main)):
        print(key, ':',  getattr(args_main, key), end='  |  ')
        cnt += 1
        if (cnt + 1) % 5 == 0:
            print('')
    print('\n')

    main(args_main)