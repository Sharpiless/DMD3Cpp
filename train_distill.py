# -*- coding: utf-8 -*-
# @File : train_amp.py
# @Project: BP-Net
# @Author : jie
# @Time : 10/27/21 3:58 PM

import torch
torch.backends.cudnn.enabled = False
from tqdm import tqdm
import hydra
import torch.distributed as dist
from utils import *


def train(run):
    for datas in tqdm(run.trainloader, desc="train", dynamic_ncols=True, leave=False, disable=run.rank):
        if run.epoch >= run.cfg.test_epoch:
            if run.iter % run.cfg.test_iter == 0:
                test(run, iter=True)
        datas = run.init_cuda(*datas)
        run.net.train()
        run.optimizer.zero_grad(set_to_none=True)
        output, da3_output = run.net(*datas[:-1], return_da3=True)
        if run.epoch >= run.cfg.test_epoch:
            return_dis = False
        else:
            return_dis = True
        loss, _ = run.criterion(output, datas[-1], torch.from_numpy(da3_output.depth[:, None]).to(datas[-1].device), return_dis=return_dis)
        loss.backward()
        if run.clip:
            grad_norm = run.clip(run.net.parameters())
        run.optimizer.step()
        run.net_ema.update(run.net)
        if run.lr_iter:
            run.lr_scheduler.step()
        if run.iter % run.cfg.vis_iter == 0:
            run.writer.add_scalar("Lr", run.optimizer.param_groups[0]['lr'], run.iter)
            if run.clip and (grad_norm is not None):
                run.writer.add_scalar("GradNorm", grad_norm.item(), run.iter)
        run.iter += 1
    if not run.lr_iter:
        run.lr_scheduler.step()
    run.writer.flush()


def test(run, iter=False):
    top1 = AverageMeter()
    net = run.net_ema.module
    best_metric_name = "best_metric_ema"
    legand = 'net_ema'
    net.eval()
    with torch.no_grad():
        for datas in tqdm(run.testloader, desc="test ", dynamic_ncols=True, leave=False, disable=run.rank):
            datas = run.init_cuda(*datas)
            output = net(*datas[:-1])
            if isinstance(output, (list, tuple)):
                output = output[-1]
            prec1 = run.metric(output, datas[-1])
            if isinstance(prec1, (list, tuple)):
                prec1 = prec1[0]
            if run.ddp:
                dist.reduce(prec1, 0, dist.ReduceOp.AVG)
            top1.update(prec1.item())
    
    run.ddp_cout(f'Epoch: {run.epoch} current_metric_name: {top1.avg:.7f}\n')
    run.save_state(name=f"result_epoch{run.epoch:04d}")
    if top1.avg < getattr(run, best_metric_name):
        setattr(run, best_metric_name, top1.avg)
        run.save_state()
        run.ddp_cout(f'Epoch: {run.epoch} {best_metric_name}: {top1.avg:.7f}\n')


@hydra.main(config_path='configs', config_name='config', version_base='1.2')
def main(cfg):
    with Trainer(cfg) as run:
        for epoch in tqdm(range(run.cfg.start_epoch, run.cfg.nepoch), desc="epoch", dynamic_ncols=True):
            run.epoch = epoch
            if run.train_sampler:
                run.train_sampler.set_epoch(epoch)
            train(run)
            torch.cuda.synchronize()
            test(run)
            torch.cuda.synchronize()


if __name__ == '__main__':
    main()
