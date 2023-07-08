# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------
import math
import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.utils import AverageMeter

from config import get_config
from data.data_cosiam import build_loader_cosiam
from models import build_model
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, auto_resume_helper

from utils import NativeScalerWithGradNormCount as NativeScaler


def parse_option():
    parser = argparse.ArgumentParser('SimMIM pre-training script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    data_loader_train = build_loader_cosiam(config, logger)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, is_pretrain=True)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model, logger, is_pretrain=True)
    # Apply SyncBN
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK],
                                                      find_unused_parameters=True,
                                                      broadcast_buffers=False)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    pretrainer = Pretrainer(config=config)
    loss_scaler = NativeScaler()
    if config.MODEL.RESUME:
        load_checkpoint(config, model_without_ddp, optimizer, loss_scaler, lr_scheduler, logger)

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        pretrainer.train_one_epoch(model, data_loader_train, optimizer, epoch, loss_scaler,
                                   lr_scheduler, config.TRAIN.BASE_MOMENTUM)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, 0., optimizer, loss_scaler, lr_scheduler, logger)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


class Pretrainer:
    def __init__(self, config):
        self.config = config
        self.F = None
        self.rho = config.MODEL.UNIGRAD.RHO
        self.lambd = config.MODEL.UNIGRAD.LAMBD

    def loss_unigrad(self, z1, z2, z1m, z2m):
        # calculate correlation matrix
        tmp_F = (torch.mm(z1m.t(), z1m) + torch.mm(z2m.t(), z2m)) / (2 * z1m.shape[0])
        torch.distributed.all_reduce(tmp_F)
        tmp_F = tmp_F / torch.distributed.get_world_size()
        if self.F is None:
            self.F = tmp_F.detach()
        else:
            self.F = self.rho * self.F + (1 - self.rho) * tmp_F.detach()

        # compute grad & loss
        grad1 = -z2m + self.lambd * torch.mm(z1, self.F)
        loss1 = (grad1.detach() * z1).sum(-1).mean()

        grad2 = -z1m + self.lambd * torch.mm(z2, self.F)
        loss2 = (grad2.detach() * z2).sum(-1).mean()

        loss = 0.5 * (loss1 + loss2)

        # compute positive similarity, just for observation
        pos_sim1 = torch.einsum('nc,nc->n', [z1, z2m]).mean().detach()
        pos_sim2 = torch.einsum('nc,nc->n', [z2, z1m]).mean().detach()
        pos_sim = 0.5 * (pos_sim1 + pos_sim2)

        return loss, pos_sim

    def train_one_epoch(self, model, data_loader, optimizer, epoch, loss_scaler, lr_scheduler, m):
        model.train()
        optimizer.zero_grad()

        num_steps = len(data_loader)
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        pos_sim_meter = AverageMeter()
        norm_meter = AverageMeter()

        start = time.time()
        end = time.time()
        for data_iter_step, sample in enumerate(data_loader):
            x1 = sample['x1']
            x2 = sample['x2']
            random_crop = sample['random_crop']
            mask = sample['mask']

            x1 = x1.cuda(non_blocking=True)
            x2 = x2.cuda(non_blocking=True)
            random_crop = random_crop.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            with torch.cuda.amp.autocast():
                z1, z2, z1m, z2m = model(x1, x2, random_crop, m, mask)

                B, L, C = z1.shape

                z1 = z1.reshape((B * L, C))
                z1m = z1m.reshape((B * L, C))
                z2 = z2.reshape((B * L, C))
                z2m = z2m.reshape((B * L, C))

                # normalize
                z1 = torch.nn.functional.normalize(z1)
                z2 = torch.nn.functional.normalize(z2)
                z1m = torch.nn.functional.normalize(z1m)
                z2m = torch.nn.functional.normalize(z2m)

                loss, pos_sim = self.loss_unigrad(z1, z2, z1m, z2m)

            loss = loss / self.config.TRAIN.ACCUMULATION_STEPS
            grad_norm = loss_scaler(loss, optimizer, parameters=model.parameters(),
                                    update_grad=(data_iter_step + 1) % self.config.TRAIN.ACCUMULATION_STEPS == 0)
            if (data_iter_step + 1) % self.config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + data_iter_step)
            torch.cuda.synchronize()

            loss_meter.update(loss.item())
            if grad_norm is not None:
                norm_meter.update(grad_norm.item())
            pos_sim_meter.update(pos_sim.item())
            batch_time.update(time.time() - end)
            end = time.time()

            if (data_iter_step + 1) % self.config.PRINT_FREQ == 0:
                lr = optimizer.param_groups[0]['lr']
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                etas = batch_time.avg * (num_steps - data_iter_step)
                logger.info(
                    f'Train: [{epoch}/{self.config.TRAIN.EPOCHS}][{data_iter_step}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'pos_sim {pos_sim_meter.val:.4f} ({pos_sim_meter.avg:.4f})\t'
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')
        epoch_time = time.time() - start
        logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


if __name__ == '__main__':
    _, config = parse_option()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
