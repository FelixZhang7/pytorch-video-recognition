import multiprocessing as mp
import argparse
import os
import time
import yaml
import logging
from easydict import EasyDict
import pprint
from tensorboardX import SummaryWriter
import subprocess
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import linklink as link

from methods import model_entry
from utils.scheduler import get_scheduler
from datasets import get_train_dataset, get_val_dataset
from utils.utils import create_logger, AverageMeter, accuracy, \
    save_result, save_checkpoint, load_state, \
    DistributedGivenIterationSampler, simple_group_split, \
    DistributedSampler, param_group_no_wd
from utils.distributed_utils import dist_init, reduce_gradients, DistModule
from utils.loss import LabelSmoothCELoss
from utils.augmentation import *

mp.set_start_method('spawn', force=True)

parser = argparse.ArgumentParser(description='PyTorch Video Analysis')
parser.add_argument('--config', default='cfgs/config_res50.yaml')
parser.add_argument('--load-path', default='', type=str)
parser.add_argument('--recover', action='store_true')
parser.add_argument('-e', '--evaluate', action='store_true')
parser.add_argument('--sync', action='store_true')
parser.add_argument('--fake', action='store_true')
parser.add_argument('--fuse-prob', action='store_true')
parser.add_argument('--fusion-list', nargs='+', help='multi model fusion list')
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--fp16-normal-bn', action='store_true')
parser.add_argument('--dynamic-loss-scale', action='store_true')
parser.add_argument('--static-loss-scale', default=1, type=float)


def main():
    global args, config, best_prec1
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

    config = EasyDict(config['common'])
    config.save_path = os.path.dirname(args.config)

    rank, world_size = dist_init()

    # create model
    bn_group_size = config.model.kwargs.bn_group_size
    bn_var_mode = config.model.kwargs.get('bn_var_mode', 'L2')
    if bn_group_size == 1:
        bn_group = None
    else:
        assert world_size % bn_group_size == 0
        bn_group = simple_group_split(
            world_size, rank, world_size // bn_group_size)

    config.model.kwargs.bn_group = bn_group
    config.model.kwargs.bn_var_mode = (link.syncbnVarMode_t.L1
                                       if bn_var_mode == 'L1'
                                       else link.syncbnVarMode_t.L2)

    # If loads model from load_path, there is no need to load imagenet
    # pretrained model

    if args.load_path:
        config.model.pretrained = False

    model = model_entry(config.model)

    if rank == 0:
        print(model)

    model.cuda()

    if args.fp16:
        # if you have modules that must use fp32 parameters, and need fp32 input
        # try use link.fp16.register_float_module(your_module)
        # if you only need fp32 parameters set cast_args=False when call this
        # function, then call link.fp16.init() before call model.half()
        # example:
        #   link.fp16.register_float_module(link.nn.SyncBatchNorm2d,
        #                                   cast_args=False)
        #   link.fp16.init()
        print("Using FP16.")
        if args.fp16_normal_bn:
            print('Using normal bn for fp16')
            link.fp16.register_float_module(
                link.nn.SyncBatchNorm2d, cast_args=False)
            link.fp16.init()
        model.half()

    model = DistModule(model, args.sync)

    if config.get('no_wd', False):
        param_group, type2num = param_group_no_wd(model)
        config.param_group_no_wd = type2num

        optimizer = torch.optim.SGD(param_group, config.lr_scheduler.base_lr,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay,
                                    nesterov=config.nesterov)
    else:
        if not config.get('policy', False):
            optimizer = torch.optim.SGD(model.parameters(),
                                        config.lr_scheduler.base_lr,
                                        momentum=config.momentum,
                                        weight_decay=config.weight_decay,
                                        nesterov=config.nesterov)
        else:
            policies = model.module.get_optim_policies()
            for group in policies:
                print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
                    group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

            optimizer = torch.optim.SGD(policies, config.lr_scheduler.base_lr,
                                        momentum=config.momentum,
                                        weight_decay=config.weight_decay,
                                        nesterov=config.nesterov)

    if args.fp16:
        optimizer = link.fp16.FP16_Optimizer(
            optimizer,
            dynamic_loss_scale=args.dynamic_loss_scale,
            static_loss_scale=args.static_loss_scale
        )

    # optionally resume from a checkpoint
    last_iter = -1
    best_prec1 = 0
    if args.load_path:
        if args.recover:
            best_prec1, last_iter = load_state(
                args.load_path, model, optimizer=optimizer)
        else:
            load_state(args.load_path, model)

    cudnn.benchmark = True

    if config.model.get('type', False) == "slowfast" or \
       config.model.get('type', False) == "classify":
        print("use slowfast")
        cudnn.benchmark = False

    # get augmentation and transforms for training
    train_aug, val_aug = get_augmentation(config, model.module)

    # train dataset
    train_dataset = get_train_dataset(
        config, transform=transforms.Compose(train_aug))

    # val dataset
    val_dataset = get_val_dataset(
        config=config, transform=transforms.Compose(val_aug))

    train_sampler = DistributedGivenIterationSampler(
        train_dataset, config.lr_scheduler.max_iter, config.batch_size,
        last_iter=last_iter)

    val_sampler = DistributedSampler(val_dataset, round_up=False)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.workers, pin_memory=True, sampler=train_sampler)

    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.workers, pin_memory=True, sampler=val_sampler)

    config.lr_scheduler['optimizer'] = optimizer if not args.fp16 else optimizer.optimizer
    config.lr_scheduler['last_iter'] = last_iter
    lr_scheduler = get_scheduler(config.lr_scheduler)

    if rank == 0:
        tb_logger = SummaryWriter(config.save_path + '/events')
        logger = create_logger('global_logger', config.save_path + '/log.txt')
        logger.info('args: {}'.format(pprint.pformat(args)))
        logger.info('config: {}'.format(pprint.pformat(config)))
    else:
        tb_logger = None

    if args.evaluate:
        if args.fusion_list is not None:
            validate(val_loader, model, fusion_list=args.fusion_list,
                     fuse_prob=args.fuse_prob)
        else:
            validate(val_loader, model)
        link.finalize()
        return

    train(train_loader, val_loader, model, optimizer,
          lr_scheduler, last_iter + 1, tb_logger)

    link.finalize()


def train(train_loader, val_loader, model, optimizer, lr_scheduler, start_iter, tb_logger):

    print("In train function")
    global best_prec1

    batch_time = AverageMeter(config.print_freq)
    fw_time = AverageMeter(config.print_freq)
    bp_time = AverageMeter(config.print_freq)
    sy_time = AverageMeter(config.print_freq)
    step_time = AverageMeter(config.print_freq)
    data_time = AverageMeter(config.print_freq)
    losses = AverageMeter(config.print_freq)
    top1 = AverageMeter(config.print_freq)
    top5 = AverageMeter(config.print_freq)

    # switch to train mode
    model.train()

    world_size = link.get_world_size()
    rank = link.get_rank()

    logger = logging.getLogger('global_logger')

    end = time.time()

    label_smooth = config.get('label_smooth', 0.0)
    if label_smooth > 0:
        logger.info('using label_smooth: {}'.format(label_smooth))
        criterion = LabelSmoothCELoss(label_smooth, 1000)
    else:
        criterion = nn.CrossEntropyLoss()

    for i, (input, target, _) in enumerate(train_loader):
        curr_step = start_iter + i
        lr_scheduler.step(curr_step)
        current_lr = lr_scheduler.get_lr()[0]

        # measure data loading time
        data_time.update(time.time() - end)

        # transfer input to gpu
        target = target.cuda()
        input = input.cuda() if not args.fp16 else input.cuda().half()

        if config.augmentation.get('mix_up', False):
            # mixup
            alpha = config.augmentation.get('mix_up')
            input, targets_a, targets_b, lam = mixup_data(input, target, alpha)
            output = model(input)
            loss_func = mixup_criterion(targets_a, targets_b, lam)
            loss = loss_func(criterion, output) / world_size
            _prec1_a, _prec5_a = accuracy(output, targets_a, topk=(1, 5))
            _prec1_b, _prec5_b = accuracy(output, targets_b, topk=(1, 5))
            prec1 = lam*_prec1_a + (1-lam)*_prec1_b
            prec5 = lam*_prec5_a + (1-lam)*_prec5_b
        else:
            # normal
            output = model(input)
            loss = criterion(output, target) / world_size
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

        reduced_loss = loss.clone()
        reduced_prec1 = prec1.clone() / world_size
        reduced_prec5 = prec5.clone() / world_size

        link.allreduce(reduced_loss)
        link.allreduce(reduced_prec1)
        link.allreduce(reduced_prec5)

        losses.update(reduced_loss.item())
        top1.update(reduced_prec1.item())
        top5.update(reduced_prec5.item())

        # backward
        optimizer.zero_grad()

        if not args.fp16:
            loss.backward()
            # sync gradients
            reduce_gradients(model, args.sync)
            # update
            optimizer.step()
        else:
            def closure():
                # backward
                optimizer.backward(loss, False)
                # sync gradients
                reduce_gradients(model, args.sync)
                # check overflow, convert to fp32 grads, downscale
                optimizer.update_master_grads()
                return loss
            optimizer.step(closure)

        # measure elapsed time
        batch_time.update(time.time() - end)

        if curr_step % config.print_freq == 0 and rank == 0:
            tb_logger.add_scalar('loss_train', losses.avg, curr_step)
            tb_logger.add_scalar('acc1_train', top1.avg, curr_step)
            tb_logger.add_scalar('acc5_train', top5.avg, curr_step)
            tb_logger.add_scalar('lr', current_lr, curr_step)
            logger.info('Iter: [{0}/{1}] '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) '
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f}) '
                        'LR {lr:.6f}'.format(
                            curr_step, len(train_loader),
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses, top1=top1, top5=top5, lr=current_lr))

        if curr_step > 0 and curr_step % config.val_freq == 0:
            val_loss, prec1, prec5 = validate(val_loader, model)

            if tb_logger is not None:
                tb_logger.add_scalar('loss_val', val_loss, curr_step)
                tb_logger.add_scalar('acc1_val', prec1, curr_step)
                tb_logger.add_scalar('acc5_val', prec5, curr_step)

            if rank == 0:
                # remember best prec@1 and save checkpoint
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                save_checkpoint({
                    'step': curr_step,
                    'arch': config.model.backbone,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict(),
                }, is_best, config.save_path + '/ckpt')

        end = time.time()


def validate(val_loader, model, fusion_list=None, fuse_prob=False):
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)
    top1 = AverageMeter(0)
    top5 = AverageMeter(0)

    # switch to evaluate mode
    if fusion_list is not None:
        model_list = []
        for i in range(len(fusion_list)):
            model_list.append(model_entry(config.model))
            model_list[i].cuda()
            model_list[i] = DistModule(model_list[i], args.sync)
            load_state(fusion_list[i], model_list[i])
            model_list[i].eval()
        if fuse_prob:
            softmax = nn.Softmax(dim=1)
    else:
        model.eval()

    rank = link.get_rank()
    world_size = link.get_world_size()

    logger = logging.getLogger('global_logger')

    criterion = nn.CrossEntropyLoss()

    file_list = []
    result_list = []
    target_list = []

    end = time.time()
    with torch.no_grad():
        for i, (input, target, img_filename) in enumerate(val_loader):
            input = input.cuda() if not args.fp16 else input.half().cuda()
            target = target.cuda()
            # compute output
            if fusion_list is not None:
                output_list = []
                for model_idx in range(len(fusion_list)):
                    output = model_list[model_idx](input)
                    if fuse_prob:
                        output = softmax(output)
                    output_list.append(output)
                output = torch.stack(output_list, 0)
                output = torch.mean(output, 0)
            else:
                output = model(input)

            # save reference result
            result_list.append(output.data)
            target_list.append(target)
            file_list.append(img_filename)
            # save_result(result_list, target_list)

            # measure accuracy and record loss
            loss = criterion(output, target) / world_size
            # loss should not be scaled here, it's reduced later?
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            num = input.size(0)
            losses.update(loss.item(), num)
            top1.update(prec1.item(), num)
            top5.update(prec5.item(), num)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0 and rank == 0:
                logger.info('Test: [{0}/{1}]\tTime {batch_time.val:.3f}({batch_time.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time))

    save_result(file_list, result_list, target_list, rank, config.save_path)

    # gather final results
    total_num = torch.Tensor([losses.count])
    loss_sum = torch.Tensor([losses.avg * losses.count])
    top1_sum = torch.Tensor([top1.avg * top1.count])
    top5_sum = torch.Tensor([top5.avg * top5.count])
    link.allreduce(total_num)
    link.allreduce(loss_sum)
    link.allreduce(top1_sum)
    link.allreduce(top5_sum)
    final_loss = loss_sum.item() / total_num.item()
    final_top1 = top1_sum.item() / total_num.item()
    final_top5 = top5_sum.item() / total_num.item()

    if rank == 0:
        logger.info(' * Prec@1 {:.3f}\tPrec@5 {:.3f}\tLoss {:.3f}\ttotal_num \
            ={}'.format(final_top1, final_top5, final_loss, total_num.item()))

        # Process the reference txt
        commands = "sh " + config.save_path + "/merge_reference_result.sh"
        subprocess.call(commands, shell=True)

    model.train()

    return final_loss, final_top1, final_top5


if __name__ == '__main__':
    main()
