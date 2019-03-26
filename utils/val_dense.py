import multiprocessing as mp
mp.set_start_method('spawn', force=True)
import argparse
import os
import time
import yaml
import numpy
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

from models_ref import model_entry
from scheduler import get_scheduler
from memcached_dataset import McDataset
from utils import create_logger, AverageMeter, accuracy, accuracy_class,  \
    save_result, save_checkpoint, load_state, DistributedGivenIterationSampler,  \
    simple_group_split, DistributedSampler, param_group_no_wd
from distributed_utils import dist_init, reduce_gradients, DistModule
from loss import LabelSmoothCELoss


from tsn.tsn import TSN
from tsn.transforms import *
#model_names = sorted(name for name in models.__dict__
#    if name.islower() and not name.startswith("__")
#    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--config', default='cfgs/config_res50.yaml')
parser.add_argument('--load-path', default='', type=str)
parser.add_argument('--recover', action='store_true')
parser.add_argument('-e', '--evaluate', action='store_true')
parser.add_argument('--sync', action='store_true')
parser.add_argument('--fake', action='store_true')
parser.add_argument('--fuse-prob', action='store_true')
parser.add_argument('--fusion-list', nargs='+', help='multi model fusion list')
parser.add_argument('--fp16', action='store_true')
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
        bn_group = simple_group_split(world_size, rank, world_size // bn_group_size)

    config.model.kwargs.bn_group = bn_group
    config.model.kwargs.bn_var_mode = (link.syncbnVarMode_t.L1
                                       if bn_var_mode == 'L1'
                                       else link.syncbnVarMode_t.L2)

    model = TSN(config.model.num_class, config.model.num_segments*10,
                base_model=config.model.arch)


    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    #policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation()

    normalize = GroupNormalize(input_mean, input_std)

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
        model.half()

    model = DistModule(model, args.sync)


    if config.get('no_wd', False):
        param_group, type2num = param_group_no_wd(model)
        config.param_group_no_wd = type2num

        optimizer = torch.optim.SGD(param_group, config.lr_scheduler.base_lr,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay, nesterov=config.nesterov)
    else:

        optimizer = torch.optim.SGD(model.parameters(), config.lr_scheduler.base_lr,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay, nesterov=config.nesterov)

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
            best_prec1, last_iter = load_state(args.load_path, model, optimizer=optimizer)
        else:
            load_state(args.load_path, model)

    cudnn.benchmark = True


    val_aug = []
    #val_aug.append(GroupScale(int(scale_size)))
    #val_aug.append(GroupCenterCrop(crop_size))
    val_aug.append(GroupOverSample(crop_size, scale_size))
    val_aug.append(Stack(roll=False))
    val_aug.append(ToTorchFormatTensor(div=False))
    val_aug.append(normalize)


    # val
    val_dataset = McDataset(
        config.val_root,
        config.val_source,
        random_shift=False,
        transform=transforms.Compose(val_aug))

    val_sampler = DistributedSampler(val_dataset, round_up=False)


    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.workers, pin_memory=True, sampler=val_sampler)

    config.lr_scheduler['optimizer'] = optimizer if not args.fp16 else optimizer.optimizer
    config.lr_scheduler['last_iter'] = last_iter
    lr_scheduler = get_scheduler(config.lr_scheduler)

    if rank == 0:
        tb_logger = SummaryWriter(config.save_path+'/events')
        logger = create_logger('global_logger', config.save_path+'/log.txt')
        logger.info('args: {}'.format(pprint.pformat(args)))
        logger.info('config: {}'.format(pprint.pformat(config)))
    else:
        tb_logger = None

    if args.evaluate:
        if args.fusion_list is not None:
            validate(val_loader, model, fusion_list=args.fusion_list, fuse_prob=args.fuse_prob)
        else:
            validate(val_loader, model)
        link.finalize()
        return


    link.finalize()

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

    res_cls = [0]*20

    # Add by sgx
    file_list = []
    result_list = []
    target_list = []

    end = time.time()
    with torch.no_grad():
        for i, (input, target, img_filename) in enumerate(val_loader):
            # # DEBUG
            # print ("i is {}".format(i))
            # print (img_filename)
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

            # # DEBUG
            # # shape of output is torch.Size([16, 155])
            print("shape of output is {}".format(output.shape))
            print("shape of target is {}".format(target.shape))

            # added by sgx
            # save reference result
            result_list.append(output.data)
            target_list.append(target)
            file_list.append(img_filename)
            #save_result(result_list, target_list)

            # measure accuracy and record loss
            loss = criterion(output, target)
            #/ world_size ## loss should not be scaled here, it's reduced later!
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            res_cls_tmp = accuracy_class(output.data, target, topk=(1,))
            res_cls = list(map(lambda x : x[0]+x[1], zip(res_cls, res_cls_tmp)))

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
    loss_sum = torch.Tensor([losses.avg*losses.count])
    top1_sum = torch.Tensor([top1.avg*top1.count])
    top5_sum = torch.Tensor([top5.avg*top5.count])
    link.allreduce(total_num)
    link.allreduce(loss_sum)
    link.allreduce(top1_sum)
    link.allreduce(top5_sum)
    final_loss = loss_sum.item()/total_num.item()
    final_top1 = top1_sum.item()/total_num.item()
    final_top5 = top5_sum.item()/total_num.item()

    if rank == 0:
        logger.info(' * Prec@1 {:.3f}\tPrec@5 {:.3f}\tLoss {:.3f}\ttotal_num \
            ={}'.format(final_top1, final_top5, final_loss, total_num.item()))

        #### Add by sgx #####
        # Process the reference txt
        commands = "sh "+config.save_path+"/merge_reference_result.sh"
        subprocess.call(commands, shell = True)

    model.train()

    return final_loss, final_top1, final_top5

if __name__ == '__main__':
    main()
