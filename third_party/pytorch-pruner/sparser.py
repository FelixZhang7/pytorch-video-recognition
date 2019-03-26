# coding:utf-8
import logging
from collections import defaultdict

logger = logging.getLogger('PRUNER')

import torch
import torch.nn as nn
from .pruner import Pruner, override_call, rollback_call

exist_syncbn = True
try:
    from linklink.nn import SyncBatchNorm2d
except Exception:
    exist_syncbn = False


class BNSparser(object):
    def __init__(self, model, l1_weight, fake_input, fake_target, to_loss_fn, optimizer, **kwargs):
        self.pruner = Pruner(model, **kwargs)
        self.bn_list = []
        self.conv2bn = None
        self.bn2conv = None
        self.l1_weight = l1_weight
        self.fake_input = fake_input
        self.fake_target = fake_target
        self.to_loss_fn = to_loss_fn
        self.optimizer = optimizer
        self._init_bn_sparser()

    def _get_conv_groups(self):
        self.pruner._analyze_network(self.fake_input, self.fake_target, self.to_loss_fn, self.optimizer)
        model = self.pruner.model
        groups = defaultdict(list)
        for name, module in model.named_modules():
            if module.prunable:
                groups[module.groupname].append(name)
        return groups

    def _init_bn_sparser(self):
        override_call()
        pruner = self.pruner
        self.convgroups = self._get_conv_groups()
        self.conv2bn = {}
        name_to_module = pruner.name_to_module

        for convname in self.convgroups.keys():
            trimtargets = {convname: [0]}
            modified = pruner._analyze_dependence(trimtargets)
            for name, mod in modified.items():
                bnmod = name_to_module[name]
                if isinstance(bnmod, torch.nn.BatchNorm2d) or (exist_syncbn and isinstance(bnmod, SyncBatchNorm2d)):
                    start = mod[False][0]
                    conv = name_to_module[convname]
                    assert isinstance(conv, torch.nn.Conv2d)
                    if convname not in self.conv2bn:
                        self.conv2bn[convname] = []
                    self.conv2bn[convname].append((name, start, start + conv.out_channels))

        self.bn2lasso = {}
        self.bn_list = []
        for convname, triads in self.conv2bn.items():
            bnname = None
            for idx, triad in enumerate(triads):
                bnname, start, end = triad
                self.bn2lasso[bnname] = (convname, start, end)
                logger.info('sparsity bn layer for {} whose index is from {} to {}'.format(bnname, start, end))
            # TODO: don't know how to choose at the present
            if bnname is not None:  # random select a bn layer
                self.bn_list.append(bnname)
        self.pruner.restore_model()
        rollback_call()

    def updateBN(self):
        assert self.bn2lasso is not None
        assert self.conv2bn is not None
        model = self.pruner.model
        for name, module in model.named_modules():
            if name in self.conv2bn:
                triads = self.conv2bn[name]
                for idx, triad in enumerate(triads):
                    bnname, start, end = triad
                    for iname, imodule in model.named_modules():
                        if iname == bnname:
                            assert isinstance(imodule, torch.nn.BatchNorm2d) or \
                                (exist_syncbn and isinstance(imodule, SyncBatchNorm2d))
                            imodule.weight.grad.data[start:end].add_(
                                self.l1_weight * torch.sign(imodule.weight.data[start:end]))
                            break

    def trim_model(self, threshold=-1, percent=-1):
        trimtargets = self._get_trim_targets(threshold, percent)
        self.pruner.rebuild_model(self.fake_input, self.fake_target, self.to_loss_fn,
                                  self.optimizer, 1, trimtargets=trimtargets)
        return self.pruner.model

    def count_channels(self):
        sum = 0
        for mod in self.pruner.model.modules():
            if isinstance(mod, nn.Conv2d):
                sum += mod.out_channels
        return sum

    def _get_trim_targets(self, threshold, percent):
        assert self.conv2bn is not None
        assert threshold <= 0 or percent <= 0
        assert not(threshold <= 0 and percent <= 0)
        model = self.pruner.model
        trimtargets = {}
        # step 1: calcuate the bn scale threshold
        if percent > 0:
            scalelist = []
            for name, module in model.named_modules():
                if name in self.bn_list or 'module.' + name in self.bn_list:
                    if isinstance(module, torch.nn.BatchNorm2d) or \
                       (exist_syncbn and isinstance(module, SyncBatchNorm2d)):
                        scalelist += torch.abs(module.weight.data.cpu()).numpy().tolist()
            scalelist = sorted(scalelist)
            threshold = scalelist[int(len(scalelist) * percent)]
            del scalelist

        # step 2: inference the trimtargets
        for name, module in model.named_modules():
                if name in self.bn_list or 'module.' + name in self.bn_list:
                    if isinstance(module, torch.nn.BatchNorm2d) or \
                       (exist_syncbn and isinstance(module, SyncBatchNorm2d)):
                        weight = torch.abs(module.weight.data.cpu()).numpy()
                        for idx in range(weight.shape[0]):
                            if weight[idx] <= threshold:
                                convname, offset = self._index2conv(name, idx)
                                if convname not in trimtargets:
                                    trimtargets[convname] = []
                                trimtargets[convname].append(idx + offset)

        return trimtargets

    def _index2conv(self, querybn, index):
        for convname, triads in self.conv2bn.items():
            for idx, triad in enumerate(triads):
                keybn, start, end = triad
                if keybn == querybn and index < end and index >= start:
                    return convname, -start
        assert False
        return convname, -start
