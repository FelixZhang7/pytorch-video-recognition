# coding:utf-8
from collections import defaultdict
import copy
import logging

logger = logging.getLogger('PRUNER')

from .utils.fake_module import FakeModule
from .utils.graph import Graph
from .utils.hooks import compute_rank_hook, bind_output_hook

import torch
import torch.nn as nn
import linklink as link
from .utils.trim import Trimmer
from .utils.trim_helper import _direct_prunable_type, _bn_type, remove_attr, create_conv, create_bn, create_fc
import numpy as np
from heapq import nsmallest
from operator import itemgetter
from .utils.disjoinset import DisJoinSet


def override_call():
    '''
        We managed to bind output.grad_fn to each module by overriding Module.__call__.
    '''
    torch.nn.Module.bind_grad_fn = FakeModule.bind_grad_fn
    FakeModule.backup_call = torch.nn.Module.__call__
    torch.nn.Module.__call__ = FakeModule.__call__
    torch.nn.Module.remove_hooks = FakeModule.remove_hooks
    torch.nn.Module.hook_handles = []


def rollback_call():
    if hasattr(torch.nn.Module, 'bind_grad_fn'):
        del torch.nn.Module.bind_grad_fn
    torch.nn.Module.__call__ = FakeModule.backup_call
    del torch.nn.Module.remove_hooks
    del torch.nn.Module.hook_handles


def is_depthwise(module):
    return module is not None and isinstance(module, _direct_prunable_type) \
        and module.groups == module.in_channels and module.groups == module.out_channels


def fixed_type(typename):
    return typename in ['Reshape', 'Slice', 'Index', 'Transpose', 'Expand']


def calc_flops(mod):
    flops = 0
    for output_shape in mod.output_shape:
        if isinstance(mod, _direct_prunable_type):
            one_channel_flops = 2 * output_shape[2] * output_shape[3] * (mod.in_channels * mod.kernel_size[0]**2 + 1)
            if is_depthwise(mod):
                flops += one_channel_flops
            else:
                flops += one_channel_flops * mod.out_channels
        elif isinstance(mod, nn.Linear):
            flops += 2 * (mod.weight.nelement() + mod.bias.nelement())
    return flops / len(mod.output_shape)


class Pruner(object):
    '''
        This pruner assumes that the model to be pruned is a static DAG,
        whose topology doesn't change through the whole traininig process.
    '''
    def __init__(self, model, distributed=False,
                 forbidden_list=[], flops_lambda=1e2, min_keep_ratio=0,
                 name_nchannel_dict={}, prune_conv_pre_fc=False):
        self.distributed = distributed
        self.forbidden_list = forbidden_list
        self.model = model
        self.graph = Graph()
        self.flops_lambda = flops_lambda

        assert min_keep_ratio == 0 or (min_keep_ratio > 0 and name_nchannel_dict != {}), \
            'when min_keep_ratio is not zero, please give name_nchannel_dict: {layername: output channels}'
        self.init_name_to_nchannels = copy.deepcopy(name_nchannel_dict)
        self.ratio = min_keep_ratio
        self.prune_conv_pre_fc = prune_conv_pre_fc
        self.grad_fn_to_module = {}
        self.module_to_name = {}
        self.name_to_module = {}
        self.channels_before_fc = defaultdict(int)

    def rebuild_model(self, batch, to_loss_fn, optimizer, prune_num,
                      cut_multi=1, trim_targets=None, param_group_func=None):
        name_to_nchannels = {}
        last_state = {}
        last_running_mean = {}
        last_running_var = {}
        channel_set = {}
        # TODO(yujie wang):
        # The BN is set to eval mode at initial implementation
        # But in new SyncBatchNorm2d implementation, the eval mode will return tensor without grad_fn
        # So we can't represent module by grad_fn, thus we set BN to be in train mode and recover it later
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Conv2d):
                name_to_nchannels[name] = mod.out_channels
            if isinstance(mod, _bn_type):
                last_state[name] = mod.training
                last_running_mean[name] = mod.running_mean.clone().detach()
                last_running_var[name] = mod.running_var.clone().detach()
                channel_set[name] = set([i for i in range(mod.num_features)])
                mod.train()
        self.name_to_nchannels = name_to_nchannels

        # step 1: add function to nn.Module
        override_call()

        # step 2: register hooks for save activation and compute rank
        self._register_hooks()

        # step 3: forward/backward for build graph, find prunable and compute rank(if register hooks)
        self._analyze_network(batch, to_loss_fn, optimizer)

        # step 4: get the pruned channels, given results by a dict {layer name: channel index(0-based)}
        modified = self.get_pruned(prune_num, cut_multi, trim_targets)

        # step 5: do prune
        self.prune(modified, optimizer, param_group_func)

        # step 5.1: calc flops
        sum_flops = 0
        name_flops = dict()
        for name, mod in self.model.named_modules():
            if isinstance(mod, _direct_prunable_type):
                flops = float(calc_flops(mod)) / pow(10, 9)
                name_flops[name] = flops
                sum_flops += flops

        # step 6: restore the model, clean all attributes we add to modules
        self.restore_model()
        rollback_call()
        for name, mod in self.model.named_modules():
            if isinstance(mod, _bn_type):
                mod.train(last_state[name])
                if name in modified:
                    removed_idx = modified[name][False]
                    remains_idx = channel_set[name] - set([i for i in range(len(removed_idx))])
                    remains_idx = list(remains_idx)
                    mod.running_mean = last_running_mean[name][remains_idx].clone().detach()
                    mod.running_var = last_running_var[name][remains_idx].clone().detach()
                else:
                    mod.running_mean = last_running_mean[name].clone().detach()
                    mod.running_var = last_running_var[name].clone().detach()
        return modified, sum_flops, name_flops

    def calc_layer_macc(self, input, verbose=False):
        # step 1: add function to nn.Module
        override_call()

        # step 2: register hooks for save activation and compute rank
        self._register_hooks()

        self.model(input)

        sum = 0
        name_flops = dict()
        for name, mod in self.model.named_modules():
            if isinstance(mod, _direct_prunable_type):
                flops = float(calc_flops(mod)) / pow(10, 9)
                name_flops[name] = flops
                sum += flops
                if verbose:
                    logger.info('{0} {1} {2}'.format(name, mod.out_channels, flops))
        if verbose:
            logger.info(sum)

        # step 6: restore the model, clean all attributes we add to modules
        self.restore_model()
        rollback_call()
        return sum, name_flops

    def _register_hooks(self):
        '''
            register hooks to compute ranks, which can be accessed by module.rank
        '''
        for name, module in self.model.named_modules():
            if isinstance(module, _direct_prunable_type):
                hooks_fb = []
                hooks_fb.append(module.register_forward_hook(bind_output_hook))
                hooks_fb.append(module.register_backward_hook(compute_rank_hook))
                module.hooks_fb = hooks_fb

    def _compress_path(self):
        '''
            compress path
            make conv layer, eltwise layer's dparent_list and dchild_list only contain
            convs layer, concate layer, reshape layer which don't include nolinear layer
        '''
        import copy
        node_list = self.graph.node_list
        dparent_list = copy.deepcopy(self.graph.parent_list)
        dchild_list = copy.deepcopy(self.graph.child_list)
        get_pytorch_type = self.graph.get_pytorch_type

        def isnolinear(typename, aux):
            # skip these types
            return typename in ['ReLU', 'Softmax', 'Pooling', 'Dropout', 'ReLU6', 'Sigmoid', 'BatchNorm',
                                'Concat', 'PSROIPooling', 'Interp'] + aux

        def backtrace(grad_fn, glist, aux=[]):
            n_id = node_list.index(grad_fn)
            typename = get_pytorch_type(grad_fn)
            atommod = True
            if grad_fn in self.grad_fn_to_module:
                module = self.grad_fn_to_module[grad_fn]
                if len(module._modules) != 0:
                    atommod = False
            if (not isnolinear(typename, aux) or len(glist[n_id]) == 0) and atommod:
                return [n_id]
            trace_list = []
            for g in glist[n_id]:
                t_list = backtrace(node_list[g], glist, aux=aux)
                for t in t_list:
                    trace_list.append(t)
            return trace_list

        def get_dlist(node_id, _list, aux=[]):
            d_list = []
            for node in _list[node_id]:
                t_list = backtrace(node_list[node], _list, aux=aux)
                for t in t_list:
                    d_list.append(t)
            return d_list

        if self.prune_conv_pre_fc:
            aux_ext = ['Reshape']
        else:
            aux_ext = []
        for name, module in self.model.named_modules():
            if isinstance(module, _direct_prunable_type):
                for grad_fn in module.grad_fn:
                    node_id = node_list.index(grad_fn)
                    dparent_list[node_id] = get_dlist(node_id, self.graph.parent_list, aux_ext)
                    dchild_list[node_id] = get_dlist(node_id, self.graph.child_list, aux_ext)
        for idx, node in enumerate(node_list):
            if get_pytorch_type(node) == 'Eltwise':
                dparent_list[idx] = get_dlist(idx, self.graph.parent_list, aux=['Eltwise'] + aux_ext)
                dchild_list[idx] = get_dlist(idx, self.graph.child_list, aux=['Eltwise'] + aux_ext)
        return dparent_list, dchild_list

    def _node_to_module(self, node):
        for name, mod in self.model.named_modules():
            if id(mod.grad_fn[0]) == id(node):
                return (name, mod)
        return None

    def _find_prunable(self):
        '''
            assign conv module prunable attribution
            assign conv module groupname attribution(means these layers need to be pruned at the same time)
        '''
        node_list = self.graph.node_list
        get_pytorch_type = self.graph.get_pytorch_type
        is_convolution = self.graph.is_convolution
        is_batchnorm = self.graph.is_batchnorm
        is_fc = self.graph.is_fc
        prunable_list = []
        # first step remove maxpool relu or other nolinear function
        dparent_list, dchild_list = self._compress_path()

        # TODO(wang yujie):
        # here need to be moved to somewhere
        # find channel numbers before fc to support cut fc's input channels
        def find_dconv(idx, plist, channels_before_fc, name, grad_fn_to_module):
            for pid in plist[idx]:
                if is_fc(node_list[pid]):
                    continue
                if get_pytorch_type(node_list[pid]) == 'Eltwise':
                    tmp = copy.deepcopy(plist)
                    tmp[pid] = [plist[pid][0]]
                    find_dconv(pid, tmp, channels_before_fc, name, grad_fn_to_module)
                    continue
                if is_convolution(node_list[pid]):
                    channels_before_fc[name] += grad_fn_to_module[node_list[pid]].out_channels
                    continue
                if is_batchnorm(node_list[pid]):
                    channels_before_fc[name] += grad_fn_to_module[node_list[pid]].num_features
                    continue
                find_dconv(pid, plist, channels_before_fc, name, grad_fn_to_module)

        self.channels_before_fc = defaultdict(int)
        if self.prune_conv_pre_fc:
            for name, mod in self.model.named_modules():
                if isinstance(mod, nn.Linear):
                    idx = self.graph.module_to_node_idx(mod)
                    find_dconv(idx, dparent_list, self.channels_before_fc, name, self.grad_fn_to_module)
        ####
        for name, module in self.model.named_modules():
            if isinstance(module, _direct_prunable_type) and name not in self.forbidden_list:
                if module.out_channels == 1:
                    module.prunable = False
                    continue
                for grad_fn in module.grad_fn:
                    node_id = node_list.index(grad_fn)
                    c_list = dchild_list[node_id]
                    module.prunable = all([True if not fixed_type(get_pytorch_type(node_list[idx]))
                                           else False for idx in c_list])
            else:
                module.prunable = False

        disjointset = DisJoinSet()

        # group share conv layer
        for name, module in self.model.named_modules():
            if isinstance(module, _direct_prunable_type) and len(module.grad_fn) > 1:
                convnames = []
                for grad_fn in module.grad_fn:
                    nid = node_list.index(grad_fn)
                    dparent = dparent_list[nid]
                    for parid in dparent:
                        par_grad_fn = node_list[parid]
                        for iname, imodule in self.model.named_modules():
                            if par_grad_fn in imodule.grad_fn:
                                convnames.append(iname)
                if (len(convnames) >= 2):
                    disjointset.append(convnames)

        # judge eltwise layer
        for idx, node in enumerate(node_list):
            module = self.grad_fn_to_module[node] if node in self.grad_fn_to_module else None
            if get_pytorch_type(node) == 'Eltwise' or is_depthwise(module):
                child_list = dchild_list[idx]
                parent_list = dparent_list[idx]
                prunable = True if module is None else module.prunable
                prunable = all([True if not fixed_type(get_pytorch_type(node_list[idx]))
                                else False for idx in child_list] + [prunable])
                prunable = all([self.grad_fn_to_module[node_list[idx]].prunable
                                for idx in parent_list if is_convolution(node_list[idx])] + [prunable])
                # remove other unprunable layer related with eltwise
                convs = []
                if module is not None:
                    for name, imodule in self.model.named_modules():
                        if id(module) == id(imodule):
                            convs = [name]
                            break
                for parent in parent_list:
                    if get_pytorch_type(node_list[parent]) == 'Convolution':
                        mod = self.grad_fn_to_module[node_list[parent]]
                        mod.prunable = prunable
                        convs.append(self.module_to_name[mod])
                disjointset.append(convs)

        # unprunable in the same group
        groups = disjointset.build_set()
        self.groups = groups

        # make sure same group conv's have same prunable/unprunable state
        for name, module in self.model.named_modules():
            if isinstance(module, _direct_prunable_type) and name in groups:
                prunable = module.prunable and self.name_to_module[groups[name]].prunable
                self.name_to_module[groups[name]].prunable = prunable

        for name, module in self.model.named_modules():
            if isinstance(module, _direct_prunable_type):
                if name in groups:
                    module.prunable = self.name_to_module[groups[name]].prunable
                if module.prunable:
                    prunable_list.append(name)
                    module.groupname = name
                    if name in groups:
                        module.groupname = groups[name]
        logger.info('There are #{}# modules are prunable:{}'.format(len(prunable_list), prunable_list))

    def _analyze_network(self, batch, to_loss_fn, optimizer):
        # step 1: forward
        loss = to_loss_fn(self.model, batch)

        # step 2: build graph by loss and find prunable
        self.graph.build_grad_graph(loss)
        self.grad_fn_to_module = {}
        self.module_to_name = {}
        self.name_to_module = {}
        for name, mod in self.model.named_modules():
            if hasattr(mod, 'grad_fn') and len(mod._modules) == 0 and len(mod._parameters) > 0:
                for grad_fn in mod.grad_fn:
                    assert grad_fn not in self.grad_fn_to_module.keys()
                    self.grad_fn_to_module[grad_fn] = mod
            self.module_to_name[mod] = name
            self.name_to_module[name] = mod
        self._find_prunable()

        # step 3: backward to compute rank
        optimizer.zero_grad()
        loss.backward()

    def _sync_rank(self):
        link.barrier()
        world_size = link.get_world_size()
        for name, mod in self.model.named_modules():
            if hasattr(mod, 'rank'):
                link.allreduce(mod.batch_size)
                link.allreduce(mod.rank)
                # For detection task, image size in different GPU is not the same,
                # so we have to sync them to avoid channel selection's difference
                sync_val = torch.cuda.LongTensor(4)
                for idx, output_shape in enumerate(mod.output_shape):
                    for i in range(4):
                        sync_val[i] = output_shape[i]
                    link.allreduce(sync_val)
                    for i in range(4):
                        mod.output_shape[idx][i] = sync_val[i] / world_size

    def _refine_rank(self):
        layer_flops = {}
        sum_flops = 0
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Conv2d):
                flops = float(calc_flops(mod))
                sum_flops += flops
                layer_flops[name] = flops

        for k, v in layer_flops.items():
            layer_flops[k] = float(v) / float(sum_flops)

        data = []
        groupset = []
        for name, module in self.model.named_modules():
            if isinstance(module, _direct_prunable_type) and module.prunable and module.groupname == name:
                groupset.append(name)

        for groupname in groupset:
            mod_group_dict = {}
            for name, module in self.model.named_modules():
                if isinstance(module, _direct_prunable_type) and module.prunable and module.groupname == groupname:
                    mod_group_dict[name] = module
            # correct the rank in a group, like residual block
            rank = 0
            for name, module in mod_group_dict.items():
                rank += module.rank
            for name, module in mod_group_dict.items():
                module.rank = rank

            # we let rank = 0 here, discard previous tensor, so the module.rank will not be 0
            rank = 0
            for name, module in mod_group_dict.items():
                v = module.rank.clone().cpu().numpy() / module.batch_size.cpu().numpy()[0]
                # l2 norm
                v = v / np.sqrt(np.sum(v * v))
                # flops
                v = v - self.flops_lambda * layer_flops[name]
                rank = rank + v / len(mod_group_dict)
            for i in range(rank.shape[0]):
                data.append((groupname, i, rank[i]))
        return data

    def _select_channels(self, prune_num, refined_rank, cut_multi=1):
        assert cut_multi >= 1
        # refined_rank: list of tuple(groupname, channel index, rank)
        # return trim_targets: dict {groupname: list(channel indexes)}
        trimitems = nsmallest(len(refined_rank), refined_rank, itemgetter(2))
        trim_candi = {}
        cnt = 0
        for name, idx, _ in trimitems:
            if self.ratio == 0:
                lower_bound = 1
            else:
                lower_bound = max(1, self.init_name_to_nchannels[name] * self.ratio)
            if cnt >= prune_num:
                break
            if self.name_to_nchannels[name] <= lower_bound:
                continue
            self.name_to_nchannels[name] -= 1
            cnt += 1
            if name not in trim_candi:
                trim_candi[name] = [idx]
            else:
                trim_candi[name].append(idx)
        if cut_multi == 1:
            return trim_candi
        # adjust pruned channel to be a multiple of cut_multi
        trim_targets = {}
        for name, idxes in trim_candi.items():
            if len(idxes) % cut_multi == 0:
                trim_targets[name] = idxes
                continue
            layer_ranks = []
            for tp in refined_rank:
                if tp[0] == name:
                    layer_ranks.append(tp)
            layer_tops = nsmallest(len(idxes) + cut_multi - len(idxes) % cut_multi, layer_ranks, itemgetter(2))
            trim_targets[name] = []
            for name, idx, _ in layer_tops:
                trim_targets[name].append(idx)
        return trim_targets

    def _analyze_dependence(self, trim_targets):
        trim = Trimmer(self.model, self.graph)
        modified = dict()
        processed_group = []
        for name, idx in trim_targets.items():
            if name in self.groups:
                name = self.groups[name]  # replace by representation of this group
                if name in processed_group:
                    continue
                processed_group.append(name)
            isok = False
            for iname, imodule in self.model.named_modules():
                if name == iname:
                    isok = True
                    break
            if not isok:
                logger.error('name mismatch {0}'.format(name))
                assert False, 'name mismatch {0}'.format(name)
            assert imodule.prunable
            assert name == imodule.groupname
            trim.rebuild_graph(name, imodule, idx, modified)
        return modified

    def get_pruned(self, prune_num=1, cut_multi=1, trim_targets=None):
        if self.distributed:
            self._sync_rank()
        if trim_targets is None or len(trim_targets) == 0:
            refined_rank = self._refine_rank()
            trim_targets = self._select_channels(prune_num, refined_rank, cut_multi)
        modified = self._analyze_dependence(trim_targets)
        return modified

    def _modify_optim(self, optimizer, mod):
        # del old param's state in optimizer's state dict
        # to avoid oom
        if optimizer is None:
            return
        if len(optimizer.state) == 0:
            return
        if len(mod._modules) > 0:
            logger.warn('remove a non-leaf node\'s optimizer state')
        for param in mod.parameters():
            if param not in optimizer.state:
                continue
            for group in optimizer.param_groups:
                for p in group['params']:
                    if id(param) == id(p):
                        optimizer.state.pop(p)
                        break

    def prune(self, modified, optimizer=None, param_group_func=None):
        for name, mod in modified.items():
            logger.info(name)
            root = self.model
            cur_name = name
            isok = False
            while not isok:
                sp = cur_name.split('.')
                for iname, imod in root.named_children():
                    if iname == sp[0] or iname == cur_name:
                        if len(sp) == 1 or iname == cur_name:
                            if isinstance(imod, _direct_prunable_type):
                                remove_in = mod[False]
                                remove_out = mod[True]
                                self._modify_optim(optimizer, root._modules[iname])
                                is_deconv = isinstance(imod, nn.ConvTranspose2d)
                                new_conv = create_conv(root._modules[iname], remove_in, remove_out, is_deconv)
                                root._modules[iname] = new_conv.cuda()
                            elif isinstance(imod, nn.Linear):
                                remove_in = mod[False]
                                remove_out = mod[True]
                                self._modify_optim(optimizer, root._modules[iname])
                                channel_num = self.channels_before_fc[name]
                                new_fc = create_fc(root._modules[iname], remove_in, remove_out, channel_num)
                                root._modules[iname] = new_fc.cuda()
                            elif isinstance(imod, _bn_type):
                                remove_index = mod[False]
                                self._modify_optim(optimizer, root._modules[iname])
                                new_bn = create_bn(root._modules[iname], remove_index)
                                root._modules[iname] = new_bn.cuda()
                            else:
                                raise NotImplementedError
                            isok = True
                            break
                        cur_name = '.'.join(sp[1:])
                        root = imod
                        break
        if optimizer is not None:
            if param_group_func is None:
                assert len(optimizer.param_groups) == 1, \
                    'if your model has multiple param_groups, \
                    please provide param_group_func to generate new param group'
                param_dict = copy.deepcopy(optimizer.param_groups[0])
                param_dict.update({'params': self.model.parameters()})
                optimizer.param_groups = []
                optimizer.add_param_group(param_dict)
            else:
                optimizer.param_groups = []
                all_param_groups = param_group_func(self.model)
                for param_group in all_param_groups:
                    optimizer.add_param_group(param_group)

    def restore_model(self):
        for name, mod in self.model.named_modules():
            remove_attr(mod)
        self.name_to_module.clear()
        self.module_to_name.clear()
        self.grad_fn_to_module.clear()
        self.channels_before_fc.clear()
