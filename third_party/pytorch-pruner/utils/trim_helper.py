import torch
import torch.nn as nn
import copy
support_syncbn = True
try:
    from linklink.nn import SyncBatchNorm2d
except Exception:
    support_syncbn = False

_direct_prunable_type = (nn.Conv2d, nn.ConvTranspose2d)
if support_syncbn:
    _bn_type = (nn.BatchNorm2d, SyncBatchNorm2d)
else:
    _bn_type = (nn.BatchNorm2d, nn.BatchNorm2d)


def remove_attr(module):
    # remove hook
    if hasattr(module, 'remove_hooks'):
        module.remove_hooks()
    # if hasattr(module, 'hook_handles'):
    #    module.__delattr__('hook_handles')
    if hasattr(module, 'hooks_fb'):
        module.hooks_fb.clear()
        module.__delattr__('hooks_fb')
    # remove attr
    if hasattr(module, 'grad_fn'):
        module.grad_fn.clear()
        module.__delattr__('grad_fn')
    if hasattr(module, 'output'):
        module.output.clear()
        module.__delattr__('output')
    if hasattr(module, 'output_shape'):
        module.output_shape.clear()
        module.__delattr__('output_shape')

    if hasattr(module, 'rank'):
        module._buffers.pop('rank')
    if hasattr(module, 'batch_size'):
        module._buffers.pop('batch_size')
    if hasattr(module, 'prunable'):
        module.__delattr__('prunable')
    if hasattr(module, 'groupname'):
        module.__delattr__('groupname')


def copyAttr(old_module, new_module, except_name=[]):
    '''
        copy module's buffer, parameter and attribution
    '''
    name_list = []
    # buffer
    for name, buffer in old_module._buffers.items():
        if name in except_name or name in new_module._buffers.keys():
            continue
        name_list.append(name)
        new_module.register_buffer(name, buffer.clone())
    # parameter
    for name, param in old_module._parameters.items():
        if name in except_name or name in new_module._parameters.keys():
            continue
        name_list.append(name)
        new_module.register_parameter(name, param.clone())
    # attributes
    name_list.extend(except_name)
    for name in old_module.__dir__():
        if name in name_list or name.startswith('_') or callable(getattr(old_module, name)):
            continue
        new_module.__setattr__(name, getattr(old_module, name))


def getAttr(model, name):
    return findModules(model, name)


def splitByModel(model, names):
    '''
        for example: names@layer1.0.conv can index a specific module,
            such as model._modules['layer1']._modules['0']._modules['conv']

        but there exists counter example such as in densenet, names@layer.norm.1 need to index a specific module
        by model._modules['layer']._modules['norm.1']

        this function returns an array which split names according model's attribute.
    '''
    if not isinstance(names, list):
        names = names.split('.')
    modulename = []
    conv = model
    index = 0
    while index < len(names):
        lastindex = index
        for i in range(len(names) - index):
            name = '.'.join(names[index:index + i + 1])
            if name in conv._modules:
                conv = conv._modules[name]
                index = index + i + 1
                modulename.append(name)
                break
        if lastindex == index:
            raise RuntimeError('Module Index Error')
    return modulename


def findModules(model, layer_name):
    '''
        index module according to layer_name
    '''
    if isinstance(layer_name, list):
        nests = layer_name
    else:
        nests = layer_name.split(".")
    conv = model
    for i in range(len(nests)):
        name = '.'.join(nests[i:])
        if name in conv._modules:
            return conv._modules[name]
        elif nests[i] in conv._modules:
            conv = conv._modules[nests[i]]
        else:
            raise RuntimeError('Module Index Error')
    return conv


def create_conv(conv, remove_in, remove_out, deconv=False):
    '''
        remove redundant conv's channels by remove_in, remove_out
    '''
    if len(remove_in) == 0 and len(remove_out) == 0:
        return conv

    if len(remove_in) >= conv.in_channels:
        remove_in = remove_in[:-(len(remove_in) - conv.in_channels + 1)]
    if len(remove_out) >= conv.out_channels:
        remove_out = remove_out[:-(len(remove_out) - conv.out_channels + 1)]

    filter_in = [i for i in range(conv.in_channels) if i not in remove_in]
    filter_out = [i for i in range(conv.out_channels) if i not in remove_out]

    weights_np = conv.weight.data.cpu().numpy()
    if not (conv.bias is None):
        bias_np = conv.bias.data.cpu().numpy()

    # support for depth-wise conv
    if conv.in_channels == conv.groups and conv.in_channels == conv.out_channels:
        assert set(remove_in) == set(remove_out)
        new_weights_np = weights_np[filter_out, :, :, :]
        groups = len(filter_in)
    else:
        if deconv:
            new_weights_np = weights_np[filter_in, :, :, :]
            new_weights_np = new_weights_np[:, filter_out, :, :]
        else:
            new_weights_np = weights_np[filter_out, :, :, :]
            new_weights_np = new_weights_np[:, filter_in, :, :]
        groups = 1

    if not (conv.bias is None):
        new_bias_np = bias_np[filter_out]
    conv_type = type(conv)
    ret = conv_type(in_channels=len(filter_in), out_channels=len(filter_out), kernel_size=conv.kernel_size,
                    stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=groups,
                    bias=conv.bias is not None)
    ret.weight.data = torch.from_numpy(new_weights_np).cuda()
    if not (conv.bias is None):
        ret.bias.data = torch.from_numpy(new_bias_np).cuda()
    copyAttr(conv, ret, except_name=['weight', 'bias', 'in_channels', 'out_channels', 'kernel_size',
                                     'stride', 'padding', 'dilation', 'groups', 'bias'])
    return ret


def create_fc(fc, remove_in, remove_out, channel_num=1, expand=True):
    '''
        remove redundant fc's channels by remove_in, remove_out
    '''
    if len(remove_in) == 0 and len(remove_out) == 0:
        return fc

    assert fc.in_features % channel_num == 0
    if expand is True:
        remove_len = fc.in_features // channel_num
        tmp = copy.deepcopy(remove_in)
        remove_in = []
        for v in tmp:
            for i in range(0, remove_len):
                remove_in.append(v * remove_len + i)

    if len(remove_in) >= fc.in_features:
        remove_in = remove_in[:-remove_len]
    if len(remove_out) >= fc.out_features:
        remove_out = remove_out[:-(len(remove_out) - fc.out_features + 1)]

    in_set = [i for i in range(fc.in_features)]
    out_set = [i for i in range(fc.out_features)]
    in_set = set(in_set) - set(remove_in)
    out_set = set(out_set) - set(remove_out)
    filter_in = list(in_set)
    filter_out = list(out_set)

    weights_np = fc.weight.data.cpu().numpy()
    if not (fc.bias is None):
        bias_np = fc.bias.data.cpu().numpy()

    new_weights_np = weights_np[filter_out, :]
    new_weights_np = new_weights_np[:, filter_in]

    if not (fc.bias is None):
        new_bias_np = bias_np[filter_out]

    fc_type = type(fc)
    ret = fc_type(in_features=len(filter_in), out_features=len(filter_out))
    ret.weight.data = torch.from_numpy(new_weights_np).cuda()
    if not (fc.bias is None):
        ret.bias.data = torch.from_numpy(new_bias_np).cuda()
    copyAttr(fc, ret, except_name=['weight', 'bias', 'in_features', 'out_features'])
    return ret


def create_bn(bn, remove_index):
    '''
        remove redundant bn's channel by remove_index
    '''
    if len(remove_index) >= bn.num_features:
        remove_index = remove_index[:-(len(remove_index) - bn.num_features + 1)]
    delta = len(remove_index)
    if delta == 0:
        return bn
    filter_index = [i for i in range(bn.num_features) if i not in remove_index]
    means = bn.running_mean.cpu().numpy()
    vares = bn.running_var.cpu().numpy()
    # means and vars are not variable
    new_means = means[filter_index]
    new_vars = vares[filter_index]
    if bn.affine:
        weights_np = bn.weight.data.cpu().numpy()
        bias_np = bn.bias.data.cpu().numpy()

        new_weights_np = weights_np[filter_index]
        new_bias_np = bias_np[filter_index]

    # SyncBN has no attribute named 'track_running_stats', when batchnorm in 0.4.1 have,
    # so we don't assign it here, but assign in copyAttr
    bn_type = type(bn)
    ret = bn_type(num_features=len(new_means), eps=bn.eps, momentum=bn.momentum,
                  affine=bn.affine)

    ret.running_mean = torch.from_numpy(new_means).cuda()
    ret.running_var = torch.from_numpy(new_vars).cuda()
    if bn.affine:
        ret.weight.data = torch.from_numpy(new_weights_np).cuda()
        ret.bias.data = torch.from_numpy(new_bias_np).cuda()
    copyAttr(bn, ret, except_name=['running_mean', 'running_var', 'weight', 'bias',
                                   'num_features', 'momentum', 'affine', 'track_running_stats'])
    return ret


def deleteAttr(model, names):
    if len(names) == 1:
        del model._modules[names[0]]
    else:
        deleteAttr(model._modules[names[0]], names[1:])


def setAttr(model, names, newInstance):
    if len(names) == 1:
        model._modules[names[0]] = newInstance
    else:
        setAttr(model._modules[names[0]], names[1:], newInstance)
