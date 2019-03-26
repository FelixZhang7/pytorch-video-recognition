import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from .trim_helper import _direct_prunable_type, _bn_type, create_conv, create_fc, create_bn

logger = logging.getLogger('PRUNER')


def find_module_by_name(module, name_split):
    if len(name_split) == 0:
        return module
    for name, mod in module.named_children():
        if name_split[0] == name:
            return find_module_by_name(mod, name_split[1:])


def replace_module_by_name(module, name_split, candidate):
    for name, mod in module.named_children():
        if name_split[0] == name:
            if len(name_split) == 1:
                module._modules[name] = candidate
                return
            else:
                return replace_module_by_name(mod, name_split[1:], candidate)


def save_model_def(model):
    def_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, _bn_type):
            bn = module
            def_dict[name] = {'num_features': bn.num_features
                              }
        elif isinstance(module, _direct_prunable_type):
            conv = module
            def_dict[name] = {'in_channels': conv.in_channels,
                              'out_channels': conv.out_channels,
                              }
        elif isinstance(module, nn.Linear):
            fc = module
            def_dict[name] = {'in_features': fc.in_features,
                              'out_features': fc.out_features
                              }
    return def_dict


def load_model_def(model, ckpt_obj, load_state=True):
    if isinstance(ckpt_obj, str):
        def map_func(storage, location):
            return storage.cuda()
        ckpt = torch.load(ckpt_obj, map_location=map_func)
    else:
        ckpt = ckpt_obj

    model_def = ckpt['model_def']
    has_prefix = False
    for name, mod in model.named_modules():
        if name.startswith('module'):
            has_prefix = True
            break
    for name, param in model_def.items():
        if has_prefix and not name.startswith('module'):
            name = 'module.' + name
        if not has_prefix and name.startswith('module'):
            name = name[7:]
        mod = find_module_by_name(model, name.split('.'))
        if isinstance(mod, _direct_prunable_type):
            is_deconv = isinstance(mod, nn.ConvTranspose2d)
            remove_in = [i for i in range(mod.in_channels - param['in_channels'])]
            remove_out = [i for i in range(mod.out_channels - param['out_channels'])]
            new_conv = create_conv(mod, remove_in, remove_out, deconv=is_deconv)
            replace_module_by_name(model, name.split('.'), new_conv)
        elif isinstance(mod, _bn_type):
            remove_index = [i for i in range(mod.num_features - param['num_features'])]
            new_bn = create_bn(mod, remove_index)
            replace_module_by_name(model, name.split('.'), new_bn)
        elif isinstance(mod, nn.Linear):
            remove_in = [i for i in range(mod.in_features - param['in_features'])]
            remove_out = [i for i in range(mod.out_features - param['out_features'])]
            new_fc = create_fc(mod, remove_in, remove_out, 1, False)
            replace_module_by_name(model, name.split('.'), new_fc)
        else:
            logger.warning('model_def {0} not found in current model'.format(name))

    if load_state is True:
        assert 'state_dict' in ckpt.keys(), 'Need load state dict, but it don''t has keyword state_dict'
        try:
            model.load_state_dict(ckpt['state_dict'], strict=True)
            logger.info('All keys of the model has been loaded')
        except Exception:
            fixed_state_dict = OrderedDict()
            for name, param in ckpt['state_dict'].items():
                if has_prefix and not name.startswith('module'):
                    name = 'module.' + name
                if not has_prefix and name.startswith('module'):
                    name = name[7:]
                fixed_state_dict[name] = param
            model.load_state_dict(fixed_state_dict, strict=False)
            model_keys = set(model.state_dict().keys())
            ckpt_keys = set(fixed_state_dict.keys())
            logger.warning('Missing keys to load to model {0}'.format(model_keys - ckpt_keys))
