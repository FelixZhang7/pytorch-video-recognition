from .tsn.tsn import get_tsn
from .tsm.tsm import get_tsm
from .slowfast.slowfast import get_slowfast

# config is config.yamal:[common.model]
# model type ['tsn', 'tsm', 'classify']


def model_entry(config):
    _type = config['type']
    if _type == 'tsn':
        return get_tsn(config)
    elif _type == 'tsm':
        return get_tsm(config)
    elif _type == 'slowfast':
        return get_slowfast(config)
    elif _type == 'classify':
        from networks import backbone_entry
        backbone = config['backbone']
        pretrained = config['pretrained']
        return backbone_entry(backbone, pretrained,
                              num_classes=config['num_class'],
                              **config['kwargs'])
    else:
        raise NameError("The model {} is not Implemented".format(_type))
