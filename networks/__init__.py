from .basics.resnet import *
from .basics.inception_v3 import *
from .basics.inception_v4 import *
from .basics.mobilenet_v2 import *
from .basics.senet import *
from .tsm.tsm import *
from .proxyless_nas import *
from .threeD import *


def backbone_entry(backbone, pretrained, **kwargs):
    if "3d" in backbone:
        return get_3d_backbone(backbone, pretrained, **kwargs)
    else:
        return globals()[backbone](pretrained, **kwargs)
