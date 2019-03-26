from collections import OrderedDict
import os
import torch
import torch.nn as nn
from torch.nn import init

from linklink.nn import SyncBatchNorm2d, syncbnVarMode_t

__all__ = ['mobilenetv2']

model_zoo_path = '/mnt/lustre/share/spring/model_zoo'

model_paths = {
    'mobilenetv2': 'mobilenetv2_t6_batch2k_epoch150_colorjitter_0.2_0.2_0.2_0.1_nesterov_labelsmooth0.1_nowd_coslr_wd4e-5',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class LinearBottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, t=6, activation=nn.ReLU6):
        super(LinearBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes * t, kernel_size=1, bias=False)
        self.bn1 = BN(inplanes * t)  # nn.BatchNorm2d(inplanes * t)
        self.conv2 = nn.Conv2d(inplanes * t, inplanes * t, kernel_size=3, stride=stride, padding=1, bias=False,
                               groups=inplanes * t)
        self.bn2 = BN(inplanes * t)  # nn.BatchNorm2d(inplanes * t)
        self.conv3 = nn.Conv2d(inplanes * t, outplanes, kernel_size=1, bias=False)
        self.bn3 = BN(outplanes)  # nn.BatchNorm2d(outplanes)
        self.activation = activation(inplace=True)
        self.stride = stride
        self.t = t
        self.inplanes = inplanes
        self.outplanes = outplanes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.stride == 1 and self.inplanes == self.outplanes:
            out += residual

        return out


class MobileNet2(nn.Module):
    """MobileNet2 implementation.
    """

    # def __init__(self, scale=1.0, input_size=224, t=2, in_channels=3, num_classes=1000, activation=nn.ReLU6,  bn_group_size=1, bn_group=None, bn_sync_stats=False):
    def __init__(self, scale=1.0, input_size=224, t=6, in_channels=3, num_classes=1000, activation=nn.ReLU,
                 bn_group_size=1, bn_group=None, bn_var_mode=syncbnVarMode_t.L2, bn_sync_stats=False,
                 use_sync_bn=True):
        """
        MobileNet2 constructor.
        :param in_channels: (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
        :param input_size:
        :param num_classes: number of classes to predict. Default
                is 1000 for ImageNet.
        :param scale:
        :param t:
        :param activation:
        """

        super(MobileNet2, self).__init__()

        global BN

        print("scale for mobilenetv2 is {}".format(scale))
        print(bn_group_size, bn_group, bn_var_mode, bn_sync_stats)

        def BNFunc(*args, **kwargs):
            return SyncBatchNorm2d(*args, **kwargs, group=bn_group, sync_stats=bn_sync_stats, var_mode=bn_var_mode)

        if use_sync_bn:
            BN = BNFunc
        else:
            BN = nn.BatchNorm2d

        self.scale = scale
        self.t = t
        self.activation_type = activation
        self.activation = activation(inplace=True)
        self.num_classes = num_classes

        self.num_of_channels = [32, 16, 24, 32, 64, 96, 160, 320]
        assert (input_size % 32 == 0)

        self.c = [_make_divisible(ch * self.scale, 8) for ch in self.num_of_channels]
        self.n = [1, 1, 2, 3, 4, 3, 3, 1]
        self.s = [2, 1, 2, 2, 2, 1, 2, 1]
        self.conv1 = nn.Conv2d(in_channels, self.c[0], kernel_size=3, bias=False, stride=self.s[0], padding=1)
        self.bn1 = BN(self.c[0])  # nn.BatchNorm2d(self.c[0])
        self.bottlenecks = self._make_bottlenecks()

        # Last convolution has 1280 output channels for scale <= 1
        self.last_conv_out_ch = 1280 if self.scale <= 1 else _make_divisible(1280 * self.scale, 8)
        self.conv_last = nn.Conv2d(self.c[-1], self.last_conv_out_ch, kernel_size=1, bias=False)
        self.bn_last = BN(self.last_conv_out_ch) # nn.BatchNorm2d(self.last_conv_out_ch)
        self.avgpool = nn.AvgPool2d(int(input_size // 32))
        # self.dropout = nn.Dropout(p=0.2, inplace=True)  # confirmed by paper authors
        # self.fc = nn.Linear(self.last_conv_out_ch, self.num_classes)
        self.fc = nn.Conv2d(self.last_conv_out_ch, self.num_classes, kernel_size=1)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, SyncBatchNorm2d) or isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_stage(self, inplanes, outplanes, n, stride, t, stage):
        modules = OrderedDict()
        stage_name = "LinearBottleneck{}".format(stage)

        # First module is the only one utilizing stride
        first_module = LinearBottleneck(inplanes=inplanes, outplanes=outplanes, stride=stride, t=t,
                                        activation=self.activation_type)
        modules[stage_name + "_0"] = first_module

        # add more LinearBottleneck depending on number of repeats
        for i in range(n - 1):
            name = stage_name + "_{}".format(i + 1)
            module = LinearBottleneck(inplanes=outplanes, outplanes=outplanes, stride=1, t=6,
                                      activation=self.activation_type)
            modules[name] = module

        return nn.Sequential(modules)

    def _make_bottlenecks(self):
        modules = OrderedDict()
        stage_name = "Bottlenecks"

        # First module is the only one with t=1
        bottleneck1 = self._make_stage(inplanes=self.c[0], outplanes=self.c[1], n=self.n[1], stride=self.s[1], t=1,
                                       stage=0)
        modules[stage_name + "_0"] = bottleneck1

        # add more LinearBottleneck depending on number of repeats
        for i in range(1, len(self.c) - 1):
            name = stage_name + "_{}".format(i)
            module = self._make_stage(inplanes=self.c[i], outplanes=self.c[i + 1], n=self.n[i + 1],
                                      stride=self.s[i + 1],
                                      t=self.t, stage=i)
            modules[name] = module

        return nn.Sequential(modules)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.bottlenecks(x)
        x = self.conv_last(x)
        x = self.bn_last(x)
        x = self.activation(x)

        # average pooling layer
        x = self.avgpool(x)
        # x = self.dropout(x)

        x = self.fc(x)
        # flatten for input to fully-connected layer
        x = x.view(x.size(0), -1)
        return x
        # return F.log_softmax(x, dim=1)


def load_pretrained_model(model, pretrained):
    """
        Load distributed model parametters.
    """
    # model.load_state_dict(pre['state_dict'])
    print("Loading pretrained model!!!")
    _model_dict = model.state_dict()
    for name, param in pretrained.items():
        name_sp = name.split('.')
        name_pre = '.'.join(name_sp[1:])
        if name_pre not in _model_dict:
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        _model_dict[name_pre].copy_(param)
    return


def mobilenetv2(pretrained=False, **kwargs):
    model = MobileNet2(**kwargs)
    if pretrained:
        # load_pretrained_model
        _pretrained = torch.load(
            os.path.join(model_zoo_path,
                         model_paths['mobilenetv2'],
                         'ckpt_best.pth.tar'))
        load_pretrained_model(model, _pretrained['state_dict'])
    return model
