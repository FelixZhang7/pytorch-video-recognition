import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial

from linklink.nn import SyncBatchNorm2d
from linklink.nn import syncbnVarMode_t


__all__ = ['resnet50_slowfast', 'resnet101_slowfast',
           'resnet152_slowfast', 'resnet200_slowfast']


BN = None


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


def bn3d_bn2d(x):
    N, C, T, H, W = x.size()
    x = x.view(N, C, T*H, W)
    return x, T


def bn2d_bn3d(x, frames):
    N, C, TH, W = x.size()
    x = x.view(N, C, frames, int(TH / frames), W)
    return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, head_conv=1):
        super(Bottleneck, self).__init__()
        if head_conv == 1:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = BN(planes)
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(
                3, 1, 1), bias=False, padding=(1, 0, 0))
            self.bn1 = BN(planes)
        else:
            raise ValueError("Unsupported head_conv!")
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1), bias=False)
        self.bn2 = BN(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BN(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out, t = bn3d_bn2d(out)
        out = self.bn1(out)
        out = bn2d_bn3d(out, t)
        out = self.relu(out)

        out = self.conv2(out)
        out, t = bn3d_bn2d(out)
        out = self.bn2(out)
        out = bn2d_bn3d(out, t)
        out = self.relu(out)

        out = self.conv3(out)
        out, t = bn3d_bn2d(out)
        out = self.bn3(out)
        out = bn2d_bn3d(out, t)

        if self.downsample is not None:
            for ll in self.downsample:
                if type(ll) == SyncBatchNorm2d:
                    x, t = bn3d_bn2d(x)
                    x = ll(x)
                    x = bn2d_bn3d(x, t)
                x = ll(x)
            #residual = self.downsample(x)
            residual = x

        out += residual
        out = self.relu(out)

        return out


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


class SlowFast(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], class_num=10, shortcut_type='B', dropout=0.5,
                 bn_group_size=1, bn_group=None, bn_var_mode=syncbnVarMode_t.L2, bn_sync_stats=False):

        super(SlowFast, self).__init__()

        global BN

        def BNFunc(*args, **kwargs):
            return SyncBatchNorm2d(*args, **kwargs, group=bn_group, sync_stats=bn_sync_stats, var_mode=bn_var_mode)

        BN = BNFunc

        self.fast_inplanes = 8
        self.fast_conv1 = nn.Conv3d(3, 8, kernel_size=(
            5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.fast_bn1 = BN(8)
        self.fast_relu = nn.ReLU(inplace=True)
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(
            1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.fast_res1 = self._make_layer_fast(
            block, 8, layers[0], shortcut_type, head_conv=3)
        self.fast_res2 = self._make_layer_fast(
            block, 16, layers[1], shortcut_type, stride=2, head_conv=3)
        self.fast_res3 = self._make_layer_fast(
            block, 32, layers[2], shortcut_type, stride=2, head_conv=3)
        self.fast_res4 = self._make_layer_fast(
            block, 64, layers[3], shortcut_type, stride=2, head_conv=3)

        self.slow_inplanes = 64
        self.slow_conv1 = nn.Conv3d(3, 64, kernel_size=(
            1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.slow_bn1 = BN(64)
        self.slow_relu = nn.ReLU(inplace=True)
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(
            1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.slow_res1 = self._make_layer_slow(
            block, 64, layers[0], shortcut_type, head_conv=1)
        self.slow_res2 = self._make_layer_slow(
            block, 128, layers[1], shortcut_type, stride=2, head_conv=1)
        self.slow_res3 = self._make_layer_slow(
            block, 256, layers[2], shortcut_type, stride=2, head_conv=1)
        self.slow_res4 = self._make_layer_slow(
            block, 512, layers[3], shortcut_type, stride=2, head_conv=1)
        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(self.fast_inplanes +
                            self.slow_inplanes, class_num, bias=False)

    def forward(self, input):

        slow = self.SlowPath(input[:, :, ::16, :, :])
        print(slow.size())
        fast = self.FastPath(input[:, :, ::2, :, :])
        print(fast.size())
        x = torch.cat([slow, fast], dim=1)
        x = self.dp(x)
        x = self.fc(x)
        return x

    def SlowPath(self, input):
        x = self.slow_conv1(input)
        x, t = bn3d_bn2d(x)
        x = self.slow_bn1(x)
        x = bn2d_bn3d(x, t)
        x = self.slow_relu(x)
        x = self.slow_maxpool(x)
        x = self.slow_res1(x)
        x = self.slow_res2(x)
        x = self.slow_res3(x)
        x = self.slow_res4(x)
        x = nn.AdaptiveAvgPool3d(1)(x)
        x = x.view(-1, x.size(1))
        return x

    def FastPath(self, input):
        x = self.fast_conv1(input)
        x, t = bn3d_bn2d(x)
        x = self.fast_bn1(x)
        x = bn2d_bn3d(x, t)
        x = self.fast_relu(x)
        x = self.fast_maxpool(x)
        x = self.fast_res1(x)
        x = self.fast_res2(x)
        x = self.fast_res3(x)
        x = self.fast_res4(x)
        x = nn.AdaptiveAvgPool3d(1)(x)
        x = x.view(-1, x.size(1))
        return x

    def _make_layer_fast(self, block, planes, blocks, shortcut_type, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.fast_inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.fast_inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), BN(planes * block.expansion))

        layers = []
        layers.append(block(self.fast_inplanes, planes,
                            stride, downsample, head_conv=head_conv))
        self.fast_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.fast_inplanes,
                                planes, head_conv=head_conv))

        return nn.Sequential(*layers)

    def _make_layer_slow(self, block, planes, blocks, shortcut_type, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.slow_inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.slow_inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), BN(planes * block.expansion))

        layers = []
        layers.append(block(self.slow_inplanes, planes,
                            stride, downsample, head_conv=head_conv))
        self.slow_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.slow_inplanes,
                                planes, head_conv=head_conv))

        return nn.Sequential(*layers)


def resnet50_slowfast(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = SlowFast(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101_slowfast(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = SlowFast(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152_slowfast(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = SlowFast(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200_slowfast(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = SlowFast(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


if __name__ == "__main__":
    num_classes = 174
    input_tensor = torch.autograd.Variable(
        torch.rand(1, 3, 64, 224, 224)).cuda()
    model = resnet152_slowfast(pretrained=False, class_num=num_classes).cuda()
    output = model(input_tensor)
    print(output.size())
