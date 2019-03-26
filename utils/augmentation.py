"""
    1. mixup
    2. color_jitter
    3. random erasing
"""
import torch
import torchvision
import numpy as np
import random
import math
from PIL import Image
from termcolor import cprint
from .transforms import *
from .spatial_transforms import *
from .temporal_transforms import *


def get_augmentation(config, model):
    """
    Entry function to get the augmenter for different methods.

    """
    if not config:
        raise "config is None in augmentation!"

    if config.augmentation.get('mix_up', False):
        cprint("Using mix_up augmentation!", 'green')

    train_aug = []
    val_aug = []

    if config.model.type == "tsn" or config.model.type == "tsm":
        input_size = model.input_size
        crop_size = model.crop_size
        scale_size = model.scale_size
        input_mean = model.input_mean
        input_std = model.input_std

        # get train augmentation
        # train_aug = [crop, stack, normalize ...]
        train_aug.append(GroupMultiScaleCrop(input_size, [1, .875, .75, .66]))
        train_aug.append(GroupRandomHorizontalFlip(is_flow=False))

        colorjitter = config.augmentation.get('colorjitter', None)
        if colorjitter:
            train_aug.append(GroupColorJitter(colorjitter))

        random_erasing = config.augmentation.get('random_erasing', None)
        if random_erasing:
            train_aug.append(GroupRandomErasing(random_erasing))

        train_aug.append(Stack(roll=False))
        train_aug.append(ToTorchFormatTensor(div=True))
        train_aug.append(GroupNormalize(input_mean, input_std))

        # get val augmentation
        # val_aug = [crop, stack, normalize ...]
        if config.get('val_dense', False):
            # dense validation
            val_aug.append(GroupOverSample(crop_size, scale_size))
        else:
            # normal validation
            val_aug.append(GroupScale(int(scale_size)))
            val_aug.append(GroupCenterCrop(crop_size))

        val_aug.append(Stack(roll=False))
        val_aug.append(ToTorchFormatTensor(div=True))
        val_aug.append(GroupNormalize(input_mean, input_std))

    elif config.model.type == "slowfast" or config.model.type == "classify":
        input_size = 224
        crop_size = 224
        scale_size = 256
        input_mean = [0.485, 0.456, 0.406]
        input_std = [0.229, 0.224, 0.225]

        # get train augmentation
        # train_aug = [crop, stack, normalize ...]
        train_aug.append(GroupMultiScaleCrop(input_size, [1, .875, .75, .66]))
        train_aug.append(GroupRandomHorizontalFlip(is_flow=False))

        colorjitter = config.augmentation.get('colorjitter', None)
        if colorjitter:
            train_aug.append(GroupColorJitter(colorjitter))

        random_erasing = config.augmentation.get('random_erasing', None)
        if random_erasing:
            train_aug.append(GroupRandomErasing(random_erasing))

        train_aug.append(Stack3D(roll=False))
        train_aug.append(ToTorchFormatTensor3D(div=True))
        train_aug.append(Normalize3D(input_mean, input_std))

        # get val augmentation
        # val_aug = [crop, stack, normalize ...]
        if config.get('val_dense', False):
            # dense validation
            val_aug.append(GroupOverSample(crop_size, scale_size))
        else:
            # normal validation
            val_aug.append(GroupScale(int(scale_size)))
            val_aug.append(GroupCenterCrop(crop_size))

        val_aug.append(Stack3D(roll=False))
        val_aug.append(ToTorchFormatTensor3D(div=True))
        val_aug.append(Normalize3D(input_mean, input_std))

    else:
        raise NotImplementedError("The augmentation is not Implemented")

    return train_aug, val_aug


class GroupRandomErasing(object):
    """docstring for GroupRandomErase
    """

    def __init__(self, arg=[1, 0.02, 0.4, 0.3]):
        print("Using random_erasing!!!!")
        super(GroupRandomErasing, self).__init__()
        self.arg = arg
        self.worker = RandomErasing(*arg)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation
    by Zhong et al.

    Args:
        probability: The probability that the operation will be performed.
        sl: min erasing area
        sh: max erasing area
        r1: min aspect ratio
        mean: erasing value
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3,
                 mean=[152, 142, 127]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            # convert PIL Image object to Numpy Array
            # PIL Image size is (224, 224) [W, H]->[x, y]->[col, row]
            # numpy array shape is (224, 224, 3) [W, H, C]
            _img = np.array(img)
            area = _img.shape[0] * _img.shape[1]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            # print("h is {}, w is {}".format(str(h), str(w)))
            if w < _img.shape[0] and h < _img.shape[1]:
                x1 = random.randint(0, _img.shape[0] - w)
                y1 = random.randint(0, _img.shape[1] - h)

                if len(_img.shape) == 2:
                    # for gray image
                    _img[x1:x1+w, y1:y1+h, 0] = self.mean[0]
                else:
                    # for RGB image
                    _img[x1:x1+w, y1:y1+h, 0] = self.mean[0]
                    _img[x1:x1+w, y1:y1+h, 1] = self.mean[1]
                    _img[x1:x1+w, y1:y1+h, 2] = self.mean[2]

                # convert Numpy Array back to PIL Image
                img = Image.fromarray(_img)
                return img

        return img


class GroupColorJitter(object):
    def __init__(self, colorjitter):
        print("Using ColorJitter!!!!!")
        self.colorjitter = colorjitter
        self.worker = torchvision.transforms.ColorJitter(*colorjitter)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class ColorAugmentation(object):
    def __init__(self, eig_vec=None, eig_val=None):
        if eig_vec is None:
            eig_vec = torch.Tensor([
                [0.4009,  0.7192, -0.5675],
                [-0.8140, -0.0045, -0.5808],
                [0.4203, -0.6948, -0.5836],
            ])
        if eig_val is None:
            eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])
        self.eig_val = eig_val  # 1*3
        self.eig_vec = eig_vec  # 3*3

    def __call__(self, tensor):
        assert tensor.size(0) == 3
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of
        targets, and lambda

    Args:
        x(tensor) : input tensor. Shape is [BS, 3*num_segments, H, W]
        y(tensor) : target tensor. Shape is [BS]
        alpha(float) : parameter of mix-up function, alpha distribution.

        '''

    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + \
        (1 - lam) * criterion(pred, y_b)


def dense_crop():
    pass
