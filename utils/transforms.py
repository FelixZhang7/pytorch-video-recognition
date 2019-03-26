import torchvision
from PIL import Image
import numpy as np
import torch


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
        rep_std = self.std * (tensor.size()[0]//len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


class Normalize3D(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # print("before", tensor[0, 0, 0, 0])
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)

        # print("after", tensor[0, 0, 0, 0])
        return tensor


class GroupColorJitter(object):
    def __init__(self, colorjitter):
        self.colorjitter = colorjitter
        self.worker = torchvision.transforms.ColorJitter(*colorjitter)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)


class Stack3D(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            raise "Not support L format images!"
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group], axis=0)
            else:
                return np.stack(img_group, axis=0)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class ToTorchFormatTensor3D(object):
    """ Converts a numpy.ndarray (T x W x H x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x T x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(3, 0, 2, 1).contiguous()
        else:
            # handle PIL Image
            raise "Must be numpy array!"
            # img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            # img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # # put it from HWC to CHW format
            # # yikes, this transpose takes 80% of the loading time/CPU
            # img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class IdentityTransform(object):

    def __call__(self, data):
        return data


if __name__ == "__main__":

    trans = torchvision.transforms.Compose([
        GroupScale(256),
        GroupRandomCrop(224),
        TemporalRandomCrop(size=64),
        Stack3D(),
        ToTorchFormatTensor3D(),
        Normalize3D(
            mean=[.485, .456, .406],
            std=[.229, .224, .225]
        )]
    )

    im = Image.open('lena_299.png')

    color_group = [im] * 300
    rst = trans(color_group)

    print(rst.shape)

    gray_group = [im.convert('L')] * 9
    gray_rst = trans(gray_group)

    trans2 = torchvision.transforms.Compose([
        GroupRandomSizedCrop(256),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(
            mean=[.485, .456, .406],
            std=[.229, .224, .225])
    ])
    print(trans2(color_group))
