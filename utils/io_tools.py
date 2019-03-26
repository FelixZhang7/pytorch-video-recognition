"""
    This module is for IO operations. Such as, save_tensor,
    load_tensor, read binary file...
"""
import torch
import struct
from PIL import Image


def get_rgb_img_from_bin(pathname, size=(224, 224)):
    """
    Get a rgb image from a binary file.

    Args:
        pathname: file name
        size: expected size of the image.

    Return:
        A PIL Image with the expected size.

    """
    with open(pathname, "rb") as f:
        byte = f.read(size[0]*size[1]*3)

    return Image.frombytes("RGB", (224, 224), byte)


def get_tensor_from_bin(pathname, shape=(3, 224, 224)):
    """
    Get a tensor from a binary file.

    Args:
        pathname: file name
        shape: expected shape of the tensor

    Return:
        A torch tensor with desired shape.

    """

    _tensor = torch.FloatTensor(shape)

    file = open(pathname, "rb")
    for i in range(0, _tensor.size(0)):
        for j in range(0, _tensor.size(1)):
            for k in range(0, _tensor.size(2)):
                _tensor[i][j][k] = struct.unpack('f', file.read(4))[0]
    return _tensor


def save_tensor_as_img(tensor, pathname):
    """
    Dump a tensor to img file [.jpg, .png].

    Args:
        tensor : A pytorch tensor, shape is [C, H, W]
        pathname(string): file name. e.g. './img.bin'

    """
    import torchvision
    torchvision.utils.save_image(tensor, pathname)


def dump_tensor_as_bin(tensor, pathname):
    """
    Dump a tensor to rgb binary file.

    Args:
        tensor : A pytorch tensor, shape is [C, H, W]
        pathname(string): file name. e.g. './img.bin'

    """

    file = open(pathname, 'wb')
    for i in range(0, tensor.size(0)):
        for j in range(0, tensor.size(1)):
            for k in range(0, tensor.size(2)):
                s = struct.pack('f', tensor[i][j][k].data.item())
                file.write(s)
    return


def dump_PIL_image_as_bin(image, pathname):
    """
    Dump a PIL.Image object to binary file.

    Args:
        image : A PIL.Image object.
        pathname(string): file name. e.g. './img.bin'

    """
    file = open(pathname, 'wb')
    file.write(image.tobytes())
    return
