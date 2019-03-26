"""
    Some useful operations on PIL images.
    Such as, save, load, mean, std ...

"""
import io
import numpy as np
import skimage.io
from PIL import Image
import torch
# IMG_FILE_EXTs = ['jpg', 'jpeg', 'png', 'bmp']


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    with Image.open(buff) as img:
        img = img.convert('RGB')
    return [img]


def get_mean_and_std(dataset):
    """
    Compute the mean and std value of dataset.

    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def save(imgPath, img):
    """
    Save an image to the path.

    Input
      imgPath  -  image path
      img      -  an image with type np.float32 in range [0, 1], h x w x nChan
    """
    skimage.io.imsave(imgPath, img)


def load(imgPath, color=True, verbose=True):
    """
    Load an image converting from grayscale or alpha as needed.

    Input
      imgPath  -  image path
      color    -  flag for color format. True (default) loads as RGB while False
                  loads as intensity (if image is already grayscale).

    Output
      image    -  an image with type np.float32 in range [0, 1]
                    of size (h x w x 3) in RGB or
                    of size (h x w x 1) in grayscale.
    """
    # load
    try:
        img0 = skimage.io.imread(imgPath)
        img = skimage.img_as_float(img0).astype(np.float32)

    except KeyboardInterrupt as e:
        raise e
    except:
        if verbose:
            print('unable to open img: {}'.format(imgPath))
        return None

    # color channel
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))

    elif img.shape[2] == 4:
        img = img[:, :, :3]

    return img


def oversample(img0s, h=224, w=224, view='mul'):
    """
    Crop images as needed. Inspired by pycaffe.

    Input
      img0s  -  n0 x, h0 x w0 x k0
      h      -  crop height, {224}
      w      -  crop width, {224}
      view   -  view, 'sin' | 'flip' | {'mul'}
                  'sin': center crop (m = 1)
                  'flip': center crop and its mirrored version (m = 2)
                  'mul': four corners, center, and their mirrored versions (m = 10)

    Output
      imgs   -  crops, (m n0) x h x w x k
    """
    # dimension
    n0 = len(img0s)
    im_shape = np.array(img0s[0].shape)
    crop_dims = np.array([h, w])
    im_center = im_shape[:2] / 2.0
    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])

    # make crop coordinates
    if view == 'sin':
        # center crop
        crops_ix = np.empty((1, 4), dtype=int)
        crops_ix[0] = np.tile(im_center, (1, 2)) + np.concatenate([
            -crop_dims / 2.0, crop_dims / 2.0
        ])

    elif view == 'flip':
        # center crop + flip
        crops_ix = np.empty((1, 4), dtype=int)
        crops_ix[0] = np.tile(im_center, (1, 2)) + np.concatenate([
            -crop_dims / 2.0, crop_dims / 2.0
        ])
        crops_ix = np.tile(crops_ix, (2, 1))

    elif view == 'mul':
        # multiple crop
        crops_ix = np.empty((5, 4), dtype=int)
        curr = 0
        for i in h_indices:
            for j in w_indices:
                crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
                curr += 1
        crops_ix[4] = np.tile(im_center, (1, 2)) + np.concatenate([
            -crop_dims / 2.0, crop_dims / 2.0
        ])
        crops_ix = np.tile(crops_ix, (2, 1))
    m = len(crops_ix)

    # extract crops
    crops = np.empty((m * n0, crop_dims[0], crop_dims[1],
                      im_shape[-1]), dtype=np.float32)
    ix = 0
    for im in img0s:
        for crop in crops_ix:
            try:
                crops[ix] = im[crop[0]: crop[2], crop[1]: crop[3], :]
            except ValueError:
                import pdb
                pdb.set_trace()
            ix += 1

        # flip for mirrors
        if view == 'flip' or view == 'mul':
            m2 = m / 2
            crops[ix - m2: ix] = crops[ix - m2: ix, :, ::-1, :]

    return crops
