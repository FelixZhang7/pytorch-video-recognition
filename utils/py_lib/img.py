#!/usr/bin/env python
"""
Image-related functions.

History
  create  -  Feng Zhou (zhfe99@gmail.com), 2016-05
"""
from . import pr
from . import cell
from . import dct
import skimage.io
import PIL
import numpy as np


IMG_FILE_EXTs = ['jpg', 'jpeg', 'png', 'bmp']


def crop(img0, box, ok_out=True):
    """
    Crop an image patch within a bounding box.

    Input
      img0     -  image, h0 x w0 x 3
      box      -  bounding box, 2 x 2 or a dict
                  if a 2 x 2 array
                    box[0, 0]: top y
                    box[0, 1]: bottom y
                    box[1, 0]: left x
                    box[1, 1]: right x
                  else
                    box.t: top y
                    box.b: bottom y
                    box.l: left x
                    box.r: right x
      ok_out   -  out of boundary is OK or not, {True} | False

    Output
      img      -  cropped image, h x w x 3 | None
    """
    # original image dimension
    h0 = img0.shape[0]
    w0 = img0.shape[1]

    # bounding box
    if isinstance(box, dct.ED):
        l = box.l
        r = box.r
        t = box.t
        b = box.b
    else:
        l = box[1, 0]
        r = box[1, 1]
        t = box[0, 0]
        b = box[0, 1]

    # out of images
    if ok_out:
        l = max(0, l)
        r = min(w0, r)
        t = max(0, t)
        b = min(h0, b)
    else:
        return None

    # invalid box
    if l >= r or t >= b:
        return None

    img = img0[t : b, l : r, :]
    return img


def imgFlip(img0, dire='lr'):
    """
    Flip an image left-to-right or up-to-down.

    Input
      img0     -  image, h x w x nChan
      dire     -  direction to flip, {'lr'} | 'ud'
                    'lr': left-to-right
                    'ud': up-to-down

    Output
      img      -  flipped image, h x w x nChan
    """
    # dimension
    h, w, nChan = img0.shape

    img = np.array(img0)
    for iChan in xrange(nChan):
      if dire == 'lr':
        img[:, :, iChan] = np.fliplr(img0[:, :, iChan])
      elif dire == 'ud':
        img[:, :, iChan] = np.flipud(img0[:, :, iChan])
      else:
        raise Exception('unknown dire: {}'.format(dire))

    return img


def imgIplCrop(imgPath0, imgPath, target_h=120, target_w=90):
    """
    Crop an image patch within a bounding box using IPL library.

    Input
      img0     -  image, h0 x w0 x 3
      isOkOut  -  flag of whether out boundary is OK, True | {False}

    Output
      img      -  cropped image, h x w x 3
    """
    img = PIL.Image.open(imgPath0)
    [w, h] = (img.size[0], img.size[1])
    xmin = 0
    ymin = 0
    xmax = w
    ymax = h
    if w / target_w > h / target_h:
      new_w = 90 * h / 120
      xmin = (w - new_w) / 2
      xmax = xmin + new_w
    else:
      new_h = 120 * w / 90
      ymin = (h - new_h) / 2
      ymax = ymin + new_h
    size = target_w, target_h
    bbox = (xmin, ymin, xmax, ymax)
    img = img.crop(bbox)
    img.thumbnail(size, PIL.Image.ANTIALIAS)
    img.convert('RGB').save(imgPath, "JPEG")


def imgCropSca(img0, h=120, w=90):
    """
    Crop an image patch within a bounding box.

    Input
      img0  -  image, h0 x w0 x 3
      h     -  height, {120} | ...
      w     -  width, {90} | ...

    Output
      img   -  cropped image, h x w x 3
    """
    # dimension
    h0, w0, nChan = img0.shape

    # get the bounding box
    if 1.0 * w0 / h0 > 1.0 * w / h:
      # crop w
      h1 = h0
      w1 = int(1.0 * w * h1 / h)
      xMi = (w0 - w1) / 2
      xMa = xMi + w1 - 1
      yMi = 0
      yMa = h0 - 1

    else:
      # crop h
      w1 = w0
      h1 = int(1.0 * h * w1 / w)
      yMi = (h0 - h1) / 2
      yMa = yMi + h1 - 1
      xMi = 0
      xMa = w0 - 1

    # crop
    box = [[yMi, yMa], [xMi, xMa]]
    img = imgCrop(img0, np.array(box), isOkOut=True)

    # scale
    img = imgSizNew(img, [h, w])

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
          crops[ix] = im[crop[0] : crop[2], crop[1] : crop[3], :]
        except ValueError:
          import pdb; pdb.set_trace()
        ix += 1

      # flip for mirrors
      if view == 'flip' or view == 'mul':
        m2 = m / 2
        crops[ix - m2 : ix] = crops[ix - m2 : ix, :, ::-1, :]

    return crops


def imgMeans(Img):
    """
    Compute different mean images for different groups of images.

    Input
      Img   -  a list of images, n x m, h x w x 3
                 n: #image in each group
                 m: #groups

    Output
      imgs  -  mean image, m (cell), h x w x 3
    """
    # dimension
    n, m = Img.shape

    imgs = cells(n)
    for i in range(n):
      imgs[i] = imgMean(Img[i])

    return imgs


def imgMean(imgs):
    """
    Compute the mean image for a group of images.

    Notice each image can have different dimensions.
    The mean image takes the largest dimension on each side.

    Input
      imgs  -  a list of images, n x, hi x wi x 3
                 n: #image in the group

    Output
      img   -  mean image, h x w x 3
    """
    # dimension
    n = len(imgs)

    # check the image size
    Siz = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
      Siz[i] = np.array(imgs[i].shape, dtype=np.uint8)
    siz = Siz.max(axis=0)

    img = np.zeros(tuple(siz))
    for imgi in imgs:
      sizi = imgi.shape
      # pad
      imgi = np.lib.pad(imgi,
                        ((0, siz[0] - sizi[0]),
                         (0, siz[1] - sizi[1]),
                         (0, siz[2] - sizi[2])),
                        'constant')

      # resize
      img += imgi

    # average
    img /= n
    return img


def imgMerge(img0s, alg='max'):
    """
    Merge multiple images of the same scale to a new image.

    Input
      img0s  -  input img, m x, h x w x nC
      alg    -  pixel merge algorithm, {'max'} | 'ave'
                  'max': pick the maximum pixel
                  'ave': compute the average pixel

    Output
      img    -  new img, h x w x nC
    """
    # dimension
    m = len(img0s)
    h, w, nC = img0s[0].shape

    # put in a big matrix
    Img0 = np.zeros((m, h, w, nC))
    for i in range(m):
      Img0[i] = img0s[i]

    if alg == 'max':
      img = Img0.max(axis=0)

    elif alg == 'ave':
      img = Img0.sum(axis=0) / m

    else:
      raise Exception('unknown alg: {}'.format(alg))

    return img


def resize(img0, siz, order=1):
    """
    Resize an image.

    Input
      img0   -  original image, h0 x w0 x nChan
      siz    -  size, [h, w]
      order  -  interpolation order, {1} | ...

    Output
      img    -  new image, h x w x nChan
    """
    siz0 = img0.shape
    if siz0[-1] == 1 or siz0[-1] == 3:
        from skimage.transform import resize

        # skimage is fast but only understands {1,3} channel images in [0, 1].
        im_min, im_max = img0.min(), img0.max()
        im_std = (img0 - im_min) / (im_max - im_min)
        resized_std = resize(im_std, siz, order=order)
        resized_im = resized_std * (im_max - im_min) + im_min

    else:
        from scipy.ndimage import zoom

        # ndimage interpolates anything but more slowly.
        scale = tuple(np.array(siz) / np.array(img0.shape[:2]))
        resized_im = zoom(img0, scale + (1,), order=order)

    return resized_im.astype(np.float32)


def fit_size_to_match_width(siz0, w):
    """
    Adjust the image size to fit with the width.

    Input
      siz0   -  original size, 2 x | 3 x
      w      -  width

    Output
      siz    -  new size, 2 x | 3 x
    """
    # original size
    h0 = siz0[0]
    w0 = siz0[1]

    # adjust height
    sca = 1.0 * h0 / w0
    w = w
    h = int(round(sca * w))

    # store
    if len(siz0) == 2:
      siz = [h, w]
    elif len(siz0) == 3:
      siz = [h, w, siz0[2]]
    else:
      raise Exception('unsupported dim: {}'.format(siz0.shape))

    return siz


def imgSizFit(siz0, sizMa):
    """
    Adjust the image size to fit with the maximum size constraint
    but keeping the ratio.

    Input
      siz0   -  original size, 2 x | 3 x
      sizMa  -  maximum size, 2 x

    Output
      siz    -  new size, 2 x | 3 x
      rat    -  ratio
    """
    # original size
    h0 = siz0[0]
    w0 = siz0[1]

    # maximum size
    hMa = sizMa[0]
    wMa = sizMa[1]

    # error
    if hMa == 0 and wMa == 0:
      siz = siz0
      rat = 1
      return siz, rat

    # fit already
    if h0 <= hMa and w0 <= wMa:
      siz = siz0
      rat = 1
      return siz, rat

    # adjust height
    if h0 > hMa:
      sca = 1.0 * w0 / h0
      h0 = hMa
      w0 = int(round(sca * h0))

    # adjust width
    if w0 > wMa:
      sca = 1.0 * h0 / w0
      w0 = wMa
      h0 = int(round(sca * w0))

    # ratio
    rat = np.mean(1.0 * np.array([h0, w0]) / np.array(siz0[:2]))

    # store
    if len(siz0) == 2:
      siz = [h0, w0]
    elif len(siz0) == 3:
      siz = [h0, w0, siz0[2]]
    else:
      raise Exception('unsupported dim: {}'.format(siz0.shape))

    return siz, rat


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
            pr.pr('unable to open img: {}'.format(imgPath))
        return None

    # color channel
    if img.ndim == 2:
      img = img[:, :, np.newaxis]
      if color:
        img = np.tile(img, (1, 1, 3))

    elif img.shape[2] == 4:
      img = img[:, :, :3]

    return img


def imgLdTxt(txtPaths):
    """
    Load image from txt file.

    Input
      txtPaths  -  txt path, 3 x

    Output
      img       -  image, h x w x 3
    """
    # dimension
    d = len(txtPaths)

    # load matrix
    As = []
    for i in range(d):
      A = np.loadtxt(txtPaths[i])
      if i == 0:
        h = A.shape[0]
        w = A.shape[1]
      else:
        assert A.shape[0] == h and A.shape[1] == w
      As.append(A)

    # merge
    img = np.zeros((h, w, d))
    for i in range(d):
      img[:, :, i] = As[i]
    img = img / 255

    return img


def imgSvTxt(img, txtPaths, fmt='%.2f'):
    """
    Save image to txt files.

    Input
      img       -  image, 3 x h x w
      txtPaths  -  txt path, 3 x
      fmt       -  format
    """
    # dimension
    d = len(txtPaths)

    # load matrix
    for i in range(d):
      np.savetxt(txtPaths[i], img[i], fmt)


def imgLdPil(imgPath):
    """
    Load an image using PIL.

    Input
      imgPath  -  image path

    Output
      image    -  image in PIL format
    """
    # load
    from PIL import Image
    img = Image.open(imgPath)
    return img


def imgLdCv(imgPath):
    """
    Load an image using OpenCV.

    Input
      imgPath  -  image path

    Output
      image    -  image, h x w x nChan
    """
    # load
    import cv2
    img = cv2.imread(imgPath)
    return img


def imgPil2Ski(im):
    """
    Convert image from PIL format to an Skimage one.

    Input
      img0   -  original PIL image

    Output
      image  -  new Ski image
    """
    img0 = np.array(im.getdata())
    if img0.shape[1] == 4:
      pix = img0[:, 0 : 3].reshape(im.size[1], im.size[0], 3) / 255.0
    elif img0.shape[1] == 3:
      pix = img0.reshape(im.size[1], im.size[0], 3) / 255.0
    im = skimage.img_as_float(pix).astype(np.float32)

    return im


def imgPil2Cv(img0):
    """
    Convert image from PIL format to an Skimage one.

    Input
      img0  -  original PIL image

    Output
      img   -  new OpenCV image
    """
    data = np.array(img0.getdata())
    img = data.reshape(img0.size[1], img0.size[0], 3)
    return img[:, :, [2, 1, 0]].astype(np.uint8)


def imgPil2Ipl(img0):
    """
    Convert image from PIL format to an OpenCV IPL one.

    Input
      img0     -  original PIL image

    Output
      image    -  image in PIL format
    """
    import cv2

    if not isinstance(img0, PIL.Image.Image):
      raise TypeError, 'must be called with PIL.Image.Image!'

    # dimension
    size = (img0.size[0], img0.size[1])

    # mode dictionary:
    # (pil_mode : (ipl_depth, ipl_channels, color model, channel Seq)
    mode_list = {
      "RGB" : (cv2.cv.IPL_DEPTH_8U, 3),
      "L"   : (cv2.cv.IPL_DEPTH_8U, 1),
      "F"   : (cv2.cv.IPL_DEPTH_32F, 1)}
    if not mode_list.has_key(img0.mode):
      raise ValueError, 'unknown or unsupported input mode'
    modes = mode_list[img0.mode]

    result = cv2.cv.CreateImageHeader(size, modes[0], modes[1])

    # set imageData
    step = size[0] * (result.depth / 8) * result.nChannels
    cv2.cv.SetData(result, img0.rotate(180).tostring()[::-1], step)

    return result


def imgDateInfo2Time(ts):
    """
    Changes EXIF date ('2005:10:20 23:22:28') to #seconds since 1970-01-01.

    Input
      ts  -  time

    Output
      ti  -  time
    """
    import time
    tpl = time.strptime(ts + 'UTC', '%Y:%m:%d %H:%M:%S%Z')
    return time.mktime(tpl)


def imgDateExif(imgPath):
    """
    Return EXIF datetime using exifread.

    Input
      imgPath  -  file path

    Output
      imgDate  -  time
    """
    import exifread

    # what tags use to redate file (use first found)
    DT_TAGS = ['Image DateTime', 'EXIF DateTimeOriginal', 'DateTime']

    dt_value = None
    f = open(imgPath, 'rb')
    try:
      tags = exifread.process_file(f)
      for dt_tag in DT_TAGS:
        try:
          dt_value = '%s' % tags[dt_tag]
          break
        except:
          continue
      if dt_value:
        exif_time = imgDateInfo2Time(dt_value)
        return exif_time
    finally:
      f.close()
    return None


def imgDatePil(imgPath):
    """
    Return EXIF datetime using PIL.

    Input
      imgPath  -  file path

    Output
      imgDate  -  time
    """
    im = PIL.Image.open(imgPath)
    if hasattr(im, '_getexif'):
      exifdata = im._getexif()
      dt_value = exifdata[0x9003]
      exif_time = imgDateInfo2Time(dt_value)
      return exif_time
    return None


def imgDate(imgPath):
    """
    Read the create date from the EXIF part of the image.

    Input
      imgPath  -  file path

    Output
      imgDate  -  image date, None | string
    """
    imgDate0 = None
    try:
      imgDate0 = imgDatePil(imgPath)
    except KeyboardInterrupt:
      raise
    except:
      try:
        imgDate0 = imgDateExif(imgPath)
      except KeyboardInterrupt:
        raise
      except:
        pr(imgPath)

    # time -> string
    import time
    imgDat = None
    if imgDate0 is not None:
      imgDat = time.strftime("%Y-%m-%d", time.gmtime(imgDate0))

    return imgDat


def imgDistort(img0, sca=1.5, rotMa=0, randSca=False):
    """
    Distort an input image.

    Input
      img0   -  input img, h0 x w0 x nC
      sca    -  scaling factor, {1.5} | 2 | ...
                  new image dimension would be
                  h = int(h0 * sca)
                  w = int(w0 * sca)
      rotMa  -  maximum random rotation, {0} | 30 | 180 | 360 | ...

    Output
      img    -  new img, h x w x nC
    """
    # dimension
    h0, w0, nC = img0.shape
    h = int(h0 * sca)
    w = int(w0 * sca)

    # random rotation
    from skimage import transform
    ang = int(np.random.rand(1) * rotMa)
    shift_y, shift_x = np.array([h0, w0]) / 2.
    rot = transform.SimilarityTransform(rotation=np.deg2rad(ang))
    tf_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = transform.SimilarityTransform(translation=[shift_x, shift_y])
    img1 = transform.warp(img0, (tf_shift + (rot + tf_shift_inv)).inverse)

    # random translate
    img = np.zeros((h, w, nC), dtype=img0.dtype)
    hD = int(np.random.rand(1) * (h - h0))
    wD = int(np.random.rand(1) * (w - w0))
    img[hD : hD + h0, wD : wD + w0, :] = img1

    return img
