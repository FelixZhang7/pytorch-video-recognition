import multiprocessing as mp
import argparse
import os
import yaml
from easydict import EasyDict
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import linklink as link
from PIL import Image
from methods import model_entry
from utils.utils import load_state
from utils.distributed_utils import dist_init, DistModule
from utils.spatial_transforms import *
from utils.transforms import *
import utils.io_tools

mp.set_start_method('spawn', force=True)

parser = argparse.ArgumentParser(description='PyTorch Video Analysis')
parser.add_argument('--config', default='cfgs/config_res50.yaml')
parser.add_argument('--load-path', default='', type=str)
parser.add_argument('--recover', action='store_true')
parser.add_argument('-e', '--evaluate', action='store_true')
parser.add_argument('--sync', action='store_true')
parser.add_argument('--fake', action='store_true')
parser.add_argument('--fuse-prob', action='store_true')
parser.add_argument('--fusion-list', nargs='+', help='multi model fusion list')


def main():
    global args, config, best_prec1
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

    config = EasyDict(config['common'])
    config.save_path = os.path.dirname(args.config)

    rank, world_size = dist_init()

    # If loads model from load_path, there is no need to load imagenet
    # pretrained model

    if args.load_path:
        config.model.pretrained = False

    model = model_entry(config.model)

    if rank == 0:
        print(model)

    model.cuda()

    model = DistModule(model, args.sync)

    # optionally resume from a checkpoint
    if args.load_path:
        load_state(args.load_path, model)

    cudnn.benchmark = True

    # frame_path = '/mnt/lustre/sunguanxiong/code/pva/video/data/douyin_data/douyin_v4_152/frames/v_篮球_54657ca0ab9745739aae3772676be26e'
    # frame_path = '/mnt/lustre/share/sunguanxiong/benchmarks/douyin/douyin_v5_152/frames/v_饺子_v0300fc30000bfrm29v82iju7bfd4ivg'
    # frame_path = '/mnt/lustre/share/jipuzhao/other/selected_frames'
    # frame_path = '/mnt/lustre/share/sunguanxiong/benchmarks/douyin/douyin_v5_152/frames/v_仓鼠_v0200fc10000berof1qkr6g14p20577g'
    # img1 = Image.open(frame_path + '/000136.jpg').convert('RGB')
    # img2 = Image.open(frame_path + '/000272.jpg').convert('RGB')
    # img3 = Image.open(frame_path + '/000408.jpg').convert('RGB')
    # img1 = Image.open(frame_path + '/000100.jpg').convert('RGB')
    # img2 = Image.open(frame_path + '/000200.jpg').convert('RGB')
    # img3 = Image.open(frame_path + '/000300.jpg').convert('RGB')
    # frame_path = '.'
    # img1 = Image.open(frame_path + '/1.png').convert('RGB')
    # img2 = Image.open(frame_path + '/2.png').convert('RGB')
    # img3 = Image.open(frame_path + '/3.png').convert('RGB')
    # test_samples = [img1, img2, img3]

    img_new = utils.io_tools.get_rgb_img_from_bin(
        "224x224_1.bgr8888hwc", (224, 224))

    print(img_new.getpixel((0, 0)))
    test_samples = [img_new, img_new, img_new]

    val_aug = []
    val_aug.append(GroupScale(int(256)))
    val_aug.append(GroupCenterCrop(224))
    val_aug.append(Stack(roll=False))
    val_aug.append(ToTorchFormatTensor(div=True))
    val_aug.append(GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    trans = transforms.Compose(val_aug)

    input_tensor = trans(test_samples)

    # """
    #     Use their input tensor binary file.

    #     Log: out put 72

    # """
    # def fill_tensor_with_bin(pathname, tensor) :
    #     import struct
    #     file = open(pathname, "rb")
    #     for i in range(0, tensor.size(0)):
    #         for j in range(0, tensor.size(1)):
    #             for k in range(0, tensor.size(2)):
    #                 tensor[i][j][k] = struct.unpack('f', file.read(4))[0]
    #     return

    # t = torch.FloatTensor(3, 224, 224)
    # fill_tensor_with_bin("3x224x224_uniform_mean_std_1.bin", t)

    # utils.io_tools.save_tensor_as_img(t, 't.jpg')
    # utils.io_tools.save_tensor_as_img(t, 't.png')

    # print(t)
    # input_tensor = torch.cat((t,t,t), 0)
    #######################################################

    print(input_tensor.shape)

    input_tensor = input_tensor.view(1, 9, 224, 224)
    input_tensor = input_tensor.to('cuda')
    print(input_tensor.shape)
    print(input_tensor)

    tensor_of_img1 = input_tensor[0, 0:3, :, :].view(3, 224, 224)
    print(tensor_of_img1.shape)

    # utils.io_tools.save_tensor_as_img(tensor_of_img1, 'tensor_of_img1.jpg')

    # utils.io_tools.dump_tensor_as_bin(tensor_of_img1, 'tensor_of_img_new.bin')

    if args.evaluate:
        inference(input_tensor, model)
        link.finalize()
        return

    link.finalize()


def inference(input_tensor, model, fusion_list=None, fuse_prob=False):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        torch.save(output, 'out.pt')

    print(output.topk(5, 1, True, True))
    print(output)
    out23d = output.view(1, 1, -1)
    utils.io_tools.dump_tensor_as_bin(out23d, 'output_new.bin')
    return


if __name__ == '__main__':
    main()
