# Pytorch Video Analysis

This repo contains state-of-the-art models for video analysis, for example, tsn, tsm, and slow-fast network. It also includes the most common scripts used in the field of video analysis, such as, frames dumping, data augmentation, and dense validation...

## 0. Prepare Data

Before using this code, you need to dump the video datasets (e.g. Kinetics400, Kinetics600, UCF101 ...) into frames. Follow the instruction of [Temporal Segments Networks](https://github.com/yjxiong/temporal-segment-networks). 

Get the original data:
### ActivityNet

* Download videos using [the official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler).
* Convert from avi to jpg files using [Temporal Segments Networks](https://github.com/yjxiong/temporal-segment-networks)

### Kinetics

* Download videos using [the official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics).
  * Locate test set in ```video_directory/test```.
* Convert from avi to jpg files using [Temporal Segments Networks](https://github.com/yjxiong/temporal-segment-networks). 

### UCF-101

* Download videos and train/test splits [here](http://crcv.ucf.edu/data/UCF101.php).
* Convert from avi to jpg files using [Temporal Segments Networks](https://github.com/yjxiong/temporal-segment-networks). 

### HMDB-51

* Download videos and train/test splits [here](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).
* Convert from avi to jpg files using [Temporal Segments Networks](https://github.com/yjxiong/temporal-segment-networks). 

## 1. structure

```
${THIS REPO ROOT}
    `-- pretrained
        |-- resnet50-19c8e357.pth
        |-- resnet101-5d3b4d8f.pth
        |-- resnet152-b121ed2d.pth
    `-- data
        `-- kin400
            `-- frames
                |   |-- v_tag_vid
                |   |-- ...
                |   |-- ...
                |   |-- ...
            `-- lists
                |   |-- train.txt
                |   |-- val.txt
                |   |-- classInd.txt
                |-- README
        `-- ucf101
            `-- frames
            |   |-- v_tag_vid
            |   |-- ...
            |   |-- ...
            |   |-- ...
            `-- lists
            |   |-- train.txt
            |   |-- val.txt
            |   |-- calssInd.txt
            |-- README
    `-- methods 视频识别的方法
        |-- tsn.py
        |-- tsm.py
        |-- ...
    `-- networks 常用backbone
        |-- resnet.py
        |-- mobilenet.py
        |-- ...
    `-- utils 实用函数 (io, loss, transform ...)
        |-- utils.py
        |-- transforms.py
        |-- loss.py
        |-- ...
    `-- datasets 不同的dataloader (img, group imgs, video clip) 
        |-- memcached_dataset.py
        |-- memcached_dataset_tsn.py
        |-- ...
    `-- anal_tools 模型结果分析，top100 errors，confusion matrix
        |-- get_reference_results.py
        |-- analyze.py
        |-- ...
    |-- main.py train/val入口
    |-- ...

```

## 2. Usage
### 2.1 Example
```
训练
./exps/exps/douyin_exps/tsn_v5_mobilenet/colorjitter_random_erasing_labelsmooth_0.1_nowd_coslr_bn16_gpu32_lr0.04_epoch100/train.sh 分区名(VI_SP_VA_1080TI) 显卡数量(32)

测试
./exps/exps/douyin_exps/tsn_v5_mobilenet/colorjitter_random_erasing_labelsmooth_0.1_nowd_coslr_bn16_gpu32_lr0.04_epoch100/eval.sh 分区名(VI_SP_VA_1080TI) 显卡数量(32)

```
### 2.2 Experiment Structure
每个实验单独存放一个文件夹, 结构如下：
```
├── res18_batch2k_epoch100_colorjitter_0.2_0.2_0.2_0.1_nesterov
│   ├── config.yaml
│   ├── eval.sh
│   └── train.sh
├── res34_batch2k_epoch100_colorjitter_0.2_0.2_0.2_0.1_nesterov
│   ├── config.yaml
│   ├── eval.sh
│   └── train.sh
├── res50_batch2k_epoch100_colorjitter_0.2_0.2_0.2_0.1_nesterov
│   ├── config.yaml
│   ├── eval.sh
│   └── train.sh
...
```
其中config.yaml是实验的配置文件，train.sh用来训练，eval.sh用来测试。

开始训练之后，会生成新文件
```
├── ckpt_best.pth.tar 到目前为止eval结果最高的checkpoint
├── ckpt.pth.tar 最新的checkpoint
├── config.yaml
├── eval.sh
├── events tensorboard events，重复启动训练会生成多个
│   └── events.out.tfevents.1539700526.SH-IDC1-10-5-36-103
├── log.txt 文本log，追加模式
└── train.sh
```
## 3. Config
一个config.yaml文件如下
```
common:
    model:
        type: tsn  # tsm, slowfast
        backbone: mobilenetv2  # backbone

        num_segments: 3
        num_class: 152
        dropout: 0.8
        pretrained: True
    
        kwargs:
            bn_group_size: 16  # sync_bn group size
            bn_sync_stats: True
            t: 6
    
    augmentation:
        input_size: 224
        test_resize: 256
        colorjitter: [0.2, 0.2, 0.2, 0.1]  # colorjitter
        random_erasing: [0.5, 0.02, 0.4, 0.3]  # random erasing
        mix_up: 0.2  # mix up

    val_dense: False  # dense validation

    workers: 3
    batch_size: 32

    lr_scheduler:  # learning schedule
        type: COSINE 

        base_lr: 0.01
        warmup_steps: 500
        warmup_lr: 0.04
        min_lr: 0.0
        max_iter: 15010

    label_smooth: 0.1  # label smooth
    no_wd: True  # without weight decay for bias
    momentum: 0.9
    weight_decay: 0.0001
    nesterov: True

    val_freq: 250
    print_freq: 10

    train_root: /
    train_source: /mnt/lustre/share/jipuzhao/benchmarks/douyin/douyin_v5/lists/rgb_train.txt
    val_root: /
    val_source: /mnt/lustre/share/jipuzhao/benchmarks/douyin/douyin_v5/lists/rgb_val.txt

```

## 4. Results

