## Update (2017/11/27)
  
We also added the following new models and their Kinetics pretrained models in this repository.  

* ResNet-50, 101, 152, 200
* Pre-activation ResNet-200
* Wide ResNet-50
* ResNeXt-101
* DenseNet-121, 201

## Pre-trained models

Pre-trained models are available [here](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M?usp=sharing).  
All models are trained on Kinetics.  


```misc
resnet-18-kinetics.pth: --model resnet --model_depth 18 --resnet_shortcut A
resnet-34-kinetics.pth: --model resnet --model_depth 34 --resnet_shortcut A
resnet-34-kinetics-cpu.pth: CPU ver. of resnet-34-kinetics.pth
resnet-50-kinetics.pth: --model resnet --model_depth 50 --resnet_shortcut B
resnet-101-kinetics.pth: --model resnet --model_depth 101 --resnet_shortcut B
resnet-152-kinetics.pth: --model resnet --model_depth 152 --resnet_shortcut B
resnet-200-kinetics.pth: --model resnet --model_depth 200 --resnet_shortcut B
preresnet-200-kinetics.pth: --model preresnet --model_depth 200 --resnet_shortcut B
wideresnet-50-kinetics.pth: --model wideresnet --model_depth 50 --resnet_shortcut B --wide_resnet_k 2
resnext-101-kinetics.pth: --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32
densenet-121-kinetics.pth: --model densenet --model_depth 121
densenet-201-kinetics.pth: --model densenet --model_depth 201
```

Some of fine-tuned models on UCF-101 and HMDB-51 (split 1) are also available.

```misc
resnext-101-kinetics-ucf101_split1.pth: --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32
resnext-101-64f-kinetics-ucf101_split1.pth: --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration 64
resnext-101-kinetics-hmdb51_split1.pth: --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32
resnext-101-64f-kinetics-hmdb51_split1.pth: --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration 64
```

### Performance of the models on Kinetics

This table shows the averaged accuracies over top-1 and top-5 on Kinetics.

| Method | Accuracies |
|:---|:---:|
| ResNet-18 | 66.1 |
| ResNet-34 | 71.0 |
| ResNet-50 | 72.2 |
| ResNet-101 | 73.3 |
| ResNet-152 | 73.7 |
| ResNet-200 | 73.7 |
| ResNet-200 (pre-act) | 73.4 |
| Wide ResNet-50 | 74.7 |
| ResNeXt-101 | 75.4 |
| DenseNet-121 | 70.8 |
| DenseNet-201 | 72.3 |

