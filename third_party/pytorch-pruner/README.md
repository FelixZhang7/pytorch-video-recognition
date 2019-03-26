# Pytorch Pruner

This project is willing to provide an extension for automatic network structure pruning.

Import this extension and add few lines of code, then pruning is ready.

This project is based on re-implementation of paper[1], but it can easily extend to other pruning methods.

[中文说明文档](http://confluence.sensetime.com/pages/viewpage.action?pageId=28667962)

[JIRA issues](http://jira.sensetime.com/projects/MXJSKJZC/issues/?filter=allopenissues)

## Contents

- [Must Read](#must-read)
- [Installation](#installation)
- [How to Use](#how-to-use)
- [Supported Network Structures](#supported-network-structures)
- [Performance](#performance)
- [Reference](#Reference)

## Must Read

### [Caution] Please `don't` set cudnn.benchmark=True

It may cause GPU Out of Memory, because its optimization for GPU Memory

### [Caution] Please `don't` set required_grad = False

Pruner needs grad to build graph, so don't set any required_grad = False

### [Caution] Please `save&load checkpoint` as below

Pruner will change the structure of your model, so we must save its structure and parameters. A simple solution as below:

**Load a pruned model**

    model = ckpt_helper.load_model_def(model, path_to_ckpt)

**Save a pruned model**

    save_checkpoint({
                    'step': ...,
                    ...
                    'model_def': ckpt_helper.save_model_def(model),
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, ... )
    

## Installation

This framework is based on an efficient internal version of PyTorch

Please refer to [http://gitlab.bj.sensetime.com/project-spring/imagenet-example](http://gitlab.bj.sensetime.com/project-spring/imagenet-example) for more detail

## How to Use

`We only provide simple useage here. FOor advanced usage, please refer our [Confluence Page]`(http://confluence.sensetime.com/pages/viewpage.action?pageId=28667962)

In your train .py file
   
    import importlib
    module_pruner = importlib.import_module('path to pytorch-pruner')
    Pruner = module_pruner.Pruner
    pruner = Pruner(model, distributed)

`model` The model to be pruned

`distributed` If your program is trained with multi-GPU, then set it `True`. Default `False`

In your train loop, when it reach you predefined prune point, do prune like below:

    if (iter + 1) % prune_freq == 0:
        pruner.rebuild_model(batch, to_loss_fn, optimizer, prune_num)
        continue

`batch` Input to `to_loss_fn`

`to_loss_fn` A function to calculate loss when given model and batch. Its prototype is

    def to_loss_fn(model, batch):
        input, label = batch
        output = model(input)
        return criterion(output, label)

Actually, you can pass anything into it via `batch`. 

You even not need to return the real loss, because the returned value is only used for building graph

`optimizer` The model's optimizer. If use lr_scheduler, you can pass lr_scheduler.optimizer

`prune_num` The number of channels you want to remove in this operation

## Supported Network Structures

### Classification
- AlexNet
- VGGNet
- Inception V3
- ResNet
- ResNet V2
- DenseNet
- SequeezeNet
- MobileNet V2

### Detection
- Faster-RCNN (FC/C4 head)
- RFCN
- RetinaNet
- FPN

### Keypoint
- Hourglass
- FPN
- Simple Baseline Net

## Performance

See results in our [Confluence Page](http://confluence.sensetime.com/display/VIBT/Classification+Performance)

## Reference

1. Molchanov P, Tyree S, Karras T, et al. Pruning convolutional neural networks for resource efficient inference[C]. ICLR 2017