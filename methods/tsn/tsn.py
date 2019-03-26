import torch
import linklink
from torch import nn
from networks import backbone_entry

__all__ = ['get_tsn']


def get_tsn(config):
    return TSN(backbone=config['backbone'], num_class=config['num_class'],
               dropout=config['dropout'], pretrained=config['pretrained'],
               num_segments=config['num_segments'], **config['kwargs'])


class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality='RGB',
                 backbone='resnet', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.5, crop_num=1, partial_bn=False,
                 pretrained=False, **kwargs):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        self.new_length = 1

        print(("""
            Initializing TSN with base model: {}.
            TSN Configurations:
            input_modality:     {}
            num_segments:       {}
            new_length:         {}
            consensus_module:   {}
            dropout_ratio:      {}
            """.format(backbone, self.modality, self.num_segments,
                       self.new_length, consensus_type, self.dropout)))

        self._prepare_base_model(backbone, pretrained, **kwargs)

        self._prepare_tsn(num_class)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        # get the input dimension of the last fc layer
        try:
            feature_dim = getattr(self.base_model,
                                  self.base_model.last_layer_name).in_features
        except AttributeError:
            print("No attribute in_features.\nTry in_channels.")
            feature_dim = getattr(self.base_model,
                                  self.base_model.last_layer_name).in_channels

        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name,
                    nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name,
                    nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        if not self.new_fc:
            nn.init.normal_(
                getattr(self.base_model,
                        self.base_model.last_layer_name).weight,
                mean=0, std=0.01)
            nn.init.constant_(
                getattr(self.base_model,
                        self.base_model.last_layer_name).bias, 0)
        else:
            nn.init.normal_(self.new_fc.weight, mean=0, std=0.01)
            nn.init.constant_(self.new_fc.bias, 0)

        return

    def _prepare_base_model(self, backbone, pretrained, **kwargs):
        print("Base model is", backbone)
        self.base_model = backbone_entry(backbone, pretrained, **kwargs)
        if 'resnet' in backbone or 'vgg' in backbone:
            self.base_model.last_layer_name = 'fc'

        elif 'proxyless' in backbone:
            self.base_model.last_layer_name = 'classifier'

        elif 'mobilenet' in backbone:
            self.base_model.last_layer_name = 'fc'

        elif 'inception' in backbone:
            # import tf_model_zoo
            print("Using inceptionv3")
            self.base_model.last_layer_name = 'fc'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise NameError('Unknown backbone: {}'.format(backbone))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                # print('the type train model : {}'.format(type(m)))
                if isinstance(m, torch.nn.BatchNorm2d) or \
                   isinstance(m, linklink.nn.syncbn_layer.SyncBatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # print('the freeze module: {} of {}th'.format(type(m), count))
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt <= 2:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    # print('the new module type :{}'.format(type(m)))
                    bn_cnt += 1
                    if not self._enable_pbn or bn_cnt <= 2:
                        bn.extend(list(m.parameters()))
                    # raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, input):
        # print("input shape is:", input.shape)
        sample_len = 3 * self.new_length

        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)
        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
        # output1 = base_out(0,0,:,:)
        # output2 = base_out(0,1,:,:)
        # output3 = base_out(0,2,:,:)
        # print(output1.topk(1,1,True))
        # print(output1)
        # print(output2.topk(1,1,True))
        # print(output2)
        # print(output3.topk(1,1,True))
        # print(output3)
        output = base_out.mean(dim=1, keepdim=True)
        return output.squeeze(1)

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224
