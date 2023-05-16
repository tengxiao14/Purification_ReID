from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
import sys
from .pooling import build_pooling_layer

from .resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a


__all__ = ['ResNetIBN', 'resnet_ibn50a', 'resnet_ibn101a']


class ResNetIBN(nn.Module):
    __factory = {
        '50a': resnet50_ibn_a,
        '101a': resnet101_ibn_a
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, pooling_type='avg'):

        print('pooling_type: {}'.format(pooling_type))
        super(ResNetIBN, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        resnet = ResNetIBN.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)

        self.gap = build_pooling_layer(pooling_type)
        self.gap2 = build_pooling_layer(pooling_type)
        self.gap3 = build_pooling_layer(pooling_type)


        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                self.feat_bn2 = nn.BatchNorm1d(self.num_features)
                self.feat_bn3 = nn.BatchNorm1d(self.num_features)

            self.feat_bn.bias.requires_grad_(False)
            self.feat_bn2.bias.requires_grad_(False)
            self.feat_bn3.bias.requires_grad_(False)

            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)

        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)
        init.constant_(self.feat_bn2.weight, 1)
        init.constant_(self.feat_bn2.bias, 0)
        init.constant_(self.feat_bn3.weight, 1)
        init.constant_(self.feat_bn3.bias, 0)


        if not pretrained:
            self.reset_params()

    def forward(self, x):
        """
        x = self.base(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        """
        bs = x.size(0)
        layer4 = self.base(x)

        x = layer4

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        margin = layer4.size(2) // 2
        feat_local1 = layer4[:, :, 0:margin, :]
        feat_local2 = layer4[:, :, margin:margin*2, :]
        feat_local1 = self.gap2(feat_local1)
        feat_local2 = self.gap3(feat_local2)
        feat_local1 = feat_local1.view(feat_local1.size(0), -1)
        feat_local2 = feat_local2.view(feat_local2.size(0), -1)


        if self.cut_at_pooling:
            return x, feat_local1, feat_local2

        if self.has_embedding:
            print('embedding')
            sys.exit()
            bn_x = self.feat_bn(self.feat(x))

        else:
            bn_x = self.feat_bn(x)
            bn_x2 = self.feat_bn2(feat_local1)
            bn_x3 = self.feat_bn3(feat_local2)

        if (self.training is False):
            bn_x = F.normalize(bn_x)
            bn_x2 = F.normalize(bn_x2)
            bn_x3 = F.normalize(bn_x3)
            return bn_x, bn_x2, bn_x3

        if self.norm:
            bn_x = F.normalize(bn_x)
            bn_x2 = F.normalize(bn_x2)
            bn_x3 = F.normalize(bn_x3)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)
            bn_x2 = F.relu(bn_x2)
            bn_x3 = F.relu(bn_x3)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)
            bn_x2 = self.drop(bn_x2)
            bn_x3 = self.drop(bn_x3)

        if self.num_classes > 0:
            print('classify')
            sys.exit()
            prob = self.classifier(bn_x)
        else:
            return bn_x, bn_x2, bn_x3

        return prob

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def resnet_ibn50a(**kwargs):
    return ResNetIBN('50a', **kwargs)


def resnet_ibn101a(**kwargs):
    return ResNetIBN('101a', **kwargs)
