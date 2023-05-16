from __future__ import absolute_import
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from .pooling import build_pooling_layer
import sys

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, pooling_type='avg'):
        print('pooling_type: {}'.format(pooling_type))
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNet.__factory[depth](pretrained=False)
        resnet.load_state_dict(torch.load('./pretrained/resnet50-19c8e357.pth'))
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

       

    def forward(self, x):
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


        return bn_x, bn_x2, bn_x3



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


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)
