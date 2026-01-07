import torch.nn as nn
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor
import torch


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    """3x3 convolution with padding"""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self,In_channel,Out_channel,Out_W,downsample=False):
        super(BasicBlock, self).__init__()
        self.stride = 1
        if downsample == True:
            self.stride = 2

        self.layer = nn.Sequential(
            nn.Conv1d(In_channel, Out_channel, 3, self.stride, padding=1),
            # nn.BatchNorm1d(Out_channel),
            nn.LayerNorm([Out_channel,Out_W]),
            nn.LeakyReLU(),
            nn.Conv1d(Out_channel, Out_channel, 3, padding=1),
            # nn.BatchNorm1d(Out_channel),
            nn.LayerNorm([Out_channel,Out_W]),
            nn.LeakyReLU(),
        )

        if In_channel != Out_channel:
            self.res_layer = nn.Conv1d(In_channel, Out_channel,1,self.stride)
        else:
            self.res_layer = None
    def forward(self,x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x)+residual

class Bottleneck(nn.Module):
    def __init__(self,In_channel,Med_channel,Out_channel,Out_W,downsample=False):
        super(Bottleneck, self).__init__()
        self.stride = 1
        if downsample == True:
            self.stride = 2

        self.layer = nn.Sequential(
            nn.Conv1d(In_channel, Med_channel, 1, self.stride),
            # nn.BatchNorm1d(Med_channel),
            nn.LayerNorm([Med_channel,Out_W]),
            nn.LeakyReLU(),
            nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
            # nn.BatchNorm1d(Med_channel),
            nn.LayerNorm([Med_channel,Out_W]),
            nn.LeakyReLU(),
            nn.Conv1d(Med_channel, Out_channel, 1),
            # nn.BatchNorm1d(Out_channel),
            nn.LayerNorm([Out_channel,Out_W]),
            nn.LeakyReLU(),
        )

        if In_channel != Out_channel:
            self.res_layer = nn.Conv1d(In_channel, Out_channel,1,self.stride)
        else:
            self.res_layer = None

    def forward(self,x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x)+residual


class ResNet(nn.Module):
    def __init__(self,in_channels=12,embed_size=512):
        super(ResNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_channels)

        self.features = nn.Sequential(
            nn.Conv1d(in_channels,32,kernel_size=7,stride=2,padding=3),
            nn.MaxPool1d(3,2,1),

            Bottleneck(32,16,64,500,False),
            Bottleneck(64,16,64,500,False),
            Bottleneck(64,16,64,500,False),
            #
            Bottleneck(64,32,128,250, True),
            Bottleneck(128,32,128,250, False),
            Bottleneck(128,32,128,250, False),
            Bottleneck(128,32,128,250, False),
            #
            Bottleneck(128,64,256,125, True),
            Bottleneck(256,64,256,125, False),
            Bottleneck(256,64,256,125, False),
            Bottleneck(256,64,256,125, False),
            Bottleneck(256,64,256,125, False),
            Bottleneck(256,64,256,125, False),
            #
            Bottleneck(256,128,512,63, True),
            Bottleneck(512,128,512,63, False),
            Bottleneck(512,128,512,63, False),
            nn.AdaptiveAvgPool1d(1)
        )


    def forward(self,x):
        x = self.norm(x)
        x = self.features(x)
        x = torch.flatten(x, 1)

        return x
