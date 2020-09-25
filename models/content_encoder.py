"""
LF-Font
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
from functools import partial
import torch.nn as nn
from .modules import LinearBlock, ConvBlock, Flatten, ResBlock


class ContentEncoder(nn.Module):
    def __init__(self, layers, sigmoid=False):
        super().__init__()
        self.net = nn.Sequential(*layers)
        self.sigmoid = sigmoid

    def forward(self, x):
        x = x.repeat((1, 1, 1, 1))
        out = self.net(x)

        if self.sigmoid:
            out = nn.Sigmoid()(out)

        return out


def content_enc_builder(C_in, C, C_out, norm='none', activ='relu', content_sigmoid=False,
                        pad_type='zero'):
    if not C_out:
        return None

    ConvBlk = partial(ConvBlock, norm=norm, activ=activ, pad_type=pad_type)
    ResBlk = partial(ResBlock, norm=norm, activ=activ, pad_type=pad_type)

    layers = [
        ConvBlk(C_in, C, 3, 1, 1, norm='none', activ='none'),
        ConvBlk(C*1, C*2, 3, 2, 1),  # 64x64
        ConvBlk(C*2, C*4, 3, 2, 1),  # 32x32
        ConvBlk(C*4, C*8, 3, 2, 1),  # 16x16
        ConvBlk(C*8, C_out, 3, 1, 1)
    ]

    return ContentEncoder(layers, content_sigmoid)
