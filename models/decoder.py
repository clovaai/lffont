"""
LF-Font
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import copy
from functools import partial
import torch
import torch.nn as nn
from .modules import ConvBlock, ResBlock
import torch.nn.functional as F


class Integrator(nn.Module):
    def __init__(self, C, norm='none', activ='none', C_in=None, C_content=0):
        super().__init__()
        C_in = (C_in or C) + C_content
        self.integrate_layer = ConvBlock(C_in, C, 1, 1, 0, norm=norm, activ=activ)

    def forward(self, comps, x=None, content=None):
        """
        Args:
            comps [B, 3, mem_shape]: component features
        """
        if content is not None:
            inputs = torch.cat([comps, content], dim=1)
        else:
            inputs = comps
        out = self.integrate_layer(inputs)

        if x is not None:
            out = torch.cat([x, out], dim=1)

        return out


class Decoder(nn.Module):
    def __init__(self, layers, skips=None, out='sigmoid'):
        super().__init__()
        self.layers = nn.ModuleList(layers)

        if skips is not None:
            self.skip_idx, self.skip_layer = skips

        if out == 'sigmoid':
            self.out = nn.Sigmoid()
        elif out == 'tanh':
            self.out = nn.Tanh()
        else:
            raise ValueError(out)

    def forward(self, x, skip_feat=None, content_feats=None):
        for i, layer in enumerate(self.layers):
            if i == self.skip_idx:
                x = self.skip_layer(skip_feat, x=x)
            if i == 0:
                x = layer(x, content=content_feats)
            else:
                x = layer(x)

        return self.out(x)


def dec_builder(C, C_out, norm='IN', activ='relu', pad_type='reflect', out='sigmoid', C_content=0):

    ConvBlk = partial(ConvBlock, norm=norm, activ=activ, pad_type=pad_type)
    ResBlk = partial(ResBlock, norm=norm, activ=activ, pad_type=pad_type)

    IntegrateBlk = partial(Integrator, norm='none', activ='none')

    layers = [
        IntegrateBlk(C*8, C_content=C_content),
        ResBlk(C*8, C*8, 3, 1),
        ResBlk(C*8, C*8, 3, 1),
        ResBlk(C*8, C*8, 3, 1),
        ConvBlk(C*8, C*4, 3, 1, 1, upsample=True),   # 32x32
        ConvBlk(C*8, C*2, 3, 1, 1, upsample=True),   # 64x64
        ConvBlk(C*2, C*1, 3, 1, 1, upsample=True),   # 128x128
        ConvBlk(C*1, C_out, 3, 1, 1)
    ]

    skips = (5, IntegrateBlk(C * 4))

    return Decoder(layers, skips, out=out)
