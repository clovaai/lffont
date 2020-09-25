"""
LF-Font
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import copy
from functools import partial
import torch.nn as nn
import torch
from .modules import ConvBlock, ResBlock, GCBlock, ParamBlock, CBAM


class ComponentConditionBlock(nn.Module):
    def __init__(self, in_shape, n_comps):
        super().__init__()
        self.in_shape = in_shape
        self.bias = nn.Parameter(torch.zeros(n_comps, in_shape[0], 1, 1), requires_grad=True)

    def forward(self, x, comp_id):
        b = self.bias[comp_id]
        out = x + b
        return out


class ComponentEncoder(nn.Module):
    def __init__(self, body, head, final_shape, skip_shape, sigmoid=False, skip_layer_idx=None):
        super().__init__()

        self.body = nn.ModuleList(body)
        self.head = nn.ModuleList(head)
        self.final_shape = final_shape
        self.skip_shape = skip_shape
        self.skip_layer_idx = skip_layer_idx
        self.sigmoid = sigmoid

    def forward(self, x, comp_id=None):
        x = x.repeat((1, 1, 1, 1))
        ret_feats = {}

        for layer in self.body:
            if isinstance(layer, ComponentConditionBlock):
                x = layer(x, comp_id)
            else:
                x = layer(x)
        for lidx, layer in enumerate(self.head):
            x = layer(x)
            if lidx == self.skip_layer_idx:
                ret_feats["skip"] = x

        ret_feats["last"] = x

        if self.sigmoid:
            ret_feats = {k: nn.Sigmoid()(v) for k, v in ret_feats.items()}

        return ret_feats


def comp_enc_builder(C_in, C, norm='none', activ='relu', pad_type='reflect', sigmoid=True, skip_scale_var=False, n_comps=None):

    ConvBlk = partial(ConvBlock, norm=norm, activ=activ, pad_type=pad_type)
    ResBlk = partial(ResBlock, norm=norm, activ=activ, pad_type=pad_type, scale_var=skip_scale_var)

    body = [
        ConvBlk(C_in, C, 3, 1, 1, norm='none', activ='none'),  # 128x128
        ConvBlk(C * 1, C * 2, 3, 1, 1, downsample=True),  # 64x64
        GCBlock(C * 2),
        ConvBlk(C * 2, C * 4, 3, 1, 1, downsample=True),  # 32x32
        CBAM(C * 4),
        ComponentConditionBlock((128, 32, 32), n_comps)
        ]
    head = [
        ResBlk(C * 4, C * 4, 3, 1),
        CBAM(C * 4),
        ResBlk(C * 4, C * 4, 3, 1),
        ResBlk(C * 4, C * 8, 3, 1, downsample=True),  # 16x16
        CBAM(C * 8),
        ResBlk(C * 8, C * 8)
    ]

    skip_layer_idx = 2

    final_shape = (C*8, 16, 16)
    skip_shape = (C*4, 32, 32)

    return ComponentEncoder(body, head, final_shape, skip_shape, sigmoid, skip_layer_idx)


def decompose_block_builder(emb_dim, in_shape, num_blocks=2):

    blocks = [ParamBlock(emb_dim, (in_shape[0], 1, 1)) for _ in range(num_blocks)]

    return blocks
