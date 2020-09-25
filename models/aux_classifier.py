""" Auxiliary component classifier
LF-Font
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
from functools import partial
import torch.nn as nn
from .modules import LinearBlock, ConvBlock, ResBlock, Flatten


class AuxClassifier(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

        self.feat_class = ()
        self.n_last = 0

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, self.feat_class):
                features.append(x)

        if self.n_last:
            features = features[-self.n_last:]

        return x, features


def aux_clf_builder(in_shape, C_out, norm='IN', gap_size=8, activ='relu',
                    pad_type='reflect', conv_dropout=0., clf_dropout=0.):

    ConvBlk = partial(ConvBlock, norm=norm, activ=activ, pad_type=pad_type, dropout=conv_dropout)
    ResBlk = partial(ResBlock, norm=norm, activ=activ, pad_type=pad_type, dropout=conv_dropout)
    LinearBlk = partial(LinearBlock, norm=norm, activ=activ, dropout=clf_dropout)

    assert in_shape[1] == in_shape[2]
    HW = in_shape[1]
    C = in_shape[0]

    layers = [
        ResBlk(C, C*2, 3, 1, downsample=True),
        ResBlk(C*2, C*2, 3, 1),
        nn.AdaptiveAvgPool2d(1),
        Flatten(1),
        nn.Dropout(clf_dropout),
        nn.Linear(C*2, C_out)
    ]

    aux_clf = AuxClassifier(layers)
    return aux_clf
