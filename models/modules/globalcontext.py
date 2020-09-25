"""
LF-Font
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import w_norm_dispatch


class GlobalContext(nn.Module):
    """ Global-context """
    def __init__(self, C, bottleneck_ratio=0.25, w_norm='none'):
        super().__init__()
        C_bottleneck = int(C * bottleneck_ratio)
        w_norm = w_norm_dispatch(w_norm)
        self.k_proj = w_norm(nn.Conv2d(C, 1, 1))
        self.transform = nn.Sequential(
            w_norm(nn.Linear(C, C_bottleneck)),
            nn.LayerNorm(C_bottleneck),
            nn.ReLU(),
            w_norm(nn.Linear(C_bottleneck, C))
        )

    def forward(self, x):
        # x: [B, C, H, W]
        context_logits = self.k_proj(x)  # [B, 1, H, W]
        context_weights = F.softmax(context_logits.flatten(1), dim=1)  # [B, HW]
        context = torch.einsum('bci,bi->bc', x.flatten(2), context_weights)
        out = self.transform(context)

        return out[..., None, None]


class GCBlock(nn.Module):
    """ Global-context block """
    def __init__(self, C, bottleneck_ratio=0.25, w_norm='none'):
        super().__init__()
        self.gc = GlobalContext(C, bottleneck_ratio, w_norm)

    def forward(self, x):
        gc = self.gc(x)
        return x + gc
