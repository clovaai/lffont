"""
LF-Font
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import torch
import torch.nn as nn


def has_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return True

    return False


def unflatten_B(t):
    """ Unflatten [B*3, ...] tensor to [B, 3, ...] tensor
    t is flattened tensor from component batch, which is [B, 3, ...] tensor
    """
    shape = t.shape
    return t.view(shape[0]//3, 3, *shape[1:])


def overwrite_weight(model, pre_weight):
    model_dict = model.state_dict()
    pre_weight = {k: v for k, v in pre_weight.items() if k in model_dict}

    model_dict.update(pre_weight)
    model.load_state_dict(model_dict)


def load_checkpoint(path, gen, disc, aux_clf, g_optim, d_optim, ac_optim, force_overwrite=False):
    ckpt = torch.load(path)

    if force_overwrite:
        overwrite_weight(gen, ckpt['generator'])
    else:
        gen.load_state_dict(ckpt['generator'])
        g_optim.load_state_dict(ckpt['optimizer'])

    if disc is not None:
        if force_overwrite:
            overwrite_weight(disc, ckpt['discriminator'])
        else:
            disc.load_state_dict(ckpt['discriminator'])
            d_optim.load_state_dict(ckpt['d_optimizer'])

    if aux_clf is not None:
        if force_overwrite:
            overwrite_weight(aux_clf, ckpt['aux_clf'])
        else:
            aux_clf.load_state_dict(ckpt['aux_clf'])
            ac_optim.load_state_dict(ckpt['ac_optimizer'])

    st_epoch = ckpt['epoch'] + 1
    loss = ckpt['loss']

    return st_epoch, loss
