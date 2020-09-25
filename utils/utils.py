"""
LF-Font
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import os
import math
import shutil
import re
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timedelta
import errno
import torch
from itertools import chain


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return "{:.3f} (val={:.3f}, count={})".format(self.avg, self.val, self.count)


class AverageMeters():
    def __init__(self, *keys):
        self.keys = keys
        for k in keys:
            setattr(self, k, AverageMeter())

    def resets(self):
        for k in self.keys:
            getattr(self, k).reset()

    def updates(self, dic, n=1):
        for k, v in dic.items():
            getattr(self, k).update(v, n)

    def __repr__(self):
        return "  ".join(["{}: {}".format(k, str(getattr(self, k))) for k in self.keys])


def accuracy(out, target):
    pred = out.max(1)[1]
    corr = (pred == target).sum().item()
    B = len(target)
    acc = float(corr) / B

    return acc


@contextmanager
def temporary_freeze(module):
    org_grads = freeze(module)
    yield
    unfreeze(module, org_grads)


def freeze(module):
    if module is None:
        return None

    org = []
    module.eval()
    for p in module.parameters():
        org.append(p.requires_grad)
        p.requires_grad_(False)

    return org


def unfreeze(module, org=None):
    if module is None:
        return

    module.train()
    if org is not None:
        org = iter(org)
    for p in module.parameters():
        grad = next(org) if org else True
        p.requires_grad_(grad)


def rm(path):
    """ remove dir recursively """
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def timestamp(fmt="%y%m%d_%H-%M-%S"):
    return datetime.now().strftime(fmt)


if __name__ == "__main__":
    import fire
    fire.Fire()
