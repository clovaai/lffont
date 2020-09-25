"""
LF-Font
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import copy
from tqdm import trange
from itertools import chain

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

import utils
from datasets import cyclize
from models import memory
from .trainer_utils import *
from pathlib import Path
try:
    from apex import amp
except ImportError:
    print('failed to import apex')


class BaseTrainer:
    def __init__(self, gen, disc, g_optim, d_optim, aux_clf, ac_optim,
                 writer, logger, evaluator, cv_loaders, cfg):
        self.gen = gen
        self.gen_ema = copy.deepcopy(self.gen)
        self.g_optim = g_optim

        self.is_bn_gen = has_bn(self.gen)
        self.disc = disc
        self.d_optim = d_optim

        self.aux_clf = aux_clf
        self.ac_optim = ac_optim

        self.cfg = cfg

        [self.gen, self.gen_ema, self.disc, self.aux_clf], [self.g_optim, self.d_optim, self.ac_optim] = self.set_model(
            [self.gen, self.gen_ema, self.disc, self.aux_clf],
            [self.g_optim, self.d_optim, self.ac_optim],
            num_losses=4,
            level=cfg.half_type
        )

        self.writer = writer
        self.logger = logger
        self.evaluator = evaluator
        self.cv_loaders = cv_loaders

        self.step = 1

        self.g_losses = {}
        self.d_losses = {}
        self.ac_losses = {}

        self.frozen_enc = copy.deepcopy(self.gen.component_encoder)
        utils.freeze(self.frozen_enc)
        self.ac_gen_decoder_only_grad = self.cfg.get('ac_gen_decoder_only_grad', False)

    def set_model(self, models, opts, num_losses, level="O2"):
        if self.cfg.use_half:
            models, opts = amp.initialize(models, opts, opt_level=level, num_losses=num_losses)

        if self.cfg.use_ddp:
            models = [DDP(m, [self.cfg.gpu]).module for m in models]

        return models, opts

    def clear_losses(self):
        """ Integrate & clear loss dict """
        # g losses
        loss_dic = {k: v.item() for k, v in self.g_losses.items()}
        loss_dic['g_total'] = sum(loss_dic.values())
        # d losses
        loss_dic.update({k: v.item() for k, v in self.d_losses.items()})
        # ac losses
        loss_dic.update({k: v.item() for k, v in self.ac_losses.items()})

        self.g_losses = {}
        self.d_losses = {}
        self.ac_losses = {}

        return loss_dic

    def accum_g(self, decay=0.999):
        par1 = dict(self.gen_ema.named_parameters())
        par2 = dict(self.gen.named_parameters())

        for k in par1.keys():
            par1[k].data.mul_(decay).add_(par2[k].data, alpha=(1 - decay))

    def sync_g_ema(self, in_style_ids, in_comp_ids, in_imgs, trg_style_ids, trg_comp_ids,
                   content_imgs):
        return

    def train(self):
        return

    def add_pixel_loss(self, out, target):
        loss = F.l1_loss(out, target, reduction="mean") * self.cfg['pixel_w']
        self.g_losses['pixel'] = loss
        return loss

    def add_gan_g_loss(self, real_font, real_uni, fake_font, fake_uni):
        if self.cfg['gan_w'] == 0.:
            return 0.

        g_loss = -(fake_font.mean() + fake_uni.mean())

        g_loss *= self.cfg['gan_w']
        self.g_losses['gen'] = g_loss

        return g_loss

    def add_gan_d_loss(self, real_font, real_uni, fake_font, fake_uni):
        if self.cfg['gan_w'] == 0.:
            return 0.

        d_loss = F.relu(1. - real_font).mean() + F.relu(1. + fake_font).mean() + \
                 F.relu(1. - real_uni).mean() + F.relu(1. + fake_uni).mean()
        d_loss *= self.cfg['gan_w']

        self.d_losses['disc'] = d_loss

        return d_loss

    def add_fm_loss(self, real_feats, fake_feats):
        if self.cfg['fm_w'] == 0.:
            return 0.

        fm_loss = 0.
        for real_f, fake_f in zip(real_feats, fake_feats):
            fm_loss += F.l1_loss(real_f.detach(), fake_f)
        fm_loss = fm_loss / len(real_feats) * self.cfg['fm_w']

        self.g_losses['fm'] = fm_loss

        return fm_loss

    def infer_ac(self, sc_feats, comp_ids):
        aux_out, aux_feats = self.aux_clf(sc_feats)
        loss = F.cross_entropy(aux_out, comp_ids)
        acc = utils.accuracy(aux_out, comp_ids)
        return aux_feats, loss, acc

    def d_backward(self):
        with utils.temporary_freeze(self.gen):
            d_loss = sum(self.d_losses.values())
            if self.cfg.use_half:
                with amp.scale_loss(d_loss, self.d_optim, loss_id=0) as scaled_d_loss:
                    scaled_d_loss.backward()
            else:
                d_loss.backward()

    def g_backward(self):
        with utils.temporary_freeze(self.disc):
            g_loss = sum(self.g_losses.values())
            if self.cfg.use_half:
                with amp.scale_loss(g_loss, self.g_optim, loss_id=1) as scaled_g_loss:
                    scaled_g_loss.backward()
            else:
                g_loss.backward()

    def ac_backward(self):
        if self.aux_clf is None:
            return

        if 'ac' in self.ac_losses:
            if self.cfg.use_half:
                with amp.scale_loss(self.ac_losses['ac'], [self.ac_optim, self.g_optim], loss_id=2) as scaled_ac_loss:
                    scaled_ac_loss.backward(retain_graph=True)
            else:
                self.ac_losses['ac'].backward(retain_graph=True)

        if 'ac_gen' in self.ac_losses:
            with utils.temporary_freeze(self.aux_clf):
                loss = self.ac_losses.get('ac_gen', 0.)
                if self.cfg.use_half:
                    with amp.scale_loss(loss, [self.ac_optim, self.g_optim], loss_id=3) as scaled_ac_g_loss:
                        scaled_ac_g_loss.backward(retain_graph=True)
                else:
                    loss.backward(retain_graph=True)

    def save(self, cur_loss, method, save_freq=None):
        """
        Args:
            method: all / last
                all: save checkpoint by step
                last: save checkpoint to 'last.pth'
                all-last: save checkpoint by step per save_freq and
                          save checkpoint to 'last.pth' always
        """
        if method not in ['all', 'last', 'all-last']:
            return

        step_save = False
        last_save = False
        if method == 'all' or (method == 'all-last' and self.step % save_freq == 0):
            step_save = True
        if method == 'last' or method == 'all-last':
            last_save = True
        assert step_save or last_save

        save_dic = {
            'generator': self.gen.state_dict(),
            'generator_ema': self.gen_ema.state_dict(),
            'optimizer': self.g_optim.state_dict(),
            'epoch': self.step,
            'loss': cur_loss
        }
        if self.disc is not None:
            save_dic['discriminator'] = self.disc.state_dict()
            save_dic['d_optimizer'] = self.d_optim.state_dict()

        if self.aux_clf is not None:
            save_dic['aux_clf'] = self.aux_clf.state_dict()
            save_dic['ac_optimizer'] = self.ac_optim.state_dict()

        ckpt_dir = self.cfg['work_dir'] / "checkpoints" / self.cfg['unique_name']
        step_ckpt_name = "{:06d}-{}.pth".format(self.step, self.cfg['name'])
        last_ckpt_name = "last.pth"
        step_ckpt_path = Path.cwd() /ckpt_dir / step_ckpt_name
        last_ckpt_path = ckpt_dir / last_ckpt_name

        log = ""
        if step_save:
            torch.save(save_dic, str(step_ckpt_path))
            log = "Checkpoint is saved to {}".format(step_ckpt_path)

            if last_save:
                utils.rm(last_ckpt_path)
                last_ckpt_path.symlink_to(step_ckpt_path)
                log += " and symlink to {}".format(last_ckpt_path)

        if not step_save and last_save:
            utils.rm(last_ckpt_path)  # last 가 symlink 일 경우 지우고 써줘야 함.
            torch.save(save_dic, str(last_ckpt_path))
            log = "Checkpoint is saved to {}".format(last_ckpt_path)

        self.logger.info("{}\n".format(log))

    def baseplot(self, losses, discs, stats):
        tag_scalar_dic = {
            'train/g_total_loss': losses.g_total.val,
            'train/pixel_loss': losses.pixel.val
        }

        if self.disc is not None:
            tag_scalar_dic.update({
                'train/d_loss': losses.disc.val,
                'train/g_loss': losses.gen.val,
                'train/d_real_font': discs.real_font.val,
                'train/d_real_uni': discs.real_uni.val,
                'train/d_fake_font': discs.fake_font.val,
                'train/d_fake_uni': discs.fake_uni.val,

                'train/d_real_font_acc': discs.real_font_acc.val,
                'train/d_real_uni_acc': discs.real_uni_acc.val,
                'train/d_fake_font_acc': discs.fake_font_acc.val,
                'train/d_fake_uni_acc': discs.fake_uni_acc.val
            })

            if self.cfg['fm_w'] > 0.:
                tag_scalar_dic['train/feature_matching'] = losses.fm.val

        if self.aux_clf is not None:
            tag_scalar_dic.update({
                'train/ac_loss': losses.ac.val,
                'train/ac_acc': stats.ac_acc.val
            })

            if self.cfg['ac_gen_w'] > 0.:
                tag_scalar_dic.update({
                    'train/ac_gen_loss': losses.ac_gen.val,
                    'train/ac_gen_acc': stats.ac_gen_acc.val
                })

        self.writer.add_scalars(tag_scalar_dic, self.step)

    def log(self, losses, discs, stats):
        self.logger.info(
            "  Step {step:7d}: L1 {L.pixel.avg:7.4f}  D {L.disc.avg:7.3f}  G {L.gen.avg:7.3f}"
            "  FM {L.fm.avg:7.3f}  AC_loss {L.ac.avg:7.3f}  AC {S.ac_acc.avg:5.1%}  AC_gen {S.ac_gen_acc.avg:5.1%}"  # "  AC_fm {L.ac_fm.avg:7.3f}"
            "  R_font {D.real_font_acc.avg:7.3f}  F_font {D.fake_font_acc.avg:7.3f}"
            "  R_uni {D.real_uni_acc.avg:7.3f}  F_uni {D.fake_uni_acc.avg:7.3f}"
            "  B_stl {S.B_style.avg:5.1f}  B_trg {S.B_target.avg:5.1f}"
                .format(step=self.step, L=losses, D=discs, S=stats))
