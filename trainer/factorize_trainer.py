"""
LF-Font
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import torch

from .base_trainer import BaseTrainer

from datasets import cyclize
import utils
import copy
from itertools import chain


class FactorizeTrainer(BaseTrainer):
    def __init__(self, gen, disc, g_optim, d_optim, aux_clf, ac_optim,
                 writer, logger, evaluator, cv_loaders, cfg):
        super().__init__(gen, disc, g_optim, d_optim, aux_clf, ac_optim,
                         writer, logger, evaluator, cv_loaders, cfg)

        self.frozen_emb_style = copy.deepcopy(self.gen.emb_style)
        self.frozen_emb_comp = copy.deepcopy(self.gen.emb_comp)
        utils.freeze(self.frozen_emb_style)
        utils.freeze(self.frozen_emb_comp)

    def sync_g_ema(self, in_style_ids, in_comp_ids, in_imgs, trg_style_ids, trg_comp_ids,
                   content_imgs):
        org_train_mode = self.gen_ema.training
        with torch.no_grad():
            self.gen_ema.train()
            self.gen_ema.encode_write_fact(in_style_ids, in_comp_ids, in_imgs)
            self.gen_ema.read_decode(trg_style_ids, trg_comp_ids, content_imgs=content_imgs,
                                     phase="fact")

        self.gen_ema.train(org_train_mode)

    def train(self, loader, st_step=1, max_step=100000):

        self.gen.train()
        self.disc.train()

        losses = utils.AverageMeters("g_total", "pixel", "disc", "gen", "fm",
                                     "ac", "ac_gen", "dec_const")
        discs = utils.AverageMeters("real_font", "real_uni", "fake_font", "fake_uni",
                                    "real_font_acc", "real_uni_acc",
                                    "fake_font_acc", "fake_uni_acc")
        # etc stats
        stats = utils.AverageMeters("B_style", "B_target", "ac_acc", "ac_gen_acc")

        self.step = st_step
        self.clear_losses()

        self.logger.info("Start training ...")

        for (in_style_ids, in_comp_ids, in_imgs,
             trg_style_ids, trg_uni_ids, trg_comp_ids, trg_imgs, content_imgs) in cyclize(loader):

            epoch = self.step // len(loader)
            if self.cfg.use_ddp and (self.step % len(loader)) == 0:
                loader.sampler.set_epoch(epoch)

            B = trg_imgs.size(0)
            stats.updates({
                "B_style": in_imgs.size(0),
                "B_target": B
            })

            in_style_ids = in_style_ids.cuda()
            in_comp_ids = in_comp_ids.cuda()
            in_imgs = in_imgs.cuda()

            trg_style_ids = trg_style_ids.cuda()
            trg_imgs = trg_imgs.cuda()

            content_imgs = content_imgs.cuda()

            if self.cfg.use_half:
                in_imgs = in_imgs.half()
                content_imgs = content_imgs.half()

            feat_styles, feat_comps = self.gen.encode_write_fact(
                in_style_ids, in_comp_ids, in_imgs, write_comb=True
            )
            feats_rc = (feat_styles * feat_comps).sum(1)
            ac_feats = feats_rc
            self.add_dec_const_loss()

            out = self.gen.read_decode(
                trg_style_ids, trg_comp_ids, content_imgs=content_imgs, phase="fact", try_comb=True
            )

            trg_uni_disc_ids = trg_uni_ids.cuda()

            real_font, real_uni, *real_feats = self.disc(
                trg_imgs, trg_style_ids, trg_uni_disc_ids, out_feats=self.cfg['fm_layers']
            )

            fake_font, fake_uni = self.disc(out.detach(), trg_style_ids, trg_uni_disc_ids)
            self.add_gan_d_loss(real_font, real_uni, fake_font, fake_uni)

            self.d_optim.zero_grad()
            self.d_backward()
            self.d_optim.step()

            fake_font, fake_uni, *fake_feats = self.disc(
                out, trg_style_ids, trg_uni_disc_ids, out_feats=self.cfg['fm_layers']
            )
            self.add_gan_g_loss(real_font, real_uni, fake_font, fake_uni)

            self.add_fm_loss(real_feats, fake_feats)

            def racc(x):
                return (x > 0.).float().mean().item()

            def facc(x):
                return (x < 0.).float().mean().item()

            discs.updates({
                "real_font": real_font.mean().item(),
                "real_uni": real_uni.mean().item(),
                "fake_font": fake_font.mean().item(),
                "fake_uni": fake_uni.mean().item(),
                'real_font_acc': racc(real_font),
                'real_uni_acc': racc(real_uni),
                'fake_font_acc': facc(fake_font),
                'fake_uni_acc': facc(fake_uni)
            }, B)

            self.add_pixel_loss(out, trg_imgs)

            self.g_optim.zero_grad()
            if self.aux_clf is not None:
                self.add_ac_losses_and_update_stats(
                    ac_feats, in_comp_ids, out, trg_comp_ids, stats
                )
                self.ac_optim.zero_grad()
                self.ac_backward()
                self.ac_optim.step()

            self.g_backward()
            self.g_optim.step()

            loss_dic = self.clear_losses()
            losses.updates(loss_dic, B)  # accum loss stats

            self.accum_g()
            if self.is_bn_gen:
                self.sync_g_ema(in_style_ids, in_comp_ids, in_imgs, trg_style_ids, trg_comp_ids,
                                content_imgs=content_imgs)

            torch.cuda.synchronize()

            if self.cfg.gpu <= 0:
                if self.step % self.cfg['tb_freq'] == 0:
                    self.baseplot(losses, discs, stats)
                    self.plot(losses)

                if self.step % self.cfg['print_freq'] == 0:
                    self.log(losses, discs, stats)
                    self.logger.debug("GPU Memory usage: max mem_alloc = %.1fM / %.1fM",
                                      torch.cuda.max_memory_allocated() / 1000 / 1000,
                                      torch.cuda.max_memory_cached() / 1000 / 1000)
                    losses.resets()
                    discs.resets()
                    stats.resets()

                if self.step % self.cfg['val_freq'] == 0:
                    epoch = self.step / len(loader)
                    self.logger.info("Validation at Epoch = {:.3f}".format(epoch))
                    if not self.is_bn_gen:
                        self.sync_g_ema(in_style_ids, in_comp_ids, in_imgs, trg_style_ids, trg_comp_ids,
                                        content_imgs=content_imgs)
                    self.evaluator.cp_validation(self.gen_ema, self.cv_loaders, self.step, phase="fact",
                                                 ext_tag="factorize")

                    self.save(loss_dic['g_total'], self.cfg['save'], self.cfg.get('save_freq', self.cfg['val_freq']))
            else:
                pass

            if self.step >= max_step:
                break

            self.step += 1

        self.logger.info("Iteration finished.")

    def add_dec_const_loss(self):
        loss = self.gen.get_fact_memory_var()
        self.g_losses['dec_const'] = loss * self.cfg["dec_const_w"]
        return loss

    def add_ac_losses_and_update_stats(self, in_sc_feats, in_comp_ids, generated, trg_comp_ids, stats):
        aux_feats, loss, acc = self.infer_ac(in_sc_feats, in_comp_ids)
        self.ac_losses['ac'] = loss * self.cfg['ac_w']
        stats.ac_acc.update(acc, in_comp_ids.numel())

        if self.cfg['ac_gen_w'] > 0.:
            enc = self.frozen_enc
            enc.load_state_dict(self.gen.component_encoder.state_dict())

            emb_style = self.frozen_emb_style
            emb_comp = self.frozen_emb_comp
            emb_style.load_state_dict(self.gen.emb_style.state_dict())
            emb_comp.load_state_dict(self.gen.emb_comp.state_dict())

            trg_comp_lens = torch.LongTensor([*map(len, trg_comp_ids)]).cuda()
            trg_comp_ids = torch.LongTensor([*chain(*trg_comp_ids)]).cuda()
            generated = generated.repeat_interleave(trg_comp_lens, dim=0)

            feats = enc(generated, trg_comp_ids)
            gen_feats = feats["last"]
            gen_emb_style = emb_style(gen_feats.unsqueeze(1))
            gen_emb_comp = emb_comp(gen_feats.unsqueeze(1))

            gen_recon = (gen_emb_style * gen_emb_comp).sum(1)

            aux_gen_feats, loss, acc = self.infer_ac(gen_recon, trg_comp_ids)
            stats.ac_gen_acc.update(acc, trg_comp_ids.numel())
            self.ac_losses['ac_gen'] = loss * self.cfg['ac_gen_w']

    def plot(self, losses):
        tag_scalar_dic = {
            "train/dec_const_loss": losses.dec_const.val
        }
        self.writer.add_scalars(tag_scalar_dic, self.step)

    def log(self, losses, discs, stats):
        self.logger.info(
            "  Step {step:7d}: L1 {L.pixel.avg:7.4f}  D {L.disc.avg:7.3f}  G {L.gen.avg:7.3f}"
            "  Dec_Const {L.dec_const.avg:3.3f}  FM {L.fm.avg:7.3f}  AC {S.ac_acc.avg:5.1%}"
            "  R_font {D.real_font_acc.avg:7.3f}  F_font {D.fake_font_acc.avg:7.3f}"
            "  R_uni {D.real_uni_acc.avg:7.3f}  F_uni {D.fake_uni_acc.avg:7.3f}"
            "  B_stl {S.B_style.avg:5.1f}  B_trg {S.B_target.avg:5.1f}"
            .format(step=self.step, L=losses, D=discs, S=stats))
