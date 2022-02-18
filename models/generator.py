"""
LF-Font
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import torch
import torch.nn as nn
from .comp_encoder import comp_enc_builder, decompose_block_builder
from .content_encoder import content_enc_builder
from .decoder import dec_builder
from .memory import Memory


class Generator(nn.Module):
    def __init__(self, C_in, C, C_out, comp_enc, emb_block, dec, content_enc, n_comps):
        super().__init__()
        self.component_encoder = comp_enc_builder(
            C_in, C, **comp_enc, n_comps=n_comps
        )
        self.mem_shape = self.component_encoder.final_shape
        assert self.mem_shape[-1] == self.mem_shape[-2]  # H == W
        self.memory = Memory()

        self.skip_shape = self.component_encoder.skip_shape
        self.skip_memory = Memory()

        if emb_block["emb_dim"]:
            self.emb_style, self.emb_comp = decompose_block_builder(
                **emb_block,
                in_shape=self.mem_shape
            )
            self.skip_emb_style, self.skip_emb_comp = decompose_block_builder(
                **emb_block,
                in_shape=self.skip_shape,
            )

        C_content = content_enc['C_out']
        self.content_encoder = content_enc_builder(
            C_in, C, **content_enc
        )

        self.decoder = dec_builder(
            C, C_out, **dec, C_content=C_content
        )

    def reset_memory(self):
        self.memory.reset_memory()
        self.skip_memory.reset_memory()

    def get_fact_memory_var(self):
        var = self.memory.get_fact_var() + self.skip_memory.get_fact_var()
        return var

    def encode_write_fact(self, style_ids, comp_ids, style_imgs, write_comb=False, reset_memory=True):
        if reset_memory:
            self.reset_memory()

        feats = self.component_encoder(style_imgs, comp_ids)

        feat_sc = feats["last"]
        feat_style = self.emb_style(feat_sc.unsqueeze(1))
        feat_comp = self.emb_comp(feat_sc.unsqueeze(1))
        self.memory.write_fact(style_ids, comp_ids, feat_style, feat_comp)
        if write_comb:
            self.memory.write_comb(style_ids, comp_ids, feat_sc)

        skip_sc = feats["skip"]
        skip_style = self.skip_emb_style(skip_sc.unsqueeze(1))
        skip_comp = self.skip_emb_comp(skip_sc.unsqueeze(1))
        self.skip_memory.write_fact(style_ids, comp_ids, skip_style, skip_comp)
        if write_comb:
            self.skip_memory.write_comb(style_ids, comp_ids, skip_sc)

        return feat_style, feat_comp

    def encode_write_comb(self, style_ids, comp_ids, style_imgs, reset_memory=True):
        if reset_memory:
            self.reset_memory()

        feats = self.component_encoder(style_imgs, comp_ids)  # [B, 3, C, H, W]

        feat_scs = feats["last"]
        self.memory.write_comb(style_ids, comp_ids, feat_scs)

        skip_scs = feats["skip"]
        self.skip_memory.write_comb(style_ids, comp_ids, skip_scs)

        return feat_scs

    def read_memory(self, target_style_ids, target_comp_ids, reset_memory=True,
                    phase="comb", try_comb=False, reduction='mean'):

        if phase == "fact" and try_comb:
            phase = "both"

        feats = self.memory.read_chars(target_style_ids, target_comp_ids, reduction=reduction, type=phase)
        skips = self.skip_memory.read_chars(target_style_ids, target_comp_ids, reduction=reduction, type=phase)

        feats = torch.stack([x.mean(0) for x in feats])
        skips = torch.stack([x.mean(0) for x in skips])

        if reset_memory:
            self.reset_memory()

        return feats, skips

    def read_decode(self, target_style_ids, target_comp_ids, content_imgs, reset_memory=True,
                    reduction='mean', phase="fact", try_comb=False):

        feat_scs, skip_scs = self.read_memory(target_style_ids, target_comp_ids, reset_memory, phase=phase,
                                              reduction=reduction, try_comb=try_comb)

        content_feats = self.content_encoder(content_imgs)

        out = self.decoder(feat_scs, skip_scs, content_feats=content_feats)

        if reset_memory:
            self.reset_memory()

        return out

    def infer(self, in_style_ids, in_comp_ids, in_imgs, trg_style_ids, trg_comp_ids, content_imgs,
              phase, reduction="mean", try_comb=False):

        in_style_ids = in_style_ids.cuda()
        in_comp_ids = in_comp_ids.cuda()
        in_imgs = in_imgs.cuda()

        trg_style_ids = trg_style_ids.cuda()
        content_imgs = content_imgs.cuda()

        if phase == "comb":
            self.encode_write_comb(in_style_ids, in_comp_ids, in_imgs)
        elif phase == "fact":
            self.encode_write_fact(in_style_ids, in_comp_ids, in_imgs, write_comb=False)
        else:
            raise NotImplementedError

        out = self.read_decode(trg_style_ids, trg_comp_ids, content_imgs=content_imgs,
                               reduction=reduction, phase=phase, try_comb=try_comb)

        return out
