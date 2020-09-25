"""
LF-Font
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
from itertools import chain
import copy

from PIL import ImageFile
import numpy as np
import random

import torch
from torch.utils.data import Dataset

from .lmdbutils import read_data_from_lmdb
from .datautils import sample

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CombTrainDataset(Dataset):
    def __init__(self, env, env_get, avails, decompose_dict, content_font, n_comps=371, n_pick=3,
                 n_sample_min=1, n_sample_max=999, transform=None):
        self.env = env
        self.env_get = env_get

        self.avails = avails
        self.unis = sorted(set.union(*map(set, self.avails.values())))
        self.fonts = list(self.avails)
        self.n_unis = len(self.unis)
        self.n_fonts = len(self.fonts)

        self.avails_rev = {}
        for fname, unilist in self.avails.items():
            for uni in unilist:
                self.avails_rev.setdefault(uni, []).append(fname)

        self.decompose_dict = decompose_dict
        self.n_comps = n_comps
        self.n_pick = n_pick
        self.n_sample_min = n_sample_min
        self.n_sample_max = n_sample_max

        self.transform = transform
        self.content_font = content_font

    def sample_style(self, font, n_pick, ex_values=None):
        avail_unis = self.avails[font]
        picked_unis = sample(avail_unis, n_pick, ex_values)
        picked_comp_ids = [self.decompose_dict[uni]
                           for uni in picked_unis]

        imgs = torch.cat([self.env_get(self.env, font, uni, self.transform) for uni in picked_unis])

        return imgs, picked_unis, picked_comp_ids

    def get_available_combinations(self, avail_unis, style_comp_ids):
        seen_comps = list(set(chain(*style_comp_ids)))
        seen_binary = np.zeros(self.n_comps)
        seen_binary[seen_comps] = 1

        avail_comb_uni = []
        avail_comb_ids = []

        for uni in avail_unis:
            comps = self.decompose_dict[uni]
            comps_binary = seen_binary[comps]
            if comps_binary.sum() == len(comps) and len(self.avails_rev[uni]) >= 2:
                avail_comb_uni.append(uni)
                avail_comb_ids.append(comps)

        return avail_comb_uni, avail_comb_ids

    def check_and_sample(self, trg_unis, trg_comp_ids):
        n_sample = len(trg_unis)
        uni_comps = list(zip(trg_unis, trg_comp_ids))

        if n_sample > self.n_sample_max:
            uni_comps = sample(uni_comps, self.n_sample_max)
        elif n_sample < self.n_sample_min:
            return None, None

        uni_comps = list(zip(*uni_comps))
        return uni_comps

    def __getitem__(self, index):
        font_idx = index % self.n_fonts
        font_name = self.fonts[font_idx]
        while True:
            (style_imgs, style_unis, style_comp_ids) = self.sample_style(font_name, n_pick=self.n_pick)

            avail_unis = set(self.avails[font_name]) - set(style_unis)
            trg_unis, trg_comp_ids = self.get_available_combinations(avail_unis, style_comp_ids)
            trg_unis, trg_comp_ids = self.check_and_sample(trg_unis, trg_comp_ids)
            if trg_unis is None:
                continue

            trg_imgs = torch.cat([self.env_get(self.env, font_name, uni, self.transform)
                                  for uni in trg_unis])
            trg_uni_ids = [self.unis.index(uni) for uni in trg_unis]

            style_comp_ids = [*map(torch.LongTensor, style_comp_ids)]
            font_idx = torch.LongTensor([font_idx])
            content_imgs = torch.cat([self.env_get(self.env, self.content_font, uni, self.transform)
                                      for uni in trg_unis]).unsqueeze_(1)

            ret = (
                font_idx.repeat(len(style_imgs)),
                style_comp_ids,
                style_imgs,
                font_idx.repeat(len(trg_imgs)),
                torch.LongTensor(trg_uni_ids),
                trg_comp_ids,
                trg_imgs,
                content_imgs
            )

            return ret

    def __len__(self):
        return sum([len(v) for v in self.avails.values()])

    @staticmethod
    def collate_fn(batch):
        (style_ids, style_comp_ids, style_imgs,
         trg_ids, trg_uni_ids, trg_comp_ids, trg_imgs, content_imgs) = zip(*batch)

        style_comp_ids = [*chain(*style_comp_ids)]
        style_comp_lens = torch.LongTensor([*map(len, style_comp_ids)])
        trg_comp_ids = [*chain(*trg_comp_ids)]

        ret = (
            torch.cat(style_ids).repeat_interleave(style_comp_lens, dim=0),
            torch.cat(style_comp_ids),
            torch.cat(style_imgs).unsqueeze_(1).repeat_interleave(style_comp_lens, dim=0),
            torch.cat(trg_ids),
            torch.cat(trg_uni_ids),
            trg_comp_ids,
            torch.cat(trg_imgs).unsqueeze_(1),
            torch.cat(content_imgs)
        )

        return ret


class CombTestDataset(Dataset):
    def __init__(self, env, env_get, target_fu, avails, decompose_dict, content_font, language="chn", n_comps=371,
                 transform=None, ret_targets=True):

        self.fonts = list(target_fu)
        self.n_uni_per_font = len(target_fu[list(target_fu)[0]])
        self.fus = [(fname, uni) for fname, unis in target_fu.items() for uni in unis]

        self.env = env
        self.env_get = env_get

        self.avails = avails
        self.decompose_dict = decompose_dict
        self.comp_dict = {}
        for uni in self.decompose_dict:
            decomposed = self.decompose_dict[uni]
            for comp_id in decomposed:
                self.comp_dict.setdefault(comp_id, [])
                self.comp_dict[comp_id].append(uni)

        self.n_comps = n_comps
        self.transform = transform
        self.ret_targets = ret_targets
        self.content_font = content_font

        to_int_dict = {"chn": lambda x: int(x, 16),
                       "kor": lambda x: ord(x),
                       "thai": lambda x: int("".join([f'{ord(each):04X}' for each in x]), 16)
                      }

        self.to_int = to_int_dict[language.lower()]

    def sample_char(self, font_name, trg_uni):
        trg_comp_ids = self.decompose_dict[trg_uni]
        style_unis = []
        style_comp_ids_list = []
        for comp in trg_comp_ids:
            avail_style_uni = sorted(set.intersection(set(self.avails[font_name]),
                                                      set(self.comp_dict[comp]) - {trg_uni}))

            style_uni = random.choice(avail_style_uni)
            style_comp_ids = self.decompose_dict[style_uni]

            style_unis.append(style_uni)

        return style_unis, trg_comp_ids

    def __getitem__(self, index):
        font_name, trg_uni = self.fus[index]
        font_idx = self.fonts.index(font_name)

        style_unis, trg_comp_ids = self.sample_char(font_name, trg_uni)
        style_imgs = torch.stack([self.env_get(self.env, font_name, uni, self.transform) for uni in style_unis])

        font_idx = torch.LongTensor([font_idx])
        trg_dec_uni = torch.LongTensor([self.to_int(trg_uni)])

        content_img = self.env_get(self.env, self.content_font, trg_uni, self.transform)

        ret = (
            font_idx.repeat(len(trg_comp_ids)),
            torch.LongTensor(trg_comp_ids),
            style_imgs,
            font_idx,
            trg_comp_ids,
            trg_dec_uni,
            content_img
        )


        if self.ret_targets:
            trg_img = self.env_get(self.env, font_name, trg_uni, self.transform)
            ret += (trg_img, )

        return ret

    def __len__(self):
        return len(self.fus)

    @staticmethod
    def collate_fn(batch):
        style_ids, style_comp_ids, style_imgs, trg_ids, trg_comp_ids, trg_unis, content_imgs, *left = \
            list(zip(*batch))

        ret = (
            torch.cat(style_ids),
            torch.cat(style_comp_ids),
            torch.cat(style_imgs),
            torch.cat(trg_ids),
            trg_comp_ids,
            torch.cat(trg_unis),
            torch.cat(content_imgs).unsqueeze_(1)
        )

        if left:
            trg_imgs = left[0]
            ret += (torch.cat(trg_imgs).unsqueeze_(1),)

        return ret


class FixedRefDataset(Dataset):
    def __init__(self, env, env_get, target_dict, ref_unis, decompose_dict, content_font, language="chn", rep_content=False,
                 decompose=False, transform=None, ret_targets=True):

        self.target_dict = target_dict
        self.fus = [(fname, uni) for fname, unis in target_dict.items() for uni in unis]

        self.ref_unis = ref_unis
        self.n_chars = min([len(ref_unis), 8])
        self.decompose_dict = decompose_dict
        self.decompose = decompose
        self.rep_content = rep_content

        self.comp_unis = {}
        for uni in self.ref_unis:
            decomposed = self.decompose_dict[uni]
            for comp_id in decomposed:
                self.comp_unis.setdefault(comp_id, [])
                self.comp_unis[comp_id].append(uni)

        self.content_font = content_font
        self.fonts = list(target_dict) + [content_font]

        self.env = env
        self.env_get = env_get

        self.transform = transform
        self.ret_targets = ret_targets

        to_int_dict = {"chn": lambda x: int(x, 16),
                       "kor": lambda x: ord(x),
                       "thai": lambda x: int("".join([f'{ord(each):04X}' for each in x]), 16)
                      }

        self.to_int = to_int_dict[language.lower()]


    def sample_char(self, fname, trg_uni):
        style_comp_ids = copy.copy(self.decompose_dict[trg_uni])

        style_fonts = []
        style_unis = []

        n_valid_uni = 0

        for comp in style_comp_ids:
            if comp in self.comp_unis:
                _font = fname
                _uni = random.choice(self.comp_unis[comp])
                if not _uni in style_unis:
                    n_valid_uni += 1
            else:
                if self.rep_content:
                    _font = fname
                    _uni = random.choice(self.ref_unis)
                else:
                    _font = self.content_font
                    _uni = trg_uni

            style_fonts.append(_font)
            style_unis.append(_uni)

        _n_remains = max([(self.n_chars - n_valid_uni), 0])
        if self.decompose and not self.rep_content:
            _unis = random.sample(self.ref_unis, _n_remains)
            for _uni in _unis:
                _comp_ids = self.decompose_dict[_uni]
                style_fonts.append(fname)
                style_unis.append(_uni)
                style_comp_ids.append(random.choice(_comp_ids))

        return style_fonts, style_unis, style_comp_ids

    def __getitem__(self, index):
        fname, trg_uni = self.fus[index]
        trg_comp_ids = self.decompose_dict[trg_uni]

        fidx = self.fonts.index(fname)

        style_fonts, style_unis, style_comp_ids = self.sample_char(fname, trg_uni)
        style_imgs = torch.cat([self.env_get(self.env, fname, uni, self.transform) for fname, uni in zip(style_fonts, style_unis)])

        if self.decompose:
            fidces = torch.LongTensor([*map(self.fonts.index, style_fonts)])
        else:
            fidces = torch.LongTensor([fidx]).repeat(len(style_fonts))

        trg_dec_uni = torch.LongTensor([self.to_int(trg_uni)])
        content_img = self.env_get(self.env, self.content_font, trg_uni, self.transform)

        ret = (
            fidces,
            torch.LongTensor(style_comp_ids),
            style_imgs,
            torch.LongTensor([fidx]),
            trg_comp_ids,
            trg_dec_uni,
            content_img
        )

        if self.ret_targets:
            trg_img = self.env_get(self.env, fname, trg_uni, self.transform)
            ret += (trg_img, )

        return ret

    def __len__(self):
        return len(self.fus)

    @staticmethod
    def collate_fn(batch):
        style_ids, style_comp_ids, style_imgs, trg_ids, trg_comp_ids, trg_unis, content_imgs, *left = \
            list(zip(*batch))

        ret = (
            torch.cat(style_ids),
            torch.cat(style_comp_ids),
            torch.cat(style_imgs).unsqueeze_(1),
            torch.cat(trg_ids),
            trg_comp_ids,
            torch.cat(trg_unis),
            torch.cat(content_imgs).unsqueeze_(1)
        )
        if left:
            trg_imgs = left[0]
            ret += (torch.cat(trg_imgs).unsqueeze_(1),)

        return ret
