"""
LF-Font
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
from itertools import chain
import random

from PIL import ImageFile
import numpy as np

import torch
from torch.utils.data import Dataset

from .lmdbutils import read_data_from_lmdb
from .datautils import sample

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FactTrainDataset(Dataset):
    def __init__(self, env, env_get, avails, decompose_dict, content_font, n_comps=371, n_in_chars=6, n_each_chars=3, n_targets=18, transform=None):
        self.env = env
        self.env_get = env_get

        self.avails = avails
        self.fus = [(fname, uni) for fname, unis in self.avails.items() for uni in unis]
        self.unis = sorted(set.union(*map(set, self.avails.values())))
        self.n_unis = len(self.unis)
        self.fonts = list(self.avails)
        self.n_fonts = len(self.fonts)
        self.avails_rev = {}
        for fname, unilist in self.avails.items():
            for uni in unilist:
                self.avails_rev.setdefault(uni, []).append(fname)

        self.decompose_dict = decompose_dict
        self.n_comps = n_comps
        self.n_in_chars = n_in_chars
        self.n_each_chars = n_each_chars
        self.n_targets = n_targets
        self.transform = transform
        self.content_font = content_font


    def get_available_fonts(self, uni_list, fonts_list=None):
        avail_font_list = []
        for uni in uni_list:
            if fonts_list is not None:
                avail_fonts = sorted(set.intersection(set(self.avails_rev[uni]), set(fonts_list)))
            else:
                avail_fonts = self.avails_rev[uni]
            if not avail_fonts:
                return None
            avail_font = random.choice(avail_fonts)
            avail_font_list.append(avail_font)
        return avail_font_list

    def sample_input(self, n_in_chars):
        picked_chars = sample(self.unis, n_in_chars)

        picked_fonts = []
        picked_unis = []
        picked_comp_ids = []

        for uni in picked_chars:
            avail_fonts = sorted(set(self.avails_rev[uni]) - {self.content_font})
            valid_fonts = random.sample(avail_fonts, self.n_each_chars)
            picked_fonts += valid_fonts
            picked_unis += [uni] * self.n_each_chars
            picked_comp_ids += [self.decompose_dict[uni]] * self.n_each_chars

        return picked_fonts, picked_unis, picked_comp_ids

    def get_available_unis(self, avail_unis, style_comp_ids):
        seen_comps = list(set(chain(*style_comp_ids)))
        seen_binary = np.zeros(self.n_comps)
        seen_binary[seen_comps] = 1

        avail_comb_uni = []
        avail_comb_ids = []

        for uni in avail_unis:
            comps = self.decompose_dict[uni]
            comps_binary = seen_binary[comps]
            if comps_binary.sum() == len(comps):
                avail_comb_uni.append(uni)
                avail_comb_ids.append(comps)

        return avail_comb_uni, avail_comb_ids

    def check_and_sample(self, in_fonts, in_unis, in_comp_ids, trg_unis, trg_comp_ids):
        trg_comp_ids_ = [*chain(*trg_comp_ids)]
        in_comp_ids = [[*filter(lambda x: x in trg_comp_ids_, comp_ids)] for comp_ids in in_comp_ids]

        in_set = [*zip(*filter(lambda x: x[2], [*zip(in_fonts, in_unis, in_comp_ids)]))]

        if in_set:
            in_fonts, in_unis, in_comp_ids = in_set
        else:
            return None

        n_sample = len(trg_unis)
        uni_comps = [*zip(trg_unis, trg_comp_ids)]

        if n_sample > self.n_targets:
            uni_comps = sample(uni_comps, self.n_targets)
        elif n_sample < self.n_targets:
            return None

        trg_unis, trg_comp_ids = [*zip(*uni_comps)]

        trg_fonts = self.get_available_fonts(trg_unis, in_fonts)
        if trg_fonts is None:
            return None

        return list(in_fonts), list(in_unis), list(in_comp_ids), list(trg_fonts), list(trg_unis), list(trg_comp_ids)

    def __getitem__(self, index):

        while True:
            (in_fonts, in_unis, in_comp_ids) = self.sample_input(self.n_in_chars)

            avail_unis = set(self.unis) - set(in_unis)
            trg_unis, trg_comp_ids = self.get_available_unis(avail_unis, in_comp_ids)
            trg_set = self.check_and_sample(in_fonts, in_unis, in_comp_ids, trg_unis, trg_comp_ids)
            if trg_set is None:
                continue

            in_fonts, in_unis, in_comp_ids, trg_fonts, trg_unis, trg_comp_ids = trg_set

            in_imgs = torch.cat([self.env_get(self.env, fname, uni, self.transform) for fname, uni in zip(in_fonts, in_unis)])
            trg_imgs = torch.cat([self.env_get(self.env, fname, uni, self.transform) for fname, uni in zip(trg_fonts, trg_unis)])

            trg_fonts = trg_fonts + in_fonts
            trg_unis = trg_unis + in_unis
            trg_comp_ids = trg_comp_ids + in_comp_ids
            trg_imgs = torch.cat([trg_imgs, in_imgs])


            in_style_ids = [*map(self.fonts.index, in_fonts)]
            trg_style_ids = [*map(self.fonts.index, trg_fonts)]

            trg_uni_ids = [*map(self.unis.index, trg_unis)]
            in_comp_ids = [*map(torch.LongTensor, in_comp_ids)]

            content_imgs = torch.cat([self.env_get(self.env, self.content_font, uni, self.transform) for uni in trg_unis])

            ret = (
                torch.LongTensor(in_style_ids),
                in_comp_ids,
                in_imgs,
                torch.LongTensor(trg_style_ids),
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
        (in_style_ids, in_comp_ids, in_imgs,
         trg_style_ids, trg_uni_ids, trg_comp_ids, trg_imgs, content_imgs) = zip(*batch)

        in_comp_ids = [*chain(*in_comp_ids)]
        in_comp_lens = torch.LongTensor([*map(len, in_comp_ids)])
        trg_comp_ids = [*chain(*trg_comp_ids)]

        ret = (
            torch.cat(in_style_ids).repeat_interleave(in_comp_lens, dim=0),
            torch.cat(in_comp_ids),
            torch.cat(in_imgs).unsqueeze_(1).repeat_interleave(in_comp_lens, dim=0),
            torch.cat(trg_style_ids),
            torch.cat(trg_uni_ids),
            trg_comp_ids,
            torch.cat(trg_imgs).unsqueeze_(1),
            torch.cat(content_imgs).unsqueeze_(1)
        )

        return ret


class FactTestDataset(Dataset):
    def __init__(self, env, env_get, target_fu, ref_unis, avails, decompose_dict, content_font, language="chn", n_comps=371, n_shots=3, transform=None, ret_targets=True):

        self.fonts = list(target_fu)
        self.n_uni_per_font = len(target_fu[list(target_fu)[0]])
        self.fus = [(fname, uni) for fname, unis in target_fu.items() for uni in unis]

        self.env = env
        self.env_get = env_get

        self.avails = avails
        self.decompose_dict = decompose_dict
        self.n_comps = n_comps
        self.n_shots = n_shots
        self.ref_unis = ref_unis

        self.transform = transform
        self.ret_targets = ret_targets
        self.content_font = content_font

        to_int_dict = {"chn": lambda x: int(x, 16),
                       "kor": lambda x: ord(x),
                       "thai": lambda x: int("".join([f'{ord(each):04X}' for each in x]), 16)
                      }

        self.to_int = to_int_dict[language.lower()]

    def __getitem__(self, index):
        trg_font, trg_uni = self.fus[index]
        trg_style_id = [self.fonts.index(trg_font)]

        if self.ref_unis is None:
            in_avail_unis = sorted(set(self.avails[trg_font]) - set(trg_uni))
        else:
            in_avail_unis = self.ref_unis

        in_unis = sample(in_avail_unis, self.n_shots)

        trg_comp_ids = self.decompose_dict[trg_uni]
        trg_dec_uni = [self.to_int(trg_uni)]
        content_img = self.env_get(self.env, self.content_font, trg_uni, self.transform)

        in_style_ids = trg_style_id * len(in_unis) + [len(self.fonts)]
        in_comp_ids = [self.decompose_dict[uni] for uni in in_unis] + [trg_comp_ids]
        in_imgs = [self.env_get(self.env, trg_font, uni, self.transform) for uni in in_unis] + [content_img]

        ret = (torch.LongTensor(in_style_ids),
               [*map(torch.LongTensor, in_comp_ids)],
               torch.cat(in_imgs),
               torch.LongTensor(trg_style_id),
               torch.LongTensor(trg_comp_ids),
               torch.LongTensor(trg_dec_uni),
               content_img
        )

        if self.ret_targets:
            trg_img = self.env_get(self.env, trg_font, trg_uni, self.transform)
            ret += (trg_img, )

        return ret

    def __len__(self):
        return len(self.fus)

    @staticmethod
    def collate_fn(batch):
        (in_style_ids, in_comp_ids, in_imgs, trg_style_ids, trg_comp_ids,
         trg_unis, content_imgs, *left) = list(zip(*batch))

        in_comp_ids = [*chain(*in_comp_ids)]
        in_comp_lens = torch.LongTensor([*map(len, in_comp_ids)])

        ret = (
            torch.cat(in_style_ids).repeat_interleave(in_comp_lens, dim=0),
            torch.cat(in_comp_ids),
            torch.cat(in_imgs).unsqueeze_(1).repeat_interleave(in_comp_lens, dim=0),
            torch.cat(trg_style_ids),
            trg_comp_ids,
            torch.cat(trg_unis),
            torch.cat(content_imgs).unsqueeze_(1)
        )
        if left:
            assert len(left) == 1
            trg_imgs = left[0]
            ret += torch.cat(trg_imgs).unsqueeze_(1),

        return ret
