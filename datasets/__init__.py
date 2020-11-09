"""
LF-Font
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import torch
import random
from .lmdbutils import (load_lmdb, load_json, read_data_from_lmdb)
from .p2dataset import FactTrainDataset, FactTestDataset
from .p1dataset import CombTrainDataset, CombTestDataset, FixedRefDataset
from .datautils import cyclize, sample, uniform_sample
from torch.utils.data import DataLoader


def get_fact_trn_loader(env, env_get, cfg, train_dict, dec_dict, transform, **kwargs):
    #  avail_fonts = [os.path.splitext(name)[0] + ".ttf" for name in avail_fonts]
    dset = FactTrainDataset(
        env,
        env_get,
        train_dict,
        dec_dict,
        content_font=cfg.content_font,
        n_comps=int(cfg.n_comps),
        n_in_chars=int(cfg.n_in_chars),
        n_each_chars=int(cfg.n_each_chars),
        n_targets=int(cfg.n_targets),
        transform=transform
    )
    if cfg.use_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(dset)
        kwargs["shuffle"] = False
    else:
        sampler = None
    loader = DataLoader(dset, batch_size=cfg.batch_size, sampler=sampler,
                        collate_fn=dset.collate_fn, **kwargs)

    return dset, loader


def get_fact_test_loader(env, env_get, target_dict, ref_unis, cfg, avails, dec_dict, transform, ret_targets=True, **kwargs):
    dset = FactTestDataset(
        env,
        env_get,
        target_dict,
        ref_unis,
        avails,
        dec_dict,
        content_font=cfg.content_font,
        language=cfg.language,
        transform=transform,
        n_comps=int(cfg.n_comps),
        n_shots=int(cfg.n_shots),
        ret_targets=ret_targets,
    )
    loader = DataLoader(dset, batch_size=cfg.batch_size, collate_fn=dset.collate_fn, **kwargs)

    return dset, loader


def get_comb_trn_loader(env, env_get, cfg, train_dict, dec_dict, transform, **kwargs):
    #  avail_fonts = [os.path.splitext(name)[0] + '.ttf' for name in avail_fonts]
    dset = CombTrainDataset(
        env,
        env_get,
        train_dict,
        dec_dict,
        content_font=cfg.content_font,
        **cfg.get('dset_args', {}),
        transform=transform,
    )
    if cfg.use_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(dset)
        kwargs["shuffle"] = False
        kwargs["num_workers"] = 0
    else:
        sampler = None
    loader = DataLoader(dset, batch_size=cfg.batch_size, sampler=sampler,
                        collate_fn=dset.collate_fn, **kwargs)

    return dset, loader


def get_comb_test_loader(env, env_get, target_dict, cfg, avails, dec_dict, transform, ret_targets=True, **kwargs):
    #  avail_fonts = [os.path.splitext(name)[0] + '.ttf' for name in avail_fonts]
    dset = CombTestDataset(
        env,
        env_get,
        target_dict,
        avails,
        dec_dict,
        content_font=cfg.content_font,
        language=cfg.language,
        transform=transform,
        n_comps=int(cfg.n_comps),
        ret_targets=ret_targets
    )
    loader = DataLoader(dset, batch_size=cfg.batch_size,
                        collate_fn=dset.collate_fn, **kwargs)

    return dset, loader


def get_fixedref_loader(env, env_get, decompose, target_dict, ref_unis, rep_content, cfg, dec_dict, transform, **kwargs):
    #  avail_fonts = [os.path.splitext(name)[0] + '.ttf' for name in avail_fonts]

    print([chr(int(uni, 16)) for uni in ref_unis])

    dset = FixedRefDataset(env,
                           env_get,
                           target_dict,
                           ref_unis,
                           rep_content=rep_content,
                           decompose=decompose,
                           content_font=cfg.content_font,
                           language=cfg.language,
                           decompose_dict=dec_dict,
                           transform=transform,
                           ret_targets=True
                           )
    loader = DataLoader(dset, batch_size=cfg.batch_size,
                        collate_fn=dset.collate_fn, **kwargs)

    return dset, loader


def get_cv_comb_loaders(env, env_get, cfg, data_meta, dec_dict, transform, **kwargs):
    n_unis = cfg.cv_n_unis
    n_fonts = cfg.cv_n_fonts

    ufs = uniform_sample(data_meta["valid"]["unseen_fonts"], n_fonts)
    sfs = uniform_sample(data_meta["valid"]["seen_fonts"], n_fonts)
    sus = uniform_sample(data_meta["valid"]["seen_unis"], n_unis)
    uus = uniform_sample(data_meta["valid"]["unseen_unis"], n_unis)

    sfuu_dict = {fname: uus for fname in sfs}
    ufsu_dict = {fname: sus for fname in ufs}
    ufuu_dict = {fname: uus for fname in ufs}

    cv_loaders = {'sfuu': get_comb_test_loader(env, env_get, sfuu_dict, cfg, data_meta['avail'], dec_dict, transform, **kwargs)[1],
                  'ufsu': get_comb_test_loader(env, env_get, ufsu_dict, cfg, data_meta['avail'], dec_dict, transform, **kwargs)[1],
                  'ufuu': get_comb_test_loader(env, env_get, ufuu_dict, cfg, data_meta['avail'], dec_dict, transform, **kwargs)[1]
                  }

    return cv_loaders


def get_cv_fact_loaders(env, env_get, cfg, data_meta, dec_dict, transform, ref_unis=None, **kwargs):
    n_unis = cfg.cv_n_unis
    n_fonts = cfg.cv_n_fonts

    ufs = uniform_sample(data_meta["valid"]["unseen_fonts"], n_fonts)
    sfs = uniform_sample(data_meta["valid"]["seen_fonts"], n_fonts)
    sus = uniform_sample(data_meta["valid"]["seen_unis"], n_unis)
    uus = uniform_sample(data_meta["valid"]["unseen_unis"], n_unis)

    sfuu_dict = {fname: uus for fname in sfs}
    ufsu_dict = {fname: sus for fname in ufs}
    ufuu_dict = {fname: uus for fname in ufs}

    if ref_unis is None:
        ref_unis = sorted(set(data_meta["valid"]["unseen_unis"]) - set(uus))[:cfg["n_shots"]]

    cv_loaders = {"sfuu": get_fact_test_loader(env, env_get, sfuu_dict, ref_unis, cfg, data_meta["avail"],
                                               dec_dict, transform, **kwargs)[1],
                  "ufsu": get_fact_test_loader(env, env_get, ufsu_dict, ref_unis, cfg, data_meta["avail"],
                                               dec_dict, transform, **kwargs)[1],
                  "ufuu": get_fact_test_loader(env, env_get, ufuu_dict, ref_unis, cfg, data_meta["avail"],
                                               dec_dict, transform, **kwargs)[1]
                  }

    return cv_loaders
