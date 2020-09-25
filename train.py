"""
LF-Font
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import sys
from pathlib import Path
import argparse

import torch

import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torchvision import transforms
import numpy as np
from sconf import Config, dump_args
import utils
from utils import Logger

from models import generator_dispatch, disc_builder, aux_clf_builder
from models.modules import weights_init
from datasets import (load_lmdb, load_json, read_data_from_lmdb,
                      get_comb_trn_loader, get_cv_comb_loaders, get_fact_trn_loader, get_cv_fact_loaders)

from trainer import load_checkpoint, CombinedTrainer, FactorizeTrainer
from evaluator import Evaluator


def setup_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("config_paths", nargs="+", help="path/to/config.yaml")
    parser.add_argument("--resume", default=None, help="path/to/saved/.pth")
    parser.add_argument("--use_unique_name", default=False, action="store_true", help="whether to use name with timestamp")

    args, left_argv = parser.parse_known_args()
    assert not args.name.endswith(".yaml")

    cfg = Config(*args.config_paths, default="cfgs/defaults.yaml",
                 colorize_modified_item=True)
    cfg.argv_update(left_argv)

    if cfg.use_ddp:
        cfg.n_workers = 0

    cfg.work_dir = Path(cfg.work_dir)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)

    if args.use_unique_name:
        timestamp = utils.timestamp()
        unique_name = "{}_{}".format(timestamp, args.name)
    else:
        unique_name = args.name

    cfg.unique_name = unique_name
    cfg.name = args.name

    (cfg.work_dir / "logs").mkdir(parents=True, exist_ok=True)
    (cfg.work_dir / "checkpoints" / unique_name).mkdir(parents=True, exist_ok=True)

    if cfg.save_freq % cfg.val_freq:
        raise ValueError("save_freq has to be multiple of val_freq.")

    return args, cfg


def setup_transforms(cfg):
    if cfg.dset_aug.random_affine:
        aug_transform = [
            transforms.ToPILImage(),
            transforms.RandomAffine(
                degrees=10, translate=(0.03, 0.03), scale=(0.9, 1.1), shear=10, fillcolor=255
            )
        ]
    else:
        aug_transform = []

    tensorize_transform = [transforms.Resize((128, 128)), transforms.ToTensor()]
    if cfg.dset_aug.normalize:
        tensorize_transform.append(transforms.Normalize([0.5], [0.5]))
        cfg.g_args.dec.out = "tanh"

    trn_transform = transforms.Compose(aug_transform + tensorize_transform)
    val_transform = transforms.Compose(tensorize_transform)

    return trn_transform, val_transform


def cleanup():
    dist.destroy_process_group()


def is_main_worker(gpu):
    return (gpu <= 0)


def train_ddp(gpu, args, cfg, world_size):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:" + str(cfg.port),
        world_size=world_size,
        rank=gpu,
    )
    cfg.batch_size = cfg.batch_size // world_size
    train(args, cfg, ddp_gpu=gpu)
    cleanup()


def train(args, cfg, ddp_gpu=-1):
    cfg.gpu = ddp_gpu
    torch.cuda.set_device(ddp_gpu)
    cudnn.benchmark = True

    logger_path = cfg.work_dir / "logs" / "{}.log".format(cfg.unique_name)
    logger = Logger.get(file_path=logger_path, level="info", colorize=True)

    image_scale = 0.6
    writer_path = cfg.work_dir / "runs" / cfg.unique_name
    image_path = cfg.work_dir / "images" / cfg.unique_name
    writer = utils.TBDiskWriter(writer_path, image_path, scale=image_scale)

    args_str = dump_args(args)
    if is_main_worker(ddp_gpu):
        logger.info("Run Argv:\n> {}".format(" ".join(sys.argv)))
        logger.info("Args:\n{}".format(args_str))
        logger.info("Configs:\n{}".format(cfg.dumps()))
        logger.info("Unique name: {}".format(cfg.unique_name))

    logger.info("Get dataset ...")

    content_font = cfg.content_font
    n_comps = int(cfg.n_comps)

    trn_transform, val_transform = setup_transforms(cfg)

    env = load_lmdb(cfg.data_path)
    env_get = lambda env, x, y, transform: transform(read_data_from_lmdb(env, f'{x}_{y}')['img'])

    data_meta = load_json(cfg.data_meta)
    dec_dict = load_json(cfg.dec_dict)

    if cfg.phase == "comb":
        get_trn_loader = get_comb_trn_loader
        get_cv_loaders = get_cv_comb_loaders
        Trainer = CombinedTrainer

    elif cfg.phase == "fact":
        get_trn_loader = get_fact_trn_loader
        get_cv_loaders = get_cv_fact_loaders
        Trainer = FactorizeTrainer

    else:
        raise ValueError(cfg.phase)

    trn_dset, trn_loader = get_trn_loader(env,
                                          env_get,
                                          cfg,
                                          data_meta["train"],
                                          dec_dict,
                                          trn_transform,
                                          num_workers=cfg.n_workers,
                                          shuffle=True)

    if is_main_worker(ddp_gpu):
        cv_loaders = get_cv_loaders(env,
                                    env_get,
                                    cfg,
                                    data_meta,
                                    dec_dict,
                                    val_transform,
                                    num_workers=cfg.n_workers,
                                    shuffle=False)
    else:
        cv_loaders = None

    logger.info("Build model ...")
    # generator
    g_kwargs = cfg.get("g_args", {})
    g_cls = generator_dispatch()
    gen = g_cls(1, cfg.C, 1, **g_kwargs, n_comps=n_comps)
    gen.cuda()
    gen.apply(weights_init(cfg.init))

    if cfg.gan_w > 0.:
        d_kwargs = cfg.get("d_args", {})
        disc = disc_builder(cfg.C, trn_dset.n_fonts, trn_dset.n_unis, **d_kwargs)
        disc.cuda()
        disc.apply(weights_init(cfg.init))
    else:
        disc = None

    if cfg.ac_w > 0.:
        aux_clf = aux_clf_builder(gen.mem_shape, n_comps, **cfg.ac_args)
        aux_clf.cuda()
        aux_clf.apply(weights_init(cfg.init))
    else:
        aux_clf = None
        assert cfg.ac_gen_w == 0., "ac_gen loss is only available with ac loss"

    g_optim = optim.Adam(gen.parameters(), lr=cfg.g_lr, betas=cfg.adam_betas)
    d_optim = optim.Adam(disc.parameters(), lr=cfg.d_lr, betas=cfg.adam_betas) \
        if disc is not None else None
    ac_optim = optim.Adam(aux_clf.parameters(), lr=cfg.ac_lr, betas=cfg.adam_betas) \
        if aux_clf is not None else None

    st_step = 1
    if args.resume:
        st_step, loss = load_checkpoint(args.resume, gen, disc, aux_clf, g_optim, d_optim, ac_optim, cfg.overwrite)
        logger.info("Resumed checkpoint from {} (Step {}, Loss {:7.3f})".format(
            args.resume, st_step - 1, loss))
        if cfg.overwrite:
            st_step = 1
        else:
            pass

    evaluator = Evaluator(env,
                          env_get,
                          logger,
                          writer,
                          cfg.batch_size,
                          val_transform,
                          content_font,
                          use_half=cfg.use_half
                          )

    trainer = Trainer(gen, disc, g_optim, d_optim,
                      aux_clf, ac_optim,
                      writer, logger,
                      evaluator, cv_loaders,
                      cfg)

    trainer.train(trn_loader, st_step, cfg[f"{cfg.phase}_iter"])


def main():
    args, cfg = setup_args_and_config()

    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    if cfg.use_ddp:
        ngpus_per_node = torch.cuda.device_count()
        world_size = ngpus_per_node
        mp.spawn(train_ddp, nprocs=ngpus_per_node, args=(args, cfg, world_size))
    else:
        train(args, cfg)


if __name__ == "__main__":
    main()
