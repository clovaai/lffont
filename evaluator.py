"""
LF-Font
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
from pathlib import Path

import torch

import utils
from utils import Logger
from datasets import load_lmdb, read_data_from_lmdb


def torch_eval(val_fn):
    @torch.no_grad()
    def decorated(self, gen, *args, **kwargs):
        gen.eval()
        ret = val_fn(self, gen, *args, **kwargs)
        gen.train()

        return ret

    return decorated


class Evaluator:
    def __init__(self, env, env_get, logger, writer, batch_size, transform,
                 content_font, use_half=False):
        torch.backends.cudnn.benchmark = True

        self.env = env
        self.env_get = env_get
        self.logger = logger
        self.writer = writer
        self.batch_size = batch_size
        self.transform = transform

        self.content_font = content_font
        self.use_half = use_half

    def cp_validation(self, gen, cv_loaders, step, phase="fact", reduction='mean', ext_tag=""):
        for tag, loader in cv_loaders.items():
            self.comparable_val_saveimg(gen, loader, step, tag=f"comparable_{tag}_{ext_tag}",
                                        phase=phase, reduction=reduction)

    @torch_eval
    def comparable_val_saveimg(self, gen, loader, step, phase="fact", tag='comparable', reduction='mean'):
        n_row = loader.dataset.n_uni_per_font
        compare_batches = self.infer_loader(gen, loader, phase=phase, reduction=reduction)
        comparable_grid = utils.make_comparable_grid(*compare_batches[::-1], nrow=n_row)
        self.writer.add_image(tag, comparable_grid, global_step=step)
        return comparable_grid

    @torch_eval
    def infer_loader(self, gen, loader, phase, reduction="mean"):
        outs = []
        trgs = []

        for i, (in_style_ids, in_comp_ids, in_imgs, trg_style_ids, trg_comp_ids,
                trg_unis, content_imgs, *trg_imgs) in enumerate(loader):

            if self.use_half:
                in_imgs = in_imgs.half()
                content_imgs = content_imgs.half()

            out = gen.infer(in_style_ids, in_comp_ids, in_imgs, trg_style_ids, trg_comp_ids,
                            content_imgs, phase, reduction=reduction)

            outs.append(out.detach().cpu())
            if trg_imgs:
                trgs.append(trg_imgs[0].detach().cpu())

        ret = (torch.cat(outs).float(),)
        if trgs:
            ret += (torch.cat(trgs).float(),)

        return ret

    @torch_eval
    def save_each_imgs(self, gen, loader, save_dir, phase, reduction='mean'):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, (in_style_ids, in_comp_ids, in_imgs, trg_style_ids, trg_comp_ids,
                trg_unis, content_imgs) in enumerate(loader):

            if self.use_half:
                in_imgs = in_imgs.half()
                content_imgs = content_imgs.half()

            out = gen.infer(in_style_ids, in_comp_ids, in_imgs, trg_style_ids, trg_comp_ids,
                            content_imgs, phase, reduction=reduction)

            out = out.float()

            dec_unis = trg_unis.detach().cpu().numpy()
            font_ids = trg_style_ids.detach().cpu().numpy()
            images = out.detach().cpu()  # [B, 1, 128, 128]
            for dec_uni, font_id, image in zip(dec_unis, font_ids, images):
                font_name = loader.dataset.fonts[font_id]  # name.ttf
                font_name = Path(font_name).stem  # remove ext
                (save_dir / font_name).mkdir(parents=True, exist_ok=True)
                uni = hex(dec_uni)[2:].upper().zfill(4)

                path = save_dir / font_name / "{}_{}.png".format(font_name, uni)
                utils.save_tensor_to_image(image, path)


def eval_ckpt():
    import argparse
    from models import generator_dispatch
    from sconf import Config
    from train import setup_transforms
    from datasets import load_json, get_fact_test_loader

    logger = Logger.get()

    parser = argparse.ArgumentParser()
    parser.add_argument("config_paths", nargs="+", help="path to config.yaml")
    parser.add_argument("--weight", help="path to weight to evaluate.pth")
    parser.add_argument("--img_dir", help="path to save images for evaluation")
    parser.add_argument("--test_meta", help="path to metafile: contains (font, chars (in unicode)) to generate and reference chars (in unicode)")
    args, left_argv = parser.parse_known_args()

    cfg = Config(*args.config_paths, default="cfgs/defaults.yaml")
    cfg.argv_update(left_argv)

    content_font = cfg.content_font
    n_comps = int(cfg.n_comps)
    trn_transform, val_transform = setup_transforms(cfg)

    env = load_lmdb(cfg.data_path)
    env_get = lambda env, x, y, transform: transform(read_data_from_lmdb(env, f'{x}_{y}')['img'])

    test_meta = load_json(args.test_meta)
    dec_dict = load_json(cfg.dec_dict)

    g_kwargs = cfg.get('g_args', {})
    g_cls = generator_dispatch()
    gen = g_cls(1, cfg['C'], 1, **g_kwargs, n_comps=n_comps)
    gen.cuda()

    weight = torch.load(args.weight)
    if "generator_ema" in weight:
        weight = weight["generator_ema"]
    gen.load_state_dict(weight)
    logger.info(f"Resumed checkpoint from {args.weight}")
    writer = None

    evaluator = Evaluator(env,
                          env_get,
                          logger,
                          writer,
                          cfg["batch_size"],
                          val_transform,
                          content_font
                          )

    img_dir = Path(args.img_dir)
    ref_unis = test_meta["ref_unis"]
    gen_unis = test_meta["gen_unis"]
    gen_fonts = test_meta["gen_fonts"]
    target_dict = {f: gen_unis for f in gen_fonts}

    loader = get_fact_test_loader(env,
                                  env_get,
                                  target_dict,
                                  ref_unis,
                                  cfg,
                                  None,
                                  dec_dict,
                                  val_transform,
                                  ret_targets=False,
                                  num_workers=cfg.n_workers,
                                  shuffle=False
                                  )[1]

    logger.info("Save CV results to {} ...".format(img_dir))
    evaluator.save_each_imgs(gen, loader, save_dir=img_dir, phase=cfg.phase, reduction='mean')


if __name__ == "__main__":
    eval_ckpt()
