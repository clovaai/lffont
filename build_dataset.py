"""
LF-Font
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import argparse
import json
import io

import lmdb
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm


def render(char, font, size=(128, 128), pad=20, pad_h=None):
    width, height = font.getsize(char)
    max_size = max(width, height)

    if pad_h is None:
        pad_h = pad

    if width < height:
        start_w = (height - width) // 2 + pad
        start_h = pad
        # start_h = 0

    else:
        start_w = pad
        start_h = (width - height) // 2 + pad

    img = Image.new("L", (max_size + pad * 2, max_size + pad * 2), 255)
    draw = ImageDraw.Draw(img)
    draw.text((start_w, start_h), char, font=font)
    img = img.resize(size, 2)

    return img


def save_lmdb(env_path, font_path_char_dict):

    env = lmdb.open(env_path, map_size=1024 ** 4)
    valid_dict = {}

    for fname in tqdm(font_path_char_dict):
        fontpath = font_path_char_dict[fname]["path"]
        charlist = font_path_char_dict[fname]["charlist"]
        ttf = ImageFont.truetype(fontpath, size=150)
        unilist = []
        for char in charlist:
            uni = hex(ord(char))[2:].upper()
            unilist.append(uni)

            char_img = render(char, ttf, pad=20)
            img = io.BytesIO()
            char_img.save(img, format="PNG")
            img = img.getvalue()

            lmdb_key = f"{fname}_{uni}".encode("utf-8")

            with env.begin(write=True) as txn:
                txn.put(lmdb_key, img)

        valid_dict[fname] = unilist

    return valid_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb_path", help="path to save lmdb environment.")
    parser.add_argument("--json_path", help="path to save json file: {fname: [available unicodes list]}.")
    parser.add_argument("--meta_path", help="path to meta file: {fname: {'path': /path/to/ttf.ttf, 'charlist': [available chars]}.")

    args = parser.parse_args()

    with open(args.meta_path) as f:
        fpc_meta = json.load(f)

    valid_dict = save_lmdb(args.lmdb_path, fpc_meta)
    with open(args.json_path, "w") as f:
        json.dump(valid_dict, f)


if __name__ == "__main__":
    main()
