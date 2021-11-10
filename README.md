# Few-shot Font Generation with Localized Style Representations and Factorization (AAAI 2021)

**NOTICE: We release the unified few-shot font generation repository ([clovaai/fewshot-font-generation](https://github.com/clovaai/fewshot-font-generation)). If you are interested in using our implementation, please visit the unified repository.**

Official PyTorch implementation of LF-Font | [paper](https://arxiv.org/abs/2009.11042)

Song Park<sup>1*</sup>, Sanghyuk Chun<sup>2*</sup>, Junbum Cha<sup>2</sup>,
Bado Lee<sup>2</sup>, Hyunjung Shim<sup>1</sup><br>
<sub>\* Equal contribution</sub>

<sup>1</sup> <sub>School of Integrated Technology, Yonsei University</sub>  
<sup>2</sup> <sub>Clova AI Research, NAVER Corp.</sub>

Automatic few-shot font generation is in high demand because manual designs are expensive and sensitive to the expertise of designers. Existing methods of few-shot font generation aims to learn to disentangle the style and content element from a few reference glyphs and mainly focus on a universal style representation for each font style. However, such approach limits the model in representing diverse local styles, and thus make it unsuitable to the most complicated letter system, e.g., Chinese, whose characters consist of a varying number of components (often called "radical") with a highly complex structure. In this paper, we propose a novel font generation method by learning localized styles, namely component-wise style representations, instead of universal styles. The proposed style representations enable us to synthesize complex local details in text designs. However, learning component-wise styles solely from reference glyphs is infeasible in the few-shot font generation scenario, when a target script has a large number of components, e.g., over 200 for Chinese. To reduce the number of reference glyphs, we simplify component-wise styles by a product of component factor and style factor, inspired by low-rank matrix factorization. Thanks to the combination of strong representation and a compact factorization strategy, our method shows remarkably better few-shot font generation results (with only 8 reference glyph images) than other state-of-the-arts, without utilizing strong locality supervision, e.g., location of each component, skeleton, or strokes.

You can find more related projects on the few-shot font generation at the following links:

- [clovaai/dmfont](https://github.com/clovaai/dmfont) (ECCV'20) | [paper](https://arxiv.org/abs/2005.10510)
- [clovaai/lffont](https://github.com/clovaai/lffont) (AAAI'21) | [paper](https://arxiv.org/abs/2009.11042)
- [clovaai/mxfont](https://github.com/clovaai/mxfont) (ICCV'21) | [paper](https://arxiv.org/abs/2104.00887)
- [clovaai/fewshot-font-generation](https://github.com/clovaai/fewshot-font-generation) The unified few-shot font generation repository

## Introduction

Pytorch implementation of ***Few-shot Font Generation with Localized Style Representations and Factorization***.

* * *

## Prerequisites

* **Python > 3.6**

  Using conda is recommended: [https://docs.anaconda.com/anaconda/install/linux/](https://docs.anaconda.com/anaconda/install/linux/)
* **pytorch >= 1.1** (recommended: 1.1)

	To install: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
	
* sconf

	To install: [https://github.com/khanrc/sconf](https://github.com/khanrc/sconf)
	
* numpy, tqdm, lmdb, yaml, jsonlib, msgpack

```
conda install numpy tqdm lmdb ruamel.yaml jsonlib-python3 msgpack
```


## Usage
### Prepare datasets
#### Build meta file to dump lmdb environment

* To build a dataset with your own truetype font files (*.ttf*), a json-format meta file is needed:
	* **structure**: *dict*
	* **format**: {fontname: {"path": path/to/.ttf", "charlist": [chars to dump.]}}
	* **example**: {"font1": {"path": "./fonts/font1.ttf", "charlist": ["春", "夏", "秋", "冬"]}}
	
The font file we used as the _content font_ can be accessed [here](https://chinesefontdesign.com/font-housekeeper-song-ming-typeface-chinese-font-simplified-chinese-fonts.html).

#### Run script
```
python build_dataset.py \
    --lmdb_path path/to/dump/lmdb \
    --meta_path path/to/meta/file \
    --json_path path/to/save/dict
```

* **arguments**
	* \-\-lmdb_path: path to save lmdb environment.
	* \-\-meta_path: path to meta file of built meta file.
	* \-\-json_path: path to save *json* file, which contains information of available fonts and characters. 
		* saved *json* file has format like this: {fontname: [saved character list in unicode format]}

#### Build meta file to train and test
* **train meta** (*dict, json format*)
	* should have keys; "train", "valid", "avail"
	* "train": {font: list of characters} pairs for training, *dict*
		* key: font name / value: list of chars in the key font.
		* example: {"font1": ["4E00", "4E01"...], "font2": ["4E00", "4E01"...]}
	* "avail": {font: list of characters} pairs which are accessible in lmdb, *dict*
		* same format with "train"
	* "valid": list of font and list characters for validation, *dict*
		* should have keys: "seen_fonts", "unseen_fonts", "seen_unis", "unseen_unis"
		* seen fonts(unis) : list of fonts(chars in unicode) in training set.
		* unseen fonts(unis): list of fonts(chars in unicode) not in training set, for validation.
	* An example of train meta file is in `meta/train.json`.

* **test meta** (*dict, json format*)
	* should have keys; "gen_fonts", "gen_unis", "ref_unis"
	* "gen_fonts": list of fonts to generate.
	* "gen_unis": list of chars to generate, in *unicode*
	* "ref_unis": list of chars to use as reference chars, in *unicode* 
	* An example of test meta file is in `meta/test.json`.

### Modify the configuration file
We recommend to modify `cfgs/custom.yaml` rather than `cfgs/default.yaml`, `cfgs/combined.yaml`, or `cfgs/factorize.yaml`.

**keys**
* use_half
	* whether to use half tensor. (*apex* is needed)
* use_ddp
	* whether to use DataDistributedParallel, for multi-gpus.
* work_dir
	* the root directory for saved results.
* data_path
	* path to data lmdb environment.
* data_meta
	* path to train meta file.
* content_font
	* the name of font you want to use as source font.
* other values are hyperparameters for training.

### Train
```
# Phase 1 training
python train.py \
    NAME_phase1 \
    cfgs/custom.yaml cfgs/combined.yaml 

# Phase 2 training
python train.py \
    NAME_phase2 \
    cfgs/custom.yaml cfgs/factorize.yaml \
    --resume ./result/checkpoints/NAME_phase1/800000-NAME_phase1.pth
```
* **arguments**
	* NAME (first argument): name for the experiment.
		* the (checkpoints, validation images, logs) are saved in ./results/(checkpoints, images, logs)/NAME
	* path/to/config (second argument): path to configration file.
		* multiple values are allowed, but their keys should not be repeated.
		* cfgs/combined.yaml : for phase 1 training.
		* cfgs/factorize.yaml: for phase 2 training.
	* \-\-resume (optional) : path to checkpoint to resume.


### Test
```
python evaluator.py \
    cfgs/factorize.yaml \
    --weight weight/generator.pth \
    --img_dir path/to/save/images \
    --test_meta meta/test.json \
    --data_path path/to/data
```
* **arguments**
  * path/to/config (first argument): path to configration file.
  * \-\-weight : path to saved weight to test.
  * \-\-img_dir: path to save generated images.
  * \-\-test_meta: path to test meta file.
  * \-\-data_path: path to lmdb dataset which contatins the reference images.

## Code license

This project is distributed under [MIT license](LICENSE), except [modules.py](models/modules/modules.py) which is adopted from https://github.com/NVlabs/FUNIT.

```
LF-Font
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## Acknowledgement

This project is based on [clovaai/dmfont](https://github.com/clovaai/dmfont).

## How to cite

```
@inproceedings{park2021lffont,
    title={Few-shot Font Generation with Localized Style Representations and Factorization},
    author={Park, Song and Chun, Sanghyuk and Cha, Junbum and Lee, Bado and Shim, Hyunjung},
    year={2021},
    booktitle={AAAI Conference on Artificial Intelligence},
}
```
