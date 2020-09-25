"""
LF-Font
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
# generators
from .generator import Generator

# discriminator builder
from .discriminator import disc_builder

# auxiliary component classifier builder
from .aux_classifier import aux_clf_builder

def generator_dispatch():
    return Generator
