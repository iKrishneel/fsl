#!/usr/bin/env python

from typing import Union
import torch
import numpy as np
from PIL import Image

from igniter.registry import engine_registry
from igniter.engine import InferenceEngine as _InferenceEngine


@engine_registry('inference')
class InferenceEngine(_InferenceEngine):
    def __init__(self, *args, **kwargs):
        super(InferenceEngine, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def __call__(self, image: Union[np.ndarray]):
        for transform in self.transforms.transforms:
            image, _ = transform(image, None)

        self.model([image])


if __name__ == '__main__':
    from fsl.model import *
    from fsl.transforms import *
    from fsl.dataset import *
    from igniter.builder import build_engine

    from PIL import Image
    from omegaconf import OmegaConf

    # import logging
    # from igniter.logger import logger
    # logger.setLevel(logging.INFO)

    cfg = OmegaConf.load('./configs/sam_relational_net.yaml')

    im = Image.open('/root/krishneel/Downloads/000000.jpg')

    e = build_engine(cfg, 'test')

    e(im)
