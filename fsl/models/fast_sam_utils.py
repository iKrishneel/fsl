#!/usr/bin/env python

import os

import numpy as np
import torch
from fsl.structures import Instances
from ultralytics.models.fastsam import FastSAM
from ultralytics.utils import DEFAULT_CFG


class FastSAMMaskGenerator(torch.nn.Module):
    def __init__(self, model: str):
        super(FastSAMMaskGenerator, self).__init__()
        self.sam = FastSAM(model=model)

    def forward(self, image: np.ndarray):
        if not self.training:
            return self.get_proposals(image)
        raise NotImplementedError('Training routing is not implemented')

    @torch.inference_mode()
    def get_proposals(self, image: np.ndarray) -> Instances:
        im_shape = image.shape[:2]
        if image.dtype != np.uint8:
            image = (image - image.min()) / (image.max() - image.min())
            image = (255 * image).astype(np.uint8)

        results = self.sam.predict(image)
        instances = Instances(
            bboxes=results[0].boxes.data[:, :4].cpu().numpy(),
            masks=torch.cat([mask.data for mask in results[0].masks]).cpu().numpy(),
            bbox_fmt='xyxy',
            image_height=im_shape[0],
            image_width=im_shape[1],
        )
        return instances

    @torch.no_grad()
    def eval(self, *args, **kwargs):
        return self

    def train(self, *args, **kwargs):
        return self


def build_fast_sam_mask_generator(model: str = None, **kwargs) -> FastSAMMaskGenerator:
    model = model or os.path.join(os.environ['HOME'], '.cache/torch/ultralytics/fastsam/FastSAM-x.pt')
    os.makedirs(os.path.dirname(model), exist_ok=True)

    DEFAULT_CFG.verbose = False
    return FastSAMMaskGenerator(model)
