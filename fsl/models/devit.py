#!/usr/bin/env python

from typing import Any, Type, Dict, List
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.ops import RoIAlign

import numpy as np
from PIL import Image

from igniter.registry import model_registry

_Image = Type[Image.Image]
_Tensor = Type[torch.Tensor]


class DeVit(nn.Module):
    def __init__(self, mask_generator, roi_pool_size: int = 16):
        super(DeVit, self).__init__()
        self.mask_generator = mask_generator
        self.roi_pool = RoIAlign(roi_pool_size, spatial_scale=1 / mask_generator.predictor.downsize, sampling_ratio=-1)

    def forward(self, images: List[_Image], targets: Dict[str, Any] = None):
        if not self.training:
            return self.forward_once(images)

        device = self.mask_generator.predictor.device  # TODO
        assert targets is not None and isinstance(targets, dict)

        gt_bboxes = [torch.stack(target['bboxes']).to(device) for target in targets]
        # ref https://github.com/mlzxy/devit/blob/main/detectron2/modeling/meta_arch/devit.py#L1016

    @torch.no_grad()
    def forward_once(self, images: List[_Image]):
        proposals = self.get_proposals(images)

        import IPython; IPython.embed()
        
    def get_proposals(self, images: List[_Image]) -> List[List[Any]]:
        return [self.mask_generator.get_proposals(image) for image in images]


@model_registry
def devit(sam_args: Dict[str, str], mask_gen_args: Dict[str, Any] = {}):
    from fsl.models.sam_relational import build_sam_auto_mask_generator

    mask_generator = build_sam_auto_mask_generator(sam_args, mask_gen_args)

    return DeVit(mask_generator)
    

if __name__ == '__main__':
    m = devit({'model': 'vit_b', 'checkpoint': None})
    m.cuda()

    im = Image.open('/root/krishneel/Downloads/000001.jpg')
    m([im])
