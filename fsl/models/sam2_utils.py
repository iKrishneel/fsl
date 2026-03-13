#!/usr/bin/env python

from typing import Dict

import torch
import numpy as np

from fsl.structures import Instances
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator as _SAMG


class SAM2AutomaticMaskGenerator(torch.nn.Module, _SAMG):
    def __init__(self, model, **kwargs):
        super(SAM2AutomaticMaskGenerator, self).__init__()
        _SAMG.__init__(self, model, **kwargs)
        
    @torch.no_grad()
    def get_proposals(self, image: np.ndarray) -> Instances:
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            masks = self.generate(image)
        instances = Instances(
             *image.shape[:2],
            bboxes=np.array([mask['bbox'] for mask in masks]),
            masks=np.array([mask['segmentation'] for mask in masks]),
            bbox_fmt='xywh'
        )
        return instances

    @property
    def device(self) -> str:
        return next(self.predictor.model.parameters()).device.type

def build_sam2_auto_mask_generator(sam_args: Dict[str, str]):
    model = sam_args['model']
    device = sam_args.get('device', 'cuda')
    if device != 'cpu':
         torch.autocast(device_type='cuda', dtype=torch.bfloat16).__enter__()

    mapping = {
        'tiny': 'facebook/sam2.1-hiera-tiny',
        'small': 'facebook/sam2.1-hiera-small',
        'large': 'facebook/sam2.1-hiera-large',
        'base': 'facebook/sam2-hiera-base-plus'
    }

    model_id = mapping[model]
    mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(model_id)
    
    return mask_generator


if __name__ == '__main__':
    import sys
    from fsl.utils import Visualizer
    import matplotlib.pyplot as plt
    import cv2 as cv

    amg = build_sam2_auto_mask_generator('tiny')

    im_path = sys.argv[1]
    im = cv.imread(im_path)
    im = cv.resize(im, dsize=None, fx=1, fy=1)

    v = Visualizer(im)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        x = amg.get_proposals(im)

    x.labels = [''] * len(x)
    x.scores = np.array([1] * len(x))
    z = v.overlay(x)

    plt.imshow(z.get_image())
    plt.show()
