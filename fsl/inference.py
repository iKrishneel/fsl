#!/usr/bin/env python

import json
import os.path as osp
from typing import List, Tuple, Type, Union

import numpy as np
import torch
from igniter.engine import InferenceEngine as _InferenceEngine
from igniter.logger import logger
from igniter.registry import engine_registry
from PIL import Image

_Image = Type[Image.Image]


@engine_registry('inference')
class InferenceEngine(_InferenceEngine):
    def __init__(self, *args, **kwargs):
        super(InferenceEngine, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def __call__(self, image: Union[np.ndarray, _Image]):
        if self.model.query_feats is None:
            logger.warning('Support images/text not set')
            return

        image, _ = self.apply_transforms(image)
        self.model([image])

    @torch.no_grad()
    def set_supports(self, root: str, anno_file: str):
        assert osp.isdir(root), f'{root} is not a valid directory!'
        assert osp.isfile(anno_file), f'{anno_file} is not a valid file!'

        with open(anno_file, 'r') as jfile:
            annos = json.load(jfile)

        assert annos

        images, bboxes = [], []
        for anno in annos:
            image = Image.open(osp.join(root, anno['file_name']))
            bboxs = torch.stack([torch.FloatTensor(bb) for bb in anno['bboxes']])

            image, bboxs = self.apply_transforms(image, bboxs)
            images.append(image)
            bboxes.append(torch.stack(bboxs))

        roi_features = self.model.get_query_roi_features(images, bboxes)
        self.model.set_query_roi_features(roi_features)

    def apply_transforms(
        self, image: _Image, bboxes: List[np.ndarray] = None
    ) -> Union[_Image, Tuple[_Image, List[np.ndarray]]]:
        for transform in self.transforms.transforms:
            image, bboxes = transform(image, bboxes)
        return image, bboxes if bboxes is not None else image


if __name__ == '__main__':
    import logging

    from igniter.builder import build_engine
    from igniter.logger import logger
    from omegaconf import OmegaConf
    from PIL import Image

    from fsl.datasets.s3_coco_dataset import *
    from fsl.model import *
    from fsl.transforms import *

    logger.setLevel(logging.INFO)

    cfg = OmegaConf.load('./configs/sam_relational_net.yaml')

    im = Image.open('/root/krishneel/Downloads/000001.jpg')

    e = build_engine(cfg, 'test')
    e.set_supports('/root/krishneel/Downloads/supports/', '/root/krishneel/Downloads/supports/annotations.json')
    e(im)
