#!/usr/bin/env python

from typing import Any, Type, List, Dict, Callable
import os
import torch
import pickle

from omegaconf import DictConfig
from PIL import Image

from igniter.registry import func_registry, engine_registry, io_registry
from igniter.engine import EvaluationEngine
from fsl.structures import Instances

_Image = Type[Image.Image]


@io_registry('prototype_writer')
def file_writer(io_cfg: Dict[str, str]) -> Callable:
    root = io_cfg.root
    os.makedirs(root, exist_ok=True)

    def write(prototypes, iid: str) -> None:
        path = os.path.join(root, f'{str(iid).zfill(12)}.pkl')
        with open(path, 'wb') as pfile:
            pickle.dump(prototypes, pfile)

    return write


@func_registry
def prototype_forward(engine, batch) -> None:
    for image, instances in zip(*batch):
        image_id = instances['image_id']
        instances = Instances(
            bboxes=instances['bboxes'],
            class_ids=instances['category_ids'],
            labels=instances['category_ids'],
            image_id=image_id,
        )
        prototypes = engine._model.build_image_prototypes(image, instances)
        engine.file_io(prototypes, image_id)


@engine_registry('prototype_engine')
class ProtoTypeEngine(EvaluationEngine):
    def __init__(self, *args, **kwargs):
        super(ProtoTypeEngine, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def __call__(self, image: _Image, instances: Dict[str, Any]) -> torch.Tensor:
        instances = Instances(bboxes=instances['bboxes'], class_ids=instances['category_ids'])
        return self.model.build_image_prototypes(image, instances)


if __name__ == '__main__':
    from omegaconf import OmegaConf
    from igniter.builder import build_engine
    from igniter.logger import logger
    import logging
    from fsl.datasets.s3_coco_dataset import collate_data  # NOQA
    from fsl.models.devit import devit  # NOQA

    logger.setLevel(logging.INFO)

    cfg = OmegaConf.load('../../configs/devit/prototypes.yaml')

    engine = build_engine(cfg, mode='val')
    engine()

    # image = Image.open('/root/krishneel/Downloads/000000.jpg')
    # engine(image)
