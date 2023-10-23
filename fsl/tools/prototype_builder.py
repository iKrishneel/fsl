#!/usr/bin/env python

import os
import pickle
from typing import Any, Callable, Dict, List, Type

import torch
from igniter.engine import EvaluationEngine
from igniter.registry import engine_registry, event_registry, func_registry, io_registry
from omegaconf import DictConfig
from PIL import Image

from fsl.structures import Instances
from fsl.utils import ProtoTypes

_Image = Type[Image.Image]


@io_registry('prototype_writer')
def file_writer(io_cfg: Dict[str, str]) -> Callable:
    root = io_cfg.root
    os.makedirs(root, exist_ok=True)

    def write(prototypes, iid: str) -> None:
        path = os.path.join(root, f'{str(iid).zfill(12)}.pkl')
        prototypes.save(path)

    return write


@func_registry
def prototype_forward(engine, batch) -> None:
    for image, instances in zip(*batch):
        image_ids = instances['image_ids']
        instances = Instances(
            bboxes=instances['bboxes'],
            class_ids=instances['category_ids'],
            labels=instances['category_names'],
            image_id=image_ids,
        )
        prototypes = engine._model.build_image_prototypes(image, instances)
        engine.file_io(prototypes, image_ids[0])


@event_registry
def collate_and_write(filename: str) -> None:
    root = engine._cfg.io.file_io.root

    def _load_pickle(filename: str) -> ProtoTypes:
        with open(filename, 'rb') as pfile:
            data = pickle.load(pfile)
        return data

    p_files = sorted(os.listdir(root))
    prototypes = None
    for i, p_file in enumerate(p_files):
        pt = _load_pickle(os.path.join(root, p_file))
        prototypes = pt if i == 0 else prototypes + pt

    average_embeddings = {}
    for embedding, label in zip(prototypes.embeddings, prototypes.labels):
        embedding = embedding[None] if len(embedding.shape) == 1 else embedding
        emb = average_embeddings.get(label, [])
        emb.append(embedding)
        average_embeddings[label] = emb

    ProtoTypes(
        torch.stack([torch.cat(value).mean(dim=0) for key, value in average_embeddings.items()]),
        list(average_embeddings.keys()),
    ).save(os.path.join(root, filename))


@engine_registry('prototype_engine')
class ProtoTypeEngine(EvaluationEngine):
    def __init__(self, *args, **kwargs):
        super(ProtoTypeEngine, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def __call__(self, image: _Image, instances: Dict[str, Any]) -> torch.Tensor:
        instances = Instances(bboxes=instances['bboxes'], class_ids=instances['category_ids'])
        return self.model.build_image_prototypes(image, instances)


if __name__ == '__main__':
    import logging

    from igniter.builder import build_engine
    from igniter.logger import logger
    from omegaconf import OmegaConf

    from fsl.datasets.s3_coco_dataset import collate_data  # NOQA
    from fsl.models.devit import devit  # NOQA

    logger.setLevel(logging.INFO)

    cfg = OmegaConf.load('../../configs/devit/prototypes.yaml')

    engine = build_engine(cfg, mode='val')
    # engine()
    import IPython

    IPython.embed()
