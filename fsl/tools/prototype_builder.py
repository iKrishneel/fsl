#!/usr/bin/env python

import os
import pickle
from typing import Any, Callable, Dict, Type

import torch
from igniter.logger import logger
from igniter.engine import EvaluationEngine
from igniter.registry import engine_registry, event_registry, func_registry, io_registry
from PIL import Image

from fsl.structures import Instances
from fsl.utils import ProtoTypes

_Image = Type[Image.Image]


@io_registry('prototype_writer')
def file_writer(io_cfg: Dict[str, str]) -> Callable:
    root = io_cfg.root

    def write(prototypes: ProtoTypes, iid: str, folder_name: str) -> None:
        write_path = os.path.join(root, folder_name)
        os.makedirs(write_path, exist_ok=True)
        write_path = os.path.join(write_path, f'{str(iid).zfill(12)}.pkl')
        prototypes.save(write_path)

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
        engine.file_io(prototypes, image_ids[0], engine._cfg.build.model)


@event_registry
def collate_and_write(engine, filename: str, clean: bool = True, per_class_avg: bool = True) -> None:
    root = os.path.join(engine._cfg.io.file_io.root, engine._cfg.build.model)

    def _load_pickle(filename: str) -> ProtoTypes:
        with open(filename, 'rb') as pfile:
            data = pickle.load(pfile)
        return data

    logger.info(f'Reading Prototypes from {root}')
    p_files = sorted(os.listdir(root))
    prototypes = None
    valid_files = []
    for i, p_file in enumerate(p_files):
        fn = os.path.join(root, p_file)
        if not os.path.isfile(fn) or 'pkl' not in fn or filename in p_file or not p_file.startswith('0'):
            logger.info(f'Skipping {p_file}')
            continue
        pt = _load_pickle(fn)
        prototypes = pt if i == 0 else prototypes + pt
        valid_files.append(fn)

    logger.info(f'Found {len(valid_files)} prototypes')

    filename = os.path.join(root, filename)
    logger.info('Saving final prototypes to {filename}')

    if per_class_avg:
        average_embeddings = {}
        for embedding, label in zip(prototypes.embeddings, prototypes.labels):
            embedding = embedding[None] if len(embedding.shape) == 1 else embedding
            emb = average_embeddings.get(label, [])
            emb.append(embedding)
            average_embeddings[label] = emb

        ProtoTypes(
            torch.stack([torch.cat(value).mean(dim=0) for key, value in average_embeddings.items()]),
            list(average_embeddings.keys()),
        ).save(filename)
    else:
        prototypes.save(filename)

    if clean:
        logger.info('Cleaning individual prototypes')
        for p_file in valid_files:
            os.remove(p_file)


@engine_registry('prototype_engine')
class ProtoTypeEngine(EvaluationEngine):
    def __init__(self, *args, **kwargs):
        super(ProtoTypeEngine, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def __call__(self, image: _Image, instances: Dict[str, Any]) -> torch.Tensor:
        instances = Instances(bboxes=instances['bboxes'], class_ids=instances['category_ids'])
        return self.model.build_image_prototypes(image, instances)


if __name__ == '__main__':
    from igniter.main import initiate

    from fsl.datasets.s3_coco_dataset import collate_data  # NOQA
    from fsl.models.devit import devit_sam  # NOQA

    # initiate('../../configs/devit/prototypes/foreground_prototypes.yaml')
    initiate('../../configs/devit/prototypes/background_prototypes.yaml')
