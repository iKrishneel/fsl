#!/usr/bin/env python

import os
import pickle
from typing import Any, Callable, Dict, Type, List

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
            masks=instances.get('masks', None),
            image_id=image_ids,
        )

        def bboxes2mask(bboxes: torch.Tensor, img_hw: List[int]) -> torch.Tensor:
            mask = torch.zeros(len(bboxes), *img_hw)
            # boxes = torch.round(bboxes).long()
            for i in range(len(bboxes)):
                x1, y1, x2, y2 = torch.round(bboxes[i]).int()
                mask[i, y1:y2, x1:x2] = 1
            return mask.to(torch.uint8)

        features = engine._model.mask_generator(image[None])
        masks = bboxes2mask(instances.bboxes, image.shape[1:])
        masks = torch.nn.functional.interpolate(masks[None], features.shape[2:], mode='nearest')[0]
        masks = masks.to(torch.bool).to(features.device)

        features = features[0]
        labels = [instances.labels[i] for i, mask in enumerate(masks) if mask.sum() > 0]

        if len(labels) == 0:
            return

        tokens = torch.stack([(features * mask).flatten(1).sum(1) / mask.sum() for mask in masks if mask.sum() > 0])
        if len(tokens.shape) != 2:
            breakpoint()

        prototypes = ProtoTypes(tokens, labels=labels)
        # prototypes = engine._model.build_image_prototypes(image, instances)
        engine.file_io(prototypes, image_ids[0], engine._cfg.build.model)


@func_registry
def bg_prototype_forward(engine, batch) -> None:
    for image, instances in zip(*batch):
        image_ids = instances['image_ids']
        instances = Instances(
            bboxes=instances['bboxes'],
            class_ids=instances['category_ids'],
            labels=instances['category_names'],
            masks=instances.get('masks', None),
            image_id=image_ids,
        )

        features = engine._model.mask_generator(image[None])
        masks = torch.nn.functional.interpolate(instances.masks[None], features.shape[2:], mode='nearest')[0]

        masks = masks.to(torch.bool)
        features = features.squeeze(0).flatten(1).permute(1, 0)

        prototypes = None
        for i, mask in enumerate(masks):
            mask = mask.flatten()
            bg_tokens = features[mask]
            if len(bg_tokens) == 0:
                continue
            bg_tokens = compress(bg_tokens, n_clst=5)
            labels = [instances.labels[i]] * bg_tokens.shape[0]
            pt = ProtoTypes(embeddings=bg_tokens, labels=labels)
            prototypes = pt if prototypes is None else prototypes + pt
        engine.file_io(prototypes, image_ids[0], engine._cfg.build.model)


def compress(tensor, n_clst=5):
    from fast_pytorch_kmeans import KMeans

    if len(tensor) <= n_clst:
        # may be normalize this, the raw tokens are not normalized
        return tensor
    else:
        kmeans = KMeans(n_clusters=n_clst, verbose=False, mode='cosine')
        kmeans.fit(tensor)
        return kmeans.centroids


@event_registry
def collate_and_write(
    engine, filename: str, clean: bool = True, reduction: str = 'per_class_avg', cluster_size: int = 10
) -> None:
    root = os.path.join(engine._cfg.io.file_io.root, engine._cfg.build.model)
    _post_process_prototypes(root, filename, clean, reduction, cluster_size)


def _post_process_prototypes(
    root: str, filename: str, clean: bool = False, reduction: str = 'per_class_avg', cluster_size: int = 1000
) -> ProtoTypes:
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
        prototypes = pt if prototypes is None else prototypes + pt
        valid_files.append(fn)

    logger.info(f'Found {len(valid_files)} prototypes')

    filename = os.path.join(root, filename)
    logger.info('Saving final prototypes to {filename}')

    if reduction in ['per_class_avg', 'per_class_cluster']:
        average_embeddings = {label: [] for label in prototypes.labels}
        for embedding, label in zip(prototypes.embeddings, prototypes.labels):
            embedding = embedding[None] if len(embedding.shape) == 1 else embedding
            average_embeddings[label].append(embedding)

        if reduction == 'per_class_avg':
            prototypes = ProtoTypes(
                torch.stack([torch.cat(value).mean(dim=0) for key, value in average_embeddings.items()]),
                list(average_embeddings.keys()),
            )
        elif reduction == 'per_class_cluster':
            assert cluster_size > 0
            prototypes = None
            for key, embeddings in average_embeddings.items():
                embeddings = compress(torch.cat(embeddings, dim=0), cluster_size).mean(0)[None]
                pt = ProtoTypes(embeddings, [key])
                prototypes = pt if prototypes is None else prototypes + pt
        prototypes.save(filename)
    elif reduction == 'inter_class_avg':
        assert cluster_size > 0
        embeddings = compress(prototypes.embeddings, cluster_size)
        ProtoTypes(embeddings, labels=['background'] * embeddings.shape[0]).save(filename)
    elif reduction.lower() == 'none':
        prototypes.save(filename)
    else:
        raise TypeError(f'Unknown reduction type: {reduction}')

    if clean:
        logger.info('Cleaning individual prototypes')
        for p_file in valid_files:
            os.remove(p_file)

    return prototypes


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

    initiate('../../configs/devit/prototypes/foreground_prototypes.yaml')
    initiate('../../configs/devit/prototypes/background_prototypes.yaml')
