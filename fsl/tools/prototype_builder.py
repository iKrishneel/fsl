#!/usr/bin/env python

import os
import pickle
from typing import Any, Callable, Dict, List, Type, Union

import torch
from igniter.engine import EvaluationEngine
from igniter.logger import logger
from igniter.registry import engine_registry, event_registry, func_registry, io_registry
from omegaconf import DictConfig
from PIL import Image

from fsl.structures import Instances
from fsl.utils import ProtoTypes

_Image = Type[Image.Image]


@io_registry('prototype_writer')
def file_writer(io_cfg: Dict[str, str], cfg: DictConfig) -> Callable:
    root = io_cfg.root
    folder_name = io_cfg.folder_name
    dataset_name = cfg.build[cfg.build.model]['dataset']

    def write(prototypes: ProtoTypes, iid: str, model_name: str, prefix: str = '') -> str:
        write_path = os.path.join(root, model_name, dataset_name, folder_name)
        os.makedirs(write_path, exist_ok=True)
        write_path = os.path.join(write_path, f'{prefix}{str(iid)}.pkl')
        prototypes.save(write_path)
        return write_path

    return write


def bboxes2mask(bboxes: torch.Tensor, img_hw: List[int]) -> torch.Tensor:
    mask = torch.zeros(len(bboxes), *img_hw)
    # boxes = torch.round(bboxes).long()
    for i in range(len(bboxes)):
        x1, y1, x2, y2 = torch.round(bboxes[i]).int()
        mask[i, y1:y2, x1:x2] = 1
    return mask.to(torch.uint8)


@func_registry
def prototype_forward(engine, batch, save: bool = True) -> Union[None, ProtoTypes]:
    all_prototypes = None

    images, instances = batch
    with torch.inference_mode():
        features = engine._model.get_features(torch.stack(images))

    for feature, image, instance in zip(features, images, instances):
        masks = bboxes2mask(instance.bboxes, image.shape[1:])
        masks = torch.nn.functional.interpolate(masks[None], feature.shape[1:], mode='nearest')[0]
        masks = masks.to(torch.bool).to(features.device)

        labels = [instance.labels[i] for i, mask in enumerate(masks) if mask.sum() > 0]

        if len(labels) == 0:
            return

        tokens = torch.stack([(feature * mask).flatten(1).sum(1) / mask.sum() for mask in masks if mask.sum() > 0])
        prototypes = ProtoTypes(tokens.float(), labels=labels)
        if save:
            engine.file_io(prototypes, instance.image_id, engine._cfg.build.model)
        else:
            all_prototypes = prototypes if all_prototypes is None else all_prototypes + prototypes

    return all_prototypes


@func_registry
def bg_prototype_forward(engine, batch) -> None:
    for image, instances in zip(*batch):
        features = engine._model.get_features(image)

        if instances.masks is not None:
            masks = instances.masks[None]
        else:
            masks = torch.zeros((1, len(instances.bboxes), *image.shape[1:]), dtype=torch.uint8)
            for i, bbox in enumerate(instances.bboxes):
                x1, y1, x2, y2 = bbox.int()
                masks[0, i, y1:y2, x1:x2].fill_(1)

        masks = torch.nn.functional.interpolate(masks, features.shape[2:], mode='nearest')[0]
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
            pt = ProtoTypes(embeddings=bg_tokens.float(), labels=labels)
            prototypes = pt if prototypes is None else prototypes + pt

        engine.file_io(prototypes, instances.image_id, engine._cfg.build.model, prefix=str(engine.state.epoch) + '_')


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
    model_name = engine._cfg.build.model
    dataset_name = engine._cfg.build[model_name]['dataset']
    root = os.path.join(engine._cfg.io.file_io.root, model_name, dataset_name, engine._cfg.io.file_io.folder_name)
    _post_process_prototypes(root, filename, clean, reduction, cluster_size)


def _post_process_prototypes(
    root: str, filename: str, clean: bool = False, reduction: str = 'per_class_avg', cluster_size: int = 1
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
        if (
            not os.path.isfile(fn) or 'pkl' not in fn or filename in p_file or not p_file[0].isdigit()
        ):  #  or not p_file[0].startswith('1'):
            logger.info(f'Skipping {p_file}')
            continue
        pt = _load_pickle(fn)
        prototypes = pt if prototypes is None else prototypes + pt
        valid_files.append(fn)

    logger.info(f'Found {len(valid_files)} prototypes')

    filename = os.path.join(root, filename)
    logger.info(f'Saving final prototypes to {filename}')

    if reduction in ['per_class_avg', 'per_class_cluster']:
        average_embeddings = {label: [] for label in prototypes.labels}
        for embedding, label in zip(prototypes.normalized_embedding, prototypes.labels):
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
                embeddings = compress(torch.cat(embeddings, dim=0), cluster_size) if cluster_size > 1 else embeddings
                pt = ProtoTypes(embeddings, [key] * embeddings.shape[0])
                prototypes = pt if prototypes is None else prototypes + pt
        prototypes.save(filename)
    elif reduction == 'inter_class_avg':
        assert cluster_size > 0
        embeddings = compress(prototypes.embeddings, cluster_size)
        ProtoTypes(embeddings, labels=['background'] * embeddings.shape[0]).save(filename)
    elif reduction is None or reduction.lower() == 'none':
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
