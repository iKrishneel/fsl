#!/usr/bin/env python

import functools
from typing import Any, Dict, List, Type, Union

import torch
import torch.nn as nn
from igniter.logger import logger
from igniter.registry import model_registry
from torchvision.ops import RoIAlign

from fsl.structures import Instances
from fsl.utils import ProtoTypes

from .fsod import MaskFSOD

_Tensor = Type[torch.Tensor]
_Module = Type[nn.Module]


class ClipMaskFSOD(MaskFSOD):
    def __init__(
        self,
        mask_generator,
        clip_model: _Module,
        backbone: _Module,
        classifier: _Module,
        roi_pooler: int,
        alpha: float = 1.0,
        **kwargs: Dict[str, Any],
    ) -> None:
        super(ClipMaskFSOD, self).__init__(
            mask_generator, backbone=backbone, classifier=classifier, roi_pooler=roi_pooler
        )
        self.alpha = alpha
        self.clip = clip_model
        self.set_descriptions({k: k for k in self.classifier._all_cids})

    def forward(self, images: _Tensor, targets: List[Dict[str, Instances]] = None):
        if not self.training:
            return self.inference(images)

        raise NotImplementedError('Training with CLIP is not supported')

        images = torch.stack(images).to(self.device)
        assert targets is not None and len(targets) == images.shape[0]

        gt_instances = [target['gt_proposal'] for target in targets]
        gt_bboxes = [gt_proposal.to_tensor().bboxes.to(self.device) for gt_proposal in gt_instances]
        class_labels = torch.cat([instance.class_ids for instance in gt_instances])
        class_labels[class_labels == -1] = self.classifier.train_class_weight.shape[0]

        # text_embeddings = [self.text_encoder.get_text_embedding(names) for names in gt_names]

        text_embeddings = self.get_text_embedding(gt_instances)
        im_embeddings = self.backbone(images)

        roi_features = self.forward_features(im_embeddings, text_embeddings, gt_bboxes)
        loss_dict = self.classifier(roi_features)

        return loss_dict

    @torch.inference_mode()
    def inference(self, image: _Tensor) -> Instances:
        instances = super().inference(image)
        if self.alpha >= 1:
            return instances

        instances = instances.to_tensor()
        roi_feats = self.clip.forward_image(image, instances.bboxes)
        scores = self.clip.similarity(roi_feats, self.text_embeddings, 100)
        instances.scores = self.alpha * instances.scores + (1 - self.alpha) * scores.cpu()
        return instances

    @torch.no_grad()
    def build_image_prototypes(self, image: _Tensor, instances: Instances) -> ProtoTypes:
        instances = instances.to_tensor(self.device)

        text_embeddings = self.get_text_embedding(instances)
        features = self.mask_generator(image[None])

        roi_feats = self.forward_features(features, text_embeddings, [instances.bboxes])
        index = 2 if len(roi_feats.shape) == 4 else 1
        roi_feats = roi_feats.flatten(index).mean(index)
        return ProtoTypes(embeddings=roi_feats, labels=instances.labels, instances=instances)

    def get_text_embeddings(self, category_names: Union[List[str], str], prefix: str = 'photo of a %s') -> _Tensor:
        category_names = [category_names] if isinstance(category_names, str) else category_names
        label_names = [prefix % name for name in category_names]
        return torch.cat([self.clip.get_text_embedding(name) for name in label_names], dim=0)

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True, assign: bool = True):
        for key, value in self.clip.state_dict().items():
            state_dict[f'clip.{key}'] = value
        return super(ClipMaskFSOD, self).load_state_dict(state_dict, strict, assign)

    def to(self, *args, **kwargs):
        ret = super().to(*args, **kwargs)
        if args[0] == torch.float16:
            from clip.model import LayerNorm

            for module in ret.clip.model.modules():
                if isinstance(module, LayerNorm):
                    module.to(torch.float32)

        return ret

    @property
    def text_embeddings(self) -> _Tensor:
        return self._text_embeddings

    def set_descriptions(self, descriptions: Dict[str, str]) -> None:
        self._descriptions = descriptions
        indices = [self.classifier._all_cids.index(key) for key in descriptions]
        text_embeddings = self.get_text_embeddings(list(descriptions.values()))
        self._text_embeddings = text_embeddings[indices]


def _build_clip_mask_fsod(
    mask_generator: nn.Module,
    clip_model: nn.Module,
    backbone: nn.Module,
    classifier: nn.Module,
    roi_pooler: RoIAlign,
    alpha: float = 1.0,
) -> ClipMaskFSOD:
    return ClipMaskFSOD(
        mask_generator,
        clip_model=clip_model,
        backbone=backbone,
        classifier=classifier,
        roi_pooler=roi_pooler,
        alpha=alpha,
    )


@model_registry
def devit_dinov2_clip_fsod(
    label_map_file: str,
    model_name: str = 'dinov2_vitb14',
    clip: Dict[str, Any] = {'model': 'ViT-B/32', 'remove_keys': []},
    roi_pool_size: int = 7,
    feature_layers: List[int] = None,
    prototype_file: str = None,
    background_prototype_file: str = None,
    rpn_args: Dict[str, Any] = None,
    alpha: float = 1.0,
) -> ClipMaskFSOD:
    import json

    from ..clip_utils import build_clip
    from .fsod import build_mask_generator, devit_dinov2_fsod

    _m = devit_dinov2_fsod(
        model_name, roi_pool_size, feature_layers, prototype_file, background_prototype_file, label_map_file, rpn_args
    )

    alpha = max(0, min(alpha, 1.0))
    if alpha == 1.0:
        logger.info('Running only few-shot')
        return _m

    clip_model = build_clip(**clip)
    if rpn_args is None:
        logger.warning('RPN args must be set when using CLIP. Building without mask generator')
        model = ClipMaskFSOD(None, clip_model, _m.backbone, _m.classifier, _m.roi_pooler)
    else:
        mask_generator = build_mask_generator(rpn_args)
        model = _build_clip_mask_fsod(
            mask_generator, clip_model, _m.backbone, _m.classifier, _m.roi_pooler, alpha=alpha
        )

    with open(label_map_file, 'r') as jfile:
        data = json.load(jfile)
        descriptions = data['descriptions']

    model.set_descriptions(descriptions)

    del _m

    return model
