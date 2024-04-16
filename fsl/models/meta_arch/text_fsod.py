#!/usr/bin/env python

from typing import Any, Dict, List, Type, Union

import torch
import torch.nn as nn
from fsl.structures import Instances
from fsl.utils import ProtoTypes
from igniter.registry import model_registry
from torchvision.ops import RoIAlign

from .fsod import FSOD, MaskFSOD

_Tensor = Type[torch.Tensor]


class TextFSOD(FSOD):
    def __init__(self, text_encoder, backbone, classifier, roi_pooler, **kwargs: Dict[str, Any]) -> None:
        super(TextFSOD, self).__init__(backbone, classifier, roi_pooler)
        self.text_encoder = text_encoder

    def forward_features(
        self, im_embeddings: _Tensor, text_embeddings: _Tensor, gt_bboxes: Union[_Tensor, List[_Tensor]]
    ) -> _Tensor:
        # TODO: the gt_bboxes can contain noisy bboxes so add the labels
        roi_features = self.roi_pooler(im_embeddings, gt_bboxes)
        text_embeddings = nn.functional.interpolate(
            text_embeddings[:, :, None, None], roi_features.shape[2:], mode='nearest'
        )
        roi_features = torch.cat([roi_features, text_embeddings], dim=1)
        return roi_features

    def forward(self, images: _Tensor, targets: List[Dict[str, Instances]] = None):
        images = torch.stack(images).to(self.device)
        if not self.training:
            return self.inference(images, targets)

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
    def inference(self, images: _Tensor, targets: List[Dict[str, Instances]] = None):
        proposals = self.mask_generator.get_proposals(images)
        bboxes = torch.cat([proposal.to_tensor().bboxes.to(self.device) for proposal in proposals])
        rois = torch.cat([torch.full((len(bboxes), 1), fill_value=0).to(self.device), bboxes], dim=1)
        return self.classifier(self.mask_generator.predictor.features, rois)

    @torch.no_grad()
    def build_image_prototypes(self, image: _Tensor, instances: Instances) -> ProtoTypes:
        instances = instances.to_tensor(self.device)

        text_embeddings = self.get_text_embedding(instances)
        features = self.mask_generator(image[None])

        roi_feats = self.forward_features(features, text_embeddings, [instances.bboxes])
        index = 2 if len(roi_feats.shape) == 4 else 1
        roi_feats = roi_feats.flatten(index).mean(index)
        return ProtoTypes(embeddings=roi_feats, labels=instances.labels, instances=instances)

    def get_text_embedding(self, instances: List[Instances], prefix: str = 'This is a photo of a %s') -> _Tensor:
        instances = [instances] if isinstance(instances, Instances) else instances
        label_names = [prefix % name for instance in instances for name in instance.labels]
        return torch.cat([self.text_encoder.get_text_embedding(name) for name in label_names], dim=0)


class TextMaskFSOD(MaskFSOD):
    def __init__(self, mask_generator, text_encoder, **kwargs: Dict[str, Any]) -> None:
        super(TextMaskFSOD, self).__init__(mask_generator, **kwargs)
        self.text_encoder = text_encoder


@model_registry
def devit_dinov2_text_fsod(
    model_name: str = 'dinov2_vitb14',
    clip_model: str = 'ViT-B/32',
    roi_pool_size: int = 7,
    prototype_file: str = None,
    background_prototype_file: str = None,
    label_map_file: str = None,
    rpn_args: Dict[str, Any] = {},
) -> Union[TextFSOD, TextMaskFSOD]:
    from fsl.models.clip_utils import build_clip

    from .fsod import devit_dinov2_fsod

    _m = devit_dinov2_fsod(model_name, roi_pool_size, prototype_file, background_prototype_file, label_map_file)

    clip_model = build_clip(clip_model, remove_keys=['visual'])

    if rpn_args is not None and len(rpn_args):
        raise NotImplementedError()
    else:
        model = TextFSOD(clip_model, _m.backbone, _m.classifier, _m.roi_pooler)

    del _m
    return model
