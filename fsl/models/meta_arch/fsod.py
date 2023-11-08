#!/usr/bin/env python

from typing import Any, Dict, List, Type, Union

import torch
import torch.nn as nn
from igniter.registry import model_registry
from torchvision.ops import RoIAlign

from fsl.structures import Instances
from fsl.utils.prototypes import ProtoTypes

_Tensor = Type[torch.Tensor]


class FSOD(nn.Module):
    def __init__(self, mask_generator, classifier, roi_pooler) -> None:
        super(FSOD, self).__init__()
        self.mask_generator = mask_generator
        self.classifier = classifier
        self.roi_pooler = roi_pooler

    def forward_features(self, im_embeddings: _Tensor, gt_bboxes: Union[_Tensor, List[_Tensor]]) -> _Tensor:
        # TODO: the gt_bboxes can contain noisy bboxes so add the labels
        roi_features = self.roi_pooler(im_embeddings, gt_bboxes)
        # roi_features = roi_features.flatten(2).mean(2)
        return roi_features

    def forward(self, images: _Tensor, targets: List[Dict[str, Instances]] = None):
        images = torch.stack(images).to(self.device)

        if not self.training:
            return self.inference(images, targets)

        assert targets is not None and len(targets) == len(images)

        gt_instances = [target['gt_proposal'] for target in targets]
        gt_bboxes = [gt_proposal.to_tensor().bboxes.to(self.device) for gt_proposal in gt_instances]
        class_labels = torch.cat([instance.class_ids for instance in gt_instances])

        class_labels[class_labels == -1] = self.classifier.train_class_weight.shape[0]

        im_embeddings = self.mask_generator(images)

        roi_features = self.forward_features(im_embeddings, gt_bboxes)
        loss_dict = self.classifier(roi_features, class_labels)

        return loss_dict

    @torch.no_grad()
    def inference(self, images: _Tensor, targets: List[Dict[str, Instances]] = None):
        if targets and 'gt_proposal' in targets[0]:
            proposals = [target['gt_proposal'] for target in targets]
            features = self.mask_generator(images)
        else:
            proposals = self.mask_generator.get_proposals(images)
            features = self.mask_generator.features

        bboxes = torch.cat([proposal.to_tensor().bboxes.to(self.device) for proposal in proposals])
        rois = torch.cat([torch.full((len(bboxes), 1), fill_value=0).to(self.device), bboxes], dim=1)
        roi_features = self.forward_features(features, rois)
        response = self.classifier(roi_features, rois)

        # breakpoint()
        return response

    @torch.no_grad()
    def build_image_prototypes(self, image: _Tensor, instances: Instances) -> ProtoTypes:
        features = self.mask_generator(image[None])
        instances = instances.to_tensor(self.device)
        roi_feats = self.roi_pooler(features, [instances.bboxes])
        index = 2 if len(roi_feats.shape) == 4 else 1
        roi_feats = roi_feats.flatten(index).mean(index)
        return ProtoTypes(embeddings=roi_feats, labels=instances.labels, instances=instances)

    @property
    def device(self) -> torch.device:
        return self.mask_generator.device


@model_registry('sam_fsod')
def build_fsod(
    sam_args: Dict[str, str],
    mask_gen_args: Dict[str, Any] = {},
    roi_pool_size: int = 16,
    prototype_file: str = None,
    background_prototype_file: str = None,
    all_classes_fn: str = None,
    seen_classes_fn: str = None,
) -> FSOD:
    from fsl.models.devit import build_devit
    from fsl.models.sam_relational import build_sam_auto_mask_generator

    mask_generator = build_sam_auto_mask_generator(sam_args, mask_gen_args)
    roi_pooler = RoIAlign(roi_pool_size, spatial_scale=1 / mask_generator.downsize, sampling_ratio=-1)
    classifier = build_devit(prototype_file, background_prototype_file, all_classes_fn, seen_classes_fn)

    return FSOD(mask_generator, classifier, roi_pooler)


@model_registry('resnet_fsod')
def build_resnet_fsod(
    roi_pool_size: int = 16,
    prototype_file: str = None,
    background_prototype_file: str = None,
    all_classes_fn: str = None,
    seen_classes_fn: str = None,
) -> FSOD:
    from timm.models import resnet50
    from fsl.models.devit import build_devit

    backbone = resnet50(pretrained=True)
    backbone.global_pool = nn.Identity()
    backbone.fc = nn.Identity()

    class Backbone(nn.Module):
        def __init__(self, backbone):
            super(Backbone, self).__init__()
            self.backbone = backbone

        @torch.no_grad()
        def forward(self, image: _Tensor) -> _Tensor:
            return self.backbone(image.to(self.device))

        @property
        def device(self) -> torch.device:
            return self.backbone.conv1.weight.device

    roi_pooler = RoIAlign(roi_pool_size, spatial_scale=1 / 32, sampling_ratio=-1)
    classifier = build_devit(prototype_file, background_prototype_file, all_classes_fn, seen_classes_fn)

    return FSOD(Backbone(backbone.eval()), classifier, roi_pooler)
