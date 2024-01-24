#!/usr/bin/env python

from typing import Any, Dict, List, Type, Union

import torch
import torch.nn as nn
from igniter.registry import model_registry
from torchvision.ops import RoIAlign

from fsl.structures import Instances
from fsl.utils import ProtoTypes

from .fsod import FSOD

_Tensor = Type[torch.Tensor]


class TextFSOD(FSOD):
    def __init__(
        self,
        mask_generator,
        text_encoder,
        classifier,
        roi_pooler,
    ) -> None:
        super(TextFSOD, self).__init__(mask_generator, classifier, roi_pooler)
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
        if not self.training:
            return self.inference(images, targets)

        assert targets is not None and len(targets) == images.shape[0]

        gt_instances = [target['gt_proposal'] for target in targets]
        gt_bboxes = [gt_proposal.to_tensor().bboxes for gt_proposal in gt_instances]
        # gt_names = [gt_proposal.labels for gt_proposal in gt_instances]
        # text_embeddings = [self.text_encoder.get_text_embedding(names) for names in gt_names]

        text_embeddings = self.get_text_embedding(gt_instances)
        im_embeddings = self.mask_generator(images)

        roi_features = self.forward_features(im_embeddings, text_embeddings, gt_bboxes)
        loss_dict = self.classifier(roi_features)

        return loss_dict

    @torch.no_grad()
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


def _build_text_fsod(
    mask_generator: Any,
    clip_args: Dict[str, Any] = {'model_name': 'ViT-B/32', 'remove_keys': []},
    roi_pool_size: int = 16,
    prototype_file: str = None,
    background_prototype_file: str = None,
    all_classes_fn: str = None,
    seen_classes_fn: str = None,
) -> TextFSOD:
    from fsl.models.clip import build_clip
    from fsl.models.devit import build_devit

    text_encoder = build_clip(**clip_args)
    roi_pooler = RoIAlign(roi_pool_size, spatial_scale=1 / mask_generator.downsize, sampling_ratio=-1)
    classifier = build_devit(prototype_file, background_prototype_file, all_classes_fn, seen_classes_fn)

    return TextFSOD(mask_generator, text_encoder, classifier, roi_pooler)


@model_registry('sam_clip_fsod')
def build_clip_sam_fsod(
    sam_args: Dict[str, str],
    mask_gen_args: Dict[str, Any] = {},
    clip_args: Dict[str, Any] = {'clip_model': 'ViT-B/32', 'remove_keys': ['visual']},
    roi_pool_size: int = 16,
    prototype_file: str = None,
    background_prototype_file: str = None,
    all_classes_fn: str = None,
    seen_classes_fn: str = None,
) -> TextFSOD:
    from fsl.models.sam_utils import build_sam_auto_mask_generator

    mask_generator = build_sam_auto_mask_generator(sam_args, mask_gen_args)
    return _build_text_fsod(
        mask_generator,
        clip_args,
        roi_pool_size,
        prototype_file,
        background_prototype_file,
        all_classes_fn,
        seen_classes_fn,
    )


@model_registry('resnet_clip_fsod')
def build_text_resnet_fsod(
    clip_args: Dict[str, Any],  #  = {'clip_model': 'ViT-B/32', 'remove_keys': ['visual']},
    roi_pool_size: int = 16,
    prototype_file: str = None,
    background_prototype_file: str = None,
    all_classes_fn: str = None,
    seen_classes_fn: str = None,
) -> TextFSOD:
    from timm.models import resnet50

    backbone = resnet50(pretrained=True)
    backbone.global_pool = nn.Identity()
    backbone.fc = nn.Identity()

    for parameter in backbone.parameters():
        parameter.requires_grad = False

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

        @property
        def downsize(self) -> int:
            return 32

    return _build_text_fsod(
        Backbone(backbone.eval()),
        clip_args,
        roi_pool_size,
        prototype_file,
        background_prototype_file,
        all_classes_fn,
        seen_classes_fn,
    )
