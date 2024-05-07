#!/usr/bin/env python

from typing import Any, Dict, List, Type, Union

import torch
import torch.nn as nn
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
        **kwargs: Dict[str, Any],
    ) -> None:
        super(ClipMaskFSOD, self).__init__(
            mask_generator, backbone=backbone, classifier=classifier, roi_pooler=roi_pooler
        )
        self.clip_model = clip_model

    # def forward_features(
    #     self, im_embeddings: _Tensor, text_embeddings: _Tensor, gt_bboxes: Union[_Tensor, List[_Tensor]]
    # ) -> _Tensor:
    #     # TODO: the gt_bboxes can contain noisy bboxes so add the labels
    #     roi_features = self.roi_pooler(im_embeddings, gt_bboxes)
    #     text_embeddings = nn.functional.interpolate(
    #         text_embeddings[:, :, None, None], roi_features.shape[2:], mode='nearest'
    #     )
    #     roi_features = torch.cat([roi_features, text_embeddings], dim=1)
    #     return roi_features

    def forward(self, images: _Tensor, targets: List[Dict[str, Instances]] = None):
        if not self.training:
            return self.inference(images)

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

        breakpoint()
        
        # proposals = self.mask_generator.get_proposals(images)
        # bboxes = torch.cat([proposal.to_tensor().bboxes.to(self.device) for proposal in proposals])
        # rois = torch.cat([torch.full((len(bboxes), 1), fill_value=0).to(self.device), bboxes], dim=1)
        # return self.classifier(self.mask_generator.predictor.features, rois)

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
        return torch.cat([self.clip_model.get_text_embedding(name) for name in label_names], dim=0)


def _build_clip_mask_fsod(
    mask_generator: nn.Module,
    clip_model: nn.Module,
    backbone: nn.Module,
    classifier: nn.Module,
    roi_pooler: RoIAlign,
) -> ClipMaskFSOD:
    return ClipMaskFSOD(
        mask_generator, clip_model=clip_model, backbone=backbone, classifier=classifier, roi_pooler=roi_pooler
    )


@model_registry
def devit_dinov2_text_fsod(
    model_name: str = 'dinov2_vitb14',
    clip: Dict[str, Any] = {'model': 'ViT-B/32', 'remove_keys': []},
    roi_pool_size: int = 7,
    feature_layers: List[int] = None,    
    prototype_file: str = None,
    background_prototype_file: str = None,
    label_map_file: str = None,
    rpn_args: Dict[str, Any] = None,
) -> ClipMaskFSOD:
    from ..clip_utils import build_clip
    from .fsod import build_mask_generator, devit_dinov2_fsod

    _m = devit_dinov2_fsod(
        model_name, roi_pool_size, feature_layers, prototype_file, background_prototype_file, label_map_file
    )
    
    clip_model = build_clip(**clip)
    if rpn_args is None:
        model = ClipMaskFSOD(clip_model, _m.backbone, _m.classifier, _m.roi_pooler)
    else:
        mask_generator = build_mask_generator(rpn_args)
        model = _build_clip_mask_fsod(mask_generator, clip_model,  _m.backbone, _m.classifier, _m.roi_pooler)

    del _m
    return model
