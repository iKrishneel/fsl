#!/usr/bin/env python

from typing import Any, Dict, Callable, Optional, Union, List
from importlib import import_module
import torch
import torch.nn as nn
from torchvision.ops import RoIAlign, box_iou
from igniter.registry import model_registry


_Tensor = torch.Tensor


class SamRPN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        mask_gen: nn.Module,
        roi_pool_size: int = 7,
        matcher: Callable = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        super(SamRPN, self).__init__()
        self.backbone = backbone
        self.mask_gen = mask_gen
        self.matcher = matcher

        spatial_scale = backbone.downsize
        self.roi_pool = RoIAlign(roi_pool_size, spatial_scale=spatial_scale, sampling_ratio=0, aligned=True)

        in_channels = kwargs.get('in_channels', 768)
        hid_channels = kwargs.get('hidden_channels', 256)
        num_classes = kwargs.get('num_classes', 2)

        self.bbox_quality = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hid_channels, hid_channels // 2, kernel_size=roi_pool_size, bias=False),
            nn.ReLU(),
            nn.Conv2d(hid_channels // 2, num_classes, kernel_size=(1, 1), bias=False),
            nn.Flatten(start_dim=1),
        )

    def forward(
        self, images: _Tensor, targets: Optional[List[Dict[str, Any]]] = None
    ) -> Union[Dict[str, _Tensor], _Tensor]:
        images = torch.stack(images).to(self.device)

        if not self.training:
            self.inference(x)

        assert targets is not None

        gt_instances = [target['gt_proposal'] for target in targets]
        gt_bboxes = torch.cat([gt_proposal.to_tensor().bboxes.to(self.device) for gt_proposal in gt_instances])

        features = self.backbone(images)
        proposals = [self.mask_gen.get_proposals(image.numpy()) for image in images.permute((0, 2, 3, 1)).cpu()]
        bboxes = torch.cat([proposal.to_tensor().bboxes.to(self.device) for proposal in proposals])

        # also add the gt bboxes to avoid nomatch
        bboxes = torch.cat([gt_bboxes, bboxes])

        match_quality_matrix = box_iou(gt_bboxes, bboxes)
        matched_idxs, matched_labels = self.matcher(match_quality_matrix)
        indices = torch.where(matched_labels != -1)
        labels = matched_labels[indices].long()

        roi_feats = self.roi_pool(features, [bboxes[indices]])
        quality = self.bbox_quality(roi_feats)

        loss = nn.functional.cross_entropy(quality, labels)
        breakpoint()
        losses = {'loss_ce': loss}
        return losses

    @torch.inference_mode()
    def inference(self, x) -> _Tensor:
        pass

    @torch.no_grad()
    def get_features(self, images: _Tensor) -> _Tensor:
        return self.backbone(images)

    @property
    def device(self):
        return self.backbone.device


def get_mask_generator_func(bb_type: str) -> Callable:
    hmap = {
        'sam': import_module('fsl.models.sam_utils').build_sam_auto_mask_generator,
        'fast_sam': import_module('fsl.models.fast_sam_utils').build_fast_sam_mask_generator,
    }
    assert bb_type in hmap, f'Key {bb_type} not found. Available are {hmap.keys()}'
    return hmap.get(bb_type)


def freeze(model: nn.Module):
    for parameter in model.parameters():
        parameter.requires_grad_(False)


@model_registry('sam_rpn')
def build_sam_rpn(
    model_name: str = 'dinov2_vitb14',
    sam_args: Dict[str, Any] = {'type': 'fast_sam'},
    matcher_args: Dict[str, Any] = {'thresholds': [0.3, 0.5], 'labels': [0, -1, 1]},
) -> nn.Module:
    from fsl.models.meta_arch.fsod import DinoV2Patch
    from fsl.utils.matcher import Matcher

    sam_args = dict(sam_args)
    bb_type = sam_args.pop('type', 'fast_sam')
    mask_generator = get_mask_generator_func(bb_type)(**sam_args)
    backbone = DinoV2Patch.build(model_name)
    freeze(backbone)

    matcher = Matcher(**dict(matcher_args))
    sam_rpn = SamRPN(backbone.eval(), mask_generator, matcher=matcher)
    # freeze(sam_rpn)
    return sam_rpn
