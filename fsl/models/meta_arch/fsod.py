#!/usr/bin/env python

import importlib
from typing import Any, Dict, List, Tuple, Type, Union

import torch
import torch.nn as nn
from igniter.registry import model_registry
from torchvision.ops import RoIAlign

from fsl.structures import Instances
from fsl.utils.prototypes import ProtoTypes

_Tensor = Type[torch.Tensor]


class FSOD(nn.Module):
    def __init__(self, backbone, classifier, roi_pooler) -> None:
        super(FSOD, self).__init__()
        self.backbone = backbone
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

        im_embeddings = self.backbone(images)

        roi_features = self.forward_features(im_embeddings, gt_bboxes)
        loss_dict = self.classifier(roi_features, class_labels)
        return loss_dict

    @torch.inference_mode()
    def inference(self, images: _Tensor, instances: Instances) -> Tuple[Dict[str, Any]]:
        features = self.backbone(images)
        instances = instances.convert_bbox_fmt('xyxy').to_tensor(self.device)
        bboxes = instances.bboxes
        rois = torch.cat([torch.full((len(bboxes), 1), fill_value=0).to(self.device), bboxes], dim=1)
        roi_features = self.forward_features(features, rois.to(features.dtype))
        response = self.classifier(roi_features)
        return response

    @torch.no_grad()
    def build_image_prototypes(self, image: _Tensor, instances: Instances) -> ProtoTypes:
        features = self.backbone(image[None].to(self.device))
        instances = instances.to_tensor(self.device)

        roi_feats = self.forward_features(features, [instances.bboxes.to(features.dtype)])
        index = 2 if len(roi_feats.shape) == 4 else 1
        roi_feats = roi_feats.flatten(index).mean(index)
        return ProtoTypes(embeddings=roi_feats, labels=instances.labels, instances=instances)

    @property
    def device(self) -> torch.device:
        return self.backbone.device

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True, assign: bool = False):
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for key in state_dict:
            new_key = (
                key
                if 'mask_generator.backbone' not in key
                else key.replace('mask_generator.backbone', 'backbone.backbone')
            )
            new_state_dict[new_key] = state_dict[key]

        return super(FSOD, self).load_state_dict(new_state_dict, strict)


class MaskFSOD(FSOD):
    def __init__(self, mask_generator: Any, **kwargs):
        super(MaskFSOD, self).__init__(**kwargs)
        self.mask_generator = mask_generator

    def forward(self, images: _Tensor, targets: List[Dict[str, Instances]] = None) -> Instances:
        if not self.training:
            assert len(images.shape) == 3, 'Batch inference is currently not supported'
            return self.inference(images)

        raise NotImplementedError('MaskFSOD training is not yet implemented')

    @torch.inference_mode()
    def inference(self, image: _Tensor) -> Instances:
        # image = image.to(self.dtype)
        # im_np = image.permute(1, 2, 0).cpu().numpy()
        # instances = self.mask_generator.get_proposals(im_np)

        instances = self.get_proposals(image)
        image = image[None] if len(image.shape) == 3 else image
        response = super(MaskFSOD, self).inference(image, instances)
        # response[0].update({'instances': instances})
        instances.scores = response[0]['scores']
        return instances

    @torch.inference_mode()
    def get_proposals(self, image: _Tensor) -> Instances:
        image = image.to(self.dtype)
        im_np = image.permute(1, 2, 0).cpu().numpy()
        instances = self.mask_generator.get_proposals(im_np)
        return instances.convert_bbox_fmt('xyxy')

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        for key in self.mask_generator.state_dict():
            state_dict[f'mask_generator.{key}'] = self.mask_generator.state_dict()[key]

        return super().load_state_dict(state_dict, strict)

    def to(self, *args, **kwargs):
        ret = super().to(*args, **kwargs)

        # currently fast sam only supports float32
        if torch.float16 in args:
            ret.mask_generator.to(torch.float32)

        return ret

    @property
    def dtype(self) -> torch.dtype:
        dtype = self.classifier.fc_other_class.weight.dtype
        if hasattr(self.mask_generator, 'dtype'):
            dtype = self.mask_generator.dtype
        return dtype


def _build_fsod(
    backbone: nn.Module,
    roi_pool_size: int = 16,
    prototype_file: str = None,
    background_prototype_file: str = None,
    label_map_file: str = None,
) -> FSOD:
    from fsl.models.devit import build_devit

    roi_pooler = RoIAlign(roi_pool_size, spatial_scale=1 / backbone.downsize, sampling_ratio=-1)
    classifier = build_devit(prototype_file, background_prototype_file, label_map_file)

    return FSOD(backbone, classifier, roi_pooler)


def _build_mask_fsod(
    mask_generator: nn.Module,
    backbone: nn.Module,
    classifier: nn.Module,
    roi_pooler: RoIAlign,
) -> MaskFSOD:
    return MaskFSOD(mask_generator, backbone=backbone, classifier=classifier, roi_pooler=roi_pooler)


"""
@model_registry('sam_fsod')
def build_sam_fsod(
    sam_args: Dict[str, str],
    mask_gen_args: Dict[str, Any] = {},
    roi_pool_size: int = 16,
    prototype_file: str = None,
    background_prototype_file: str = None,
    label_map_file: str = None,
) -> FSOD:
    from fsl.models.sam_utils import build_sam_auto_mask_generator

    backbone = build_sam_auto_mask_generator(sam_args, mask_gen_args)
    return _build_fsod(
        backbone,
        roi_pool_size,
        prototype_file,
        background_prototype_file,
        label_map_file,
    )
"""


@model_registry('resnet_fsod')
def build_resnet_fsod(
    roi_pool_size: int = 16,
    prototype_file: str = None,
    background_prototype_file: str = None,
    label_map_file: str = None,
) -> FSOD:
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
            image = image.to(self.backbone.conv1.weight.dtype)
            return self.backbone(image.to(self.device))

        @property
        def device(self) -> torch.device:
            return self.backbone.conv1.weight.device

        @property
        def downsize(self) -> int:
            return 32

    return _build_fsod(
        Backbone(backbone.eval()),
        roi_pool_size,
        prototype_file,
        background_prototype_file,
        label_map_file,
    )


@model_registry('cie_fsod')
def build_cie_fsod(
    model_name: str = 'ViT-B/32',
    roi_pool_size: int = 16,
    prototype_file: str = None,
    background_prototype_file: str = None,
    label_map_file: str = None,
) -> FSOD:
    import clip

    assert model_name in clip.available_models(), f'{model_name} not found in Clip Model'

    class Backbone(nn.Module):
        def __init__(self, model):
            super(Backbone, self).__init__()
            self.model = model

        def interpolate_embeddings(self, size: List[int]) -> _Tensor:
            size = (size[0] - 1, size[1])
            pos_embedding_token = self.model.positional_embedding[:1]
            pos_embeddings = self.model.positional_embedding[1:][None][None]
            new_pos_embeddings = nn.functional.interpolate(pos_embeddings, size, mode='bicubic', align_corners=True)
            new_pos_embeddings = new_pos_embeddings[0, 0]
            new_pos_embeddings = torch.cat([pos_embedding_token, new_pos_embeddings], dim=0)
            return new_pos_embeddings

        @torch.no_grad()
        def forward(self, x: _Tensor) -> _Tensor:
            x = self.model.conv1(x.to(self.dtype).to(self.device))
            hw = x.shape[2:]

            x = x.flatten(2).permute(0, 2, 1)
            class_embedding = self.model.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            )
            x = torch.cat([class_embedding, x], dim=1)
            positional_embedding = self.interpolate_embeddings(x.shape[1:])
            x = x + positional_embedding.to(x.dtype)
            x = self.model.ln_pre(x)

            x = x.permute(1, 0, 2)
            x = self.model.transformer(x)
            x = x[1:].permute(1, 2, 0)
            x = x.reshape(*x.shape[:2], *hw)
            return x.float()

        @property
        def device(self) -> torch.device:
            return self.model.conv1.weight.device

        @property
        def dtype(self) -> torch.dtype:
            return self.model.conv1.weight.dtype

    model, _ = clip.load(model_name)

    for name in ['transformer', 'token_embedding', 'ln_final']:
        setattr(model, name, nn.Identity())
    model.visual.ln_post = nn.Identity()
    model.visual.proj = None

    for parameter in model.parameters():
        parameter.requires_grad = False

    return _build_fsod(
        Backbone(model.visual.eval()),
        roi_pool_size,
        prototype_file,
        background_prototype_file,
        label_map_file,
    )


@model_registry('dinov2_fsod')
def build_dinov2_fsod(
    model_name: str = 'dinov2_vitb14',
    roi_pool_size: int = 16,
    prototype_file: str = None,
    background_prototype_file: str = None,
    label_map_file: str = None,
) -> FSOD:
    class DinoV2Patch(nn.Module):
        def __init__(self, backbone):
            super(DinoV2Patch, self).__init__()
            self.backbone = backbone.eval()

        @torch.no_grad()
        def forward(self, image: _Tensor) -> _Tensor:
            image = image.to(self.backbone.patch_embed.proj.weight.dtype)
            outputs = self.backbone.get_intermediate_layers(image, n=[self.backbone.n_blocks - 1], reshape=True)
            return outputs[0].float()

        @property
        def downsize(self) -> int:
            return self.backbone.patch_size

        @property
        def device(self) -> torch.device:
            return self.backbone.patch_embed.proj.weight.device

    backbone = torch.hub.load('facebookresearch/dinov2', model_name)

    for param in backbone.parameters():
        param.requires_grad = False

    backbone = backbone.to(torch.float16)

    return _build_fsod(DinoV2Patch(backbone), roi_pool_size, prototype_file, background_prototype_file, label_map_file)


class DinoV2Patch(nn.Module):
    def __init__(self, backbone):
        super(DinoV2Patch, self).__init__()
        self.backbone = backbone
        self.backbone.eval()

    @torch.no_grad()
    def forward(self, image: _Tensor) -> _Tensor:
        im_dtype = image.dtype
        image = image.to(self.device).to(self.dtype)
        outputs = self.backbone.get_intermediate_layers(image, n=[self.backbone.n_blocks - 1], reshape=True)
        return outputs[0].to(im_dtype) if self.training else outputs[0]

    @property
    def downsize(self) -> int:
        return self.backbone.patch_size

    @property
    def device(self):
        return self.backbone.patch_embed.proj.weight.device

    @property
    def dtype(self):
        return self.backbone.patch_embed.proj.weight.dtype

    @classmethod
    def build(cls, model_name: str):
        backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        return cls(backbone)


@model_registry
def devit_dinov2_fsod(
    model_name: str = 'dinov2_vitb14',
    roi_pool_size: int = 7,
    prototype_file: str = None,
    background_prototype_file: str = None,
    label_map_file: str = None,
    rpn_args: Dict[str, Any] = None,
) -> FSOD:
    backbone = torch.hub.load('facebookresearch/dinov2', model_name)

    for param in backbone.parameters():
        param.requires_grad = False

    model = _build_fsod(
        DinoV2Patch.build(model_name).to(torch.float16),  # (backbone.to(torch.float16)),
        roi_pool_size,
        prototype_file,
        background_prototype_file,
        label_map_file,
    )

    if rpn_args is not None:
        rpn_args = dict(rpn_args)
        rpn_type = rpn_args.pop('type', 'sam').lower()

        if rpn_type == 'sam':
            build_func = importlib.import_module('fsl.models.sam_utils').build_sam_auto_mask_generator
        elif rpn_type == 'fast_sam':
            build_func = importlib.import_module('fsl.models.fast_sam_utils').build_fast_sam_mask_generator
        else:
            raise TypeError(f'Unknown type {rpn_type}')

        mask_generator = build_func(**rpn_args)
        model = _build_mask_fsod(
            mask_generator,
            model.backbone,
            model.classifier,
            model.roi_pooler,
        )

    return model
