#!/usr/bin/env python

import importlib
from typing import Any, Dict, List, Tuple, Type, Union

import torch
import torch.nn as nn
from igniter.registry import model_registry
from omegaconf import DictConfig
from torchvision.ops import RoIAlign

from fsl.structures import Instances
from fsl.utils.prototypes import ProtoTypes

_Tensor = Type[torch.Tensor]


class FSOD(nn.Module):
    def __init__(self, backbone, classifier, roi_pooler, use_mask: bool = False) -> None:
        super(FSOD, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.roi_pooler = roi_pooler
        self.use_mask = use_mask

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

        class_labels = [cid for instance in gt_instances for cid in instance.class_ids]
        names = [label for gt_instance in gt_instances for label in gt_instance.labels]

        for i, name in enumerate(names):
            if name not in self.classifier._seen_cids:
                continue
            class_labels[i] = self.classifier._seen_cids.index(name)

        class_labels = torch.IntTensor(class_labels)
        class_labels[class_labels == -1] = self.classifier.train_class_weight.shape[0]
        
        features = self.get_features(images)
        roi_features = torch.cat(
            [self.get_roi_features(feat[None], gt_instance) for feat, gt_instance in zip(features, gt_instances)]
        )
        
        loss_dict = self.classifier(roi_features, class_labels)
        return loss_dict

    @torch.inference_mode()
    def inference(self, images: _Tensor, instances: Instances) -> Tuple[Dict[str, Any]]:
        features = self.get_features(images)        
        roi_features = self.get_roi_features(features, instances)
        response = self.classifier(roi_features)
        return response

    @torch.no_grad()
    def get_roi_features(self, features: _Tensor, instances: Instances) -> Tuple[Dict[str, Any]]:
        instances = instances.convert_bbox_fmt('xyxy') # .to_tensor(self.device)
        bboxes = instances.bboxes.to(features.device, non_blocking=True)

        if self.use_mask:
            assert hasattr(instances, 'masks') and instances.masks is not None
            masks = instances.masks.to(features.device, non_blocking=True)
            masks = nn.functional.interpolate(masks[:, None], features.shape[2:], mode='nearest')
            features = features * masks.to(features.dtype)

            batch_indices = torch.arange(len(bboxes), device=bboxes.device, dtype=bboxes.dtype).unsqueeze(1)
            rois = torch.cat([batch_indices, bboxes], dim=1)
        else:
            rois = torch.cat([torch.full((len(bboxes), 1), fill_value=0, device=self.device), bboxes], dim=1)

        roi_features = self.forward_features(features, rois.to(features.dtype))
        return roi_features

    @torch.no_grad()
    def build_image_prototypes(self, image: _Tensor, instances: Instances) -> ProtoTypes:
        # features = self.backbone(image[None].to(self.device))
        features = self.get_features(image)
        instances = instances.to_tensor(self.device)

        roi_feats = self.forward_features(features, [instances.bboxes.to(features.dtype)])
        index = 2 if len(roi_feats.shape) == 4 else 1
        roi_feats = roi_feats.flatten(index).mean(index)
        return ProtoTypes(embeddings=roi_feats, labels=instances.labels, instances=instances)

    def get_features(self, images: _Tensor) -> _Tensor:
        images = images if len(images.shape) == 4 else images[None]
        assert len(images.shape) == 4

        images = images.to(self.device, non_blocking=True)
        features = self.backbone(images, norm=False)
        features = features[0] if len(features) == 1 else features

        if isinstance(features, (list, tuple)):
            # TODO: fusion strategy
            features = torch.add(torch.stack(features), dim=0)
            # features = torch.cat(features, dim=1)
        return features

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

        has_bg1 = any(
            [
                'bg_' in key and isinstance(getattr(self.classifier, key.split('.')[0]), torch.Tensor)
                for key in self.classifier.state_dict()
            ]
        )
        has_bg2 = any(['classifier.bg_' in key for key in state_dict])

        for key in state_dict:
            name = key.replace('classifier.', '')
            if '_class_weight' in key or 'bg_tokens' in key:
                if hasattr(self.classifier, name):
                    if isinstance(getattr(self.classifier, name), torch.Tensor):
                        new_state_dict[key] = getattr(self.classifier, name)
                        # delattr(self.classifier, name)
                    else:
                        delattr(self.classifier, name)

                self.classifier.register_buffer(name, state_dict[key])

        if not has_bg1 and has_bg2:
            self.classifier._init_bg_layers()

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

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True, assign: bool = True):
        for key in self.mask_generator.state_dict():
            state_dict[f'mask_generator.{key}'] = self.mask_generator.state_dict()[key]

        return super(MaskFSOD, self).load_state_dict(state_dict, strict, assign)

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
    use_mask: bool = False,
) -> FSOD:
    from fsl.models.devit import build_devit

    roi_pooler = RoIAlign(roi_pool_size, spatial_scale=1 / backbone.downsize, sampling_ratio=-1)
    classifier = build_devit(prototype_file, background_prototype_file, label_map_file)

    return FSOD(backbone, classifier, roi_pooler, use_mask)


def _build_mask_fsod(
    mask_generator: nn.Module,
    backbone: nn.Module,
    classifier: nn.Module,
    roi_pooler: RoIAlign,
    use_mask: bool = False,
) -> MaskFSOD:
    return MaskFSOD(
        mask_generator, backbone=backbone, classifier=classifier, roi_pooler=roi_pooler, use_mask=use_mask
    )


@model_registry('dinov2_fsod')
def build_dinov2_fsod(
    model_name: str = 'dinov2_vitb14',
    roi_pool_size: int = 16,
    feature_layers: List[int] = None,
    prototype_file: str = None,
    background_prototype_file: str = None,
    label_map_file: str = None,
    use_mask: bool = False,
) -> FSOD:
    from ..backbone import DinoV2Backbone as DinoV2Patch

    backbone = DinoV2Patch.build(model_name=model_name, frozen=True, feat_layers=feature_layers)
    backbone = backbone.to(torch.float16)
    return _build_fsod(
        backbone, roi_pool_size, prototype_file, background_prototype_file, label_map_file, use_mask=use_mask
    )


def build_mask_generator(rpn_args: DictConfig) -> nn.Module:
    if rpn_args is None:
        return

    rpn_args = dict(rpn_args)
    rpn_type = rpn_args.pop('type', 'sam').lower()

    if rpn_type == 'sam':
        build_func = importlib.import_module('fsl.models.sam_utils').build_sam_auto_mask_generator
    elif rpn_type == 'fast_sam':
        build_func = importlib.import_module('fsl.models.fast_sam_utils').build_fast_sam_mask_generator
    else:
        raise TypeError(f'Unknown type {rpn_type}')

    mask_generator = build_func(**rpn_args)
    return mask_generator


@model_registry
def devit_dinov2_fsod(
    model_name: str = 'dinov2_vitb14',
    roi_pool_size: int = 7,
    feature_layers: List[int] = None,
    prototype_file: str = None,
    background_prototype_file: str = None,
    label_map_file: str = None,
    rpn_args: Dict[str, Any] = None,
    use_mask: bool = False,
) -> Union[FSOD, MaskFSOD]:
    model = build_dinov2_fsod(
        model_name=model_name,
        roi_pool_size=roi_pool_size,
        feature_layers=feature_layers,
        prototype_file=prototype_file,
        background_prototype_file=background_prototype_file,
        label_map_file=label_map_file,
        use_mask=use_mask
    )

    if rpn_args is not None:
        mask_generator = build_mask_generator(rpn_args)
        model = _build_mask_fsod(
            mask_generator,
            model.backbone,
            model.classifier,
            model.roi_pooler,
            use_mask,
        )

    return model
