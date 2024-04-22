#!/usr/bin/env python

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np
import torch
import torchvision
from igniter.registry import transform_registry
from PIL import Image

torchvision.disable_beta_transforms_warning()  # NOQA

from torchvision import transforms
from torchvision.ops.boxes import box_iou
from torchvision.transforms.v2 import functional as TF

from fsl.datasets.utils import prepare_noisy_boxes
from fsl.utils import version
from fsl.utils.matcher import Matcher


def tv_version_lte(ver: int = 15):
    return version.minor_version(torchvision.__version__) <= ver


if tv_version_lte(15):
    from torchvision.datapoints import BoundingBoxFormat
else:
    from torchvision.tv_tensors import BoundingBoxFormat


_Tensor = Type[torch.Tensor]
_Image = Type[Image.Image]


@transform_registry
@dataclass
class ResizeLongestSide(object):
    size: int

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image, bboxes = [data.get(key) for key in ['image', 'bboxes']]
        img_hw = image.shape[1:]

        target_size = self.get_preprocess_shape(*img_hw, self.size)
        image = TF.resize(image, target_size, antialias=True)

        if 'masks' in data:
            data['masks'] = TF.resize(
                data['masks'], target_size, interpolation=TF.InterpolationMode.NEAREST, antialias=True
            )

        if bboxes is not None:
            func = TF.resize_bounding_box if tv_version_lte(15) else TF.resize_bounding_boxes
            bboxes = [func(bbox, img_hw, size=target_size)[0] for bbox in bboxes]
            data['bboxes'] = bboxes

        data['image'] = image
        return data

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


@transform_registry
@dataclass
class ResizeLongestSide2(ResizeLongestSide):
    long_size: int
    short_size: int

    def get_preprocess_shape(self, oldh: int, oldw: int, **kwargs) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale_long = self.long_size * 1.0 / max(oldh, oldw)
        scale_short = self.short_size * 1.0 / min(oldh, oldw)

        newh = oldh * (scale_long if oldh > oldw else scale_short)
        neww = oldw * (scale_short if oldh > oldw else scale_long)

        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


@transform_registry
@dataclass
class Resize(object):
    size: Union[List[int], int]

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image, bboxes = [data.get(key) for key in ['image', 'bboxes']]
        img_hw = image.shape[1:]
        image = TF.resize(image, self.size, antialias=True)

        if 'masks' in data:
            data['masks'] = TF.resize(
                data['masks'], self.size, interpolation=TF.InterpolationMode.NEAREST, antialias=True
            )

        if bboxes is not None:
            func = TF.resize_bounding_box if tv_version_lte(15) else TF.resize_bounding_boxes
            bboxes = [func(bbox, img_hw, size=image.shape[1:])[0] for bbox in bboxes]
            data['bboxes'] = bboxes

        data['image'] = image
        return data


@transform_registry
@dataclass
class PadToSize(object):
    size: int
    bbox_fmt: BoundingBoxFormat = BoundingBoxFormat.XYXY

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image, bboxes = [data.get(key) for key in ['image', 'bboxes']]
        img_hw = image.shape[1:]

        # LRTB --> LTRB
        padding = (0, 0, self.size - img_hw[1], self.size - img_hw[0])

        func = TF.pad_image_tensor if tv_version_lte(15) else TF.pad_image
        image = func(image, padding)

        if 'masks' in data:
            data['masks'] = func(data['masks'], padding)

        if bboxes is not None:
            func = TF.pad_bounding_box if tv_version_lte(15) else TF.pad_bounding_boxes
            bboxes = [func(bbox, self.bbox_fmt, img_hw, padding)[0] for bbox in bboxes]
            data['bboxes'] = bboxes

        data['image'] = image
        return data


@transform_registry
@dataclass
class VHFlip(object):
    hflip: bool = True
    vflip: bool = True
    bbox_fmt: BoundingBoxFormat = BoundingBoxFormat.XYXY

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image, bboxes = [data.get(key) for key in ['image', 'bboxes']]
        mask = data.get('masks', None)

        hflipper = TF.horizontal_flip_image_tensor if isinstance(image, torch.Tensor) else TF.horizontal_flip_image_pil
        vflipper = TF.vertical_flip_image_tensor if isinstance(image, torch.Tensor) else TF.vertical_flip_image_pil

        if self.hflip and np.random.choice([True, False]):
            image = hflipper(image)
            mask = hflipper(mask) if mask is not None else mask

            if bboxes is not None:
                spatial_size = image.shape[1:]
                bboxes = [TF.horizontal_flip_bounding_box(bbox, self.bbox_fmt, spatial_size) for bbox in bboxes]

        if self.vflip and np.random.choice([True, False]):
            image = vflipper(image)
            mask = vflipper(mask) if mask is not None else mask
            if bboxes is not None:
                spatial_size = image.shape[1:]
                bboxes = [TF.vertical_flip_bounding_box(bbox, self.bbox_fmt, spatial_size) for bbox in bboxes]

        data['image'] = image

        if bboxes is not None:
            data['bboxes'] = bboxes

        if mask is not None:
            data['masks'] = mask

        return data


@transform_registry
@dataclass
class ConvertFormatBoundingBox(object):
    old_fmt: str
    new_fmt: str

    def __post_init__(self):
        self.old_fmt, self.new_fmt = [getattr(BoundingBoxFormat, fmt) for fmt in [self.old_fmt, self.new_fmt]]

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image, bboxes = [data.get(key) for key in ['image', 'bboxes']]
        convert_bounding_box_format = (
            TF.convert_format_bounding_box
            if version.minor_version(torchvision.__version__) < 16
            else TF.convert_bounding_box_format
        )
        if bboxes is not None:
            bboxes = [convert_bounding_box_format(bbox, self.old_fmt, self.new_fmt) for bbox in bboxes]
            data['bboxes'] = bboxes

        data['image'] = image
        return data


@transform_registry
@dataclass
class Normalize(object):
    mean: List[float] = field(default_factory=lambda: [0.48145466, 0.4578275, 0.40821073])
    std: List[float] = field(default_factory=lambda: [0.26862954, 0.26130258, 0.27577711])

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image = [data.get(key) for key in ['image']][0]
        func = TF.to_image_tensor if tv_version_lte(15) else TF.to_image
        image = TF.to_image(image).float() / 255.0
        image = TF.normalize(image, self.mean, self.std)
        data['image'] = image
        return data


@transform_registry
@dataclass
class ResizeToDivisible(object):
    factor: float

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image = [data[key] for key in ['image']][0]
        h, w = image.shape[1:]
        h, w = h - h % self.factor, w - w % self.factor
        return Resize([h, w])(data)


@transform_registry
class RandomResizeCrop(transforms.RandomResizedCrop):
    def __init__(self, *args, **kwargs):
        self.min_size = kwargs.pop('min_size', 10)
        super(RandomResizeCrop, self).__init__(*args, **kwargs)

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not np.random.choice([True, False]):
            return Resize(self.size)(data)

        image = data['image']

        func_bb = TF.resized_crop_bounding_box if tv_version_lte(15) else TF.resized_crop_bounding_boxes

        i, j, h, w = self.get_params(image, self.scale, self.ratio)
        data['image'] = TF.resized_crop(image, i, j, h, w, self.size, self.interpolation, antialias=self.antialias)

        if 'masks' in data:
            data['masks'] = TF.resized_crop(
                data['masks'], i, j, h, w, self.size, transforms.InterpolationMode.NEAREST, antialias=self.antialias
            )

        if 'bboxes' in data:
            bboxes = data['bboxes']
            bboxes = torch.stack(bboxes) if isinstance(bboxes, list) else bboxes
            nbboxes, _ = func_bb(bboxes, BoundingBoxFormat.XYXY, i, j, h, w, self.size)

            diff = nbboxes[:, 2:] - nbboxes[:, :2]
            flags = torch.prod(diff > self.min_size, dim=1).bool()
            data['bboxes'] = nbboxes[flags]
            data['category_ids'] = data['category_ids'][flags]
            data['category_names'] = [name for i, name in enumerate(data['category_names']) if flags[i]]

        return data


@transform_registry
@dataclass
class ArgumentNoisyBBoxes(object):
    sample_size: int = 5
    pos_ratio: float = 0.25
    proposal_thresh: List[float] = field(default_factory=lambda: [0.3, 0.7])
    labels: List[int] = field(default_factory=lambda: [0, -1, 1])
    background_id: int = -1
    batch_size_per_image: int = 128

    def __post_init__(self):
        self.proposal_matcher = Matcher(self.proposal_thresh, self.labels)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.sample_size <= 0 or not np.random.choice([True, False]):
            return data

        image, gt_bboxes, class_ids, names = [
            data[key] for key in ['image', 'bboxes', 'category_ids', 'category_names']
        ]

        gt_bboxes = [gt_bbox.reshape(-1, 4) for gt_bbox in gt_bboxes]
        img_hw = image.shape[1:]
        noisy_bboxes = prepare_noisy_boxes(
            gt_bboxes, img_hw, n=self.sample_size, random_center=np.random.choice([True, False])
        )
        bboxes = torch.cat([torch.cat([gt_bboxes[i], noisy_bboxes[i]]) for i in range(len(gt_bboxes))])

        match_quality_matrix = box_iou(torch.cat(gt_bboxes), bboxes)
        matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
        class_labels = class_ids[matched_idxs]

        if len(class_labels) == 0:
            # no annotation on this image
            assert torch.all(matched_labels == 0)
            class_labels = torch.zeros_like(matched_idxs)

        temp_label = -100
        class_labels[matched_labels == -1] = temp_label
        class_labels[matched_labels == 0] = self.background_id

        positive = ((class_labels != temp_label) & (class_labels != self.background_id)).nonzero().flatten()
        negative = (class_labels == self.background_id).nonzero().flatten()

        num_pos = int(self.batch_size_per_image * self.pos_ratio)
        # protect against not enough positive examples
        num_pos = min(positive.numel(), num_pos)
        num_neg = self.batch_size_per_image - num_pos
        # protect against not enough negative examples
        num_neg = min(negative.numel(), num_neg)

        perm1 = torch.randperm(positive.numel())[:num_pos]
        perm2 = torch.randperm(negative.numel())[:num_neg]
        pos_idx = positive[perm1]
        neg_idx = negative[perm2]
        sampled_idxs = torch.cat([pos_idx, neg_idx], dim=0)

        # bboxes =
        class_labels = class_labels[sampled_idxs]

        mapping = {int(class_ids[i]): names[i] for i in range(len(class_ids))}
        mapping[self.background_id] = 'background'
        names = [mapping[int(i)] for i in class_labels]

        data['image'] = image
        data['bboxes'] = bboxes[sampled_idxs]
        data['category_ids'] = torch.Tensor(class_labels)
        data['category_names'] = names
        return data
