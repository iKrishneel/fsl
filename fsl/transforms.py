#!/usr/bin/env python

from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision
from igniter.registry import transform_registry
from PIL import Image

torchvision.disable_beta_transforms_warning()  # NOQA

from torchvision.datapoints import BoundingBoxFormat
from torchvision.transforms.v2 import functional as TF

_Tensor = torch.Tensor
_Image = Image.Image


@transform_registry
class ResizeLongestSide(nn.Module):
    def __init__(self, size: int) -> None:
        super(ResizeLongestSide, self).__init__()
        self.size = size

    def forward(self, image: Union[_Image, np.ndarray], bboxes: List[_Tensor] = None) -> Tuple[_Image, _Tensor]:
        image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        img_hw = image.size[::-1]

        target_size = self.get_preprocess_shape(*img_hw, self.size)
        image = TF.resize(image, target_size)

        if bboxes is not None:
            bboxes = [TF.resize_bounding_box(bbox, spatial_size=img_hw, size=image.size[::-1])[0] for bbox in bboxes]

        return image, bboxes

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
class PadToSize(nn.Module):
    def __init__(self, size: int, bbox_fmt: BoundingBoxFormat = BoundingBoxFormat.XYXY):
        super(PadToSize, self).__init__()
        self.size = size
        self.bbox_fmt = bbox_fmt

    def forward(self, image: Union[np.ndarray, _Image], bboxes: List[_Tensor] = None) -> Tuple[_Image, _Tensor]:
        image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        img_hw = image.size[::-1]

        # padh, padw = self.size - img_hw[0], self.size - img_hw[1]
        # image = torch.from_numpy(np.array(image)).permute((2, 0, 1))
        # image = nn.functional.pad(image, (0, padw, 0, padh))

        # LRTB --> LTRB
        padding = (0, 0, self.size - img_hw[1], self.size - img_hw[0])
        image = TF.pad_image_pil(image, padding)
        if bboxes is not None:
            bboxes = [TF.pad_bounding_box(bbox, self.bbox_fmt, img_hw, padding)[0] for bbox in bboxes]

        return image, bboxes


@transform_registry
class VHFlip(nn.Module):
    def __init__(self, hflip: bool = True, vflip: bool = True, bbox_fmt: BoundingBoxFormat = BoundingBoxFormat.XYXY):
        super(VHFlip, self).__init__()
        self.bbox_fmt = bbox_fmt
        self.vflip = vflip
        self.hflip = hflip

    def forward(self, image: Union[_Image, _Tensor], bboxes: List[_Tensor] = None):
        hflipper = TF.horizontal_flip_image_tensor if isinstance(image, torch.Tensor) else TF.horizontal_flip_image_pil
        vflipper = TF.vertical_flip_image_tensor if isinstance(image, torch.Tensor) else TF.vertical_flip_image_pil

        if self.hflip and np.random.choice([True, False]):
            image = hflipper(image)
            if bboxes is not None:
                spatial_size = image.shape[:1] if isinstance(image, torch.Tensor) else image.size[::-1]
                bboxes = [TF.horizontal_flip_bounding_box(bbox, self.bbox_fmt, spatial_size) for bbox in bboxes]

        if self.vflip and np.random.choice([True, False]):
            image = vflipper(image)
            if bboxes is not None:
                spatial_size = image.shape[:1] if isinstance(image, torch.Tensor) else image.size[::-1]
                bboxes = [TF.vertical_flip_bounding_box(bbox, self.bbox_fmt, spatial_size) for bbox in bboxes]
        return image, bboxes


@transform_registry
class ConvertFormatBoundingBox(nn.Module):
    def __init__(self, old_fmt: str, new_fmt: str):
        super(ConvertFormatBoundingBox, self).__init__()
        self.old_fmt, self.new_fmt = [getattr(BoundingBoxFormat, fmt) for fmt in [old_fmt, new_fmt]]

    def forward(self, image: _Image, bboxes: List[_Tensor]):
        bboxes = [TF.convert_format_bounding_box(bbox, self.old_fmt, self.new_fmt) for bbox in bboxes]
        return image, bboxes


@transform_registry
class ClipPreprocess(nn.Module):
    def __init__(
        self,
        mean: List[float] = [0.48145466, 0.4578275, 0.40821073],
        std: List[float] = [0.26862954, 0.26130258, 0.27577711],
    ):
        super(ClipPreprocess, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, image: _Image, bboxes: List[_Tensor] = None):
        image = TF.to_image_tensor(image)
        image = image.float() / 255.0
        image = TF.normalize(image, self.mean, self.std)
        return image, bboxes
