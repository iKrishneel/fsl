#!/usr/bin/env python

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torchvision

torchvision.disable_beta_transforms_warning()

from torchvision.transforms.v2 import functional

from fsl.utils import version

if version.minor_version(torchvision.__version__) <= 15:
    from torchvision.datapoints import BoundingBoxFormat
else:
    from torchvision.tv_tensors import BoundingBoxFormat


Coord = Union[List[int], np.ndarray, torch.Tensor]
Array = Union[np.ndarray, torch.Tensor]


@dataclass
class Instances(object):
    image_height: int
    image_width: int
    bboxes: List[Coord] = field(default_factory=lambda: [])
    masks: Array = None
    labels: List[str] = field(default_factory=lambda: [])
    class_ids: List[int] = field(default_factory=lambda: [])
    scores: List[float] = field(default_factory=lambda: [])
    bbox_fmt: Union[BoundingBoxFormat, str] = BoundingBoxFormat.XYWH
    image_id: str = ""

    def __post_init__(self):
        if isinstance(self.bbox_fmt, str):
            self.bbox_fmt = getattr(BoundingBoxFormat, self.bbox_fmt.upper())

        size = max(len(self.bboxes), self.masks.shape[0] if self.masks is not None else 0, len(self.class_ids))
        if len(self.bboxes) > 0:
            assert len(self.bboxes) == size, f'Incorect size {len(self.bboxes)} != {size}'
        if len(self.class_ids) > 0:
            assert len(self.class_ids) == size, f'Incorect size {len(self.class_ids)} != {size}'
        if self.labels:
            assert len(self.labels) == size, f'Incorect size {len(self.labels)} != {size}'
        if self.masks is not None:
            assert self.masks.shape[0] == size, f'Incorect size {len(self.masks)} != {size}'

            self.image_width, self.image_height = (
                (self.masks.shape[1:][::-1])
                if any(dim == -1 for dim in (self.image_width, self.image_height))
                else (self.image_width, self.image_height)
            )

    def convert_bbox_fmt(self, bbox_fmt: Union[BoundingBoxFormat, str]) -> 'Instances':
        assert len(self.bboxes) > 0, 'No bounding box instance'

        if isinstance(bbox_fmt, str):
            bbox_fmt = getattr(BoundingBoxFormat, bbox_fmt.upper())

        instances = deepcopy(self)
        for i, bbox in enumerate(instances.bboxes):
            bbox = torch.as_tensor(bbox) if not isinstance(bbox, torch.Tensor) else bbox
            bbox = functional.convert_format_bounding_box(bbox, instances.bbox_fmt, bbox_fmt)
            instances.bboxes[i] = bbox if isinstance(instances.bboxes[i], torch.Tensor) else bbox.cpu().numpy()
        instances.bbox_fmt = bbox_fmt
        return instances

    def to_tensor(self, device: str = 'cpu') -> 'Instances':
        instances = deepcopy(self)
        if len(instances.bboxes) > 0:
            instances.bboxes = torch.stack([torch.Tensor(bbox) for bbox in instances.bboxes]).to(device)
        if instances.masks is not None:
            instances.masks = torch.as_tensor(instances.masks).to(device)
        if len(instances.class_ids) > 0:
            instances.class_ids = torch.as_tensor(instances.class_ids).to(device)
        if len(instances.scores) > 0:
            instances.scores = torch.as_tensor(instances.scores).to(device)
        return instances

    def repeat(self, n: int) -> List['Instances']:
        assert n > 0
        return [deepcopy(self) for i in range(n)]

    def resize(self, new_hw: List[int]) -> 'Instances':
        assert len(new_hw) == 2
        if any(dim == -1 for dim in (self.image_width, self.image_height)):
            print('Cannot resize as current instance image size is unknown')
            return

        instance = self.to_tensor()
        if len(instance.bboxes):
            instance.bboxes = functional.resize_bounding_box(
                instance.bboxes, size=new_hw, spatial_size=(self.image_height, self.image_width)
            )[0]

        if instance.masks is not None:
            instance.masks = functional.resize(
                instance.masks, new_hw, functional.InterpolationMode.NEAREST, antialias=True
            )

        instance.image_height, instance.image_width = new_hw
        return instance
