#!/usr/bin/env python

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torchvision

torchvision.disable_beta_transforms_warning()

from torchvision.datapoints import BoundingBoxFormat
from torchvision.transforms.v2 import functional

Coord = Union[List[int], np.ndarray, torch.Tensor]
Array = Union[np.ndarray, torch.Tensor]


@dataclass
class Instances(object):
    bboxes: List[Coord] = field(default_factory=lambda: [])
    masks: Array = None
    labels: List[str] = field(default_factory=lambda: [])
    class_ids: List[int] = field(default_factory=lambda: [])
    bbox_fmt: BoundingBoxFormat = BoundingBoxFormat.XYWH
    image_id: str = ""

    def __post_init__(self):
        size = max(len(self.bboxes), self.masks.shape[0] if self.masks is not None else 0, len(self.class_ids))
        if self.bboxes:
            assert len(self.bboxes) == size
        if self.class_ids:
            assert len(self.class_ids) == size
        if self.labels:
            assert len(self.labels) == size
        if self.masks is not None:
            assert self.masks.shape[0] == size

    def convert_bbox_fmt(self, bbox_fmt: BoundingBoxFormat) -> 'Instances':
        assert len(self.bboxes) > 0, 'No bounding box instance'
        for i, bbox in enumerate(self.bboxes):
            bbox = torch.as_tensor(bbox) if not isinstance(bbox, torch.Tensor) else bbox
            bbox = functional.convert_format_bounding_box(bbox, self.bbox_fmt, bbox_fmt)
            self.bboxes[i] = bbox if isinstance(bbox, torch.Tensor) else bbox.cpu().numpy()
            self.bbox_fmt = bbox_fmt
        return self

    def to_tensor(self, device: str = 'cpu') -> 'Instances':
        instances = deepcopy(self)
        if len(instances.bboxes) > 0:
            instances.bboxes = torch.stack([torch.Tensor(bbox) for bbox in instances.bboxes]).to(device)
        if instances.masks is not None:
            instances.masks = torch.as_tensor(instances.masks).to(device)
        if len(instances.class_ids) > 0:
            instances.class_ids = torch.as_tensor(instances.class_ids).to(device)
        return instances

    def repeat(self, n: int) -> List['Instances']:
        assert n > 0
        return [deepcopy(self) for i in range(n)]
