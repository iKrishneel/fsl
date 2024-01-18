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
    bboxes: List[Coord] = field(default_factory=lambda: [])
    masks: Array = None
    labels: List[str] = field(default_factory=lambda: [])
    class_ids: List[int] = field(default_factory=lambda: [])
    bbox_fmt: Union[BoundingBoxFormat, str] = BoundingBoxFormat.XYWH
    image_id: str = ""
    image_height: int = -1
    image_width: int = -1

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
        return instances

    def repeat(self, n: int) -> List['Instances']:
        assert n > 0
        return [deepcopy(self) for i in range(n)]
