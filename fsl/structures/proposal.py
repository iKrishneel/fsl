#!/usr/bin/env python

from typing import Any, Dict, List, Union
from dataclasses import dataclass

import numpy as np
import torch

import torchvision

torchvision.disable_beta_transforms_warning()

from torchvision.transforms.v2 import functional
from torchvision.datapoints import BoundingBoxFormat

Coord = Union[List[int], np.ndarray, torch.Tensor]
Array = Union[np.ndarray, torch.Tensor]


@dataclass
class Proposal(object):
    bbox: Coord = None
    mask: Array = None
    label: str = ""
    class_id: int = -1
    bbox_fmt: BoundingBoxFormat = BoundingBoxFormat.XYWH

    def convert_bbox_fmt(self, bbox_fmt: BoundingBoxFormat) -> 'Proposal':
        assert self.bbox is not None
        bbox = torch.as_tensor(self.bbox) if not isinstance(self.bbox, torch.Tensor) else self.bbox
        bbox = functional.convert_format_bounding_box(bbox, self.bbox_fmt, bbox_fmt)
        self.bbox = bbox if isinstance(self.bbox, torch.Tensor) else bbox.cpu().numpy()
        self.bbox_fmt = bbox_fmt
        return self

    def to_tensor(self) -> 'Proposal':
        if self.bbox is not None:
            self.bbox = torch.Tensor(self.bbox)
        if self.mask is not None:
            self.mask = torch.as_tensor(self.mask)
        return self

    def __repr__(self) -> str:
        return f'{self.label} | {self.bbox} | {self.bbox_fmt}'


@dataclass
class Proposals(object):
    bboxes: List[Coord] = None
    masks: List[Array] = None
    class_ids: List[int] = None
    bbox_fmt: BoundingBoxFormat = BoundingBoxFormat.XYWH

    def __call__(self, annotations: Dict[str, Any]) -> 'Proposals':
        for key in ['bboxes', 'masks', 'category_id', 'class_id']:
            setattr(self, key, annotations.get(key, None))
