#!/usr/bin/env python

from copy import deepcopy
from dataclasses import dataclass
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

    def repeat(self, n: int) -> List['Proposal']:
        assert n > 0
        return [deepcopy(self) for i in range(n)]

    def __repr__(self) -> str:
        return f'{self.bbox}'


@dataclass
class Proposals(object):
    bboxes: List[Coord] = None
    masks: List[Array] = None
    class_ids: List[int] = None
    bbox_fmt: BoundingBoxFormat = BoundingBoxFormat.XYWH

    def __call__(self, annotations: Dict[str, Any]) -> 'Proposals':
        for key in ['bboxes', 'masks', 'category_id', 'class_id']:
            setattr(self, key, annotations.get(key, None))
