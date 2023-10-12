#!/usr/bin/env python

from typing import List, Union
from dataclasses import dataclass

import numpy as np
import torch

import torchvision
torchvision.disable_beta_transforms_warning()

from torchvision.transforms.v2 import functional
from torchvision.datapoints import BoundingBoxFormat


@dataclass
class Proposal(object):
    bbox: Union[List[int], np.ndarray, torch.Tensor] = None
    mask: Union[np.ndarray, torch.Tensor] = None
    label: str = ""
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
