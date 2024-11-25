#!/usr/bin/env python

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
from functools import singledispatchmethod

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

    def __post_init__(self) -> None:
        if isinstance(self.bbox_fmt, str):
            self.bbox_fmt = getattr(BoundingBoxFormat, self.bbox_fmt.upper())

        size = max(len(self.bboxes), len(self.masks) if self.masks is not None else 0, len(self.class_ids))
        if len(self.bboxes) > 0:
            assert len(self.bboxes) == size, f'Incorect size {len(self.bboxes)} != {size}'
            setattr(self, 'sort_by_area', self._sort_by_area)
        if len(self.class_ids) > 0:
            assert len(self.class_ids) == size, f'Incorect size {len(self.class_ids)} != {size}'
        if self.labels:
            assert len(self.labels) == size, f'Incorect size {len(self.labels)} != {size}'
        if self.masks is not None:
            assert len(self.masks) == size, f'Incorect size {len(self.masks)} != {size}'

            self.image_width, self.image_height = (
                (self.masks.shape[1:][::-1])
                if any(dim == -1 for dim in (self.image_width, self.image_height))
                else (self.image_width, self.image_height)
            )
        self._size = size

    def convert_bbox_fmt(self, bbox_fmt: Union[BoundingBoxFormat, str]) -> 'Instances':
        assert len(self.bboxes) > 0, 'No bounding box instance'
        
        bbox_fmt = getattr(BoundingBoxFormat, bbox_fmt.upper()) if isinstance(bbox_fmt, str) else bbox_fmt

        func = getattr(functional, 'convert_bounding_box_format', None)
        func = func or getattr(functional, 'convert_format_bounding_box', None)
        assert func is not None

        self.bboxes = torch.stack(self.bboxes) if isinstance(self.bboxes, list) else self.bboxes
        self.bboxes = torch.from_numpy(self.bboxes) if isinstance(self.bboxes, np.ndarray) else self.bboxes
        bb = func(self.bboxes, self.bbox_fmt, bbox_fmt)

        # instances = deepcopy(self) if copy else self
        # for i, bbox in enumerate(instances.bboxes):
        #     bbox = torch.as_tensor(bbox) if not isinstance(bbox, torch.Tensor) else bbox
        #     try:
        #         bbox = functional.convert_format_bounding_box(bbox, instances.bbox_fmt, bbox_fmt)
        #     except AttributeError:
        #         bbox = functional.convert_bounding_box_format(bbox, instances.bbox_fmt, bbox_fmt)
        #     instances.bboxes[i] = bbox if isinstance(instances.bboxes[i], torch.Tensor) else bbox.cpu().numpy()

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
        if len(instances.scores) > 0:
            instances.scores = torch.as_tensor(instances.scores).to(device)
        return instances

    def numpy(self):
        instance = deepcopy(self)
        if isinstance(instance.bboxes, torch.Tensor):
            instance.bboxes = instance.bboxes.cpu().numpy()
        if isinstance(instance.masks, torch.Tensor):
            instance.masks = instance.masks.cpu().numpy()
        return instance

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
            try:
                instance.bboxes = functional.resize_bounding_box(
                    instance.bboxes, size=new_hw, spatial_size=(self.image_height, self.image_width)
                )[0]
            except AttributeError:
                instance.bboxes = functional.resize_bounding_boxes(
                    instance.bboxes, size=new_hw, canvas_size=(self.image_height, self.image_width)
                )[0]

        if instance.masks is not None:
            instance.masks = functional.resize(
                instance.masks, new_hw, functional.InterpolationMode.NEAREST, antialias=True
            )

        instance.image_height, instance.image_width = new_hw
        return instance

    def _sort_by_area(self, descending: bool = True) -> 'Instances':
        if len(self.bboxes) == 0:
            return self
        instance = self.convert_bbox_fmt('xywh') if self.bbox_fmt.value != 'XYWH' else deepcopy(self)
        instance = instance.numpy()
        areas = np.prod(instance.bboxes[:, 2:], axis=1)
        order = -1 if descending else 1
        sorted_indices = np.argsort(order * areas).tolist()
        instance.bboxes = instance.bboxes[sorted_indices] if instance.bboxes is not None else None
        instance.masks = instance.masks[sorted_indices] if instance.masks is not None else None
        instance.labels = [instance.labels[i] for i in sorted_indices] if len(instance.labels) else None
        instance.class_ids = instance.class_ids[sorted_indices] if len(instance.class_ids) else None
        instance.scores = instance.scores[sorted_indices] if len(instance.scores) else None
        return instance.convert_bbox_fmt(self.bbox_fmt)

    @singledispatchmethod
    def filter(self, thresh: float = 0.5) -> 'Instances':
        # instance = deepcopy(self)
        instance = self
        if len(instance.scores) == 0:
            return instance

        indices = (
            torch.where(self.scores >= thresh)
            if isinstance(self.scores, torch.Tensor)
            else np.where(self.scores >= thresh)
        )
        indices = indices[0]
        return self._filter_by_indices(instance, indices)

    @filter.register(list)
    def _(self, names: List[str]):
        if not len(self.labels):
            return self

        indices = np.array([i for i, label in enumerate(self.labels) if label not in names])
        return self._filter_by_indices(self, indices)

    @filter.register(str)
    def _(self, name: str):
        return self.filter([name])

    def filter_bboxes(self, crop_box: List[float], iou_thresh: float = 0.0) -> 'Instances':
        if len(crop_box) != 4 or crop_box[2] < crop_box[0] or crop_box[3] < crop_box[1]:
            return self
        crop_box = torch.tensor(crop_box) if not isinstance(crop_box, torch.Tensor) else crop_box
        iou_scores = torchvision.ops.box_iou(self.bboxes, crop_box.reshape(1, -1))

        indices = torch.where(iou_scores > iou_thresh)
        return self._filter_by_indices(self, indices[0])
        
    @staticmethod
    def _filter_by_indices(instance, indices):
        if instance.bboxes is not None:
            instance.bboxes = instance.bboxes[indices]
            instance._size = len(instance.bboxes)
        if instance.masks is not None:
            instance.masks = instance.masks[indices]
            instance._size = len(instance.masks)
        if len(instance.labels):
            instance.labels = [instance.labels[int(i)] for i in indices]
        if len(instance.class_ids):
            instance.class_ids = instance.class_ids[indices]
        if len(instance.scores):
            instance.scores = instance.scores[indices]
        return instance
    
    def __len__(self) -> int:
        return self._size
