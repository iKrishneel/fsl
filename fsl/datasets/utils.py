#!/usr/bin/env python

from typing import List, Union

import numpy as np
import torch

from torchvision.transforms.v2 import functional
from torchvision.datapoints import BoundingBoxFormat

from fsl.structures import Proposal


def prepare_noisy_boxes(
    gt_boxes: Union[List[Proposal], Proposal], im_shape: List[int], box_noise_scale: float = 1.0, n: int = 5
) -> Union[List[Proposal], Proposal]:
    gt_boxes = [gt_boxes] if not isinstance(gt_boxes, list) else gt_boxes
    noisy_boxes = []
    h, w = np.array(im_shape, dtype=np.float32)
    for proposal in gt_boxes:
        proposals = proposal.to_tensor().repeat(n)
        box_ccwh = torch.stack([p.convert_bbox_fmt(BoundingBoxFormat.CXCYWH).bbox for p in proposals])

        diff = torch.zeros_like(box_ccwh)
        diff[:, :2] = box_ccwh[:, 2:] / 2
        diff[:, 2:] = box_ccwh[:, 2:] / 2

        rand_sign = torch.randint_like(box_ccwh, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
        rand_part = torch.rand_like(box_ccwh) * rand_sign
        box_ccwh = box_ccwh + torch.mul(rand_part, diff) * box_noise_scale

        noisy_box = functional.convert_format_bounding_box(box_ccwh, BoundingBoxFormat.CXCYWH, BoundingBoxFormat.XYXY)

        noisy_box[:, 0].clamp_(min=0.0, max=im_shape[1])
        noisy_box[:, 2].clamp_(min=0.0, max=im_shape[1])
        noisy_box[:, 1].clamp_(min=0.0, max=im_shape[0])
        noisy_box[:, 3].clamp_(min=0.0, max=im_shape[0])

        for b in noisy_box:
            noisy_boxes.append(Proposal(bbox=b, bbox_fmt=BoundingBoxFormat.XYXY))

    return noisy_boxes
