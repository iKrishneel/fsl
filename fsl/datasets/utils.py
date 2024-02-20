#!/usr/bin/env python

from typing import List, Type, Union

import numpy as np
import torch
import torchvision
from torchvision.transforms.v2 import functional

from fsl.utils import version

if version.minor_version(torchvision.__version__) <= 15:
    from torchvision.datapoints import BoundingBoxFormat as BBFmt
else:
    from torchvision.tv_tensors import BoundingBoxFormat as BBFmt


_Tensor = Type[torch.Tensor]


def prepare_noisy_boxes(
    gt_boxes: List[_Tensor],
    im_shape: List[int],
    bb_fmt: BBFmt = BBFmt.XYXY,
    box_noise_scale: List[float] = [1.0, 2.5],
    n: int = 5,
    random_center: bool = True,
) -> List[_Tensor]:
    noisy_boxes = []
    h, w = np.array(im_shape, dtype=np.float32)

    scale_range = torch.abs(torch.sub(*box_noise_scale))
    box_noise_scale = torch.rand((n, 1)) * scale_range + 0.5 if isinstance(box_noise_scale, list) else 1.0

    for box in gt_boxes:
        box = box.repeat(n, 1)
        box_ccwh = functional.convert_format_bounding_box(box, bb_fmt, BBFmt.CXCYWH)

        diff = torch.zeros_like(box_ccwh)
        if random_center:
            diff[:, :2] = box_ccwh[:, 2:] / 2
        diff[:, 2:] = box_ccwh[:, 2:] / 2

        rand_sign = torch.randint_like(box_ccwh, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
        rand_part = torch.rand_like(box_ccwh) * rand_sign
        box_ccwh = box_ccwh + torch.mul(rand_part, diff) * box_noise_scale

        noisy_box = functional.convert_format_bounding_box(box_ccwh, BBFmt.CXCYWH, bb_fmt)

        noisy_box[:, 0].clamp_(min=0.0, max=im_shape[1])
        noisy_box[:, 2].clamp_(min=0.0, max=im_shape[1])
        noisy_box[:, 1].clamp_(min=0.0, max=im_shape[0])
        noisy_box[:, 3].clamp_(min=0.0, max=im_shape[0])

        noisy_boxes.append(noisy_box)

    return noisy_boxes
