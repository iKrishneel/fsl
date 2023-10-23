#!/usr/bin/env python

import torchvision

torchvision.disable_beta_transforms_warning()

from .s3_coco_dataset import *  # NOQA: F401, F403
