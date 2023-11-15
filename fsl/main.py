#!/usr/bin/env python

from igniter import initiate

from fsl.datasets import *  # NOQA: F401, F403
from fsl.models import *  # NOQA: F401, F403
from fsl.transforms import *  # NOQA: F401, F403


config_file = '../configs/devit/sam_vitb_trainval_30shot.yaml'
config_file = '../configs/devit/dinov2_trainval_5shot.yaml'
config_file = '../configs/devit/resnet_trainval_30shot.yaml'
config_file = '../configs/devit/dinov2_trainval_30shot.yaml'
config_file = '../configs/devit/resnet_clip_vitb_trainval_30shot.yaml'

initiate(config_file)
