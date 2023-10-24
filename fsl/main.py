#!/usr/bin/env python

from igniter import initiate

from fsl.datasets import *  # NOQA: F401, F403
from fsl.models import *  # NOQA: F401, F403
from fsl.transforms import *  # NOQA: F401, F403


# initiate('./configs/sam_relational_net.yaml')
initiate('../configs/devit/sam_vitb_trainval_5shot.yaml')
