#!/usr/bin/env python

from igniter import initiate

from fsl.model import *  # NODA: F401, F403
from fsl.dataset import *  # NODA: F401, F403
from fsl.transforms import *  # NODA: F401, F403


initiate('./configs/sam_relational_net.yaml')
