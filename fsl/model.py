#!/usr/bin/evn python

from typing import Any, Dict, List, Type, Callable, Iterator, Union, Tuple

import os.path as osp
import numpy as np
from PIL import Image

import torch

from fsl.models import sam_relational_network


if __name__ == '__main__':
    m = sam_relational_network(
        {'model': 'default', 'checkpoint': '/root/.cache/torch/segment_anything/checkpoints/sam_vit_h_4b8939.pth'}
    )
    # m.sam_predictor.model.cuda()
    m.cuda()

    im = Image.open('/root/krishneel/Downloads/000000.jpg')

    bb = [torch.Tensor(np.array([0, 0, 100, 200]))] * 2
    r = m([im], [{'bboxes': bb}])

    import IPython

    IPython.embed()
