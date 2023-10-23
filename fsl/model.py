#!/usr/bin/evn python

import os.path as osp
from typing import Any, Callable, Dict, Iterator, List, Tuple, Type, Union

import numpy as np
import torch
from PIL import Image

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
