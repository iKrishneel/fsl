#!/usr/bin/env python

from typing import Any, Dict

from igniter.registry import func_registry

from .prototypes import ProtoTypes  # NOQA: F401
from .visualizer import Visualizer


@func_registry('fsl_visualizer')
def visualize(data: Dict[str, Any], color=None, show: bool = True):
    import matplotlib.pyplot as plt

    image, pred = data['image'], data['pred']
    im_viz = Visualizer(image)(pred.numpy(), color=color)

    plt.imshow(im_viz.get_image())
    plt.show()
