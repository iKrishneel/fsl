#!/usr/bin/env python

from igniter.registry import func_registry

from .prototypes import ProtoTypes  # NOQA: F401
from .visualizer import Visualizer


@func_registry('fsl_visualizer')
def visualize(image, pred, color=None, show:bool = True):
    import matplotlib.pyplot as plt
    
    im_viz = Visualizer(image)(pred, color=color)

    plt.imshow(im_viz.get_image())
    plt.show()

