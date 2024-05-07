#!/usr/bin/env python

from typing import Any, Dict, List, Union

import numpy as np
import torch
from igniter.engine import InferenceEngine as _InferenceEngine
from igniter.registry import engine_registry
from PIL import Image

from fsl.utils.visualizer import Visualizer


@engine_registry('fsl_inference_engine')
class InferenceEngine(_InferenceEngine):
    def __init__(self, *args, **kwargs):
        super(InferenceEngine, self).__init__(*args, **kwargs)

        self._labels = lambda indices: [self.model.classifier._all_cids[i] for i in indices]

        self._class_ids = lambda values: [self.model.classifier._all_cids.index(val) for val in values]

    @torch.inference_mode()
    def __call__(self, image: Union[np.array, Image.Image]) -> Dict[str, Any]:
        im_shape = image.shape[:2] if isinstance(image, np.ndarray) else image.size[::-1]

        im_tensor = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).to(self.model.device)
        im_tensor = self.transforms({'image': im_tensor})['image']

        instances = self.model(im_tensor)

        confidences, indices = torch.topk(instances.scores, 1, dim=1)
        instances.labels = self._labels(indices)
        instances.class_ids = self._class_ids(instances.labels)
        instances.scores = confidences.squeeze()
        return instances.resize(im_shape).to_tensor('cpu')
