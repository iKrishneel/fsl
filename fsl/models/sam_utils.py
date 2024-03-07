#!/usr/bin/evn python

from typing import Any, Dict, List, Type, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator as _SAMG
from segment_anything import SamPredictor as _SamPredictor
from segment_anything import sam_model_registry
from torchvision.datapoints import BoundingBoxFormat

from fsl.models import utils
from fsl.structures import Instances

_Tensor = Type[torch.Tensor]
_Module = Type[nn.Module]
_Image = Type[Image.Image]


class SamPredictor(nn.Module, _SamPredictor):
    def __init__(self, sam):
        super(SamPredictor, self).__init__()

        for parameter in sam.parameters():
            parameter.requires_grad = False

        _SamPredictor.__init__(self, sam_model=sam)

    @torch.no_grad()
    def forward(self, images: List[np.ndarray]) -> torch.Tensor:
        return self.set_images(images)

    @torch.no_grad()
    def set_images(self, images: List[Union[np.ndarray, _Tensor]], image_format: str = 'RGB') -> None:
        if isinstance(images[0], np.ndarray):
            assert image_format in ['RGB', 'BGR'], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
            if image_format != self.model.image_format:
                images = [image[..., ::-1] for image in images]
            input_images = [
                torch.as_tensor(self.transform.apply_image(image)).permute(2, 0, 1).contiguous() for image in images
            ]
            images = torch.stack([self.model.preprocess(image.to(self.device)) for image in input_images]).to(
                self.device
            )

        self.model.image_encoder.to(self.device)
        return self.model.image_encoder(images.to(self.device))

    @property
    def img_size(self) -> List[int]:
        return [self.model.image_encoder.img_size] * 2

    @property
    def out_channels(self) -> int:
        return self.model.image_encoder.neck[-1].weight.shape[0]

    @property
    def downsize(self) -> int:
        return self.model.image_encoder.img_size // self.model.image_encoder.patch_embed.proj.kernel_size[0]

    def reset_image(self) -> None:
        self.is_image_set = False
        # self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None


class SamAutomaticMaskGenerator(nn.Module, _SAMG):
    def __init__(self, model, **kwargs):
        super(SamAutomaticMaskGenerator, self).__init__()
        _SAMG.__init__(self, model, **kwargs)

        self.predictor = SamPredictor(model)

    def forward(self, images: List[Union[np.ndarray, _Image, _Tensor]]) -> _Tensor:
        dtype = type(images[0])
        for image in images:
            assert isinstance(image, dtype), 'All image must be of same type'

        if dtype != torch.Tensor:
            images = [np.asarray(image) for image in images]
        return self.predictor.set_images(images)

    def get_proposals(self, image: Union[_Image, np.ndarray]) -> List[Instances]:
        image = np.asarray(image)
        if image.dtype == np.float32:
            image = (image - image.min()) / (image.max() - image.min())
            image = (255 * image).astype(np.uint8)

        masks = self.generate(image)
        instances = Instances(
            *image.shape[:2],
            bboxes=[mask['bbox'] for mask in masks],
            masks=np.array([mask['segmentation'] for mask in masks]),
            bbox_fmt='xywh',
        )
        return instances

    @property
    def downsize(self) -> int:
        return self.predictor.downsize

    @property
    def device(self) -> torch.device:
        return self.predictor.device

    @property
    def dtype(self) -> torch.dtype:
        return self.predictor.model.image_encoder.patch_embed.proj.weight.dtype


def get_sam_model(name: str = 'default'):
    import os
    import subprocess

    url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_%s.pth'
    sam_checkpoint_registry = {'default': 'h_4b8939', 'vit_h': 'h_4b8939', 'vit_l': 'l_0b3195', 'vit_b': 'b_01ec64'}

    directory = os.path.join(os.environ['HOME'], '.cache/torch/segment_anything/checkpoints/')
    checkpoint = os.path.join(directory, f'sam_vit_{sam_checkpoint_registry[name]}.pth')

    if not os.path.isfile(checkpoint):
        checkpoint_url = url % sam_checkpoint_registry[name]
        utils.download(checkpoint_url, directory)

    print(f'Loading checkpoint from {checkpoint}')
    sam_model = sam_model_registry[name](checkpoint)
    return sam_model


def build_sam_predictor(model: str, checkpoint: str = None) -> SamPredictor:
    return SamPredictor(
        get_sam_model(model) if checkpoint is None else sam_model_registry[model](checkpoint=checkpoint)
    )


def build_sam_auto_mask_generator(
    sam_args: Dict[str, str], mask_gen_args: Dict[str, Any] = {}
) -> SamAutomaticMaskGenerator:
    sam_predictor = build_sam_predictor(**sam_args)
    mask_gen_args = mask_gen_args or {}
    return SamAutomaticMaskGenerator(sam_predictor.model, **mask_gen_args)
