#!/usr/bin/env python

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import clip
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose

from fsl.models.sam_utils import SamAutomaticMaskGenerator

_Tensor = Type[torch.Tensor]
_Image = Type[Image.Image]


def normalize(x: _Tensor, dim: int = -1, keepdim: bool = True):
    return x / x.norm(dim=dim, keepdim=keepdim)


class CLIP(nn.Module):
    def __init__(self, clip_model: str, remove_keys: List[str] = []):
        super(CLIP, self).__init__()
        assert clip_model in clip.available_models(), f'{clip_model} not found. Available are {clip.available_models()}'
        self.clip_model, self.preprocessing = self.get_clip_model(clip_model, remove_keys=remove_keys)

    def encode_text(self, text: _Tensor) -> _Tensor:
        x = self.clip_model.token_embedding(text).type(self.dtype)
        x = x + self.clip_model.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip_model.text_projection

    def forward(self, image: Union[_Tensor, _Image], bboxes: List[_Tensor]):
        img = Image.fromarray(image.permute(1, 2, 0).cpu().numpy()) if not isinstance(image, Image.Image) else image
        im_crops = torch.stack([self.preprocessing(img.crop(bbox.int().cpu().numpy())) for bbox in bboxes])
        roi_feats = self.clip_model.encode_image(im_crops.to(self.device)).float()
        return roi_feats

    def forward_text(self, text_tokens: _Tensor) -> _Tensor:
        text_features = self.encode_text(text_tokens).float()
        return text_features

    @torch.no_grad()
    def get_text_embedding(self, texts: Union[str, List[str]], is_normalized: bool = True) -> _Tensor:
        text_tokens = clip.tokenize(texts).to(self.device)
        text_embeddings = self.forward_text(text_tokens)
        return normalize(text_embeddings) if is_normalized else text_embeddings

    @property
    def device(self) -> torch.device:
        if getattr(self.clip_model, 'visual', None):
            layer = self.clip_model.visual.conv1
        else:
            layer = self.clip_model.transformer.resblocks[0].attn.out_proj
        return layer.weight.device

    @property
    def dtype(self):
        if hasattr(self.clip_model, 'visual'):
            return self.clip_model.dtype
        return self.clip_model.transformer.resblocks[0].attn.out_proj.weight.dtype

    @staticmethod
    def get_clip_model(name: str, remove_keys: List[str] = []) -> Tuple[clip.model.CLIP, Compose]:
        assert name in clip.available_models(), f'{name} not found. Options are {clip.available_models()}'
        model, preprocessng = clip.load(name)
        for key in remove_keys:
            delattr(model, key)

        for parameter in model.parameters():
            parameter.requires_grad = False
        return model, preprocessng


def build_clip(model: str, remove_keys: List[str] = []) -> CLIP:
    return CLIP(model, remove_keys)


class SamPlusCLIP(nn.Module):
    def __init__(self, mask_generator: SamAutomaticMaskGenerator, clip_model: CLIP):
        super(SamPlusCLIP, self).__init__()
        self.mask_generator = mask_generator
        self.clip = clip_model

        self._text_descriptions = None
        self._text_features = None
        self._text_tokens = None

    def forward_sam(self, image: np.ndarray):
        self.mask_generator.predictor.set_image(image)
        masks = self.mask_generator.generate(image)
        return masks

    def forward(self, images: _Tensor, text_descriptions: List[str] = None) -> Tuple[_Tensor]:
        assert images.shape[0] == len(text_descriptions), f'Size mismatch {images.shape} {len(text_descriptions)}'
        image_features = self.mask_generator(images)
        text_features = self.get_text_embedding(text) if text is not None else None

        if self.training:
            return image_features, text_features

        raise NotImplementedError('Infernce not yet implemented!')

    @torch.no_grad()
    def set_text_descriptions(self, text_descriptions: List[str]):
        self._text_tokens = clip.tokenize(text_descriptions).to(self.device)
        self._text_features = normalize(self.clip.forward_text(self._text_tokens))
        self._text_descriptions = text_descriptions

    @property
    def text_descriptions(self):
        return self._text_descriptions

    @property
    def features(self):
        return self.mask_generator.predictor.features

    @property
    def device(self) -> torch.device:
        return self.mask_generator.predictor.model.device

    @staticmethod
    def visualize(image, anns) -> np.ndarray:
        if len(anns) == 0:
            return
        image = np.asarray(image) if not isinstance(image, np.ndarray) else image
        anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        im_mask = np.zeros(image.shape, dtype=np.uint8)
        for i, ann in enumerate(anns, 1):
            m = ann['segmentation'].astype(np.uint8)
            contours, _ = cv.findContours(m * 255, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            color_mask = tuple(np.random.randint(0, 255, 3).tolist())
            cv.drawContours(im_mask, contours, 0, color_mask, -1)
            cv.drawContours(image, contours, 0, color_mask, 3)

            indices = np.where(m != 0)
            centroid = np.column_stack(indices[::-1]).mean(axis=0).astype(np.int32)
            cv.circle(image, tuple(centroid), 3, color_mask, -1)

        image = cv.addWeighted(image, 0.65, im_mask, 0.35, 0.0)
        return image

    def to(self, device):
        self.mask_generator.predictor.model.to(device)
        self.clip.to(device)
        super(SamPlusCLIP, self).to(device)
