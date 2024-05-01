# /usr/bin/env python

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type

import torch
import torch.nn as nn
from igniter.registry import model_registry
from omegaconf import ListConfig

_Tensor = Type[torch.Tensor]
_Module = Type[nn.Module]


class BackboneWrapper(nn.Module, ABC):
    def __init__(self, backbone: _Module):
        super(BackboneWrapper, self).__init__()
        self.backbone = backbone

    @abstractmethod
    def downsize(self) -> int:
        raise NotImplementedError

    def device(self):
        raise NotImplementedError

    def dtype(self):
        raise NotImplementedError

    @classmethod
    def build(cls, *args, **kwargs: Dict[str, Any]) -> 'BackboneWrapper':
        raise NotImplementedError


class DinoV2Backbone(BackboneWrapper):
    def __init__(self, backbone, feat_layers: List[int] = None) -> None:
        super(DinoV2Backbone, self).__init__(backbone)

        if feat_layers is not None:
            assert isinstance(feat_layers, (tuple, list, ListConfig)), f'Expects {list} but got {type(feat_layers)}'

        # n = len(backbone.blocks) - 1
        # feat_layers = [*feat_layers, n] if n not in feat_layers else feat_layers
        self.feat_layers = feat_layers or [self.backbone.n_blocks - 1]

    @torch.inference_mode()
    def forward(self, image: _Tensor, norm: bool = True, reshape: bool = True) -> List[_Tensor]:
        im_dtype = image.dtype
        image = image.to(self.device).to(self.dtype)
        outputs = self.backbone.get_intermediate_layers(image, n=self.feat_layers, reshape=reshape, norm=norm)
        outputs = [output.to(im_dtype) for output in outputs] if self.training else outputs
        return outputs

    @property
    def downsize(self) -> int:
        return self.backbone.patch_size

    @property
    def device(self) -> torch.device:
        return self.backbone.patch_embed.proj.weight.device

    @property
    def dtype(self):
        return self.backbone.patch_embed.proj.weight.dtype

    @classmethod
    def build(
        cls,
        model_name: str = 'dinov2_vitb14',
        frozen: bool = True,
        feat_layers: List[int] = None,
    ) -> 'DinoV2Backbone':
        backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        if frozen:
            for param in backbone.parameters():
                param.requires_grad_(False)
        return cls(backbone, feat_layers)
