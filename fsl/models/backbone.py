# /usr/bin/env python

from abc import ABC, abstractmethod
from typing import Any, Dict, Type

import torch
import torch.nn as nn
from igniter.registry import model_registry

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
    def __init__(self, backbone):
        super(DinoV2Backbone, self).__init__(backbone)

    def forward(self, image: _Tensor) -> _Tensor:
        im_dtype = image.dtype
        image = image.to(self.device).to(self.dtype)
        outputs = self.backbone.get_intermediate_layers(image, n=[self.backbone.n_blocks - 1], reshape=True)
        return outputs[0].to(im_dtype) if self.training else outputs[0]

    @property
    def downsize(self) -> int:
        return self.backbone.patch_size

    @property
    def device(self):
        return self.backbone.patch_embed.proj.weight.device

    @property
    def dtype(self):
        return self.backbone.patch_embed.proj.weight.dtype

    @classmethod
    def build(cls, model_name: str = 'dinov2_vitb14', frozen: bool = True):
        backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        if frozen:
            for param in backbone.parameters():
                param.requires_grad_(False)
        return cls(backbone)
