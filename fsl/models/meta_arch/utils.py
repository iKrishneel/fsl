#!/usr/bin/env python

from typing import Any, Dict

import torch
import torch.nn as nn


class SineCosinePositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, dropout: int = 0, max_len: int = 5000) -> None:
        super(SineCosinePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.FloatTensor([10000.0])) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

    @property
    def embed_dim(self) -> int:
        return self.pe.shape[-1]


class VisualPromptEncoder(nn.Module):
    def __init__(self, pos_encoding: nn.Module, n_tokens: int, **kwargs: Dict[str, Any]) -> None:
        super(VisualPromptEncoder, self).__init__()
        self.position_encoding = pos_encoding
        embed_dim = kwargs.get('embed_dim') or pos_encoding.embed_dim
        self.proj = nn.Linear(embed_dim, n_tokens)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_tokens))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.position_encoding(x)
        x = self.proj(x)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        return x


def build_position_encoder(encoder_type: str, embed_dim: int, **kwargs: Dict[str, Any]) -> nn.Module:
    enc_cls_lut = {
        'sine_cosine': SineCosinePositionalEncoding
    }
    assert encoder_type in enc_cls_lut, f'{encoder_type} not in {enc_cls_lut.keys()}'
    return enc_cls_lut[encoder_type](embed_dim, **kwargs)


def build_visual_prompt_encoder(
    position_encoding_args: Dict[str, Any] = {'encoder_type': 'sine_cosine', 'embed_dim': 4},
    prompt_encoder_args: Dict[str, Any] = {'n_tokens': 256},
) -> VisualPromptEncoder:
    pos_encoding = build_position_encoder(**position_encoding_args)
    prompt_encoder = VisualPromptEncoder(pos_encoding, **prompt_encoder_args)
    return prompt_encoder
