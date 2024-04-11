#!/usr/bin/env python

from typing import Any, Dict

import torch
import torch.nn as nn


class SineCosinePositionalEncoding(nn.Module):
    def __init__(self, dim: int, dropout: int = 0, max_len: int = 5000) -> None:
        super(SineCosinePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(torch.log(torch.FloatTensor([10000.0])) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class VisualPromptEncoder(nn.Module):
    def __init__(n_tokens: int, embed_dim: int = 4, **kwargs: Dict[str, Any]) -> None:
        super(VisualPromptEncoder, self).__init__()
        self.pos_embed = SineCosinePositionalEncoding(embed_dim, dropout=kwargs.get('dropout', 0.1))
        self.proj = nn.Linear(embed_dim, n_tokens)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_tokens))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_embed(x)
        x = self.proj(x)

        cls_token = self.cls_token.expand(x.size(0), x.size(1), -1)
        # x = torch.cat((cls_token, x), dim=)
        return x


def build_visual_prompt_encoder():
    pass
