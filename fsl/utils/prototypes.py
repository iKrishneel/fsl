#!/usr/bin/env python
import os.path as osp
import pickle
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Type, Union

import numpy as np
import torch

# from fsl.structures import Instances

_Tensor = Type[torch.Tensor]


@dataclass
class ProtoTypes(object):
    embeddings: _Tensor
    labels: List[str]
    instances: 'Instances' = None

    def __post_init__(self):
        if isinstance(self.embeddings, np.ndarray):
            self.embeddings = torch.as_tensor(self.embeddings)
        assert isinstance(self.embeddings, torch.Tensor)

        assert self.embeddings.shape[0] == len(
            self.labels
        ), f'Size mismatch {self.embeddings.shape} != {len(self.labels)}'

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {'embedding': self.embeddings[index], 'label': self.labels[index]}

    def __add__(self, other: 'ProtoTypes') -> 'ProtoTypes':
        if not isinstance(other, ProtoTypes):
            raise TypeError(f'Invalid object type {other}, expects {type(self)}')
        embeddings = torch.cat([self.embeddings, other.embeddings.to(self.embeddings.device)], dim=0)
        labels = self.labels + other.labels
        return ProtoTypes(embeddings, labels)

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __copy__(self) -> 'ProtoTypes':
        return ProtoTypes(self.embeddings.clone(), deepcopy(self.labels))

    def check(self, labels: List[str], insert_missing: bool = True) -> 'ProtoTypes':
        pt = deepcopy(self.to('cpu'))
        for label in labels:
            if label in pt.labels or not insert_missing:
                continue
            pt.labels.append(label)
            pt.embeddings = torch.cat([pt.embeddings, torch.zeros(1, *pt.embeddings.shape[1:])], dim=0)
        return pt

    def to(self, device: str) -> 'ProtoTypes':
        self.embeddings = self.embeddings.to(device)
        return self

    def save(self, filename: str) -> None:
        import os

        os.makedirs(osp.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as pfile:
            pickle.dump(self.to('cpu'), pfile)

    @property
    def normalized_embedding(self) -> _Tensor:
        embeddings = self.embeddings.mean(dim=1) if len(self.embeddings.shape) == 3 else self.embeddings
        return torch.nn.functional.normalize(embeddings, dim=-1)

    @classmethod
    def load(
        cls,
        filenames: Union[List[str], str],
        keys: List[str] = ['embeddings', 'labels'],
        loader: Callable = pickle.load,
    ) -> Union[List['ProtoTypes'], 'ProtoTypes']:
        is_list = isinstance(filenames, list)
        filenames = [filenames] if not is_list else filenames
        prototypes = []
        for filename in filenames:
            assert osp.isfile(filename)
            with open(filename, 'rb') as pfile:
                data = loader(pfile)

            if isinstance(data, ProtoTypes):
                prototypes.append(data)
                continue

            if isinstance(data, torch.Tensor):
                data = data.flatten(0, 1) if len(data.shape) == 3 else data
                args = [data, [-1] * data.shape[0]]
            elif isinstance(data, dict):
                args = [data[key] for key in keys]
            else:
                raise TypeError(f'Unsupported data type {type(data)}')
            prototypes.append(cls(*args))
        return prototypes if is_list else prototypes[0]
