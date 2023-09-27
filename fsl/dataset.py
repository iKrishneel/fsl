#!/usr/bin/env python

import os
import time
from io import BytesIO
from typing import Any, Dict, List

import numpy as np
import torch
from igniter.datasets import S3CocoDataset
from igniter.logger import logger
from igniter.registry import dataset_registry, func_registry


class S3CocoDatasetSam(S3CocoDataset):
    def __init__(self, *args, **kwargs):
        super(S3CocoDatasetSam, self).__init__(*args, **kwargs)

        self.im_size = kwargs.get('image_size', None)

        size = len(self.ids)
        self.ids = [iid for iid in self.ids if len(self.coco.getAnnIds(iid)) != 0]

        if len(self.ids) != size:
            logger.info(f'Removing {size - len(self.ids)} empty annotations')

    def __getitem__(self, index: int) -> Dict[str, Any]:
        while True:
            time.sleep(0.1)
            try:
                iid = self.ids[index]
                image, targets = self._load(iid)

                assert len(targets) > 0

                # FIXME: Add targets transform parsing directly from config
                bboxes = [
                    torch.Tensor(self.xywh_to_xyxy(target['bbox']))
                    for target in targets
                    if all(side > 20 for side in target['bbox'][2:])
                ]

                if len(bboxes) == 0:
                    bboxes = [torch.FloatTensor([0, 0, *self.im_size]).reshape(-1, 4)]

                filename = f'perception/sam/coco/{self.root.split(os.sep)[-1]}/features/'
                filename = filename + f'{str(iid).zfill(12)}.pt'

                contents = self.client.get(filename, False)
                buffer = BytesIO(contents)
                sam_feats = torch.load(buffer, map_location=torch.device('cpu'))

                for transform in self.transforms.transforms:
                    image, bboxes = transform(image, bboxes)

                # import matplotlib.pyplot as plt
                # plt.imshow(image); plt.show()
                # import IPython, sys; IPython.embed(); sys.exit()

                break
            except Exception as e:
                logger.warning(f'{e} for iid: {iid} index: {index}')
                index = np.random.choice(np.arange(len(self.ids)))

        return {'image': image, 'sam_feats': sam_feats, 'filename': filename, 'bboxes': bboxes}

    @staticmethod
    def xywh_to_xyxy(bbox: List[float]) -> List[float]:
        x, y, w, h = bbox
        return [x, y, x + w, y + h]


@dataset_registry('coco')
class S3CocoDatasetFSLEpisode(S3CocoDatasetSam):
    def __init__(self, *args, **kwargs):
        super(S3CocoDatasetFSLEpisode, self).__init__(*args, **kwargs)

        # remap ids
        labels = set()
        for iid in self.ids:
            targets = self.coco.loadAnns(self.coco.getAnnIds(iid))
            for target in targets:
                cat_id = target['category_id']
                labels.add(cat_id)

        labels = list(labels)
        self.label_mapping = {labels[i]: i for i in range(len(labels))}
        self.instances_per_batch = kwargs.get('instances_per_batch', 10)

        # import IPython, sys; IPython.embed(); sys.exit()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        while True:
            try:
                iid = self.ids[index]
                image, targets = self._load(iid)

                assert len(targets) > 0

                # FIXME: Add targets transform parsing directly from config
                category_ids = []
                bboxes = []
                for target in targets:
                    # if any(side < 20 for side in target['bbox'][2:]):
                    #     continue
                    category_ids.append(self.label_mapping[target['category_id']])
                    bboxes.append(torch.Tensor(self.xywh_to_xyxy(target['bbox'])))

                assert len(bboxes) > 0, 'Empty bounding boxes'

                for transform in self.transforms.transforms:
                    image, bboxes = transform(image, bboxes)

                break
            except Exception as e:
                logger.warning(f'{e} for iid: {iid} index: {index}')
                index = np.random.choice(np.arange(len(self.ids)))

        if self.instances_per_batch > 0:
            indices = np.random.choice(len(bboxes), self.instances_per_batch, replace=True)
            bboxes = [bboxes[index] for index in indices]
            category_ids = [category_ids[index] for index in indices]
        return {'image': image, 'bboxes': bboxes, 'category_ids': category_ids}


@func_registry('collate_data')
def collate_data(batches: List[Dict[str, Any]]) -> List[Any]:    
    images, targets = [], []
    for batch in batches:
        images.append(batch['image'])
        targets.append({'bboxes': batch['bboxes'], 'category_ids': batch['category_ids']})
    return images, targets
