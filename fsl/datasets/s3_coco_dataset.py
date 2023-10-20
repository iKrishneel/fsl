#!/usr/bin/env python

import os
import time
import io
import contextlib

from typing import Any, Dict, List

import json
from omegaconf import OmegaConf, DictConfig
import numpy as np
import torch
import torchvision

torchvision.disable_beta_transforms_warning()

from torchvision.datapoints import BoundingBoxFormat
from igniter.datasets import S3CocoDataset, S3Dataset
from igniter.logger import logger
from igniter.registry import dataset_registry, func_registry

from fsl.structures import Proposal


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
                buffer = io.BytesIO(contents)
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
                time.sleep(0.1)

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
        return {'image': image, 'bboxes': bboxes, 'category_ids': category_ids, 'image_ids': iid}


@dataset_registry('fs_coco')
class S3CocoDatasetFS(S3CocoDataset):
    def __init__(
        self,
        bucket_name: str,
        root: str,
        json_file: str,
        shot: int = 5,
        filename_signature:str = 'full_box_%sshot_%s_trainval.json',
        **kwargs
    ) -> None:
        split_dir = os.path.join(os.path.dirname(json_file), 'cocosplit2017/seed1')
        assert os.path.isdir(split_dir), f'Directory not found {split_dir}'
        anno_dict = load_json(json_file)

        fileids = {}
        for idx, class_name in enumerate(anno_dict.thing_classes):
            filename = os.path.join(split_dir, filename_signature % (shot, class_name))
            coco_api = load_coco(filename)
            img_ids = sorted(list(coco_api.imgs.keys()))
            imgs = coco_api.loadImgs(img_ids)
            anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
            fileids[idx] = list(zip(imgs, anns))

        dataset_dicts = []
        ann_keys = ['iscrowd', 'bbox', 'category_id', 'segmentation', 'area']
        categories = {}
        for _, fileids_ in fileids.items():
            dicts = []
            for (img_dict, anno_dict_list) in fileids_:
                for anno in anno_dict_list:
                    record = {}
                    record['file_name'] = img_dict['file_name']
                    record['height'] = img_dict['height']
                    record['width'] = img_dict['width']
                    image_id = record['image_id'] = img_dict['id']

                    assert anno['image_id'] == image_id
                    assert anno.get('ignore', 0) == 0

                    obj = {key: anno[key] for key in ann_keys if key in anno}
                    
                    # obj['bbox_mode'] = BoundingBoxFormat.XYWH.name
                    category_id = anno_dict.thing_dataset_id_to_contiguous_id[str(obj['category_id'])]

                    obj['category_id'] = category_id
                    categories[category_id] = anno_dict.thing_classes[category_id]

                    record['annotations'] = [obj]
                    dicts.append(record)
            if len(dicts) > int(shot):
                dicts = np.random.choice(dicts, int(shot), replace=False)
            dataset_dicts.extend(dicts)

        categories = [{'id': key, 'name': val} for key, val in categories.items()]
            
        images, annotations = [], []
        anno_id = 1
        for dataset_dict in dataset_dicts:
            images.append(
                {
                    'id': dataset_dict['image_id'],
                    'file_name': dataset_dict['file_name'],
                    'height': dataset_dict['height'],
                    'width': dataset_dict['width']}
            )
            for anno in dataset_dict['annotations']:
                anno['id'] = anno_id
                anno['image_id'] = dataset_dict['image_id']
                annotations.append(anno)
                anno_id += 1

        dataset = {'images': images, 'annotations': annotations, 'categories': categories}

        anno_fn = '/tmp/fs_coco_anno.json'
        with open(anno_fn, 'w') as json_file:
            json.dump(dataset, json_file)
        super(S3CocoDatasetFS, self).__init__(bucket_name, root, anno_fn, **kwargs)
        self.apply_transforms = False

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image, targets = super().__getitem__(index)
        bboxes = [torch.FloatTensor(target['bbox']) for target in targets]
        category_ids = [target['category_id'] for target in targets]
        image_ids = [target['image_id'] for target in targets]

        assert len(bboxes) > 0, 'Empty bounding boxes'

        for transform in self.transforms.transforms:
            image, bboxes = transform(image, bboxes)

        return {'image': image, 'bboxes': bboxes, 'category_ids': category_ids, 'image_ids': image_ids}


def load_coco(json_file):
    from pycocotools.coco import COCO

    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    return coco_api
    
def load_json(filename: str) -> DictConfig:
    assert os.path.isfile(filename), f'Invalid filename {filename}'
    return OmegaConf.load(filename)


@func_registry('collate_data')
def collate_data(batches: List[Dict[str, Any]]) -> List[Any]:
    images, targets = [], []
    for batch in batches:
        images.append(batch['image'])
        targets.append(
            {'bboxes': batch['bboxes'], 'category_ids': batch['category_ids'], 'image_ids': batch['image_ids']}
        )
    return images, targets


if __name__ == '__main__':
    bucket_name = 'sr-shokunin'
    root = 'perception/datasets/coco/train2017/'
    json_file = '/root/krishneel/Downloads/coco/fs_coco_trainval_novel_5shot.json'

    s = S3CocoDatasetFS(bucket_name, root, json_file=json_file)
    x = s[0]
    import IPython; IPython.embed()
