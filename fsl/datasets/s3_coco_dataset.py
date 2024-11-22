#!/usr/bin/env python

import contextlib
import io
import json
import os
import re
import time
from typing import Any, Dict, List

import numpy as np
import torch
import torchvision
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision.transforms.v2 import functional
from torchvision.datasets import CocoDetection

torchvision.disable_beta_transforms_warning()

from igniter.datasets import S3CocoDataset, S3Dataset
from igniter.logger import logger
from igniter.registry import dataset_registry, func_registry
from pycocotools import mask as mask_utils

from fsl.utils import version

if version.minor_version(torchvision.__version__) <= 15:
    from torchvision.datapoints import BoundingBoxFormat
else:
    from torchvision.tv_tensors import BoundingBoxFormat


class S3CocoDatasetSam(torch.utils.data.Dataset):
    def __init__(self, root: str, anno_fn: str, transforms: Any = None, bucket_name: str = None, **kwargs):
        if os.path.isdir(root) and os.path.isfile(anno_fn):
            self.coco = CocoDetection(root=root, annFile=anno_fn, transform=None)
        elif bucket_name is not None:
            self.coco = S3CocoDataset(bucket_name=bucket_name, root=root, anno_fn=anno_fn)
        else:
            raise TypeError

        self.transforms = transforms
        self.im_size = kwargs.get('image_size', None)

        size = len(self.coco.ids)
        self.ids = [iid for iid in self.coco.ids if len(self.coco.coco.getAnnIds(iid)) != 0]

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

    def __len__(self) -> int:
        return len(self.coco)
    
    @staticmethod
    def xywh_to_xyxy(bbox: List[float]) -> List[float]:
        x, y, w, h = bbox
        return [x, y, x + w, y + h]


@dataset_registry('coco')
class S3CocoDatasetFSLEpisode(S3CocoDatasetSam):
    def __init__(self, *args, **kwargs):
        super(S3CocoDatasetFSLEpisode, self).__init__(*args, **kwargs)
        self.min_bbox_size = kwargs.get('min_bbox_size', 10)
        self.use_mask = kwargs.get('use_mask', False)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        label_name_mapping = {v['id']: v['name'] for v in self.coco.coco.dataset['categories']}
        target_mask_decoder = (
            lambda target: mask_utils.decode(target['segmentation'])
            if isinstance(target['segmentation'], dict)
            else self.coco.coco.annToMask(target)
        )

        while True:
            try:
                iid = self.ids[index]
                if isinstance(self.coco, CocoDetection):
                    image, targets = self.coco[index]
                else:
                    image, targets = self._load(iid)

                masks = (
                    [
                        torch.as_tensor(target_mask_decoder(target))
                        for target in targets
                        if np.all(np.array(target['bbox'][2:]) > self.min_bbox_size)
                    ]
                    if self.use_mask
                    else None
                )

                assert len(targets) > 0, 'No target(s) found!'

                bboxes = [
                    torch.Tensor(target['bbox'])
                    for target in targets
                    if np.all(np.array(target['bbox'][2:]) > self.min_bbox_size)
                ]
                assert len(bboxes) > 0, 'No bounding box found!'

                if masks:
                    assert len(masks) == len(bboxes), 'bboxes and masks are not same lenght'
                    masks = torch.stack(masks)

                break
            except Exception as e:
                index = int(np.random.choice(np.arange(len(self.ids))))
                if not isinstance(e, AssertionError):
                    logger.warning(f'{e} for iid: {iid} index: {index}')

        # FIXME: Add targets transform parsing directly from config
        category_ids = torch.IntTensor(
            [target['category_id'] for target in targets if np.all(np.array(target['bbox'][2:]) > self.min_bbox_size)]
        )
        # bboxes = [torch.Tensor(target['bbox']) for target in targets if np.all(np.array(target['bbox'][2:]) > 10)]
        category_names = [
            label_name_mapping[target['category_id']]
            for target in targets
            if np.all(np.array(target['bbox'][2:]) > self.min_bbox_size)
        ]

        assert len(bboxes) > 0, 'Empty bounding boxes'

        if image.mode != 'RBG':
            image = image.convert('RGB')

        image = functional.pil_to_tensor(image)

        data = {
            'image': image,
            'bboxes': bboxes,
            'category_ids': category_ids,
            'category_names': category_names,
            'image_ids': [iid],
        }

        if self.use_mask:
            data['masks'] = masks

        data = self.transforms(data)
        return data


@dataset_registry('coco_detection')
class S3CocoDatasetForDetection(S3CocoDataset):
    def __init__(self, bucket_name: str, root: str, anno_fn: str, **kwargs):
        super(S3CocoDatasetForDetection, self).__init__(bucket_name, root, anno_fn, **kwargs)
        self.apply_transforms = False

    def __getitem__(self, index: int):
        image, targets = super().__getitem__(index)

        image = image.convert('RGB') if image.mode != 'RBG' else image
        image = functional.pil_to_tensor(image)

        bboxes = torch.as_tensor([target['bbox'] for target in targets])
        category_ids = [target['category_id'] for target in targets]
        iids = [target['image_id'] for target in targets]

        data = {
            'image': image,
            'bboxes': bboxes,
            'category_ids': category_ids,
            'image_ids': iids,
        }

        data = self.transforms(data)
        return data


@dataset_registry('fs_coco')
class S3CocoDatasetFS(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        json_file: str,
        label_map_file: str,
        bucket_name: str = None,            
        filename_signature: str = 'full_box_%sshot_%s_trainval.json',
        transforms: Any = None,
        **kwargs,
    ) -> None:
        self.transforms = transforms
        json_filename = os.path.splitext(os.path.basename(json_file))[0]

        if 'shot' in json_filename:
            match = re.search(r'\d+shot', json_filename)
            assert match
            shot = int(json_filename[match.start() : match.end()].replace('shot', ''))
        else:
            shot = kwargs.get('shot', None)
            assert shot

        split_dir = os.path.join(os.path.dirname(json_file), 'cocosplit2017/seed1')
        assert os.path.isdir(split_dir), f'Directory not found {split_dir}'
        anno_dict = load_json(json_file)

        coco_label_map = load_json(label_map_file)['all_classes']
        anno_dict.thing_classes = list(coco_label_map.values())
        anno_dict.thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(coco_label_map.keys())}
        # anno_dict.thing_classes = anno_dict.base_classes
        # anno_dict.thing_dataset_id_to_contiguous_id = anno_dict.base_dataset_id_to_contiguous_id

        dataset = self._prepare_data_catalog(anno_dict, shot, split_dir, filename_signature)

        anno_fn = '/tmp/fs_coco_anno.json'
        with open(anno_fn, 'w') as json_file:
            json.dump(dataset, json_file)

        if bucket_name is not None:
            self.coco = S3CocoDatasetFS(bucket_name, root, anno_fn, **kwargs)
        else:
            self.coco = CocoDetection(root=root, annFile=anno_fn, transform=None)

        # self.apply_transforms = False
        # import IPython, sys; IPython.embed(); sys.exit()

    def _prepare_data_catalog(self, anno_dict, shot: int, split_dir: str, filename_signature: str):
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
            for img_dict, anno_dict_list in fileids_:
                for anno in anno_dict_list:
                    record = {}
                    record['file_name'] = img_dict['file_name']
                    record['height'] = img_dict['height']
                    record['width'] = img_dict['width']
                    image_id = record['image_id'] = img_dict['id']

                    assert anno['image_id'] == image_id
                    assert anno.get('ignore', 0) == 0

                    obj = {key: anno[key] for key in ann_keys if key in anno}
                    category_id = anno_dict.thing_dataset_id_to_contiguous_id[str(obj['category_id'])]

                    obj['category_name'] = anno_dict.thing_classes[category_id]
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
                    'width': dataset_dict['width'],
                }
            )
            for anno in dataset_dict['annotations']:
                anno['id'] = anno_id
                anno['image_id'] = dataset_dict['image_id']
                annotations.append(anno)
                anno_id += 1

        dataset = {'images': images, 'annotations': annotations, 'categories': categories}
        return dataset

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image, targets = self.coco[index]
        bboxes = [torch.FloatTensor(target['bbox']) for target in targets]
        category_ids = [target['category_id'] for target in targets]
        category_names = [target['category_name'] for target in targets]
        image_ids = [target['image_id'] for target in targets]

        assert len(bboxes) > 0, 'Empty bounding boxes'

        masks = np.array(
            [
                mask_utils.decode(mask_utils.frPyObjects(target['segmentation'], *image.size))[..., 0]
                for target in targets
            ]
        )

        masks = torch.as_tensor(masks)        
        image = functional.pil_to_tensor(image)

        data = {
            'image': image,
            'masks': masks,
            'bboxes': bboxes,
            'category_ids': category_ids,
            'image_ids': image_ids,
            'category_names': category_names,
        }
    
        if self.transforms is not None:
            data = self.transforms(data)

        return data

    def __len__(self) -> int:
        return len(self.coco)


def load_coco(json_file: str):
    from pycocotools.coco import COCO

    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    return coco_api


def load_json(filename: str) -> DictConfig:
    assert os.path.isfile(filename), f'Invalid filename {filename}'
    return OmegaConf.load(filename)


@func_registry
def collate_data(batches: List[Dict[str, Any]]) -> List[Any]:
    images, targets = collate_data_instances(batches)
    targets = [target['gt_proposal'] for target in targets]
    return images, targets


@func_registry
def collate_data_instances(batches: List[Dict[str, Any]]) -> List[Any]:
    from fsl.structures import Instances

    images, targets = [], []
    for batch in batches:
        image = batch.pop('image')
        images.append(image)
        instances = Instances(
            bboxes=torch.stack(batch['bboxes']),
            class_ids=batch['category_ids'],
            labels=batch['category_names' if 'category_names' in batch else 'category_ids'],
            bbox_fmt=BoundingBoxFormat.XYXY,
            image_id=batch['image_ids'][0],
            image_width=image.shape[-1],
            image_height=image.shape[-2],
            masks=batch['masks'] if 'masks' in batch else None,
        )
        targets.append({'gt_proposal': instances})
    return images, targets


if __name__ == '__main__':
    bucket_name = 'sr-shokunin'
    root = 'perception/datasets/coco/train2017/'
    json_file = '/root/krishneel/Downloads/coco/fs_coco_trainval_novel_5shot.json'

    s = S3CocoDatasetFS(bucket_name, root, json_file=json_file)
    x = s[0]
    import IPython

    IPython.embed()
