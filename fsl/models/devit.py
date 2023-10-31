#!/usr/bin/env python

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from igniter.registry import model_registry
from PIL import Image
from torchvision.ops import RoIAlign
from torchvision.ops.boxes import box_area, box_iou

from fsl.datasets import utils
from fsl.structures import Instances
from fsl.utils.prototypes import ProtoTypes

_Image = Type[Image.Image]
_Tensor = Type[torch.Tensor]


class PropagateNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        has_input_mask: bool = False,
        num_layers: int = 3,
        dropout: float = 0.5,
        mask_temperature: float = 0.1,
    ) -> None:
        super(PropagateNet, self).__init__()
        self.has_input_mask = has_input_mask
        start_mask_dim = 1 if has_input_mask else 0
        self.mask_temperature = mask_temperature
        self.main_layers = nn.ModuleList()
        self.mask_layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        self.num_layers = num_layers

        for i in range(num_layers):
            channels = input_dim if i == 0 else hidden_dim
            self.main_layers.append(
                nn.Sequential(
                    nn.Conv2d(channels + start_mask_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(),
                )
            )
            self.mask_layers.append(nn.Conv2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1))
            start_mask_dim += 1

        # more proj for regression
        self.class_proj = nn.Linear(hidden_dim, 1)

    def forward(self, embedding: _Tensor, mask: _Tensor = None):
        masks = []
        if self.has_input_mask:
            assert mask is not None
            masks.append(mask.float())

        outputs = []
        for i in range(self.num_layers):
            if len(masks) > 0:
                embedding = torch.cat([embedding] + masks, dim=1)
            embedding = self.main_layers[i](embedding)

            mask_logits = self.mask_layers[i](embedding) / self.mask_temperature
            mask_weights = mask_logits.sigmoid()
            masks.insert(0, mask_weights)

            # classification
            mask_weights = mask_weights / mask_weights.sum(dim=[2, 3], keepdim=True)
            latent = (embedding * mask_weights).sum(dim=[2, 3])

            latent = self.dropout(latent)
            outputs.append({'class': self.class_proj(latent)})

        results = [o['class'] for o in outputs]
        return results if self.training else results[-1]


class DeVit(nn.Module):
    def __init__(
        self,
        mask_generator,
        proposal_matcher,
        roi_pool_size: int = 16,
        fg_prototypes: ProtoTypes = None,
        bg_prototypes: ProtoTypes = None,
        all_cids: List[str] = None,
        seen_cids: List[str] = None,
    ):
        super(DeVit, self).__init__()
        self.mask_generator = mask_generator
        self.proposal_matcher = proposal_matcher
        self.roi_pool = RoIAlign(roi_pool_size, spatial_scale=1 / mask_generator.downsize, sampling_ratio=-1)

        # TODO: Configure this
        self.batch_size_per_image = 128
        self.pos_ratio = 0.25
        self.num_sample_class = 10
        self.t_len = 128  # 256
        self.temb = 128
        self.t_pos_emb = 128
        self.t_bg_emb = 128
        self.cls_temp = 0.1
        self.hidden_dim = 256
        self.num_cls_layers = 3

        cls_input_dim = self.temb * 2

        if fg_prototypes:
            self._setup_prototypes(fg_prototypes, all_cids, seen_cids, is_bg=False)
        if bg_prototypes:
            self._setup_prototypes(bg_prototypes, all_cids, seen_cids, is_bg=True)
            self.bg_cls_weight = 0.2
            cls_input_dim += self.t_bg_emb
        else:
            self.bg_tokens = None
            self.fc_back_class = None
            self.bg_cnn = None
            self.bg_cls_weight = 0.0

        self.fc_other_class = nn.Linear(self.t_len, self.temb)
        self.fc_intra_class = nn.Linear(self.t_pos_emb, self.temb)
        self.per_cls_cnn = PropagateNet(cls_input_dim, self.hidden_dim, num_layers=self.num_cls_layers)

    def _setup_prototypes(
        self, prototypes: ProtoTypes, all_cids: List[str] = None, seen_cids: List[str] = None, is_bg: bool = False
    ) -> None:
        if is_bg:
            self.register_buffer('bg_tokens', prototypes.embeddings)
            self.fc_bg_class = nn.Linear(self.t_len, self.temb)
            self.fc_back_class = nn.Linear(len(self.bg_tokens), self.t_bg_emb)
            bg_input_dim = self.temb + self.t_bg_emb
            self.bg_cnn = PropagateNet(bg_input_dim, self.hidden_dim, num_layers=self.num_cls_layers)
        else:
            pt = prototypes.check(all_cids)
            train_class_order = [pt.labels.index(c) for c in seen_cids]
            test_class_order = [pt.labels.index(c) for c in all_cids]
            assert -1 not in train_class_order and -1 not in test_class_order

            self.register_buffer('train_class_weight', pt.normalized_embedding[torch.as_tensor(train_class_order)])
            self.register_buffer('test_class_weight', pt.normalized_embedding[torch.as_tensor(test_class_order)])

    def forward(self, images: List[_Tensor], targets: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.training:
            return self.forward_once(images, targets)

        num_classes = len(self.train_class_weight)
        assert targets is not None and len(targets) == len(images)

        # proposals
        img_hw = images[0].shape[1:] if isinstance(images[0], torch.Tensor) else images[0].size[::-1]
        gt_instances = [target['gt_proposal'] for target in targets]
        gt_bboxes = [gt_proposal.to_tensor().bboxes for gt_proposal in gt_instances]
        noisy_proposals = utils.prepare_noisy_boxes(gt_bboxes, img_hw)
        boxes = [torch.cat([gt_bboxes[i], noisy_proposals[i]]).to(self.device) for i in range(len(targets))]

        # embedding of the images
        images = torch.stack(images).to(self.device) if isinstance(images[0], torch.Tensor) else images
        features = self.mask_generator(images)

        class_labels, matched_gt_boxes, resampled_proposals = [], [], []
        num_bg_samples, num_fg_samples, gt_masks = [], [], []
        for i, (proposals_per_image, targets_per_image) in enumerate(zip(boxes, gt_instances)):
            targets_per_image = targets_per_image.to_tensor(self.device)

            match_quality_matrix = box_iou(targets_per_image.bboxes, proposals_per_image)
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)

            class_labels_i = targets_per_image.class_ids[matched_idxs]

            if len(class_labels_i) == 0:
                # no annotation on this image
                assert torch.all(matched_labels == 0)
                class_labels_i = torch.zeros_like(matched_idxs)

            class_labels_i[matched_labels == 0] = num_classes
            class_labels_i[matched_labels == -1] = -1

            positive = ((class_labels_i != -1) & (class_labels_i != num_classes)).nonzero().flatten()
            negative = (class_labels_i == num_classes).nonzero().flatten()

            batch_size_per_image = self.batch_size_per_image  # 512
            num_pos = int(batch_size_per_image * self.pos_ratio)
            # protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)
            num_neg = batch_size_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            perm1 = torch.randperm(positive.numel())[:num_pos]
            perm2 = torch.randperm(negative.numel())[:num_neg]
            pos_idx = positive[perm1]
            neg_idx = negative[perm2]
            sampled_idxs = torch.cat([pos_idx, neg_idx], dim=0)

            proposals_per_image = proposals_per_image[sampled_idxs]
            class_labels_i = class_labels_i[sampled_idxs]

            resampled_proposals.append(proposals_per_image)
            class_labels.append(class_labels_i)

            num_bg_samples.append((class_labels_i == num_classes).sum().item())
            num_fg_samples.append(class_labels_i.numel() - num_bg_samples[-1])

        class_labels = torch.cat(class_labels)

        rois = []
        for bid, box in enumerate(resampled_proposals):
            batch_index = torch.full((len(box), 1), fill_value=float(bid)).to(self.device)
            rois.append(torch.cat([batch_index, box.to(self.device)], dim=1))
        rois = torch.cat(rois)

        roi_features = self.roi_pool(features, rois)  # N, C, k, k

        # sample topk classes
        class_topk = self.num_sample_class if self.num_sample_class > 0 else num_classes
        sample_class_enabled = class_topk > 0
        num_active_classes, class_indices = num_classes, None

        if sample_class_enabled:
            num_active_classes = class_topk
            init_scores = nn.functional.normalize(roi_features.flatten(2).mean(2), dim=1) @ self.train_class_weight.T
            topk_class_indices = torch.topk(init_scores, class_topk, dim=1).indices

            class_indices = []
            for i in range(roi_features.shape[0]):
                curr_label = class_labels[i].item()
                topk_class_indices_i = topk_class_indices[i].cpu()
                if curr_label in topk_class_indices_i or curr_label == num_classes:
                    curr_indices = topk_class_indices_i
                else:
                    curr_indices = torch.cat([torch.as_tensor([curr_label]), topk_class_indices_i[:-1]])
                class_indices.append(curr_indices)
            class_indices = torch.stack(class_indices).to(self.device)
            class_indices = torch.sort(class_indices, dim=1).values

        logits = self.get_logits(
            self.train_class_weight,
            roi_features,
            class_indices,
            class_topk,
            sample_class_enabled,
            num_classes,
            num_active_classes,
        )

        # loss
        class_labels = class_labels.long().to(self.device)
        if sample_class_enabled:
            class_labels[class_labels != num_classes] = (class_indices == class_labels.view(-1, 1)).nonzero()[:, 1]
            class_labels[class_labels == num_classes] = num_active_classes

        if self.bg_tokens is not None and self.fc_back_class is not None:
            indices = torch.where(class_labels != num_active_classes)
            class_labels = class_labels[indices]
            logits = [logit[indices] for logit in logits]

        loss_dict = {}
        if isinstance(logits, list):
            for i, logit in enumerate(logits):
                loss_dict[f'focal_loss_{i}'] = self.focal_loss(
                    logit, class_labels, num_classes=num_active_classes, bg_weight=self.bg_cls_weight
                )
        else:
            loss_dict['focal_loss'] = self.focal_loss(
                logits, class_labels, num_classes=num_active_classes, bg_weight=self.bg_cls_weight
            )

        import IPython, sys

        IPython.embed(header="Forward")
        sys.exit()
        return loss_dict

    @torch.no_grad()
    def forward_once(self, images: List[_Tensor], targets: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        if targets and 'gt_proposal' in targets[0]:
            proposals = [target['gt_proposal'] for target in targets]
        else:
            proposals = self.get_proposals(images)

        assert len(proposals) == 1

        # class_weights = self.test_class_weight
        num_classes = len(self.test_class_weight)

        bboxes = torch.cat([proposal.to_tensor().bboxes.to(self.device) for proposal in proposals])
        rois = torch.cat([torch.full((len(bboxes), 1), fill_value=0).to(self.device), bboxes], dim=1)

        images = torch.stack(images).to(self.device)
        features = self.mask_generator(images)

        roi_features = self.roi_pool(features, rois)  # N, C, k, k

        # sample topk classes
        class_topk = self.num_sample_class if self.num_sample_class > 0 else num_classes
        sample_class_enabled = class_topk > 0
        num_active_classes, class_indices = num_classes, None

        if sample_class_enabled:
            num_active_classes = class_topk
            init_scores = nn.functional.normalize(roi_features.flatten(2).mean(2), dim=1) @ self.test_class_weight.T
            topk_class_indices = torch.topk(init_scores, class_topk, dim=1).indices
            class_indices = torch.sort(topk_class_indices, dim=1).values

        logits = self.get_logits(
            self.test_class_weight,
            roi_features,
            class_indices,
            class_topk,
            sample_class_enabled,
            num_classes,
            num_active_classes,
        )

        scores = torch.softmax(logits, dim=-1)
        output = {'scores': scores[:, :-1]}

        if sample_class_enabled:
            full_scores = torch.zeros(len(scores), num_classes + 1, device=self.device)
            full_scores.scatter_(1, class_indices, scores)

            scores = full_scores
            full_scores[:, -1] = scores[:, -1]
            output['scores'] = full_scores[:, :-1]

        # import IPython, sys; IPython.embed(header="Forward Once"); sys.exit()
        return output, {'loss': 0.0}  # loss is not yet computed

    def get_logits(
        self,
        class_weights,
        roi_features,
        class_indices,
        class_topk,
        sample_class_enabled,
        num_classes,
        num_active_classes,
    ) -> Union[_Tensor, List[_Tensor]]:
        roi_features = roi_features.flatten(2) if len(roi_features.shape) == 4 else roi_features
        feats = roi_features.transpose(-2, -1) @ class_weights.T
        bs, spatial_size = roi_features.shape[0], roi_features.shape[-1]

        other_classes = []
        if sample_class_enabled:
            indexes = torch.arange(0, num_classes, device=self.device)[None, None, :].repeat(bs, spatial_size, 1)
            for i in range(class_topk):
                cmask = indexes != class_indices[:, i].view(-1, 1, 1)
                _ = torch.gather(
                    feats, 2, indexes[cmask].view(bs, spatial_size, num_classes - 1)
                )  # N x spatial x classes-1
                other_classes.append(_[:, :, None, :])
        else:
            for c in range(num_classes):  # TODO: change to classes sampling during training for LVIS type datasets
                cmask = torch.ones(num_classes, device=self.device, dtype=torch.bool)
                cmask[c] = False
                _ = feats[:, :, cmask]  # # N x spatial x classes-1
                other_classes.append(_[:, :, None, :])

        other_classes = torch.cat(other_classes, dim=2)  # N x spatial x classes x classes-1
        other_classes = other_classes.permute(0, 2, 1, 3)  # N x classes x spatial x classes-1
        other_classes = other_classes.flatten(0, 1)  # (Nxclasses) x spatial x classes-1
        other_classes, _ = torch.sort(other_classes, dim=-1)
        other_classes = self.interpolate(other_classes, self.t_len, mode='linear')  # (Nxclasses) x spatial x T
        other_classes = self.fc_other_class(other_classes)  # (Nxclasses) x spatial x emb
        other_classes = other_classes.permute(0, 2, 1)  # (Nxclasses) x emb x spatial
        # (Nxclasses) x emb x S x S
        roi_pool_size = [self.roi_pool.output_size] * 2
        inter_dist_emb = other_classes.reshape(bs * num_active_classes, -1, *roi_pool_size)

        intra_feats = (
            torch.gather(feats, 2, class_indices[:, None, :].repeat(1, spatial_size, 1))
            if sample_class_enabled
            else feats
        )
        intra_dist_emb = self.distance_embed(intra_feats.flatten(0, 1), num_pos_feats=self.t_pos_emb)
        intra_dist_emb = self.fc_intra_class(intra_dist_emb)
        intra_dist_emb = intra_dist_emb.reshape(bs, spatial_size, num_active_classes, -1)

        # (Nxclasses) x emb x S x S
        intra_dist_emb = (
            intra_dist_emb.permute(0, 2, 3, 1)
            .flatten(0, 1)
            .reshape(bs * num_active_classes, -1, *[self.roi_pool.output_size] * 2)
        )

        # N x 1
        # feats: N x spatial x class
        cls_dist_feats = self.interpolate(torch.sort(feats, dim=2).values, self.t_len, mode='linear')  # N x spatial x T

        if self.bg_tokens is not None and self.fc_back_class is not None:
            # N x spatial x back
            bg_feats = roi_features.transpose(-2, -1) @ self.bg_tokens.T
            bg_dist_emb = self.fc_back_class(bg_feats)  # N x spatial x emb
            bg_dist_emb = bg_dist_emb.permute(0, 2, 1).reshape(bs, -1, *roi_pool_size)
            # N x emb x S x S
            bg_dist_emb_c = bg_dist_emb[:, None, :, :, :].expand(-1, num_active_classes, -1, -1, -1).flatten(0, 1)
            # (Nxclasses) x emb x S x S

            # (Nxclasses) x EMB x S x S
            per_cls_input = torch.cat([intra_dist_emb, inter_dist_emb, bg_dist_emb_c], dim=1)

            bg_cls_dist_emb = self.fc_bg_class(cls_dist_feats)  # N x spatial x emb
            bg_cls_dist_emb = bg_cls_dist_emb.permute(0, 2, 1).reshape(bs, -1, *roi_pool_size)
            bg_logits = self.bg_cnn(torch.cat([bg_cls_dist_emb, bg_dist_emb], dim=1))
        else:
            per_cls_input = torch.cat([intra_dist_emb, inter_dist_emb], dim=1)
            bg_logits = None

        # (Nxclasses) x 1
        cls_logits = self.per_cls_cnn(per_cls_input)

        # N x classes
        if isinstance(cls_logits, list):
            cls_logits = [v.reshape(bs, num_active_classes) for v in cls_logits]
        else:
            cls_logits = cls_logits.reshape(bs, num_active_classes)

        if bg_logits is not None:
            if isinstance(bg_logits, list):
                logits = []
                for c, b in zip(cls_logits, bg_logits):
                    logits.append(torch.cat([c, b], dim=1) / self.cls_temp)
            else:
                # N x (classes + 1)
                logits = torch.cat([cls_logits, bg_logits], dim=1) / self.cls_temp
        else:
            logits = cls_logits / self.cls_temp

        return logits

    @torch.no_grad()
    def build_image_prototypes(self, image: _Image, instances: Instances) -> ProtoTypes:
        features = self.mask_generator([image])
        instances = instances.to_tensor(features.device)
        roi_feats = self.roi_pool(features, [instances.bboxes])
        index = 2 if len(roi_feats.shape) == 4 else 1
        roi_feats = roi_feats.flatten(index).mean(index)
        return ProtoTypes(embeddings=roi_feats, labels=instances.labels, instances=instances)

    def get_proposals(self, images: List[_Image]) -> List[Instances]:
        return [self.mask_generator.get_proposals(image) for image in images]

    def interpolate(self, seq, x, mode='linear', force=False) -> _Tensor:
        return nn.functional.interpolate(seq, x, mode=mode) if (seq.shape[-1] < x) or force else seq[:, :, -x:]

    def distance_embed(self, x, temperature=10000, num_pos_feats=128, scale=10.0) -> _Tensor:
        # x: [bs, n_dist]
        x = x[..., None]
        scale = 2 * torch.pi * scale
        dim_t = torch.arange(num_pos_feats)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
        sin_x = x * scale / dim_t.to(x.device)
        emb = torch.stack((sin_x[:, :, 0::2].sin(), sin_x[:, :, 1::2].cos()), dim=3).flatten(2)
        return emb  # [bs, n_dist, n_emb]

    def focal_loss(self, inputs, targets, gamma=0.5, reduction='mean', bg_weight=0.0, num_classes=None):
        """Inspired by RetinaNet implementation"""
        if targets.numel() == 0 and reduction == "mean":
            return input.sum() * 0.0  # connect the gradient

        # focal scaling
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        p = nn.functional.softmax(inputs, dim=-1)
        p_t = p[torch.arange(p.size(0)).to(p.device), targets]  # get prob of target class
        p_t = torch.clamp(p_t, 1e-7, 1 - 1e-7)  # prevent NaN
        loss = ce_loss * ((1 - p_t) ** gamma)

        # bg loss weight
        if bg_weight >= 0:
            assert num_classes is not None
            loss_weight = torch.ones(loss.size(0)).to(p.device)
            loss_weight[targets == num_classes] = bg_weight
            loss = loss * loss_weight

        if reduction == 'mean':
            loss = loss.mean()

        return loss

    @property
    def device(self) -> torch.device:
        return self.fc_other_class.weight.device


def read_text_file(filename: str) -> List[str]:
    with open(filename, 'r') as txt_file:
        lines = txt_file.readlines()
    return [line.strip('\n') for line in lines]


def build_devit(
    generator: Any,
    roi_pool_size: int = 16,
    prototype_file: str = None,
    background_prototype_file: str = None,
    all_classes_fn: str = None,
    seen_classes_fn: str = None,
) -> DeVit:
    from fsl.utils.matcher import Matcher

    proposal_matcher = Matcher([0.3, 0.7], [0, -1, 1])

    if all_classes_fn and seen_classes_fn and prototype_file:
        prototypes = ProtoTypes.load(prototype_file)
        all_cids = read_text_file(all_classes_fn)
        seen_cids = read_text_file(seen_classes_fn)
    else:
        prototypes, all_cids, seen_cids = None, None, None

    bg_prototypes = ProtoTypes.load(background_prototype_file) if background_prototype_file else None
    return DeVit(generator, proposal_matcher, roi_pool_size, prototypes, bg_prototypes, all_cids, seen_cids)


@model_registry
def devit_sam(
    sam_args: Dict[str, str],
    mask_gen_args: Dict[str, Any] = {},
    roi_pool_size: int = 16,
    prototype_file: str = None,
    background_prototype_file: str = None,
    all_classes_fn: str = None,
    seen_classes_fn: str = None,
) -> DeVit:
    import pickle

    from fsl.models.sam_relational import build_sam_auto_mask_generator

    mask_generator = build_sam_auto_mask_generator(sam_args, mask_gen_args)
    return build_devit(
        mask_generator,
        roi_pool_size,
        prototype_file,
        background_prototype_file,
        all_classes_fn,
        seen_classes_fn,
    )


@model_registry
def devit_dinov2(
    model_name: str = 'dinov2_vitb14',
    roi_pool_size: int = 16,
    prototype_file: str = None,
    background_prototype_file: str = None,
    all_classes_fn: str = None,
    seen_classes_fn: str = None,
) -> DeVit:
    backbone = torch.hub.load('facebookresearch/dinov2', model_name)

    class DinoV2Patch(nn.Module):
        def __init__(self, backbone):
            super(DinoV2Patch, self).__init__()
            self.backbone = backbone.eval()

        @torch.no_grad()
        def forward(self, x: _Tensor) -> _Tensor:
            outputs = self.backbone.get_intermediate_layers(x, n=[self.backbone.n_blocks - 1], reshape=True)
            return outputs[0]

        @property
        def downsize(self) -> int:
            return self.backbone.patch_size

    for param in backbone.parameters():
        param.requires_grad_ = False

    backbone = DinoV2Patch(backbone)

    return build_devit(
        backbone,
        roi_pool_size,
        prototype_file,
        background_prototype_file,
        all_classes_fn,
        seen_classes_fn,
    )


if __name__ == '__main__':
    from torchvision.datapoints import BoundingBoxFormat

    # fn = '/root/krishneel/Downloads/fs_coco_trainval_novel_10shot.vitl14.pkl'
    fn = '/root/krishneel/Downloads/fsl/prototypes/fs_coco_trainval_novel_5shot.pkl'
    bg = '/root/krishneel/Downloads/background_prototypes.vitb14.pth'
    an = '../../data/coco/all_classes.txt'
    sn = '../../data/coco/seen_classes.txt'

    m = devit_sam(
        {'model': 'vit_b', 'checkpoint': None},
        prototype_file=fn,
        # background_prototype_file=bg,
        all_classes_fn=an,
        seen_classes_fn=sn,
    )
    m.cuda()

    im = Image.open('/root/krishneel/Downloads/000000.jpg')
    proposal = Instances(
        bboxes=[[750, 75, 1800, 1040], [750, 75, 1800, 1040]],
        bbox_fmt=BoundingBoxFormat.XYXY,
        class_ids=[1, 1],
        labels=['plate', 'plate'],
    )
    targets = [{'gt_proposal': proposal}]

    x = m([im], targets=targets)
    # x = m.build_image_prototypes(im, proposal)
    print(x)
