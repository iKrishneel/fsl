#!/usr/bin/env python

from typing import Any, Type, Dict, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.ops import RoIAlign
from torchvision.ops.boxes import box_area, box_iou

import numpy as np
from PIL import Image

from igniter.registry import model_registry

from fsl.structures import Proposal
from fsl.datasets import utils
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
                embedding = torch.cat(
                    [
                        embedding,
                    ]
                    + masks,
                    dim=1,
                )
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

        if not self.training:
            results = results[-1]
        return results


class DeVit(nn.Module):
    def __init__(
        self,
        mask_generator,
        proposal_matcher,
        prototypes: ProtoTypes,
        background_prototypes: ProtoTypes = None,
        all_cids = [],
        seen_cids = [],             
        roi_pool_size: int = 16
    ):
        super(DeVit, self).__init__()
        self.mask_generator = mask_generator
        self.proposal_matcher = proposal_matcher
        self.roi_pool = RoIAlign(roi_pool_size, spatial_scale=1 / mask_generator.predictor.downsize, sampling_ratio=-1)

        self.setup_prototypes(prototypes, all_cids, seen_cids)

        if background_prototypes is not None:
            self.register_buffer('bg_tokens', background_prototypes.embeddings)
        
        # TODO: Configure this
        self.batch_size_per_image = 128
        self.pos_ratio = 0.25
        self.num_sample_class = 10
        self.t_len = 128
        self.temb = 128
        self.t_pos_emb = 128
        self.t_bg_emb = 128
        self.cls_temp = 0.1
        self.bg_cls_weight = 0.2        
        hidden_dim = 256

        self.fc_other_class = nn.Linear(self.t_len, self.temb)
        self.fc_intra_class = nn.Linear(self.t_pos_emb, self.temb)
        self.fc_back_class = nn.Linear(len(self.bg_tokens), self.t_bg_emb)
        self.fc_bg_class = nn.Linear(self.t_len, self.temb)

        cls_input_dim = self.temb * 2 + self.t_bg_emb
        bg_input_dim = self.temb + self.t_bg_emb
        num_cls_layers = 3
        self.per_cls_cnn = PropagateNet(cls_input_dim, hidden_dim, num_layers=num_cls_layers)
        self.bg_cnn = PropagateNet(bg_input_dim, hidden_dim, num_layers=num_cls_layers)

    def setup_prototypes(self, prototypes: ProtoTypes, all_cids: List[str], seen_cids: List[str] = None):
        pt = prototypes.check(all_cids)
        train_class_order = [pt.labels.index(c) for c in seen_cids]
        test_class_order = [pt.labels.index(c) for c in all_cids]
        assert -1 not in train_class_order and -1 not in test_class_order

        self.register_buffer('train_class_weight', pt.normalized_embedding[torch.as_tensor(train_class_order)])
        self.register_buffer('test_class_weight', pt.normalized_embedding[torch.as_tensor(test_class_order)])
        
    def forward(self, images: List[_Image], targets: Dict[str, Any] = None):
        if not self.training:
            return self.forward_once(images)

        num_classes = len(self.train_class_weight)
        device = self.mask_generator.predictor.device  # TODO
        assert targets is not None and isinstance(targets, list)

        # proposals
        gt_proposals = [target['gt_proposal'] for target in targets]
        print(gt_proposals)
        
        # pred_proposals = self.get_proposals(images)
        noisy_proposals = [
            utils.prepare_noisy_boxes(gt_proposal, im.size[::-1]) for gt_proposal, im in zip(gt_proposals, images)
        ]
        
        noisy_proposals = torch.stack(
            [p.bbox.reshape(-1, 4) for noisy_proposal in noisy_proposals for p in noisy_proposal]
        )
        gt_proposals = [gt_proposal.bbox.reshape(-1, 4) for gt_proposal in gt_proposals]

        boxes = [torch.cat([gt_proposals[i], noisy_proposals[i]]) for i in range(len(targets))]
        
        # embedding of the images
        features = self.mask_generator(images)

        class_labels, matched_gt_boxes, resampled_proposals = [], [], []
        num_bg_samples, num_fg_samples, gt_masks = [], [], []
        for i, (proposals_per_image, targets_per_image) in enumerate(zip(boxes, gt_proposals)):
            match_quality_matrix = box_iou(targets_per_image, proposals_per_image)
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)

            # class_labels_i = targets_per_image[matched_idxs]
            class_labels_i = torch.Tensor([0, 0])
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
            
            gt_boxes_i = (
                targets_per_image[matched_idxs[sampled_idxs]]
                if len(targets_per_image) > 0
                else torch.zeros(len(sampled_idxs), 4, device=device)
            )  # not used anyway

            resampled_proposals.append(proposals_per_image)
            class_labels.append(class_labels_i)
            matched_gt_boxes.append(gt_boxes_i)

            num_bg_samples.append((class_labels_i == num_classes).sum().item())
            num_fg_samples.append(class_labels_i.numel() - num_bg_samples[-1])

        class_labels = torch.cat(class_labels)
        matched_gt_boxes = torch.cat(matched_gt_boxes)

        rois = []
        for bid, box in enumerate(resampled_proposals):
            batch_index = torch.full((len(box), 1), fill_value=float(bid)).to(device)
            rois.append(torch.cat([batch_index, box.to(device)], dim=1))
            rois = torch.cat(rois)            

        roi_features = self.roi_pool(features, rois)  # N, C, k, k
        roi_bs = len(roi_features)        
        
        # classification
        roi_features = roi_features.flatten(2)
        bs, spatial_size = roi_features.shape[0], roi_features.shape[-1]

        class_weight = self.train_class_weight[:, :roi_features.shape[-1]]
        # (N x spatial x emb) @ (emb x class) = N x spatial x class
        feats = roi_features.transpose(-2, -1) @ class_weight.T

        # sample topk classes
        class_topk = self.num_sample_class
        class_indices = None

        if class_topk < 0:
            class_topk = num_classes
            sample_class_enabled = False
        else:
            class_topk = num_classes if class_topk == 0 else class_topk
            sample_class_enabled = True
             
        if sample_class_enabled:
            num_active_classes = class_topk
            init_scores = nn.functional.normalize(roi_features.flatten(2).mean(2), dim=1) @ class_weight.T
            topk_class_indices = torch.topk(init_scores, class_topk, dim=1).indices

            class_indices = []
            for i in range(roi_bs):
                curr_label = class_labels[i].item()
                topk_class_indices_i = topk_class_indices[i].cpu()
                if curr_label in topk_class_indices_i or curr_label == num_classes:
                    curr_indices = topk_class_indices_i
                else:
                    curr_indices = torch.cat([torch.as_tensor([curr_label]), topk_class_indices_i[:-1]])
                class_indices.append(curr_indices)
            class_indices = torch.stack(class_indices).to(device)
            class_indices = torch.sort(class_indices, dim=1).values
        else:
            num_active_classes = num_classes        

        other_classes = []
        if sample_class_enabled:
            indexes = torch.arange(0, num_classes, device=device)[None, None, :].repeat(bs, spatial_size, 1)
            for i in range(class_topk):
                cmask = indexes != class_indices[:, i].view(-1, 1, 1)
                _ = torch.gather(
                    feats, 2, indexes[cmask].view(bs, spatial_size, num_classes - 1)
                )  # N x spatial x classes-1
                other_classes.append(_[:, :, None, :])
        else:
            for c in range(num_classes):  # TODO: change to classes sampling during training for LVIS type datasets
                cmask = torch.ones(num_classes, device=device, dtype=torch.bool)
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

        # TODO: REMOVE
        bg_feats = roi_features.transpose(-2, -1) @ self.bg_tokens[:, :roi_features.shape[-1]].T  # N x spatial x back
        bg_dist_emb = self.fc_back_class(bg_feats)  # N x spatial x emb
        bg_dist_emb = bg_dist_emb.permute(0, 2, 1).reshape(bs, -1, *roi_pool_size)
        # N x emb x S x S
        bg_dist_emb_c = bg_dist_emb[:, None, :, :, :].expand(-1, num_active_classes, -1, -1, -1).flatten(0, 1)
        # (Nxclasses) x emb x S x S

        # (Nxclasses) x EMB x S x S
        per_cls_input = torch.cat([intra_dist_emb, inter_dist_emb, bg_dist_emb_c], dim=1)
        # (Nxclasses) x 1
        cls_logits = self.per_cls_cnn(per_cls_input)

        # N x classes
        if isinstance(cls_logits, list):
            cls_logits = [v.reshape(bs, num_active_classes) for v in cls_logits]
        else:
            cls_logits = cls_logits.reshape(bs, num_active_classes)
            
        # N x 1
        # feats: N x spatial x class
        cls_dist_feats = self.interpolate(torch.sort(feats, dim=2).values, self.t_len, mode='linear')  # N x spatial x T
        bg_cls_dist_emb = self.fc_bg_class(cls_dist_feats)  # N x spatial x emb
        bg_cls_dist_emb = bg_cls_dist_emb.permute(0, 2, 1).reshape(bs, -1, *roi_pool_size)
        bg_logits = self.bg_cnn(torch.cat([bg_cls_dist_emb, bg_dist_emb], dim=1))
        
        if isinstance(bg_logits, list):
            logits = []
            for c, b in zip(cls_logits, bg_logits):
                logits.append(torch.cat([c, b], dim=1) / self.cls_temp)
        else:
            # N x (classes + 1)
            logits = torch.cat([cls_logits, bg_logits], dim=1)
            logits = logits / self.cls_temp
            
        # loss
        class_labels = class_labels.long().to(device)
        if sample_class_enabled:
            bg_indices = class_labels == num_classes
            fg_indices = class_labels != num_classes

            class_labels[fg_indices] = (class_indices == class_labels.view(-1, 1)).nonzero()[:, 1]
            class_labels[bg_indices] = num_active_classes

        loss_dict = {}
        if isinstance(logits, list):
            for i, l in enumerate(logits):
                loss = self.focal_loss(l, class_labels, num_classes=num_active_classes, bg_weight=self.bg_cls_weight)
                loss_dict[f'focal_loss_{i}'] = loss
        else:
            loss = self.focal_loss(logits, class_labels, num_classes=num_active_classes, bg_weight=self.bg_cls_weight)
            loss_dict['focal_loss'] = loss

        import IPython, sys; IPython.embed(header="Forward"); sys.exit()

            
    @torch.no_grad()
    def forward_once(self, images: List[_Image]) -> Dict[str, Any]:
        proposals = self.get_proposals(images)
        return {'features': self.mask_generator.predictor.features, 'proposals': proposals}        

    def get_proposals(self, images: List[_Image]) -> List[List[Proposal]]:
        return [self.mask_generator.get_proposals(image) for image in images]

    def interpolate(self, seq, T, mode='linear', force=False) -> _Tensor:
        return nn.functional.interpolate(seq, T, mode=mode) if (seq.shape[-1] < T) or force else seq[:, :, -T:]

    def distance_embed(self, x, temperature=10000, num_pos_feats=128, scale=10.0) -> _Tensor:
        # x: [bs, n_dist]
        x = x[..., None]
        scale = 2 * torch.pi * scale
        dim_t = torch.arange(num_pos_feats)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
        sin_x = x * scale / dim_t.to(x.device)
        emb = torch.stack((sin_x[:, :, 0::2].sin(), sin_x[:, :, 1::2].cos()), dim=3).flatten(2)
        return emb  # [bs, n_dist, n_emb]

    def focal_loss(self, inputs, targets, gamma=0.5, reduction="mean", bg_weight=0.2, num_classes=None):
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


def read_text_file(filename: str) -> List[str]:
    with open(filename, 'r') as txt_file:
        lines = txt_file.readlines()
    return [line.strip('\n') for line in lines]


@model_registry
def devit(
    sam_args: Dict[str, str],
    mask_gen_args: Dict[str, Any] = {},
    prototype_file: str = None,
    background_prototype_file: str = None,
    all_classes_fn: str = None,
    seen_classes_fn: str = None,
):
    from fsl.models.sam_relational import build_sam_auto_mask_generator
    from fsl.utils.matcher import Matcher

    mask_generator = build_sam_auto_mask_generator(sam_args, mask_gen_args)
    proposal_matcher = Matcher([0.3, 0.7], [0, -1, 1])

    prototypes = ProtoTypes.load(prototype_file)
    background_prototypes = ProtoTypes.load(background_prototype_file)
    
    all_cids = read_text_file(all_classes_fn)
    seen_cids = read_text_file(seen_classes_fn)

    return DeVit(
        mask_generator,
        proposal_matcher=proposal_matcher,
        prototypes=prototypes,
        background_prototypes=background_prototypes,
        all_cids=all_cids,
        seen_cids=seen_cids
    )


if __name__ == '__main__':
    from torchvision.datapoints import BoundingBoxFormat
    
    fn = '/root/krishneel/Downloads/fs_coco_trainval_novel_10shot.vitl14.pkl'
    bg = '/root/krishneel/Downloads/background_prototypes.vitb14.pth'    
    an = '../../data/coco/all_classes.txt'
    sn = '../../data/coco/seen_classes.txt'

    
    m = devit(
        {'model': 'vit_b', 'checkpoint': None},
        prototype_file=fn,
        background_prototype_file=bg,
        all_classes_fn=an,
        seen_classes_fn=sn
    )
    m.cuda()

    im = Image.open('/root/krishneel/Downloads/000000.jpg')
    proposal = Proposal(bbox=[750, 75, 1800, 1040], bbox_fmt=BoundingBoxFormat.XYXY, label=1)
    targets = [{'gt_proposal': proposal}]

    m([im], targets=targets)
