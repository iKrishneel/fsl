#!/usr/bin/env python

from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from igniter.registry import model_registry
from PIL import Image
from torchvision.ops import RoIAlign
from torchvision.ops.boxes import box_iou

from fsl.structures import Instances
from fsl.utils.matcher import Matcher
from fsl.utils.prototypes import ProtoTypes

_Image = Type[Image.Image]
_Tensor = Type[torch.Tensor]
_Module = Type[nn.Module]


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
        fg_prototypes: ProtoTypes = None,
        bg_prototypes: ProtoTypes = None,
        all_cids: List[str] = None,
        seen_cids: List[str] = None,
    ):
        super(DeVit, self).__init__()

        # TODO: Configure this
        self.batch_size_per_image = 128
        self.pos_ratio = 0.25
        self.num_sample_class = 10
        self.t_len = 256 // 2
        self.temb = 128
        self.t_pos_emb = 128
        self.t_bg_emb = 128
        self.cls_temp = 0.1
        self.hidden_dim = 256
        self.num_cls_layers = 3

        self.bg_cls_weight = 0.0

        if fg_prototypes:
            self._setup_prototypes(fg_prototypes, all_cids, seen_cids, is_bg=False)

        if bg_prototypes:
            self._setup_prototypes(bg_prototypes, all_cids, seen_cids, is_bg=True)
        else:
            cls_input_dim = self.temb * 2
            self.bg_tokens, self.fc_back_class, self.bg_cnn = None, None, None
            self.per_cls_cnn = PropagateNet(cls_input_dim, self.hidden_dim, num_layers=self.num_cls_layers)

        self.fc_other_class = nn.Linear(self.t_len, self.temb)
        self.fc_intra_class = nn.Linear(self.t_pos_emb, self.temb)

        self._all_cids = all_cids

    def _setup_prototypes(
        self, prototypes: ProtoTypes, all_cids: List[str] = None, seen_cids: List[str] = None, is_bg: bool = False
    ) -> None:
        if is_bg:
            self.register_buffer('bg_tokens', prototypes.normalized_embedding)
            self._init_bg_layers()
        else:
            pt = prototypes.check(all_cids)
            train_class_order = [pt.labels.index(c) for c in seen_cids]
            test_class_order = [pt.labels.index(c) for c in all_cids]
            assert -1 not in train_class_order and -1 not in test_class_order

            self.register_buffer('train_class_weight', pt.normalized_embedding[torch.as_tensor(train_class_order)])
            self.register_buffer('test_class_weight', pt.normalized_embedding[torch.as_tensor(test_class_order)])

    def _init_bg_layers(self) -> None:
        self.bg_cls_weight = 0.2
        self.fc_bg_class = nn.Linear(self.t_len, self.temb)
        self.fc_back_class = nn.Linear(len(self.bg_tokens), self.t_bg_emb)
        bg_input_dim = self.temb + self.t_bg_emb
        self.bg_cnn = PropagateNet(bg_input_dim, self.hidden_dim, num_layers=self.num_cls_layers)
        cls_input_dim = self.temb * 2 + self.t_bg_emb
        self.per_cls_cnn = PropagateNet(cls_input_dim, self.hidden_dim, num_layers=self.num_cls_layers)

    def forward(self, roi_features: _Tensor, class_labels: _Tensor = None) -> Dict[str, Any]:
        if not self.training:
            return self.forward_once(roi_features)

        num_classes = len(self.train_class_weight)

        # sample topk classes
        class_topk = self.num_sample_class if self.num_sample_class > 0 else num_classes
        class_topk = min(class_topk, num_classes)

        sample_class_enabled = class_topk > 1
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
            bg_indices = class_labels == num_classes
            fg_indices = class_labels != num_classes

            class_labels[fg_indices] = (class_indices == class_labels.view(-1, 1)).nonzero()[:, 1]
            class_labels[bg_indices] = num_active_classes

        if self.bg_tokens is None:
            indices = torch.where(class_labels != num_active_classes)
            class_labels = class_labels[indices]
            logits = [logit[indices] for logit in logits]

        if isinstance(logits, list):
            loss_dict = {
                f'focal_loss_{i}': self.focal_loss(
                    logit, class_labels, num_classes=num_active_classes, bg_weight=self.bg_cls_weight
                )
                for i, logit in enumerate(logits)
            }
        else:
            loss_dict = {
                'focal_loss': self.focal_loss(
                    logits, class_labels, num_classes=num_active_classes, bg_weight=self.bg_cls_weight
                )
            }

        del logits
        return loss_dict

    @torch.no_grad()
    def forward_once(self, roi_features: _Tensor, class_labels: _Tensor = None) -> Dict[str, Any]:
        self.test_class_weight.to(roi_features.dtype)
        num_classes = len(self.test_class_weight)

        # sample topk classes
        class_topk = self.num_sample_class if self.num_sample_class > 0 else num_classes
        class_topk = min(class_topk, num_classes)

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
            full_scores = torch.zeros(len(scores), num_classes + 1, device=self.device, dtype=roi_features.dtype)
            full_scores.scatter_(1, class_indices, scores)

            scores = full_scores
            full_scores[:, -1] = scores[:, -1]
            output['scores'] = full_scores[:, :-1]

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
        hw = np.sqrt(spatial_size)
        assert hw.is_integer(), 'Only square roi shape is support!'
        roi_pool_size = [int(hw), int(hw)]

        other_classes = []
        if sample_class_enabled:
            indexes = torch.arange(0, num_classes, device=self.device)[None, None, :].repeat(bs, spatial_size, 1)
            for i in range(class_topk):
                cmask = indexes != class_indices[:, i].view(-1, 1, 1)
                fts = torch.gather(
                    feats, 2, indexes[cmask].view(bs, spatial_size, num_classes - 1)
                )  # N x spatial x classes-1
                other_classes.append(fts[:, None, :, :])
        else:
            for c in range(num_classes):
                cmask = torch.ones(num_classes, device=self.device, dtype=torch.bool)
                cmask[c] = False
                fts = feats[:, :, cmask]  # # N x spatial x classes-1
                other_classes.append(fts[:, None, :, :])

        other_classes = torch.cat(other_classes, dim=1)  # N x spatial x classes x classes-1
        other_classes = other_classes.flatten(0, 1)  # (Nxclasses) x spatial x classes-1
        other_classes, _ = torch.sort(other_classes, dim=-1)
        other_classes = self.interpolate(other_classes, self.t_len, mode='linear')  # (Nxclasses) x spatial x T
        other_classes = self.fc_other_class(other_classes)  # (Nxclasses) x spatial x emb
        other_classes = other_classes.permute(0, 2, 1)  # (Nxclasses) x emb x spatial
        # (Nxclasses) x emb x S x S
        inter_dist_emb = other_classes.reshape(bs * num_active_classes, -1, *roi_pool_size)

        intra_feats = (
            torch.gather(feats, 2, class_indices[:, None, :].repeat(1, spatial_size, 1))
            if sample_class_enabled
            else feats
        )
        intra_dist_emb = self.distance_embed(intra_feats.flatten(0, 1), num_pos_feats=self.t_pos_emb)
        intra_dist_emb = self.fc_intra_class(intra_dist_emb.to(roi_features.dtype))
        intra_dist_emb = intra_dist_emb.reshape(bs, spatial_size, num_active_classes, -1)

        # (Nxclasses) x emb x S x S
        intra_dist_emb = (
            intra_dist_emb.permute(0, 2, 3, 1).flatten(0, 1).reshape(bs * num_active_classes, -1, *roi_pool_size)
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
        cls_logits = (
            [v.reshape(bs, num_active_classes) for v in cls_logits]
            if isinstance(cls_logits, list)
            else cls_logits.reshape(bs, num_active_classes)
        )

        if bg_logits is not None:
            logits = (
                [torch.cat([c, b], dim=1) / self.cls_temp for c, b in zip(cls_logits, bg_logits)]
                if isinstance(bg_logits, list)
                else torch.cat([cls_logits, bg_logits], dim=1) / self.cls_temp
            )
            # N x (classes + 1)
        else:
            logits = (
                [logit / self.cls_temp for logit in cls_logits]
                if isinstance(cls_logits, list)
                else cls_logits / self.cls_temp
            )

        return logits

    def interpolate(self, seq: _Tensor, size: int, mode: str = 'linear', force: bool = False) -> _Tensor:
        return nn.functional.interpolate(seq, size, mode=mode) if (seq.shape[-1] < size) or force else seq[:, :, -size:]

    def distance_embed(
        self, x: _Tensor, temperature: int = 10000, num_pos_feats: int = 128, scale: float = 10.0
    ) -> _Tensor:
        # x: [bs, n_dist]
        x = x[..., None]
        scale = 2 * torch.pi * scale
        dim_t = torch.arange(num_pos_feats)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
        sin_x = x * scale / dim_t.to(x.device)
        emb = torch.stack((sin_x[:, :, 0::2].sin(), sin_x[:, :, 1::2].cos()), dim=3).flatten(2)
        return emb  # [bs, n_dist, n_emb]

    def focal_loss(
        self,
        inputs: _Tensor,
        targets: _Tensor,
        gamma: int = 0.5,
        reduction: str = 'mean',
        bg_weight: float = 0.0,
        num_classes: List[int] = None,
    ):
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

    # def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True, assign: bool = False) -> Any:
    #    return super(DeVit, self).load_state_dict(state_dict, strict)

    @property
    def device(self) -> torch.device:
        return self.fc_other_class.weight.device


class DeVitSam(DeVit):
    def __init__(
        self,
        mask_generator: _Module,
        proposal_matcher: Matcher,
        roi_pool_size: int = 16,
        fg_prototypes: ProtoTypes = None,
        bg_prototypes: ProtoTypes = None,
        all_cids: List[str] = None,
        seen_cids: List[str] = None,
    ):
        super(DeVitSam, self).__init__(fg_prototypes, bg_prototypes, all_cids, seen_cids)
        self.mask_generator = mask_generator
        self.proposal_matcher = proposal_matcher
        self.roi_pool = RoIAlign(roi_pool_size, spatial_scale=1 / mask_generator.downsize, sampling_ratio=-1)
        self.use_noisy_bboxes = False

    def sample_noisy_rois(self, images: List[_Tensor], targets: List[Dict[str, Any]]) -> Tuple[List, _Tensor]:
        from fsl.datasets import utils
        
        num_classes = len(self.train_class_weight)
        img_hw = images[0].shape[1:] if isinstance(images[0], torch.Tensor) else images[0].size[::-1]
        gt_instances = [target['gt_proposal'] for target in targets]
        gt_bboxes = [gt_proposal.to_tensor().bboxes for gt_proposal in gt_instances]

        if self.use_noisy_bboxes:
            noisy_proposals = utils.prepare_noisy_boxes(gt_bboxes, img_hw)
            boxes = [torch.cat([gt_bboxes[i], noisy_proposals[i]]).to(self.device) for i in range(len(targets))]
        else:
            boxes = [torch.cat([gt_bboxes[i]]).to(self.device) for i in range(len(targets))]

        class_labels, resampled_proposals = [], []
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

        class_labels = torch.cat(class_labels)
        return resampled_proposals, class_labels

    def forward(self, images: List[_Tensor], targets: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.training:
            return self.forward_once(images, targets)

        num_classes = len(self.train_class_weight)
        assert targets is not None and len(targets) == len(images)

        # proposals
        resampled_proposals, class_labels = self.sample_noisy_rois(images, targets)

        rois = []
        for bid, box in enumerate(resampled_proposals):
            batch_index = torch.full((len(box), 1), fill_value=float(bid)).to(self.device)
            rois.append(torch.cat([batch_index, box.to(self.device)], dim=1))
        rois = torch.cat(rois)

        # embedding of the images
        images = torch.stack(images).to(self.device) if isinstance(images[0], torch.Tensor) else images
        features = self.mask_generator(images)
        roi_features = self.roi_pool(features, rois)  # N, C, k, k

        return super().forward(roi_features, class_labels)

    @torch.no_grad()
    def forward_once(self, images: List[_Tensor], targets: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        if targets and 'gt_proposal' in targets[0]:
            proposals = [target['gt_proposal'] for target in targets]
        else:
            proposals = self.get_proposals(images)

        assert len(proposals) == 1

        # num_classes = len(self.test_class_weight)

        bboxes = torch.cat([proposal.to_tensor().bboxes.to(self.device) for proposal in proposals])
        rois = torch.cat([torch.full((len(bboxes), 1), fill_value=0).to(self.device), bboxes], dim=1)

        images = torch.stack(images).to(self.device)
        features = self.mask_generator(images)

        roi_features = self.roi_pool(features, rois)  # N, C, k, k
        predictions = super().forward_once(roi_features)
        return predictions

    @torch.no_grad()
    def build_image_prototypes(self, image: _Tensor, instances: Instances) -> ProtoTypes:
        features = self.mask_generator(image[None])
        instances = instances.to_tensor(features.device)
        roi_feats = self.roi_pool(features, [instances.bboxes])
        index = 2 if len(roi_feats.shape) == 4 else 1
        roi_feats = roi_feats.flatten(index).mean(index)
        return ProtoTypes(embeddings=roi_feats, labels=instances.labels, instances=instances)

    def get_proposals(self, images: List[_Image]) -> List[Instances]:
        return [self.mask_generator.get_proposals(image) for image in images]


def build_devit(
    prototype_file: str = None,
    background_prototype_file: str = None,
    label_map_file: str = None,
) -> DeVit:
    if label_map_file:
        import json

        with open(label_map_file, 'r') as jfile:
            label_map = json.load(jfile)

        all_cids = list(label_map['all_classes'].values())
        seen_cids = list(label_map['seen_classes'].values())
    else:
        all_cids, seen_cids = None, None

    prototypes = ProtoTypes.load(prototype_file) if prototype_file else None
    bg_prototypes = ProtoTypes.load(background_prototype_file) if background_prototype_file else None
    return DeVit(prototypes, bg_prototypes, all_cids, seen_cids)


def build_devit_sam(
    generator: Any,
    roi_pool_size: int = 16,
    prototype_file: str = None,
    background_prototype_file: str = None,
    label_map_file: str = None,
) -> DeVitSam:
    proposal_matcher = Matcher([0.3, 0.7], [0, -1, 1])

    if label_map_file:
        import json

        with open(label_map_file, 'r') as jfile:
            label_map = json.load(jfile)

        all_cids = list(label_map['all_classes'].values())
        seen_cids = list(label_map['seen_classes'].values())

    if prototype_file:
        prototypes = ProtoTypes.load(prototype_file)
    else:
        prototypes, all_cids, seen_cids = None, None, None

    bg_prototypes = ProtoTypes.load(background_prototype_file) if background_prototype_file else None
    return DeVitSam(generator, proposal_matcher, roi_pool_size, prototypes, bg_prototypes, label_map_file)


@model_registry
def devit_sam(
    sam_args: Dict[str, str],
    mask_gen_args: Dict[str, Any] = {},
    roi_pool_size: int = 16,
    prototype_file: str = None,
    background_prototype_file: str = None,
    label_map_file: str = None,
) -> DeVitSam:
    from fsl.models.sam_utils import build_sam_auto_mask_generator

    mask_generator = build_sam_auto_mask_generator(sam_args, mask_gen_args)
    return build_devit_sam(
        mask_generator,
        roi_pool_size,
        prototype_file,
        background_prototype_file,
        label_map_file,
    )
