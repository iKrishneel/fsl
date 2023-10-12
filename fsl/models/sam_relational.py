#!/usr/bin/evn python

from typing import Any, Dict, List, Type, Callable, Iterator, Union, Tuple

import os.path as osp
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import RoIAlign

from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator as _SAMG, SamPredictor as _SamPredictor
from igniter.registry import model_registry

from fsl.dataset import S3CocoDatasetSam
from fsl.structures import Proposal

_Tensor = Type[torch.Tensor]
_Module = Type[nn.Module]
_Image = Type[Image.Image]


class Conv2dBN(nn.Sequential):
    def __init__(self, in_channels: int, hidden_channels: int, stride: int = 1, activation: Callable = None) -> None:
        super(Conv2dBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden_channels)
        self.activation = activation

    def forward(self, x: _Tensor) -> _Tensor:
        x = self.bn(self.conv(x))
        x = self.activation(x) if self.activation else x
        return x


class AttentionPool2d(nn.Module):
    """
    Performs attention pooling on the feature maps
    Ref: https://github.com/openai/CLIP/blob/main/clip/model.py#L58
    """

    def __init__(self, spatial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super(AttentionPool2d, self).__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spatial_dim**2 + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class Relational(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, spatial_dim: int = 16):
        super(Relational, self).__init__()

        self.conv1 = Conv2dBN(in_channels, hidden_channels, activation=nn.ReLU())
        self.conv2 = Conv2dBN(hidden_channels, hidden_channels, activation=nn.ReLU())
        self.attnpool = AttentionPool2d(spatial_dim, hidden_channels, 8, hidden_channels)
        self.fc1 = nn.Linear(hidden_channels, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x: _Tensor) -> _Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attnpool(x)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x


class SamPredictor(nn.Module, _SamPredictor):
    def __init__(self, sam):
        super(SamPredictor, self).__init__()

        for parameter in sam.parameters():
            parameter.requires_grad = False

        _SamPredictor.__init__(self, sam_model=sam)

    @torch.no_grad()
    def forward(self, images: List[np.ndarray]) -> torch.Tensor:
        return self.set_images(images)

    @torch.no_grad()
    def set_images(self, images: List[np.ndarray], image_format: str = 'RGB') -> None:
        assert image_format in ['RGB', 'BGR'], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            images = [image[..., ::-1] for image in images]

        input_images = [
            torch.as_tensor(self.transform.apply_image(image)).permute(2, 0, 1).contiguous() for image in images
        ]
        input_images = [self.model.preprocess(image.to(self.device)) for image in input_images]
        input_images_torch = torch.stack(input_images).to(self.device)
        self.model.image_encoder.to(self.device)
        return self.model.image_encoder(input_images_torch)

    """
    def to(self, device):
        self.model.to(device)

    def children(self) -> Iterator[_Module]:
        for name, module in self.model.named_children():
            yield module

    def modules(self) -> Iterator[_Module]:
        for _, module in self.model.named_modules():
            yield module

    def buffers(self, *args, **kwargs):
        return self.model.buffers(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)

    def named_modules(self, *args, **kwargs):
        return self.model.named_modules(*args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        return self.model.named_buffers(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        return self.model.named_parameters(*args, **kwargs)
    """

    @property
    def img_size(self) -> List[int]:
        return [self.model.image_encoder.img_size] * 2

    @property
    def out_channels(self) -> int:
        return self.model.image_encoder.neck[-1].weight.shape[0]

    @property
    def downsize(self) -> int:
        return self.model.image_encoder.img_size // self.model.image_encoder.patch_embed.proj.kernel_size[0]

    def reset_image(self) -> None:
        self.is_image_set = False
        # self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None


class SamAutomaticMaskGenerator(nn.Module, _SAMG):
    def __init__(self, model, **kwargs):
        super(SamAutomaticMaskGenerator, self).__init__()
        _SAMG.__init__(self, model, **kwargs)
        
        self.predictor = SamPredictor(model)

    def forward(self, images: List[Union[np.ndarray, _Image]]) -> _Tensor:
        x = [np.asarray(image) for image in images]
        return self.predictor.set_images(x)

    def get_proposals(self, image: Union[_Image, np.ndarray]) -> List[Dict[str, Any]]:
        image = np.asarray(image)
        masks = self.generate(image)
        proposals = [
            Proposal(*[mask[k] for k in ['bbox', 'segmentation']]).convert_bbox_fmt(BoundingBoxFormat.XYXY)
            for mask in masks
        ]
        return proposals
        

class SamRelationNetwork(nn.Module):
    def __init__(
        self, mask_generator: SamAutomaticMaskGenerator, relational_in_size: int = 128, roi_pool_size: int = 16
    ) -> None:
        super(SamRelationNetwork, self).__init__()
        self.mask_generator = mask_generator

        self.condenser = nn.Conv2d(
            self.sam_predictor.out_channels * 2, relational_in_size, kernel_size=(3, 3), stride=1, padding=1
        )
        self.relation_net = Relational(relational_in_size, hidden_channels=relational_in_size // 2)
        self.roi_pool = RoIAlign(
            roi_pool_size, spatial_scale=1 / self.sam_predictor.downsize, sampling_ratio=0, aligned=True
        )
        self._query_feats = None

    def forward(self, images: List[Union[np.ndarray, _Image]], targets: Dict[str, Any] = None):
        if not self.training:
            return self.forward_inference(images)

        assert self.training and targets is not None

        query_index = 0
        labels = torch.Tensor([i for target in targets for i in target['category_ids']])
        # y_true = torch.Tensor([1]) if labels.shape[0] == 1 else labels.eq(labels[query_index]).float()[1:]

        sam_feats = self.forward_sam(images)
        bboxes = self.get_roi_bboxes(targets if self.training else sam_feats[1])
        roi_feats = self.roi_pool(sam_feats, bboxes)

        if labels.shape[0] == 1:
            y_true = torch.Tensor([1])
            query_feats = roi_feats.clone()
        else:
            mode, index = torch.mode(labels)
            query_index = int(index)
            y_true = labels.eq(mode).float()
            y_true = torch.cat((y_true[:query_index], y_true[query_index + 1 :]), dim=0)

            k = roi_feats.shape[0]
            query_feats = roi_feats[query_index][None].repeat(k - 1, 1, 1, 1)
            roi_feats = self.remove(roi_feats, query_index)

        roi_feats = torch.cat([roi_feats, query_feats], dim=1)
        roi_feats = self.condenser(roi_feats)
        y_pred = self.relation_net(roi_feats)

        assert (
            y_true.shape[0] == roi_feats.shape[0]
        ), f'Shape mismatch of label {y_true.shape} and roi {roi_feats.shape}'

        y_true = y_true.reshape(-1, 1).to(self.device)
        return self.losses(y_pred, y_true)

    @torch.no_grad()
    def forward_inference(self, images: List[Union[np.ndarray, _Image]]) -> Any:
        masks = self.forward_sam(images)
        bboxes = [torch.FloatTensor(S3CocoDatasetSam.xywh_to_xyxy(mask['bbox'])) for mask in masks]
        bboxes = [torch.stack(bboxes).to(self.sam_predictor.features.device)]
        roi_feats = self.roi_pool(self.sam_predictor.features, bboxes)

        query_feats = self.query_feats.repeat(roi_feats.shape[0], 1, 1, 1)
        roi_feats = torch.cat([roi_feats, query_feats], dim=1)
        roi_feats = self.condenser(roi_feats)
        y_pred = self.relation_net(roi_feats)
        
        import IPython, sys; IPython.embed(header="forward"); sys.exit()

        raise NotImplementedError('Inference is not yet implemented!')

    @torch.no_grad()
    def forward_sam(
        self, images: List[np.ndarray], only_feats: bool = False
    ) -> Union[_Tensor, Tuple[_Tensor, List[Dict[str, Any]]]]:
        x = [np.asarray(image) for image in images]
        if self.training or only_feats:
            return self.sam_predictor.set_images(x)

        self.mask_generator.predictor.set_image(x[0])
        masks = self.mask_generator.generate(x[0])
        return masks

    def set_query_images(self, images: List[Union[np.ndarray, _Image]], bboxes: List[np.ndarray]):
        features = self.forward_sam(images)
        self._query_feats = self.roi_pool(features, bboxes)

    def set_query_roi_features(self, features: _Tensor) -> None:
        expected_size = torch.Size([self.roi_pool.output_size] * 2)
        assert features.shape[2:] == expected_size, f'Expected spatial size {expected_size} but got features.shape[2:]'
        self._query_feats = features.to(self.device)

    def get_query_roi_features(self, images: List[Union[np.ndarray, _Image]], bboxes: List[np.ndarray]):
        features = self.forward_sam(images, only_feats=True)
        bboxes = [bbox.to(self.device) for bbox in bboxes]
        return self.roi_pool(features, bboxes)

    def get_roi_bboxes(self, targets: List[Dict[str, _Tensor]] = None):
        if targets is not None:
            bboxes = [torch.stack(target['bboxes']).to(self.device) for target in targets]
        return bboxes

    def losses(self, y_pred: _Tensor, y_true: _Tensor) -> Dict[str, _Tensor]:
        return {'loss': F.mse_loss(y_pred, y_true)}

    def to(self, *args, **kwargs):
        self.sam_predictor.model.to(*args, **kwargs)
        return super(SamRelationNetwork, self).to(*args, **kwargs)

    def cuda(self, device: Union[int, torch.device] = None):
        self.sam_predictor.model.cuda(device)
        return super(SamRelationNetwork, self).cuda(device)

    def eval(self):
        self.sam_predictor.model.eval()
        return super(SamRelationNetwork, self).eval()

    @staticmethod
    def remove(tensor: _Tensor, index: int) -> _Tensor:
        return torch.cat((tensor[:index], tensor[index + 1 :]), dim=0)

    @property
    def query_feats(self) -> _Tensor:
        return self._query_feats

    @property
    def sam_predictor(self) -> SamPredictor:
        return self.mask_generator.predictor

    @property
    def device(self) -> torch.device:
        return self.sam_predictor.model.image_encoder.patch_embed.proj.weight.device


def get_sam_model(name: str = 'default'):
    import os
    import subprocess

    url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_%s.pth'
    sam_checkpoint_registry = {'default': 'h_4b8939', 'vit_h': 'h_4b8939', 'vit_l': 'l_0b3195', 'vit_b': 'b_01ec64'}

    directory = os.path.join(os.environ['HOME'], '.cache/torch/segment_anything/checkpoints/')
    checkpoint = os.path.join(directory, f'sam_vit_{sam_checkpoint_registry[name]}.pth')

    if not os.path.isfile(checkpoint):
        os.makedirs(directory, exist_ok=True)
        try:
            checkpoint_url = url % sam_checkpoint_registry[name]
            command = ['wget', checkpoint_url, '-P', directory, '--quiet', '--show-progress', '--progress=dot']
            subprocess.run(command, check=True)
            print(f'Downloaded {checkpoint_url}')
        except subprocess.CalledProcessError as e:
            print(f'Error downloading {checkpoint_url}: {e}')
            checkpoint = None

    print(f'Loading checkpoint from {checkpoint}')
    sam_model = sam_model_registry[name](checkpoint)
    return sam_model


def build_sam_predictor(model: str, checkpoint: str = None) -> SamPredictor:
    return SamPredictor(get_sam_model(model) if checkpoint is None else sam_model_registry[model](checkpoint=checkpoint))


def build_sam_auto_mask_generator(sam_args: Dict[str, str], mask_gen_args: Dict[str, Any]) -> SamAutomaticMaskGenerator:
    sam_predictor = build_sam_predictor(**sam_args)    
    return SamAutomaticMaskGenerator(sam_predictor.model, **mask_gen_args)


@model_registry('relational_network')
def sam_relational_network(
    sam_args: Dict[str, str] = {'model': 'default', 'checkpoint': None}, mask_gen_args: Dict[str, Any] = {}
) -> SamRelationNetwork:
    mask_generator = build_sam_auto_mask_generator(sam_args, mask_gen_args)
    return SamRelationNetwork(mask_generator)
