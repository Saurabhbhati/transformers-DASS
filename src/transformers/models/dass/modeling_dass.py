# coding=utf-8
"""Distilled Audio State-Space Model (DASS) model"""

import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from functools import partial
from typing import Optional, Callable, Any, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from ...modeling_outputs import SequenceClassifierOutput

from ...utils import logging
from ...modeling_utils import PreTrainedModel

from .configuration_dass import DASSConfig

from .vmamba_utils import cross_scan_fn, selective_scan_fn, cross_merge_fn

logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "DASSConfig"

class DASSLinear2d(nn.Linear):
    def __init__(self, *args, groups=1, **kwargs):
        nn.Linear.__init__(self, *args, **kwargs)
        self.groups = groups
    
    def forward(self, x: torch.Tensor):
        if len(x.shape) == 4:
            return F.conv2d(x, self.weight[:, :, None, None], self.bias, groups=self.groups)
        elif len(x.shape) == 3:
            return F.conv1d(x, self.weight[:, :, None], self.bias, groups=self.groups)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self_state_dict = self.state_dict()
        load_state_dict_keys = list(state_dict.keys())
        if prefix + "weight" in load_state_dict_keys:
            state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view_as(self_state_dict["weight"])
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class DASSLayerNorm2d(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        nn.LayerNorm.__init__(self, *args, **kwargs)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.LayerNorm.forward(self, x)
        x = x.permute(0, 3, 1, 2)
        return x


class DASSPatchEmbeddings(nn.Module):
    """
    This class turns `input_values` into the initial `hidden_states` (patch embeddings) of shape `(batch_size,
    seq_length, hidden_size)` to be consumed by a State-space model.
    """

    def __init__(self, patch_size=4,embed_dim=96):
        super().__init__()

        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1

        self.projection = nn.Sequential(
            nn.Conv2d(1, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            DASSLayerNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            DASSLayerNorm2d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.projection(x)
        return x


class DASSDowsample(nn.Module):
    """
    This class downsamples the input tensor using a convolutional layer followed by a layer normalization.
    """
    def __init__(self, dim, out_dim, use_norm=True):
        super().__init__()
        self.down = nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.norm = DASSLayerNorm2d(out_dim) if use_norm else nn.Identity()

    def forward(self, x):
        x = self.down(x)
        x = self.norm(x)
        return x


class DASSMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = DASSLinear2d(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = DASSLinear2d(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SS2D(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3,
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # forward_type="v05_noz" is always used
        # ======================
        **kwargs,
    ):
        super().__init__()
        self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.forward_core = partial(self.forward_corev2, force_fp32=False, no_einsum=True)
        self.with_dconv = d_conv > 1

        # In projection
        self.in_proj = DASSLinear2d(self.d_model, self.d_inner, bias=bias)
        self.act: nn.Module = act_layer()

        # Convolution
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
            )

        # x_proj and dt_proj
        self.x_proj = DASSLinear2d(self.d_inner, self.k_group * (self.dt_rank + self.d_state * 2), groups=self.k_group, bias=False)
        self.dt_projs = DASSLinear2d(self.dt_rank, self.k_group * self.d_inner, groups=self.k_group, bias=False)

        # out projection
        self.out_proj = DASSLinear2d(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # Initialization
        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = self.init_dt_A_D(
            self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=self.k_group,
        )
        self.dt_projs.weight.data = self.dt_projs_weight.data.view(self.dt_projs.weight.shape)
        # self.dt_projs.bias.data = self.dt_projs_bias.data.view(self.dt_projs.bias.shape)
        del self.dt_projs_weight
        # del self.dt_projs_bias
        # Define out_norm directly with "LN2D"
        self.out_norm = DASSLayerNorm2d(self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))
        del dt_projs
            
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias

    def forward_corev2(
        self,
        x: torch.Tensor,
        force_fp32=False,
        no_einsum=True,
    ):
        B, D, H, W = x.shape
        N = self.d_state
        L = H * W

        xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True)
        x_dbl = self.x_proj(xs.view(B, -1, L))
        dts, Bs, Cs = torch.split(x_dbl.view(B, self.k_group, -1, L), [self.dt_rank, N, N], dim=2)
        dts = dts.contiguous().view(B, -1, L)
        dts = self.dt_projs(dts)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -self.A_logs.to(torch.float32).exp()
        Ds = self.Ds.to(torch.float32)
        Bs = Bs.contiguous().view(B, self.k_group, N, L)
        Cs = Cs.contiguous().view(B, self.k_group, N, L)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float32)
        
        ys = selective_scan_fn(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus=True, backend="mamba"
        ).view(B, self.k_group, -1, H, W)
        
        y = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True)
        y = y.view(B, -1, H, W)
        y = self.out_norm(y)
        return y.to(x.dtype)

    def forward(self, x: torch.Tensor):
        x = self.in_proj(x)
        x = self.conv2d(x)
            
        x = self.act(x)
        y = self.forward_core(x)
        
        out = self.dropout(self.out_proj(y))
        return out


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        ssm_d_state: int = 1,
        ssm_ratio=1.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=False,
        ssm_drop_rate: float = 0,
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        use_checkpoint: bool = False,
        post_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = DASSLayerNorm2d(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
            )
        
        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            self.norm2 = DASSLayerNorm2d(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = DASSMlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate)

    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x))) 
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

class DASSLayer(nn.Module):

    def __init__(
        self,
        input_dim,
        depth,
        drop_path=0.0,
        norm_layer=DASSLayerNorm2d,
        downsample=nn.Identity(),
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                VSSBlock(hidden_dim=input_dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,use_checkpoint=use_checkpoint,**kwargs,
                )
            )
        
        self.downsample = downsample

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        x = self.downsample(x)
        return x

class DASSPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = DASSConfig
    base_model_prefix = "dass"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)


class DASSModel(DASSPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        dims = config.dims
        if isinstance(dims, int):
            dims = [int(dims * 2**i_layer) for i_layer in range(self.num_layers)]

        self.dims = dims
        self.patch_embeddings = DASSPatchEmbeddings(patch_size=config.patch_size,
                                                     embed_dim=dims[0])
        
        self.num_layers = len(config.depths)
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        self.num_features = dims[-1]

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = DASSLayer(
                input_dim=self.dims[i],
                depth=config.depths[i],
                drop_path=dpr[sum(config.depths[:i]):sum(config.depths[:i+1])],
                downsample=DASSDowsample(self.dims[i], self.dims[i+1]) if i < self.num_layers - 1 else nn.Identity(),
                use_checkpoint=config.use_checkpoint,
            )
            self.layers.append(layer)
        
        self.norm = DASSLayerNorm2d(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def get_input_embeddings(self) -> DASSPatchEmbeddings:
        return self.patch_embeddings
    
    def forward(self, input_values: torch.Tensor):
        x = self.patch_embeddings(input_values)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x).flatten(1)
        return x


class DASSForAudioClassification(DASSPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_classes = config.num_classes
        self.dass = DASSModel(config)
        self.head = nn.Linear(self.dass.num_features, self.num_classes) if self.num_classes > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ):

        outputs = self.dass(
            input_values,
        )

        logits = self.head(outputs)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.loss_type == "ce":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "bce":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if return_dict:
            output = (logits,) + (outputs,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs,
        )

__all__ = [
    "DASSModel",
    "DASSPreTrainedModel",
    "DASSForAudioClassification",
]