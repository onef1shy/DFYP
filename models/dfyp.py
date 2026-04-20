from pathlib import Path
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from vit_pytorch import ViT as BaseViT

from .base import ModelBase
from data.loader import DataWrapper, SentinelDataset, USDADataset


VALID_OPERATORS = {"sobel", "scharr", "learnable"}


class DFYPModel(ModelBase):
    """Wrapper for MODIS DFYP training and evaluation."""

    def __init__(
        self,
        in_channels=9,
        num_bins=32,
        hidden_size=128,
        dense_features=None,
        dropout=0.1,
        savedir=Path("checkpoints/modis"),
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        use_locations=False,
        time=32,
        default_operator="learnable",
        year_operator_map=None,
    ):
        self.dropout = dropout
        self.savedir_temp = savedir
        self.default_operator = normalize_operator_name(default_operator)
        self.year_operator_map = normalize_year_operator_map(year_operator_map)

        model = DFYPNet(
            time=time,
            num_periods=32,
            num_bins=num_bins,
            num_classes=1,
            channels=in_channels,
            dense_features=dense_features,
            dim=256,
            depth=4,
            heads=8,
            mlp_dim=512,
            dropout=dropout,
            emb_dropout=0.1,
            default_operator=self.default_operator,
            year_operator_map=self.year_operator_map,
        )

        super().__init__(
            model,
            "DFYP",
            self.savedir_temp,
            device,
        )

    def reinitialize_model(self, time=None):
        """Rebuild the MODIS model for a new temporal input length."""
        if time is None:
            time = 32
        self.__init__(
            dropout=self.dropout,
            savedir=self.savedir_temp,
            time=time,
            default_operator=self.default_operator,
            year_operator_map=self.year_operator_map,
        )


def normalize_operator_name(name):
    """Normalize operator aliases to the public operator names."""
    aliases = {
        "adaptive": "learnable",
        "adaptive_sobel": "learnable",
        "operator_library": "learnable",
    }
    normalized = aliases.get(str(name).lower(), str(name).lower())
    if normalized not in VALID_OPERATORS:
        valid = ", ".join(sorted(VALID_OPERATORS))
        raise ValueError(f"Unknown operator '{name}'. Choose from: {valid}")
    return normalized


def normalize_year_operator_map(mapping):
    """Normalize year-to-operator mappings to integer keys and canonical names."""
    if mapping is None:
        return {}
    normalized = {}
    for year, operator_name in mapping.items():
        normalized[int(year)] = normalize_operator_name(operator_name)
    return normalized


class ConvNet(nn.Module):
    """CNN backbone for the MODIS branch."""

    def __init__(self, in_channels=9, dropout=0.5, dense_features=None, time=32):
        super().__init__()
        self.operator_router = YearOperatorRouter(in_channels=in_channels)
        self.operator_weight = nn.Parameter(torch.tensor(0.5))

        in_out_channels_list = [in_channels, 128, 256, 256, 512, 512, 512]
        stride_list = [None, 1, 2, 1, 2, 1, 2]

        num_divisors = sum(1 if i == 2 else 0 for i in stride_list)
        for _ in range(num_divisors):
            if time % 2 != 0:
                time += 1
            time /= 2

        if dense_features is None:
            dense_features = [2048, 1]
        dense_features = list(dense_features)
        dense_features.insert(0, int(in_out_channels_list[-1] * time * 4))

        self.convblocks = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=in_out_channels_list[i - 1],
                    out_channels=in_out_channels_list[i],
                    kernel_size=3,
                    stride=stride_list[i],
                    dropout=dropout,
                )
                for i in range(1, len(stride_list))
            ]
        )

        self.dense_layers = nn.ModuleList(
            [
                nn.Linear(dense_features[i - 1], dense_features[i])
                for i in range(1, len(dense_features))
            ]
        )

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize convolution and dense layers."""
        for convblock in self.convblocks:
            nn.init.kaiming_uniform_(convblock.conv.weight.data)
            nn.init.constant_(convblock.conv.bias.data, 0)
        for dense_layer in self.dense_layers:
            nn.init.kaiming_uniform_(dense_layer.weight.data)
            nn.init.constant_(dense_layer.bias.data, 0)

    def forward(self, x, return_last_dense=False, year=None):
        """Run the CNN branch with the configured operator router."""
        operator_output = self.operator_router(x, year=year)
        x = self.operator_weight * operator_output + (1 - self.operator_weight) * x

        for block in self.convblocks:
            x = block(x)

        x = x.view(x.shape[0], -1)

        output = None
        for layer_number, dense_layer in enumerate(self.dense_layers):
            x = dense_layer(x)
            if return_last_dense and layer_number == len(self.dense_layers) - 2:
                output = x
        if return_last_dense:
            return x, output
        return x


class ConvBlock(nn.Module):
    """Convolution, normalization, activation, and dropout block."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout):
        super().__init__()
        self.conv = Conv2dSamePadding(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Apply the convolution block."""
        x = self.relu(self.batchnorm(self.conv(x)))
        return self.dropout(x)


class Conv2dSamePadding(nn.Conv2d):
    def forward(self, input):
        """Apply convolution with TensorFlow-style same padding."""
        return conv2d_same_padding(
            input, self.weight, self.bias, self.stride, self.dilation, self.groups
        )


def conv2d_same_padding(input, weight, bias=None, stride=1, dilation=1, groups=1):
    """Run 2D convolution with explicit same-padding behavior."""
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(
        0, (out_rows - 1) * stride[0] + effective_filter_size_rows - input_rows
    )
    rows_odd = padding_rows % 2 != 0

    input_cols = input.size(3)
    filter_cols = weight.size(3)
    effective_filter_size_cols = (filter_cols - 1) * dilation[1] + 1
    out_cols = (input_cols + stride[1] - 1) // stride[1]
    padding_cols = max(
        0, (out_cols - 1) * stride[1] + effective_filter_size_cols - input_cols
    )
    cols_odd = padding_cols % 2 != 0

    if rows_odd or cols_odd:
        input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(
        input,
        weight,
        bias,
        stride,
        padding=(padding_rows // 2, padding_cols // 2),
        dilation=dilation,
        groups=groups,
    )


def pair(t):
    """Convert a scalar into a square tuple."""
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    """Transformer feed-forward block."""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """Apply the feed-forward network."""
        return self.net(x)


class Attention(nn.Module):
    """Multi-head self-attention block."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        """Apply self-attention to token embeddings."""
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
        )
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    """Stacked transformer encoder layers."""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
                )

    def forward(self, x):
        """Run the transformer encoder."""
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViT(nn.Module):
    """Vision Transformer used in the MODIS branch."""

    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {"cls", "mean"}

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        """Encode images into predictions with a ViT backbone."""
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)


class SobelOperator(nn.Module):
    """Fixed Sobel edge operator."""

    def __init__(self, in_channels):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        )
        self.register_buffer(
            "sobel_x", sobel_x.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
        )
        self.register_buffer(
            "sobel_y", sobel_y.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
        )

    def forward(self, x):
        """Apply Sobel filtering to each input channel."""
        grad_x = F.conv2d(x, self.sobel_x, padding=1, groups=x.shape[1])
        grad_y = F.conv2d(x, self.sobel_y, padding=1, groups=x.shape[1])
        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)


class ScharrOperator(nn.Module):
    """Fixed Scharr edge operator."""

    def __init__(self, in_channels):
        super().__init__()
        scharr_x = torch.tensor(
            [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32
        )
        scharr_y = torch.tensor(
            [[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32
        )
        self.register_buffer(
            "scharr_x",
            scharr_x.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1),
        )
        self.register_buffer(
            "scharr_y",
            scharr_y.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1),
        )

    def forward(self, x):
        """Apply Scharr filtering to each input channel."""
        grad_x = F.conv2d(x, self.scharr_x, padding=1, groups=x.shape[1])
        grad_y = F.conv2d(x, self.scharr_y, padding=1, groups=x.shape[1])
        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)


class LearnableOperator(nn.Module):
    """Learnable edge operator interpolating between Sobel and Scharr kernels."""

    def __init__(self, in_channels):
        super().__init__()
        sobel_kernel = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        scharr_kernel = torch.tensor(
            [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        self.eps = 1e-6
        self.in_channels = in_channels

        self.register_buffer("sobel_x", sobel_kernel)
        self.register_buffer("sobel_y", sobel_kernel.transpose(0, 1))
        self.register_buffer("scharr_x", scharr_kernel)
        self.register_buffer("scharr_y", scharr_kernel.transpose(0, 1))

    def forward(self, x):
        """Apply the learned edge operator to the input."""
        alpha = torch.sigmoid(self.alpha)
        kernel_x = alpha * self.sobel_x + (1 - alpha) * self.scharr_x
        kernel_y = alpha * self.sobel_y + (1 - alpha) * self.scharr_y
        kernel_x = kernel_x / torch.abs(kernel_x).max()
        kernel_y = kernel_y / torch.abs(kernel_y).max()
        kernel_x = kernel_x.unsqueeze(0).unsqueeze(0).repeat(self.in_channels, 1, 1, 1)
        kernel_y = kernel_y.unsqueeze(0).unsqueeze(0).repeat(self.in_channels, 1, 1, 1)
        grad_x = F.conv2d(x, kernel_x, padding=1, groups=x.shape[1])
        grad_y = F.conv2d(x, kernel_y, padding=1, groups=x.shape[1])
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + self.eps)
        return grad_magnitude * torch.sigmoid(self.scale)


class YearOperatorRouter(nn.Module):
    """Select the operator used for each MODIS prediction year."""

    def __init__(self, in_channels):
        super().__init__()
        self.operators = nn.ModuleDict(
            {
                "sobel": SobelOperator(in_channels),
                "scharr": ScharrOperator(in_channels),
                "learnable": LearnableOperator(in_channels),
            }
        )
        self.default_operator = "learnable"
        self.year_operator_map = {}

    def configure(self, default_operator="learnable", year_operator_map=None):
        """Update default and year-specific operator assignments."""
        self.default_operator = normalize_operator_name(default_operator)
        self.year_operator_map = normalize_year_operator_map(year_operator_map)

    def forward(self, x, year=None):
        """Route each sample through the configured operator."""
        outputs = {name: op(x) for name, op in self.operators.items()}
        if year is None:
            return outputs[self.default_operator]

        if not isinstance(year, torch.Tensor):
            year_tensor = torch.as_tensor(year, device=x.device)
        else:
            year_tensor = year.to(x.device)
        year_tensor = year_tensor.view(-1).long()

        mixed = torch.zeros_like(next(iter(outputs.values())))
        assigned = torch.zeros(year_tensor.shape[0], dtype=torch.bool, device=x.device)
        for year_value, operator_name in self.year_operator_map.items():
            mask = year_tensor == int(year_value)
            if mask.any():
                mixed[mask] = outputs[operator_name][mask]
                assigned[mask] = True

        if (~assigned).any():
            mixed[~assigned] = outputs[self.default_operator][~assigned]
        return mixed


class SEBlock(nn.Module):
    """Squeeze-and-excitation block for channel reweighting."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced_channels = max(channels // reduction, 1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Apply channel attention to the input tensor."""
        b, c, _, _ = x.size()
        y = self.max_pooling(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DFYPNet(nn.Module):
    """Combined CNN and ViT model for the MODIS branch."""

    expects_year_input = True

    def __init__(
        self,
        *,
        num_periods,
        num_bins,
        num_classes,
        time,
        dense_features=None,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0,
        emb_dropout=0,
        use_locations=False,
        default_operator="learnable",
        year_operator_map=None,
    ):
        super().__init__()
        self.cnn_model = ConvNet(
            in_channels=channels,
            dropout=0.5,
            dense_features=dense_features,
            time=time,
        )
        self.cnn_model.operator_router.configure(
            default_operator=default_operator,
            year_operator_map=year_operator_map,
        )
        self.vit_model = ViT(
            image_size=32,
            patch_size=4,
            num_classes=num_classes,
            channels=channels,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )
        self.se_block = SEBlock(channels=channels)
        self.a1 = nn.Parameter(torch.tensor(0.5))
        self.a2 = nn.Parameter(torch.tensor(0.5))

    def forward(self, img, return_last_dense=False, year=None):
        """Run the MODIS DFYP model."""
        x = self.se_block(img)
        if return_last_dense:
            cnn_result, last_dense = self.cnn_model(
                x, return_last_dense=True, year=year
            )
        else:
            cnn_result = self.cnn_model(x, year=year)
        vit_result = self.vit_model(x)
        output = self.a1 * cnn_result + self.a2 * vit_result
        if return_last_dense:
            return output, last_dense
        return output


class SentinelSobelGate(nn.Module):
    """Sobel-based operator gate for Sentinel-2 inputs."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "kernel_x",
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "kernel_y",
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3),
        )

    def forward(self, x):
        """Apply Sobel filtering to grayscale Sentinel-2 imagery."""
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        gray = gray.unsqueeze(1)
        gx = F.conv2d(gray, self.kernel_x, padding=1)
        gy = F.conv2d(gray, self.kernel_y, padding=1)
        grad = torch.sqrt(gx**2 + gy**2 + 1e-8)
        return grad.expand_as(x)


class SentinelScharrGate(nn.Module):
    """Scharr-based operator gate for Sentinel-2 inputs."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "kernel_x",
            torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "kernel_y",
            torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32).view(1, 1, 3, 3),
        )

    def forward(self, x):
        """Apply Scharr filtering to grayscale Sentinel-2 imagery."""
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        gray = gray.unsqueeze(1)
        gx = F.conv2d(gray, self.kernel_x, padding=1)
        gy = F.conv2d(gray, self.kernel_y, padding=1)
        grad = torch.sqrt(gx**2 + gy**2 + 1e-8)
        return grad.expand_as(x)


class SentinelCropOperatorGate(nn.Module):
    """Apply the crop-specific fixed operator to Sentinel-2 inputs."""

    def __init__(self, operator_type="sobel"):
        super().__init__()
        self.operator_type = operator_type
        self.operator_weight = nn.Parameter(torch.tensor(0.5))
        self.sobel = SentinelSobelGate()
        self.scharr = SentinelScharrGate()

    def forward(self, x):
        """Blend the selected operator response with the original input."""
        grad = self.sobel(x) if self.operator_type == "sobel" else self.scharr(x)
        return self.operator_weight * grad + (1 - self.operator_weight) * x


class SentinelConvNet(nn.Module):
    """CNN backbone for the Sentinel-2 branch."""

    def __init__(self, in_channels=3, out_dim=2, dropout=0.5, batch_size=16, operator_type="sobel"):
        super().__init__()
        self.batch_size = batch_size
        dim = 128
        self.operator_gate = SentinelCropOperatorGate(operator_type=operator_type)

        channels = [in_channels, 32, 64, 128, dim]
        cnn_layers = []
        for i in range(len(channels) - 1):
            cnn_layers.extend(
                [
                    nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(dropout / 2) if i > 0 else nn.Identity(),
                    nn.MaxPool2d(2) if i < len(channels) - 2 else nn.AdaptiveAvgPool2d((1, 1)),
                ]
            )
        self.cnn = nn.Sequential(*cnn_layers)

        fusion_dims = [(dim, dim), (dim, dim // 2), (dim // 2, out_dim)]
        fusion_layers = []
        for in_dim, out_dim_ in fusion_dims[:-1]:
            fusion_layers.extend([nn.Linear(in_dim, out_dim_), nn.ReLU(), nn.Dropout(dropout)])
        fusion_layers.append(nn.Linear(fusion_dims[-1][0], fusion_dims[-1][1]))
        self.fusion = nn.Sequential(*fusion_layers)
        self.dropout = nn.Dropout(dropout)
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize convolution and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        """Extract CNN features from Sentinel-2 time series inputs."""
        b, t, g = x.shape[:3]
        x = rearrange(x, "b t g c h w -> (b t g) c h w")
        B = x.shape[0]
        n = B // self.batch_size if B % self.batch_size == 0 else B // self.batch_size + 1
        x_hat = []
        for i in range(n):
            start, end = i * self.batch_size, (i + 1) * self.batch_size
            x_tmp = self.operator_gate(x[start:end])
            x_hat.append(self.cnn(x_tmp))
        return torch.cat(x_hat, dim=0).squeeze(-1).squeeze(-1)

    def forward(self, x):
        """Run the Sentinel-2 CNN branch."""
        b, t, g = x.shape[:3]
        x = self.forward_features(x)
        x = rearrange(x, "(b t g) d -> b (t g) d", b=b, t=t, g=g)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fusion(x)


class SentinelSEBlock(nn.Module):
    """Squeeze-and-excitation block for Sentinel-2 inputs."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced_channels = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Apply channel attention to Sentinel-2 images."""
        b, c, h, w = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SentinelViT(BaseViT):
    """Vision Transformer backbone for the Sentinel-2 branch."""

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool="cls", channels=3, dim_head=32, dropout=0, emb_dropout=0, batch_size=16):
        super().__init__(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool, channels=channels, dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout)
        self.batch_size = batch_size
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize the linear layers in the ViT."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        """Extract ViT features from Sentinel-2 time series inputs."""
        b, t, g = x.shape[:3]
        x = rearrange(x, "b t g c h w -> (b t g) c h w")
        B = x.shape[0]
        n = B // self.batch_size if B % self.batch_size == 0 else B // self.batch_size + 1
        x_hat = []
        for i in range(n):
            start, end = i * self.batch_size, (i + 1) * self.batch_size
            x_tmp = x[start:end]
            x_tmp = self.to_patch_embedding(x_tmp)
            batch, num_tokens, _ = x_tmp.shape
            cls_token = self.cls_token
            if cls_token.dim() == 2:
                cls_token = cls_token.unsqueeze(1)
            cls_tokens = repeat(cls_token, "1 1 d -> b 1 d", b=batch)
            x_tmp = torch.cat((cls_tokens, x_tmp), dim=1)
            x_tmp += self.pos_embedding[:, : (num_tokens + 1)]
            x_tmp = self.dropout(x_tmp)
            x_tmp = self.transformer(x_tmp)
            x_tmp = x_tmp.mean(dim=1) if self.pool == "mean" else x_tmp[:, 0]
            x_tmp = self.to_latent(x_tmp)
            x_hat.append(x_tmp)
        return torch.cat(x_hat, dim=0)

    def forward(self, x):
        """Run the Sentinel-2 ViT branch."""
        b, t, g = x.shape[:3]
        x = self.forward_features(x)
        x = rearrange(x, "(b t g) d -> b (t g) d", b=b, t=t, g=g)
        x = x.mean(dim=1)
        return self.mlp_head(x)


class SentinelDFYPNet(nn.Module):
    """Combined CNN and ViT model for the Sentinel-2 branch."""

    def __init__(self, in_channels=3, out_dim=2, dim=128, depth=6, heads=6, mlp_dim=256, dropout=0.5, batch_size=8, operator_type="sobel"):
        super().__init__()
        self.se = SentinelSEBlock(in_channels)
        self.cnn_model = SentinelConvNet(in_channels=in_channels, out_dim=out_dim, dropout=dropout, batch_size=batch_size, operator_type=operator_type)
        self.vit_model = SentinelViT(image_size=224, patch_size=16, num_classes=out_dim, channels=in_channels, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout, batch_size=batch_size)
        self.a1 = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.a2 = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize learnable parameters in the Sentinel-2 model."""
        for m in self.se.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.a1, 0.5)
        nn.init.constant_(self.a2, 0.5)
        self.cnn_model.initialize_weights()
        self.vit_model.initialize_weights()

    def forward(self, x):
        """Run the Sentinel-2 DFYP model."""
        b, t, g, c, h, w = x.shape
        x = rearrange(x, "b t g c h w -> (b t g) c h w")
        x = self.se(x)
        x = rearrange(x, "(b t g) c h w -> b t g c h w", b=b, t=t, g=g)
        cnn_result = self.cnn_model(x)
        vit_result = self.vit_model(x)
        return self.a1 * cnn_result + self.a2 * vit_result


class SentinelDFYPRunner:
    """Training and evaluation runner for Sentinel-2 experiments."""

    def __init__(self, crop, operator_type, savedir, device):
        self.crop = crop.lower()
        self.operator_type = operator_type
        self.device = device
        self.savedir = Path(savedir) / self.crop
        self.savedir.mkdir(parents=True, exist_ok=True)
        self.model = SentinelDFYPNet(operator_type=operator_type)
        if device.type != "cpu":
            self.model = self.model.cuda()
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    def _resolve_json_file(self, json_dir, split):
        """Resolve train or test split files for a specific crop."""
        json_dir = Path(json_dir)
        candidates = [
            json_dir / self.crop / f"{self.crop}_{split}.json",
            json_dir / f"{self.crop}_{split}.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def run(self, root_dir, json_dir, train_steps=25000, batch_size=32, starter_learning_rate=1e-4, weight_decay=1e-5, patience=10):
        """Train the Sentinel-2 model and save the final checkpoint."""
        year = 2022
        begin_time = time.time()
        self.model.initialize_weights()
        rmse, me, mae, r2 = self._run_prediction(
            target_year=year,
            train_steps=train_steps,
            batch_size=batch_size,
            starter_learning_rate=starter_learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            json_dir=json_dir,
            root_dir=root_dir,
        )
        end_time = time.time()
        pd.DataFrame(
            {"year": [year], "RMSE": [rmse], "ME": [me], "MAE": [mae], "R_2": [r2], "time": [end_time - begin_time]}
        ).to_csv(self.savedir / "results.csv", index=False)

    def _run_prediction(self, target_year, train_steps, batch_size, starter_learning_rate, weight_decay, patience, json_dir, root_dir):
        """Run a single Sentinel-2 training/evaluation cycle."""
        train_file = self._resolve_json_file(json_dir, "train")
        test_file = self._resolve_json_file(json_dir, "test")

        train_loader_sentinel = DataLoader(SentinelDataset(root_dir, train_file, is_train=True), batch_size=1, shuffle=False, drop_last=True)
        train_loader_usda = DataLoader(USDADataset(root_dir, train_file, crop_type=self.crop, is_train=True), batch_size=1, drop_last=True)
        val_loader_sentinel = DataLoader(SentinelDataset(root_dir, train_file, is_train=False), batch_size=1, shuffle=False, drop_last=True)
        val_loader_usda = DataLoader(USDADataset(root_dir, train_file, crop_type=self.crop, is_train=False), batch_size=1, drop_last=True)
        test_loader_sentinel = DataLoader(SentinelDataset(root_dir, test_file, is_train=None), batch_size=1, shuffle=False, drop_last=True)
        test_loader_usda = DataLoader(USDADataset(root_dir, test_file, crop_type=self.crop, is_train=None), batch_size=1, drop_last=True)

        criterion = nn.MSELoss().to(device=self.device)
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=starter_learning_rate, weight_decay=weight_decay)

        train_scores, val_scores = self._train(
            criterion, optimizer, train_loader_sentinel, train_loader_usda, val_loader_sentinel, val_loader_usda, train_steps, batch_size, patience
        )
        results = self._test(criterion, test_loader_sentinel, test_loader_usda)
        model_information = {"state_dict": self.model.state_dict(), "val_loss": val_scores["loss"], "train_loss": train_scores["loss"], **results}
        torch.save(model_information, self.savedir / f"model_{target_year}.pth")
        return self.analyze_results(model_information["test_real"], model_information["test_pred"])

    def evaluate_checkpoint(self, checkpoint_path, root_dir, json_dir):
        """Evaluate a saved Sentinel-2 checkpoint on the test split."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        test_file = self._resolve_json_file(json_dir, "test")
        test_loader_sentinel = DataLoader(
            SentinelDataset(root_dir, test_file, is_train=None),
            batch_size=1,
            shuffle=False,
            drop_last=True,
        )
        test_loader_usda = DataLoader(
            USDADataset(root_dir, test_file, crop_type=self.crop, is_train=None),
            batch_size=1,
            drop_last=True,
        )

        criterion = nn.MSELoss().to(device=self.device)
        results = self._test(criterion, test_loader_sentinel, test_loader_usda)
        rmse, me, mae, r_2 = self.analyze_results(
            results["test_real"],
            results["test_pred"],
        )
        return {
            "test_real": results["test_real"],
            "test_pred": results["test_pred"],
            "rmse": rmse,
            "me": me,
            "mae": mae,
            "r_2": r_2,
        }

    def _train(self, criterion, optimizer, data_loader_sentinel, data_loader_usda, val_loader_sentinel, val_loader_usda, train_steps, batch_size, patience):
        """Train the Sentinel-2 model with validation-based early stopping."""
        best_val_loss = float("inf")
        best_model = None
        patience_counter = 0
        step_number = 0
        train_scores = defaultdict(list)
        val_scores = defaultdict(list)
        num_epochs = int(train_steps / (len(data_loader_sentinel) / batch_size))

        for _ in range(num_epochs):
            self.model.train()
            total_step = len(data_loader_sentinel) - 1
            total_loss = 0
            optimizer.zero_grad()
            data_wrapper = DataWrapper()

            for x, z in tqdm(zip(data_loader_sentinel, data_loader_usda), total=total_step, desc="Training", leave=True):
                if z[0].size(1) == 0:
                    continue
                x = x[0].to(self.device, non_blocking=True)
                z = z[0].to(self.device, non_blocking=True)
                b, t, g, _, _, _ = x.shape
                x = rearrange(x, "b t g h w c -> (b t g) c h w")
                x, _ = data_wrapper(x)
                x = rearrange(x, "(b t g) c h w -> b t g c h w", b=b, t=t, g=g)
                z_hat = self.model(x)
                loss = criterion(z, z_hat)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                step_number += 1
                if step_number in [4000, 20000]:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] /= 10

            avg_train_loss = total_loss / total_step
            train_scores["loss"].append(avg_train_loss)

            self.model.eval()
            total_val_loss = 0
            val_steps = 0
            with torch.no_grad():
                for x, z in tqdm(zip(val_loader_sentinel, val_loader_usda), total=len(val_loader_sentinel), desc="Validating", leave=True):
                    if z[0].size(1) == 0:
                        continue
                    x = x[0].to(self.device, non_blocking=True)
                    z = z[0].to(self.device, non_blocking=True)
                    b, t, g, _, _, _ = x.shape
                    x = rearrange(x, "b t g h w c -> (b t g) c h w")
                    x, _ = data_wrapper(x)
                    x = rearrange(x, "(b t g) c h w -> b t g c h w", b=b, t=t, g=g)
                    z_hat = self.model(x)
                    total_val_loss += criterion(z, z_hat).item()
                    val_steps += 1

            avg_val_loss = total_val_loss / val_steps
            val_scores["loss"].append(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter == patience:
                    break

        self.model.load_state_dict(best_model)
        return train_scores, val_scores

    def _test(self, criterion, test_loader_sentinel, test_loader_usda):
        """Run inference on the Sentinel-2 test split."""
        self.model.eval()
        total_test_loss = 0
        test_steps = 0
        all_outputs = []
        all_targets = []
        data_wrapper = DataWrapper()
        with torch.no_grad():
            for x, z in zip(test_loader_sentinel, test_loader_usda):
                if z[0].size(1) == 0:
                    continue
                x = x[0].to(self.device, non_blocking=True)
                z = z[0].to(self.device, non_blocking=True)
                b, t, g, _, _, _ = x.shape
                x = rearrange(x, "b t g h w c -> (b t g) c h w")
                x, _ = data_wrapper(x)
                x = rearrange(x, "(b t g) c h w -> b t g c h w", b=b, t=t, g=g)
                z_hat = self.model(x)
                total_test_loss += criterion(z, z_hat).item()
                test_steps += 1
                all_outputs.append(z_hat.cpu().numpy())
                all_targets.append(z.cpu().numpy())

        return {
            "test_loss": total_test_loss / test_steps,
            "test_real": np.concatenate(all_targets, axis=0),
            "test_pred": np.concatenate(all_outputs, axis=0),
        }

    @staticmethod
    def analyze_results(true, pred):
        """Compute RMSE, ME, MAE, and R^2 for Sentinel-2 predictions."""
        rmse = np.sqrt(np.mean((true - pred) ** 2))
        me = np.mean(true - pred)
        mae = np.mean(np.abs(true - pred))
        r_2 = 1 - np.sum((true - pred) ** 2) / np.sum((true - np.mean(true)) ** 2)
        return rmse, me, mae, r_2
