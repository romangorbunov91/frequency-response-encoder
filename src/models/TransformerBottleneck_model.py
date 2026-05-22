import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from typing_extensions import Self

class ResConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout=0.1
        ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(
                num_groups=1,
                num_channels=in_channels
                ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
                ),
            nn.GroupNorm(
                num_groups=1,
                num_channels=out_channels
                ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
                ),
            nn.Dropout1d(p=dropout)
        )
        self.skip = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
            ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.skip(x)

class AttentionGate(nn.Module):
    """Attention Gate for Skip Connections (Attention U-Net style)"""
    def __init__(
        self,
        F_g: int,
        F_l: int,
        F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(
                in_channels=F_g,
                out_channels=F_int,
                kernel_size=1,
                bias=False),
            nn.GroupNorm(
                num_groups=1,
                num_channels=F_int)
            )
        self.W_x = nn.Sequential(
            nn.Conv1d(
                in_channels=F_l,
                out_channels=F_int,
                kernel_size=1,
                bias=False),
            nn.GroupNorm(
                num_groups=1,
                num_channels=F_int)
            )
        self.psi = nn.Sequential(
            nn.Conv1d(
                in_channels=F_int,
                out_channels=1,
                kernel_size=1,
                bias=False),
            nn.GroupNorm(
                num_groups=1,
                num_channels=1),
            nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):    
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class TransformerBottleneck(nn.Module):
    """Lightweight Transformer for Global Context Modeling"""
    def __init__(
        self,
        channels: int,
        num_heads: int=8,
        mlp_ratio=4,
        dropout=0.1
        ):
        super().__init__()
        self.norm1 = nn.GroupNorm(
            num_groups=1,
            num_channels=channels
            )
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.GroupNorm(
            num_groups=1,
            num_channels=channels
            )
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=channels,
                out_features=int(channels * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=int(channels * mlp_ratio),
                out_features=channels),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        # x: (B, C, L)
        x_norm = self.norm1(x)
        x_t = x_norm.transpose(1, 2)  # (B, L, C)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        x = x + attn_out.transpose(1, 2)
        
        x_norm2 = self.norm2(x)
        mlp_out = self.mlp(x_norm2.transpose(1, 2)).transpose(1, 2)
        return x + mlp_out

class UpSample(nn.Module):
    """Artifact-free Upsampling: Interpolation + Conv"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int=2):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='nearest'),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.GroupNorm(
                num_groups=1,
                num_channels=out_channels),
            nn.GELU()
        )
    def forward(self, x): return self.up(x)


class _TransformerBottleneck_model(nn.Module):

    def __init__(self,
                in_channels: int,
                out_channels: int,
                features: List[int],
                deep_supervision: bool=True
                ):
        super(_TransformerBottleneck_model, self).__init__()
        
        self.deep_supervision = deep_supervision
        self.input_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=features[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
                ),
            nn.GroupNorm(
                num_groups=1,
                num_channels=features[0]),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=features[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
                )
        )
        # Encoder.
        self.enc1 = ResConvBlock(features[0], features[1])
        self.enc2 = ResConvBlock(features[1], features[2])
        self.enc3 = ResConvBlock(features[2], features[3])
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Bottleneck: Project to higher dim + Global Transformer.
        self.bottleneck_proj = nn.Sequential(
            nn.Conv1d(
                in_channels=features[3],
                out_channels=features[4],
                kernel_size=1,
                bias=False),
            nn.GroupNorm(
                num_groups=1,
                num_channels=features[4]),
            nn.GELU()
        )
        self.bottleneck_attn = TransformerBottleneck(
            channels=features[4],
            num_heads=8,
            mlp_ratio=4,
            dropout=0.1)

        # Decoder with Attention Gated Skip Connections.
        self.up3 = UpSample(features[4], features[3])
        self.att3 = AttentionGate(features[3], features[3], features[3]//2)
        self.dec3 = ResConvBlock(features[3]*2, features[3])

        self.up2 = UpSample(features[3], features[2])
        self.att2 = AttentionGate(features[2], features[2], features[2]//2)
        self.dec2 = ResConvBlock(features[2]*2, features[2])

        self.up1 = UpSample(features[2], features[1])
        self.att1 = AttentionGate(features[1], features[1], features[1]//2)
        self.dec1 = ResConvBlock(features[1]*2, features[1])

        # Prediction Heads
        self.final_conv = nn.Conv1d(
            in_channels=features[1],
            out_channels=out_channels,
            kernel_size=1)
        self.ds_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=features[3],
                out_channels=out_channels,
                kernel_size=1),
            nn.Conv1d(
                in_channels=features[2],
                out_channels=out_channels,
                kernel_size=1)
        ])

    def forward(self, x):
        # Encoder
        e1 = self.enc1(self.input_conv(x))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck_attn(self.bottleneck_proj(self.pool(e3)))

        # Decoder
        g3 = self.up3(b)
        d3 = self.dec3(torch.cat([g3, self.att3(g3, e3)], dim=1))

        g2 = self.up2(d3)
        d2 = self.dec2(torch.cat([g2, self.att2(g2, e2)], dim=1))

        g1 = self.up1(d2)
        d1 = self.dec1(torch.cat([g1, self.att1(g1, e1)], dim=1))

        out_main = self.final_conv(d1)

        if self.training and self.deep_supervision:
            ds_outs = [
                self.ds_convs[0](d3),
                self.ds_convs[1](d2)
            ]
            return out_main, ds_outs
        return out_main

def TransformerBottleneck_model(
    in_channels: int,
    out_channels: int,
    features: List[int]
    ):
    return _TransformerBottleneck_model(
        in_channels = in_channels,
        out_channels = out_channels,
        features = features
    )