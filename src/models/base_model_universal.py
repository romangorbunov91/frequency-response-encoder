import torch
import torch.nn as nn
from typing import List, Optional

class ResConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float=0.1
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.skip(x)

class AttentionGate(nn.Module):
    def __init__(
        self,
        F_g: int,
        F_l: int,
        F_int: Optional[int] = None
        ):
        super().__init__()
        F_int = F_int or (F_g // 2)
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

    def forward(self,
            g: torch.Tensor,
            x: torch.Tensor
            ) -> torch.Tensor:   
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))

class TransformerBottleneck(nn.Module):
    def __init__(self,
            channels: int,
            num_heads: int=8,
            mlp_ratio: float=4,
            dropout: float=0.1
            ):
        super().__init__()
        self.norm_attn = nn.GroupNorm(
            num_groups=1,
            num_channels=channels
            )
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.GroupNorm(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        x_norm = self.norm_attn(x)
        x_t = x_norm.transpose(1, 2) # (B, L, C)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        x = x + attn_out.transpose(1, 2)
        
        x_norm_mlp = self.norm_mlp(x)
        mlp_out = self.mlp(x_norm_mlp.transpose(1, 2)).transpose(1, 2)
        return x + mlp_out

class UpSample(nn.Module):
    def __init__(self,
            in_channels: int,
            out_channels: int,
            scale_factor: int=2
            ):
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)

class _base_model(nn.Module):

    def __init__(self,
            in_channels: int,
            out_channels: int,
            features: List[int],
            input_conv_kernel_size: int,
            deep_supervision: bool=True
            ):
        super(_base_model, self).__init__()
        self.deep_supervision = deep_supervision
        self.num_levels = len(features) - 1
        
        # Initial projection.
        self.input_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=features[0],
            kernel_size=input_conv_kernel_size,
            stride=1,
            padding=input_conv_kernel_size // 2,
            bias=False
            )
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Encoder.
        self.encoders = nn.ModuleList()
        self.encoders.append(ResConvBlock(features[0], features[0]))
        for i in range(1, self.num_levels):
            self.encoders.append(ResConvBlock(features[i-1], features[i]))

        # Bottleneck: Project to higher dim + Global Transformer.
        self.bottleneck_proj = nn.Sequential(
            nn.Conv1d(
                in_channels=features[-2],
                out_channels=features[-1],
                kernel_size=1,
                bias=False),
            nn.GroupNorm(
                num_groups=1,
                num_channels=features[-1]),
            nn.GELU()
        )
        self.bottleneck_attn = TransformerBottleneck(
            channels=features[-1],
            num_heads=8,
            mlp_ratio=4,
            dropout=0.1
            )

        # Decoder with Attention Gated Skip Connections.
        self.upsamples = nn.ModuleList()
        self.att_gates = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ds_convs = nn.ModuleList()
        
        for i in range(self.num_levels - 1, -1, -1):
            self.upsamples.append(UpSample(features[i+1], features[i]))
            self.att_gates.append(AttentionGate(features[i], features[i]))  # F_int defaults to features[i] // 2
            self.decoders.append(ResConvBlock(features[i] * 2, features[i]))
            
            # Deep supervision for all decoder stages except the final one (i=0)
            if i > 0:
                self.ds_convs.append(nn.Conv1d(features[i], out_channels, 1))
        '''
        self.up4 = UpSample(features[4], features[3])
        self.att4 = AttentionGate(features[3], features[3], features[3]//2)
        self.dec4 = ResConvBlock(features[3]*2, features[3])

        self.up3 = UpSample(features[3], features[2])
        self.att3 = AttentionGate(features[2], features[2], features[2]//2)
        self.dec3 = ResConvBlock(features[2]*2, features[2])

        self.up2 = UpSample(features[2], features[1])
        self.att2 = AttentionGate(features[1], features[1], features[1]//2)
        self.dec2 = ResConvBlock(features[1]*2, features[1])

        self.up1 = UpSample(features[1], features[0])
        self.att1 = AttentionGate(features[0], features[0], features[0]//2)
        self.dec1 = ResConvBlock(features[0]*2, features[0])

        self.ds_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=features[3],
                out_channels=out_channels,
                kernel_size=1),
            nn.Conv1d(
                in_channels=features[2],
                out_channels=out_channels,
                kernel_size=1),
            nn.Conv1d(
                in_channels=features[1],
                out_channels=out_channels,
                kernel_size=1)
        ])
        '''
        # Prediction Head.
        self.final_conv = nn.Conv1d(
            in_channels=features[0],
            out_channels=out_channels,
            kernel_size=1)

    def forward(self, x):
        # Encoder.
        x = self.input_conv(x)
        enc_outs = []
        for enc in self.encoders:
            x = enc(x)
            enc_outs.append(x)
            x = self.pool(x)
        
        # Bottleneck.
        b = self.bottleneck_attn(self.bottleneck_proj(x))
        
        # Decoder
        ds_outs = []
        d_prev = b
        for i, (up, att, dec) in enumerate(zip(self.upsamples, self.att_gates, self.decoders)):
            skip_idx = len(enc_outs) - 1 - i
            g = up(d_prev)
            d_prev = dec(torch.cat([g, att(g, enc_outs[skip_idx])], dim=1))
            
            # Collect deep supervision outputs
            if i < len(self.ds_convs):
                ds_outs.append(self.ds_convs[i](d_prev))

        out_main = self.final_conv(d_prev)

        if self.training and self.deep_supervision:
            return out_main, ds_outs
        return out_main
    
    
        e1 = self.enc1(self.input_conv(x))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck.
        b = self.bottleneck_attn(self.bottleneck_proj(self.pool(e4)))

        # Decoder.
        g4 = self.up4(b)
        d4 = self.dec4(torch.cat([g4, self.att4(g4, e4)], dim=1))

        g3 = self.up3(d4)
        d3 = self.dec3(torch.cat([g3, self.att3(g3, e3)], dim=1))

        g2 = self.up2(d3)
        d2 = self.dec2(torch.cat([g2, self.att2(g2, e2)], dim=1))

        g1 = self.up1(d2)
        d1 = self.dec1(torch.cat([g1, self.att1(g1, e1)], dim=1))

        out_main = self.final_conv(d1)

        if self.training and self.deep_supervision:
            ds_outs = [
                self.ds_convs[0](d4),
                self.ds_convs[1](d3),
                self.ds_convs[2](d2)
            ]
            return out_main, ds_outs
        return out_main

def base_model(
    in_channels: int,
    out_channels: int,
    features: List[int],
    input_conv_kernel_size: int,
    deep_supervision: bool = True
    ) -> _base_model:
    return _base_model(
        in_channels = in_channels,
        out_channels = out_channels,
        features = features,
        input_conv_kernel_size = input_conv_kernel_size,
        deep_supervision = deep_supervision
    )