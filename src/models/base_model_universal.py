import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class ResConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int=3,
        stride: int=1,
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
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
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
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
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
            mlp_ratio: float=4.0,
            dropout: float=0.1
            ):
        super().__init__()
        '''
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.dropout = dropout
        
        # 1. Manual Linear projections for Q, K, V and Output
        self.q_proj = nn.Linear(
            in_features=channels,
            out_features=channels,
            bias=False
        )
        self.k_proj = nn.Linear(
            in_features=channels,
            out_features=channels,
            bias=False
        )
        self.v_proj = nn.Linear(
            in_features=channels,
            out_features=channels,
            bias=False
        )
        self.out_proj = nn.Linear(
            in_features=channels,
            out_features=channels,
            bias=False
        )
        '''

        # BEGIN delete section.
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        # END delete section.
        
        self.norm_attn = nn.GroupNorm(
            num_groups=1,
            num_channels=channels
            )

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
        B, C, L = x.shape
        x_norm = self.norm_attn(x)
        x_t = x_norm.transpose(1, 2) # (B, L, C)
        
        # BEGIN delete section.
        attn_out, _ = self.attn(x_t, x_t, x_t)
        # END delete section.

        '''
        # Project and reshape for multi-head attention
        # (B, L, C) -> (B, L, num_heads, head_dim) -> (B, num_heads, L, head_dim).
        q = self.q_proj(x_t).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_t).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_t).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_out = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout if self.training else 0.0, 
            is_causal=False
        )
        
        # Reshape back and apply output projection.
        # (B, num_heads, L, head_dim) -> (B, L, C)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, C)
        attn_out = self.out_proj(attn_out)
        '''
        
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
            num_heads: int=8,
            mlp_ratio: float=4.0,
            transformer_dropout: float=0.1,
            conv_dropout: float=0.1,
            deep_supervision: bool=True,
            use_attention_gate: bool=True,
            use_skip_connection: bool=True
            ):
        super(_base_model, self).__init__()
        self.deep_supervision = deep_supervision
        self.use_attention_gate = use_attention_gate
        self.use_skip_connection = use_skip_connection
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
        self.encoders.append(
            ResConvBlock(
                in_channels=features[0],
                out_channels=features[0],
                dropout=conv_dropout
                ))
        for i in range(1, self.num_levels):
            self.encoders.append(
                ResConvBlock(
                    in_channels=features[i-1],
                    out_channels=features[i],
                    dropout=conv_dropout
                ))

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
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=transformer_dropout
            )

        # Decoder with Attention Gated Skip Connections.
        self.upsamples = nn.ModuleList()
        self.att_gates = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ds_convs = nn.ModuleList()
        
        for i in range(self.num_levels - 1, -1, -1):
            self.upsamples.append(
                UpSample(
                    in_channels=features[i+1],
                    out_channels=features[i],
                    scale_factor=2
                ))
            
            if self.use_skip_connection:
                if self.use_attention_gate:
                    self.att_gates.append(
                        AttentionGate(
                            F_g=features[i],
                            F_l=features[i],
                            F_int=features[i] // 2
                        ))
                decoder_in_channels = features[i] * 2
            else:
                decoder_in_channels = features[i]
            
            self.decoders.append(
                ResConvBlock(
                    in_channels=decoder_in_channels,
                    out_channels=features[i],
                    dropout=conv_dropout
                ))
            
            # Deep supervision for all decoder stages except the final one (i=0)
            if i > 0:
                self.ds_convs.append(
                    nn.Conv1d(
                        in_channels=features[i],
                        out_channels=out_channels,
                        kernel_size=1
                        ))
        
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
        if self.deep_supervision:
            ds_outs = []
        
        d_prev = b
        for i, (up, dec) in enumerate(zip(self.upsamples, self.decoders)):
            skip_idx = len(enc_outs) - 1 - i
            g = up(d_prev)
            
            if self.use_skip_connection:
                skip = enc_outs[skip_idx]
                if self.use_attention_gate:
                    skip = self.att_gates[i](g, skip)
                d_prev = dec(torch.cat([g, skip], dim=1))
            else:
                d_prev = dec(g)

            if self.deep_supervision:
                # Collect deep supervision outputs.
                if i < len(self.ds_convs):
                    ds_outs.append(self.ds_convs[i](d_prev))

        out_main = self.final_conv(d_prev)

        if self.training and self.deep_supervision:
            return out_main, ds_outs
        return out_main

def base_model(
    in_channels: int,
    out_channels: int,
    features: List[int],
    input_conv_kernel_size: int=5,
    num_heads: int=8,
    mlp_ratio: float=4.0,
    transformer_dropout: float=0.1,
    conv_dropout: float=0.1,
    deep_supervision: bool=True,
    use_attention_gate: bool=True,
    use_skip_connection: bool=True
    ) -> _base_model:
    return _base_model(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        input_conv_kernel_size=input_conv_kernel_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        transformer_dropout=transformer_dropout,
        conv_dropout=conv_dropout,
        deep_supervision=deep_supervision,
        use_attention_gate=use_attention_gate,
        use_skip_connection=use_skip_connection
        )