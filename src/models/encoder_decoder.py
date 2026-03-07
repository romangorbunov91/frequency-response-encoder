import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from typing_extensions import Self

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class _customUNet(nn.Module):
    
    def freeze_encoder(self):
        """Freeze all encoder (backbone) parameters."""
        for block in self.encoder_blocks:
            for param in block.parameters():
                param.requires_grad = False
        self.encoder_frozen = True
        
    def unfreeze_encoder(self):
        """Unfreeze all encoder (backbone) parameters."""
        for block in self.encoder_blocks:
            for param in block.parameters():
                param.requires_grad = True
        self.encoder_frozen = False
    '''
    def train(self, mode: bool = True) -> Self:
        """
        Sets the module in training mode.
        Overrides default behavior to keep frozen encoder in eval mode.
        """
        super().train(mode)

        if self.encoder_frozen:
            for block in self.encoder_blocks:
                block.eval()
        return self
    '''
    def __init__(self,
                in_channels: int,
                out_channels: int,
                features: List[int]
                ):
        super(_customUNet, self).__init__()
        
        self.encoder_frozen = False
        
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        prev_channels = in_channels
        for feature in features:
            self.encoder_blocks.append(
                DoubleConv(prev_channels, feature)
                )
            prev_channels = feature

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for feature in reversed(features):
            self.decoder_blocks.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(
                DoubleConv(feature * 2, feature)
            )

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        print(f"Encoder features by level: {features}")
    
    def forward(self, x):

        skip_connections = []

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)

        # Reverse order.
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoder_blocks), 2):
            x = self.decoder_blocks[idx](x)
            
            skip_connection = skip_connections[idx // 2]

            x = torch.cat([skip_connection, x], dim=1)
            
            x = self.decoder_blocks[idx + 1](x)

        x = self.final_conv(x)

        return x

def customUNet(
    in_channels: int,
    out_channels: int,
    features: List[int]
    ):
    return _customUNet(
        in_channels = in_channels,
        out_channels = out_channels,
        features = features
    )