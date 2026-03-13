import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from typing_extensions import Self

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class _base_model(nn.Module):

    def __init__(self,
                in_channels: int,
                out_channels: int,
                features: List[int]
                ):
        super(_base_model, self).__init__()


        self.initial_conv = DoubleConv(in_channels, features[0])
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(features[0], features[0] * 2)
        self.transpose = nn.ConvTranspose1d(features[0] * 2, features[-1], kernel_size=2, stride=2)
        self.final_conv = nn.Conv1d(features[-1], out_channels, kernel_size=1)
        
        print(f"Encoder features by level: {features}")
    
    def forward(self, x):

        x = self.initial_conv(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        x = self.transpose(x)
        x = self.final_conv(x)

        return x

def base_model(
    in_channels: int,
    out_channels: int,
    features: List[int]
    ):
    return _base_model(
        in_channels = in_channels,
        out_channels = out_channels,
        features = features
    )