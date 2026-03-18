import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from typing_extensions import Self

class _base_model(nn.Module):

    def __init__(self,
                in_channels: int,
                out_channels: int,
                features: List[int]
                ):
        super(_base_model, self).__init__()

        self.initial_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=features[0],
            kernel_size=3,
            padding=1
            )
        self.BN1 = nn.BatchNorm1d(num_features=features[0])
        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.transpose = nn.ConvTranspose1d(features[0], features[-1], kernel_size=2, stride=2)
        self.final_conv = nn.Conv1d(features[-1], out_channels, kernel_size=1)
        
        print(f"Encoder features by level: {features}")
    
    def forward(self, x):

        x = self.initial_conv(x)
        x = self.BN1(x)
        x = self.activation(x)
        x = self.pool(x)
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