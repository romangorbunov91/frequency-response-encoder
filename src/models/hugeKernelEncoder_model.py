import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from typing_extensions import Self

class _hugeKernelEncoder_model(nn.Module):

    def __init__(self,
                in_channels: int,
                out_channels: int,
                features: List[int]
                ):
        super(_hugeKernelEncoder_model, self).__init__()

        self.input_norm = nn.InstanceNorm1d(
            num_features=in_channels, 
            affine=True  # Allows model to learn optimal scale/shift.
        )
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(num_features=in_channels),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=features[0],
                kernel_size=511,
                stride=1,
                padding=0
            ),
            nn.BatchNorm1d(num_features=features[0]),
            nn.ReLU(inplace=True)
        )

        self.bottleneck = nn.Sequential(
            nn.Conv1d(
                in_channels=features[-1],
                out_channels=2*features[-1],
                kernel_size=2,
                stride=2,
                padding=0
            ),
            nn.BatchNorm1d(num_features=2*features[-1]),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                    in_channels=2*features[-1],
                    out_channels=features[-1],
                    kernel_size=2,
                    stride=2
                ),
            nn.Conv1d(
                in_channels=features[-1],
                out_channels=features[-1],
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(num_features=features[-1]),
            nn.ReLU(inplace=True),
        )

        self.final_conv =  nn.Sequential(
            nn.ConvTranspose1d(
                    in_channels=2*features[0],
                    out_channels=out_channels,
                    kernel_size=511,
                    stride=1
                ),
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            )
        )
        
        print(f"Encoder features by level: {features}")
    
    def forward(self, x):
        
        input = self.input_norm(x)
        #print('input', input.shape)
        
        enc = self.encoder(input)
        #print('enc', enc.shape)

        enc_out = self.bottleneck(enc)
        #print('enc_out', enc_out.shape)
        
        dec = self.decoder(enc_out)
        #print('dec', dec.shape)
        
        out = self.final_conv(torch.cat([dec, enc], dim=1))
        #print('out', out.shape)

        return out

def hugeKernelEncoder_model(
    in_channels: int,
    out_channels: int,
    features: List[int]
    ):
    return _hugeKernelEncoder_model(
        in_channels = in_channels,
        out_channels = out_channels,
        features = features
    )