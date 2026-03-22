import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from typing_extensions import Self

class baseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(baseConv, self).__init__()
        self.base_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ),
            #nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.base_conv(x)


class _base_model(nn.Module):

    def __init__(self,
                in_channels: int,
                out_channels: int,
                features: List[int]
                ):
        super(_base_model, self).__init__()

        self.input_norm = nn.InstanceNorm1d(
            num_features=in_channels, 
            affine=True  # Allows model to learn optimal scale/shift.
        )

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        prev_channels = in_channels
        for feature in features:
            self.encoder_blocks.append(
                baseConv(
                    in_channels=prev_channels,
                    out_channels=feature)
                )
            prev_channels = feature
        '''
        self.initial_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=features[0],
            kernel_size=3,
            padding=1
        )
        '''
        self.pool = nn.MaxPool1d(
            kernel_size=2,
            stride=2
        )
        
        self.bottleneck = baseConv(
            in_channels=features[-1],
            out_channels=features[-1] * 2)

        for feature in reversed(features):
            self.decoder_blocks.append(
                nn.ConvTranspose1d(
                    in_channels=feature * 2,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2
                )
            )
            self.decoder_blocks.append(
                baseConv(
                    in_channels=feature * 2,
                    out_channels=feature
                )
            )

        #self.BN1 = nn.BatchNorm1d(num_features=features[0])
        #self.activation = nn.ReLU(inplace=True)
        self.final_conv = nn.Conv1d(
            in_channels=features[0],
            out_channels=out_channels,
            kernel_size=1
        )
        
        print(f"Encoder features by level: {features}")
    
    def forward(self, x):
        x = self.input_norm(x)

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