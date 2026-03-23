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

        self.input_norm = nn.InstanceNorm1d(
            num_features=in_channels, 
            affine=True  # Allows model to learn optimal scale/shift.
        )
        self.encoder1_1 = nn.Sequential(
            #nn.BatchNorm1d(num_features=in_channels),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=features[0],
                kernel_size=32,
                stride=2,
                padding=15
            ),
            #nn.BatchNorm1d(num_features=features[0]),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=features[0],
                out_channels=features[0],
                kernel_size=16,
                stride=2,
                padding=7
            ),
        )

        self.encoder1_2 = nn.Sequential(
            #nn.BatchNorm1d(num_features=features[0]),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=features[0],
                out_channels=features[1],
                kernel_size=16,
                stride=2,
                padding=7
            ),
            #nn.BatchNorm1d(num_features=features[1]),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=features[1],
                out_channels=features[1],
                kernel_size=8,
                stride=2,
                padding=3
            ),     
        )

        self.encoder1_3 = nn.Sequential(
            #nn.BatchNorm1d(num_features=features[1]),
            #nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=features[1],
                out_channels=features[2],
                kernel_size=8,
                stride=2,
                padding=3
            ),
            #nn.BatchNorm1d(num_features=features[2]),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=features[2],
                out_channels=features[2],
                kernel_size=4,
                stride=2,
                padding=1
            ),
        )

        self.encoder2_1 = nn.Sequential(
            #nn.BatchNorm1d(num_features=in_channels),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=features[0],
                kernel_size=16,
                stride=2,
                padding=7
            ),
            #nn.BatchNorm1d(num_features=features[0]),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=features[0],
                out_channels=features[0],
                kernel_size=8,
                stride=2,
                padding=3
            ),
        )

        self.encoder2_2 = nn.Sequential(
            #nn.BatchNorm1d(num_features=features[0]),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=features[0],
                out_channels=features[1],
                kernel_size=8,
                stride=2,
                padding=3
            ),
            #nn.BatchNorm1d(num_features=features[1]),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=features[1],
                out_channels=features[1],
                kernel_size=4,
                stride=2,
                padding=1
            ),
        )

        self.encoder2_3 = nn.Sequential(
            #nn.BatchNorm1d(num_features=features[1]),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=features[1],
                out_channels=features[2],
                kernel_size=4,
                stride=2,
                padding=1
            ),
            #nn.BatchNorm1d(num_features=features[2]),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=features[2],
                out_channels=features[2],
                kernel_size=2,
                stride=2,
                padding=0
            ),
        )

        '''
        self.pool = nn.MaxPool1d(
            kernel_size=2,
            stride=2
        )
        '''
        self.bottleneck = nn.Sequential(
            nn.Conv1d(
                in_channels=features[-1],
                out_channels=2*features[-1],
                kernel_size=2,
                stride=2,
                padding=0
            ),
            #nn.BatchNorm1d(num_features=features[-1]),
            nn.ReLU(inplace=True),
        )

        self.decoder1 = nn.Sequential(
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
            #nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose1d(
                    in_channels=2*features[-1],
                    out_channels=features[-2],
                    kernel_size=4,
                    stride=4,
                    padding=0
                ),
            nn.Conv1d(
                in_channels=features[-2],
                out_channels=features[-2],
                kernel_size=3,
                padding=1
            ),
            #nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose1d(
                    in_channels=2*features[-2],
                    out_channels=features[-3],
                    kernel_size=4,
                    stride=4,
                    padding=0
                ),
            nn.Conv1d(
                in_channels=features[-3],
                out_channels=features[-3],
                kernel_size=3,
                padding=1
            ),
            #nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

        #self.BN1 = nn.BatchNorm1d(num_features=features[0])
        self.activation = nn.ReLU(inplace=False)

        self.final_conv =  nn.Sequential(
            nn.ConvTranspose1d(
                    in_channels=2*features[-3],
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=4,
                    padding=0
                ),
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ),
            #nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )
        
        print(f"Encoder features by level: {features}")
    
    def forward(self, x):
        
        input = self.input_norm(x)

        enc1_1 = self.encoder1_1(input)
        enc2_1 = self.encoder2_1(input)
        enc12_1 = self.activation(enc1_1+enc2_1)
        #print('enc12_1', enc12_1.shape)

        enc1_2 = self.encoder1_2(enc1_1)
        enc2_2 = self.encoder2_2(enc2_1)
        enc12_2 = self.activation(enc1_2+enc2_2)
        #print('enc12_2', enc12_2.shape)

        enc1_3 = self.encoder1_3(enc1_2)
        enc2_3 = self.encoder1_3(enc2_2)    
        enc12_3 = self.activation(enc1_3+enc2_3)
        #print('enc12_3', enc12_3.shape)

        enc_out = self.bottleneck(
            self.activation(enc12_3)
            )
        #print('enc_out', enc_out.shape)
        dec1 = self.decoder1(enc_out)
        #print('dec1', dec1.shape)
        dec2 = self.decoder2(torch.cat([dec1, enc12_3], dim=1))
        #print('dec2', dec2.shape)
        dec3 = self.decoder3(torch.cat([dec2, enc12_2], dim=1))
        #print('dec3', dec3.shape)
        out = self.final_conv(torch.cat([dec3, enc12_1], dim=1))
        #print('out', out.shape)

        return out

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