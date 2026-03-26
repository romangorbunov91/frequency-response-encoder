import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from typing_extensions import Self

class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int],
        kernel_size: List[int],
        stride: List[int],
        padding: List[int] = [0, 0]
        ):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels[0],
                out_channels=out_channels[0],
                kernel_size=kernel_size[0],
                stride=stride[0],
                padding=padding[0]
            ),
            nn.BatchNorm1d(num_features=out_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=out_channels[0],
                out_channels=out_channels[1],
                kernel_size=kernel_size[1],
                stride=stride[1],
                padding=padding[1]
            ),
        )

    def forward(self, x):
        return self.double_conv(x)


class EncoderCh(nn.Module):
    def __init__(
        self,
        in_channels: int,
        features: List[int],
        kernel_size: int,
        stride: int,
        padding: int
        ):
        super(EncoderCh, self).__init__()
        
        self.block_sequence = nn.ModuleList()
        self.batchNorm = nn.ModuleList()

        prev_channels = in_channels
        for feature in features:
            self.block_sequence.append(
                DoubleConv(
                    in_channels = [prev_channels, feature],
                    out_channels = [feature, feature],
                    kernel_size = [kernel_size+1, kernel_size],
                    stride = [stride, stride+1],
                    padding = [padding, padding-1]
                )
            )
            prev_channels = feature
        
            self.batchNorm.append(nn.BatchNorm1d(num_features=feature))
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        outputs = []
        for block, BN in zip(self.block_sequence, self.batchNorm):
            x = block(x)
            outputs.append(x)
            x = BN(x)
            x = self.activation(x)
            
        return outputs


class _parallelEncoder_model(nn.Module):

    def __init__(self,
                in_channels: int,
                out_channels: int,
                features: List[int]
                ):
        super(_parallelEncoder_model, self).__init__()

        self.input_norm = nn.InstanceNorm1d(
            num_features=in_channels, 
            affine=True  # Allows model to learn optimal scale/shift.
        )
        
        self.encoder1 = EncoderCh(
            in_channels = in_channels,
            features = features,
            kernel_size = 64,
            stride = 1,
            padding = 32
        )        
        
        self.encoder2 = EncoderCh(
            in_channels = in_channels,
            features = features,
            kernel_size = 32,
            stride = 1,
            padding = 16
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
            nn.BatchNorm1d(num_features=features[-1]),
            nn.ReLU(inplace=True),
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose1d(
                    in_channels=2*features[-1],
                    out_channels=features[-2],
                    kernel_size=2,
                    stride=2
                ),
            nn.Conv1d(
                in_channels=features[-2],
                out_channels=features[-2],
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(num_features=features[-2]),
            nn.ReLU(inplace=True),
        )
        
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose1d(
                    in_channels=2*features[-2],
                    out_channels=features[-3],
                    kernel_size=2,
                    stride=2
                ),
            nn.Conv1d(
                in_channels=features[-3],
                out_channels=features[-3],
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(num_features=features[-3]),
            nn.ReLU(inplace=True),
        )

        self.batchNorm = nn.ModuleList()
        for feature in features:
            self.batchNorm.append(nn.BatchNorm1d(num_features=feature))

        self.activation = nn.ReLU(inplace=False)

        self.final_conv =  nn.Sequential(
            nn.ConvTranspose1d(
                    in_channels=2*features[-3],
                    out_channels=out_channels,
                    kernel_size=2,
                    stride=2
                ),
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.apply(self._init_weights)
        print(f"Encoder features by level: {features}")
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.InstanceNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        
        input = self.input_norm(x)
        #print('input', input.shape)
        
        enc1_out = self.encoder1(input)
        enc2_out = self.encoder2(input)
        
        enc_sum = []
        for enc1, enc2, BN in zip(enc1_out, enc2_out, self.batchNorm):
            enc_sum.append(self.activation(BN(enc1+enc2)))
            #print('enc_sum', enc_sum[-1].shape)

        enc_out = self.bottleneck(
            self.activation(enc_sum[-1])
            )
        #print('enc_out', enc_out.shape)
        
        dec1 = self.decoder1(enc_out)
        #print('dec1', dec1.shape)
        dec2 = self.decoder2(torch.cat([dec1, enc_sum[-1]], dim=1))
        #print('dec2', dec2.shape)
        dec3 = self.decoder3(torch.cat([dec2, enc_sum[1]], dim=1))
        #print('dec3', dec3.shape)
        out = self.final_conv(torch.cat([dec3, enc_sum[0]], dim=1))
        #print('out', out.shape)

        return out

def parallelEncoder_model(
    in_channels: int,
    out_channels: int,
    features: List[int]
    ):
    return _parallelEncoder_model(
        in_channels = in_channels,
        out_channels = out_channels,
        features = features
    )