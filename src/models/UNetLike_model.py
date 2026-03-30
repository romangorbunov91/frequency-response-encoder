import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from typing_extensions import Self

class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int=1,
        padding: int=0
        ):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(inplace=True)
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

        prev_channels = in_channels
        for feature in features:
            self.block_sequence.append(
                DoubleConv(
                    in_channels=prev_channels,
                    out_channels=feature,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
            )
            prev_channels = feature
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        outputs = []
        outputs.append(self.block_sequence[0](x))
        for block in self.block_sequence[1:]:
            outputs.append(block(self.pool(outputs[-1])))
            
        return outputs

class _UNetLike_model(nn.Module):

    def __init__(self,
                in_channels: int,
                out_channels: int,
                features: List[int]
                ):
        super(_UNetLike_model, self).__init__()

        self.input_norm = nn.InstanceNorm1d(
            num_features=in_channels, 
            affine=True  # Allows model to learn optimal scale/shift.
        )
        
        print(f"Encoder features by level: {features}")
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.encoder = EncoderCh(
            in_channels=in_channels,
            features=features,
            kernel_size=3,
            stride=1,
            padding=1
        )        
      
        self.bottleneck = DoubleConv(
            in_channels=features[-1],
            out_channels=2*features[-1],
            kernel_size=3,
            stride=1,
            padding=1
        )
       
        features = features[::-1]
        prev_channels = 2*features[0]
        
        self.upsample = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for feature in features:
            self.upsample.append(nn.ConvTranspose1d(
                    in_channels=prev_channels,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2
                )
            )
            
            self.decoder.append(DoubleConv(
                in_channels=prev_channels,
                out_channels=feature,
                kernel_size=3,
                stride=1,
                padding=1
                )
            )

            prev_channels = feature

        self.final_conv = nn.Conv1d(
                    in_channels=features[-1],
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                    )

        self.apply(self._init_weights)
        
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.InstanceNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        
        #x_input = self.input_norm(x)
        #print('input', input.shape)
        
        enc_outputs = self.encoder(x)
        
        # Reverse order.
        enc_outputs = enc_outputs[::-1]
        
        dec_out = self.bottleneck(self.pool(enc_outputs[0]))
        #print('dec_out', dec_out.shape)
        
        for dec_layer, up_layer, enc_out_item in zip(self.decoder, self.upsample, enc_outputs):
            dec_out = dec_layer(torch.cat([up_layer(dec_out), enc_out_item], dim=1))
            #print('dec_out', dec_out.shape)
        
        return self.final_conv(dec_out)

def UNetLike_model(
    in_channels: int,
    out_channels: int,
    features: List[int]
    ):
    return _UNetLike_model(
        in_channels = in_channels,
        out_channels = out_channels,
        features = features
    )