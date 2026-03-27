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
        padding: int=1
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
            nn.ReLU(inplace=True),
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
        
        self.double_conv_inter = nn.ModuleList()
        self.double_conv_final = nn.ModuleList()

        self.batchNorm = nn.ModuleList()

        prev_channels = in_channels
        for feature in features:
            self.double_conv_inter.append(
                DoubleConv(
                    in_channels=prev_channels,
                    out_channels=feature,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
            )
            prev_channels = feature

            self.double_conv_final.append(
                DoubleConv(
                    in_channels=feature,
                    out_channels=feature,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
            )
            self.batchNorm.append(nn.BatchNorm1d(num_features=feature))

        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        outputs = []
        prev_input = x
        for layer_inter, layer_final,  BN in zip(self.double_conv_inter, self.double_conv_final, self.batchNorm):
            out_inter = layer_inter(prev_input)
            prev_input = self.pool(self.activation(BN(out_inter + layer_final(out_inter))))
            outputs.append(prev_input)

        return outputs


class DecoderTransConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int=1,
        padding: int=0
        ):
        super(DecoderTransConv, self).__init__()
        self.decoder_trans_conv = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2
                ),
            
            DoubleConv(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
        )

    def forward(self, x):
        return self.decoder_trans_conv(x)


class _deepEncoder_model(nn.Module):

    def __init__(self,
                in_channels: int,
                out_channels: int,
                features: List[int]
                ):
        super(_deepEncoder_model, self).__init__()

        self.input_norm = nn.InstanceNorm1d(
            num_features=in_channels, 
            affine=True  # Allows model to learn optimal scale/shift.
        )
        
        self.encoder = EncoderCh(
            in_channels = in_channels,
            features = features,
            kernel_size = 3,
            stride = 1,
            padding = 1
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
      
        print(f"Encoder features by level: {features}")

        features = features[::-1]
        prev_channels = features[0]
        self.decoder = nn.ModuleList()
        self.decoder.append(
                DecoderTransConv(
                    in_channels = 2*prev_channels,
                    out_channels = prev_channels,
                    kernel_size = 3,
                    stride = 1,
                    padding = 1
                )
            )

        for feature in features[1:]:
            self.decoder.append(
                DecoderTransConv(
                    in_channels = 2*prev_channels,
                    out_channels = feature,
                    kernel_size = 3,
                    stride = 1,
                    padding = 1
                )
            )
            prev_channels = feature        

        self.decoder.append(nn.Sequential(
                nn.ConvTranspose1d(
                        in_channels=2*prev_channels,
                        out_channels=out_channels,
                        kernel_size=2,
                        stride=2
                    ),
                nn.Conv1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1
                )
            )
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
        
        input_x = self.input_norm(x)
        #print('input_x', input_x.shape)
        
        enc_sum = self.encoder(input_x)

        enc_out = self.bottleneck(
            enc_sum[-1]
            )
        
        # Reverse order.
        enc_sum = enc_sum[::-1]
        
        dec_out = self.decoder[0](enc_out)
        for dec_layer, enc_sum_item in zip(self.decoder[1:], enc_sum):
            dec_input = torch.cat([dec_out, enc_sum_item], dim=1)
            dec_out = dec_layer(dec_input)
            #print('dec_out', dec_out.shape)
        
        return dec_out

def deepEncoder_model(
    in_channels: int,
    out_channels: int,
    features: List[int]
    ):
    return _deepEncoder_model(
        in_channels = in_channels,
        out_channels = out_channels,
        features = features
    )