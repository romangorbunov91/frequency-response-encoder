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
        #self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        outputs = []
        for block, BN in zip(self.block_sequence, self.batchNorm):
            x = block(x)
            outputs.append(x)
            x = BN(x)
            x = self.activation(x)
            
        return outputs


class DecoderTransConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int
        ):
        super(DecoderTransConv, self).__init__()
        self.decoder_trans_conv = nn.Sequential(
            nn.ConvTranspose1d(
                    in_channels=in_channels,
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

    def forward(self, x):
        return self.decoder_trans_conv(x)


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

        self.batchNorm = nn.ModuleList([nn.BatchNorm1d(num_features=feature) for feature in features])

        features = features[::-1]
        prev_channels = features[0]
        self.decoder = nn.ModuleList()
        self.decoder.append(
                DecoderTransConv(
                    in_channels = 2*prev_channels,
                    out_channels = prev_channels
                )
            )

        for feature in features[1:]:
            self.decoder.append(
                DecoderTransConv(
                    in_channels = 2*prev_channels,
                    out_channels = feature
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
                ),
                nn.BatchNorm1d(num_features=out_channels),
                nn.ReLU(inplace=True),
            )
        )

        self.activation = nn.ReLU(inplace=False)

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
        
        # Reverse order.
        enc_sum = enc_sum[::-1]
        
        dec_out = self.decoder[0](enc_out)
        for dec_layer, enc_sum_item in zip(self.decoder[1:], enc_sum):
            dec_input = torch.cat([dec_out, enc_sum_item], dim=1)
            dec_out = dec_layer(dec_input)
            #print('dec_out', dec_out.shape)
        
        return dec_out

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