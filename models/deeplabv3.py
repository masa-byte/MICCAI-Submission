import torch.nn as nn
import torch
import torch.nn.functional as F

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module.
    """
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        
        # 1x1 convolution
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        )
        
        # Atrous convolutions with different rates
        for rate in atrous_rates:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Global average pooling
        modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        )
        
        self.convs = nn.ModuleList(modules)
        
        # Projection layer
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(F.interpolate(conv(x), size=x.shape[2:], mode='bilinear', align_corners=False))
        res = torch.cat(res, dim=1)
        return self.project(res)


class Deeplabv3Plus(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(Deeplabv3Plus, self).__init__()
        
        # Encoder (Backbone)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # ResNet-like blocks
            self._make_layer(64, 64, 3, stride=1),  # No downsampling
            self._make_layer(64, 128, 4, stride=2),  # Downsample by 2
            self._make_layer(128, 256, 6, stride=2),  # Downsample by 2
            self._make_layer(256, 512, 3, stride=2)  # Downsample by 2
        )
        
        # ASPP module
        self.aspp = ASPP(in_channels=512, out_channels=256, atrous_rates=[6, 12, 18])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=1)
        )
        
        # Upsampling layer
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        )
        for _ in range(1, num_blocks):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        
        # ASPP
        x = self.aspp(x)
        
        # Decoder
        x = self.decoder(x)
        
        # Upsample to input resolution
        x = self.upsample(x)
        
        return x