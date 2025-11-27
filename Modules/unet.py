import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        
        # Encoder (downsampling path)
        self.encoder1 = self._double_conv(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.encoder2 = self._double_conv(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.encoder3 = self._double_conv(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.encoder4 = self._double_conv(features[2], features[3])
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self._double_conv(features[3], features[3] * 2)
        
        # Decoder (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(features[3] * 2, features[3], 2, 2)
        self.decoder4 = self._double_conv(features[3] * 2, features[3])  # *2 for skip connection
        
        self.upconv3 = nn.ConvTranspose2d(features[3], features[2], 2, 2)
        self.decoder3 = self._double_conv(features[2] * 2, features[2])
        
        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], 2, 2)
        self.decoder2 = self._double_conv(features[1] * 2, features[1])
        
        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], 2, 2)
        self.decoder1 = self._double_conv(features[0] * 2, features[0])
        
        # Final output layer
        self.out = nn.Sequential(
            nn.Conv2d(features[0], out_channels, 1),
            nn.Sigmoid()  # Forces [0,1] output
        )
    
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder with skip connections
        enc1 = self.encoder1(x)      # 256x256
        enc2 = self.encoder2(self.pool1(enc1))  # 128x128
        enc3 = self.encoder3(self.pool2(enc2))  # 64x64
        enc4 = self.encoder4(self.pool3(enc3))  # 32x32
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))  # 16x16
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)  # 32x32
        dec4 = torch.cat([dec4, enc4], dim=1)  # Skip connection
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)  # 64x64
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)  # 128x128
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)  # 256x256
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.out(dec1)
    