import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)
        
        attn = torch.softmax(torch.bmm(q, k) / (C ** 0.5), dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        
        return self.gamma * out + x


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        
        # Encoder (downsampling path)
        self.encoder1 = self._double_conv(in_channels, features[0])
        self.pool1 = nn.Conv2d(features[0], features[0], kernel_size=3, stride=2, padding=1)
        
        self.encoder2 = self._double_conv(features[0], features[1])
        self.pool2 = nn.Conv2d(features[1], features[1], kernel_size=3, stride=2, padding=1)
        
        self.encoder3 = self._double_conv(features[1], features[2])
        self.pool3 = nn.Conv2d(features[2], features[2], kernel_size=3, stride=2, padding=1)
        
        self.encoder4 = self._double_conv(features[2], features[3])
        self.pool4 = nn.Conv2d(features[3], features[3], kernel_size=3, stride=2, padding=1)
        
        # Bottleneck
        self.bottleneck = self._double_conv(features[3], features[3] * 2)
        self.attention = SelfAttention(features[3] * 2)
        
        # Decoder (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(features[3] * 2, features[3], 2, 2)
        self.decoder4 = self._double_conv(features[3] * 2, features[3])
        
        self.upconv3 = nn.ConvTranspose2d(features[3], features[2], 2, 2)
        self.decoder3 = self._double_conv(features[2] * 2, features[2])
        
        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], 2, 2)
        self.decoder2 = self._double_conv(features[1] * 2, features[1])
        
        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], 2, 2)
        self.decoder1 = self._double_conv(features[0] * 2, features[0])
        
        # Spectrogram output head
        self.spec_out = nn.Sequential(
            nn.Conv2d(features[0], out_channels, 1),
            nn.Sigmoid()
        )
        
        # # Mask output head
        # self.mask_out = nn.Sequential(
        #     nn.Conv2d(features[0], 32, 3, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(32, out_channels, 1),
        #     nn.Sigmoid()
        # )
    
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x, return_mask=False):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        bottleneck = self.attention(bottleneck)

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        # Two heads
        spec = self.spec_out(dec1)
        # mask = self.mask_out(dec1)
        
        # if return_mask:
        #     return spec * mask, mask
        # return spec * mask
        return torch.clamp(x - spec, 0, 1) # residual difference