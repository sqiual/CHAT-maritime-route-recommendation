import math
import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 1, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x 


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        return x   
    
class Encoder(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.inc = DoubleConv(c_in, 16) 
        self.bn1 = nn.BatchNorm2d(16)
        self.down1 = Down(16, 32) 
        self.sa1 = SelfAttention(32, 64) 
        self.bn2 = nn.BatchNorm2d(32)
        self.down2 = Down(32, 64) 
        self.sa2 = SelfAttention(64, 32) 
        self.bn3 = nn.BatchNorm2d(64)
        self.down3 = Down(64, 64)
        self.sa3 = SelfAttention(64, 16)

        self.bot1 = DoubleConv(64, 128)
        self.bot2 = DoubleConv(128, 128)
        self.bot3 = DoubleConv(128, 64)

        self.up1 = Up(128, 32) 
        self.sa4 = SelfAttention(32, 32) 
        self.up2 = Up(64, 16)
        self.sa5 = SelfAttention(16, 64) 
        self.up3 = Up(32, 8) 
        self.sa6 = SelfAttention(8, 128) 
        self.outc = nn.Conv2d(16, c_out, kernel_size=1)
        
        self.act = nn.LeakyReLU(0.2)
            
    def forward(self, lr):
        x1 = self.inc(lr) 
        x1 = self.bn1(x1)
        x1 = self.act(x1)
        
        x2= self.down1(x1) 
        x2 = self.sa1(x2)
        x2 = self.bn2(x2)
        x2 = self.act(x2)
        
        x3= self.down2(x2)
        x3 = self.sa2(x3)
        x3 = self.bn3(x3)
        x3 = self.act(x3)
        
        x4= self.down3(x3)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3)
        x = self.sa4(x)
        x = self.up2(x, x2)
        x = self.sa5(x)
        output = self.outc(x)

        return output
