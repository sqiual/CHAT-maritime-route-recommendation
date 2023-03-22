import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import time
import matplotlib.pyplot as plt


def sr_images(xt, lr, model, epoch, mode):
    noise_steps = 500

    beta_start = -6
    beta_end = 6
    betas = torch.linspace(beta_start, beta_end, noise_steps)
    betas = torch.sigmoid(betas) * (0.5e-4) + 5e-7 # * (0.5e-3 - 1e-6) + 1e-6
    # print('===betas===', betas)
    alpha = 1. - betas
    alpha_hat = torch.cumprod(alpha, dim=0)
    # print('alpha_hat',alpha_hat.shape) 1000

    sqrt_alpha_hat = torch.sqrt(alpha_hat)[:, None, None, None].cuda()
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat)[:, None, None, None].cuda()

    cur_x = xt
    #start = time.time()
    for i in reversed(range(noise_steps)):
        cur_x = sr_images_step(model, cur_x, lr, i, betas, sqrt_alpha_hat, sqrt_one_minus_alpha_hat)
    return cur_x

def sr_images_test(xt, lr, model, epoch):
    noise_steps = 500

    beta_start = -6
    beta_end = 6
    betas = torch.linspace(beta_start, beta_end, noise_steps)
    betas = torch.sigmoid(betas) * (0.5e-4) + 5e-7
    alpha = 1. - betas
    alpha_hat = torch.cumprod(alpha, dim=0)

    sqrt_alpha_hat = torch.sqrt(alpha_hat)[:, None, None, None].cuda()
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat)[:, None, None, None].cuda()

    cur_x = xt
    for i in reversed(range(noise_steps)):
        cur_x = sr_images_step(model, cur_x, lr, i, betas, sqrt_alpha_hat, sqrt_one_minus_alpha_hat) 
    return cur_x


def sr_images_step(model, x, lr, i, betas, sqrt_alpha_hat, sqrt_one_minus_alpha_hat):
    model.eval()
    with torch.no_grad():
        i = torch.tensor([i])
        t = np.repeat(i, x.shape[0]).cuda()
        coeff = betas[i].cuda() / sqrt_one_minus_alpha_hat[i].cuda()
        epsilon = model(x = x, lr = lr, t = t)
        mean = (1/sqrt_alpha_hat[i].cuda()) * (x - (coeff.cuda() * epsilon))
        mean = mean.cuda()
        z = torch.randn_like(x).cuda()
        sig_t = torch.sqrt(betas[i]).cuda()
        x = mean + sig_t * z
    model.train()
    return x
    


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

    def forward(self, x, lr):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        lr = lr.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        y_ln = self.ln(lr)
        attention_value, _ = self.mha(y_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

class SelfAttention_pre(nn.Module):
    def __init__(self, channels, num_top, size):
        super(SelfAttention_pre, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 1, batch_first=True)
        self.ln_1 = nn.LayerNorm([channels])
        self.ln_2 = nn.Sequential(
            nn.LayerNorm([num_top]),
            nn.Linear(num_top, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.LayerNorm([channels])
        )
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )


    def forward(self, top_sim):
        #[5, 1, 128, 128]
        org = top_sim[0]
        others = top_sim[1:]
        org = org.reshape(-1, self.channels, self.size * self.size).swapaxes(0, 2) #[128*128, 1, 1]
        others = others.reshape(-1, self.channels, self.size * self.size).swapaxes(0, 2) #[128*128, 1, 4]
        # org = torch.from_numpy(org).cuda()
        # others = torch.from_numpy(others).cuda()
        org_ln = self.ln_1(org) #[128*128, 1, 1]
        others_ln = self.ln_2(others) #[128*128, 1, 1]
        attention_value, _ = self.mha(org_ln, others_ln, others_ln)
        attention_value = attention_value + org
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)
    
# 这是给 channel 用的
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
        
        self.emb_lr = nn.Sequential(
            nn.Tanh(),
            nn.Linear(
                out_channels,
                out_channels
            ),
        )

    def forward(self, x, lr, t):
        x = self.maxpool_conv(x)
        lr = self.maxpool_conv(lr)
        emb_t = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        emb_lr = self.emb_lr(lr.permute(0,2,3,1)).permute(0,3,1,2) 
        emb = emb_t + emb_lr
        return x + emb, lr


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
        
        self.emb_lr = nn.Sequential(
            nn.Tanh(),
            nn.Linear(
                int(in_channels/2),
                out_channels
            ),
        )

    def forward(self, x, skip_x, lr, t):
        x = self.up(x)
        lr = self.up(lr)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb_t = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        emb_lr = self.emb_lr(lr.permute(0,2,3,1)).permute(0,3,1,2)
        emb = emb_t + emb_lr
        return x + emb, emb_lr



class UNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, device="cuda"):
        # 在原始的代码中，输入的图片就是64*64，我们这里是128
        # 由于down的maxpool会减一半的像素，所以down后面的sa的像素输入直接就是64
        #只有SA的第二个参数是分辨率，其他都是channel
        # 由由于报OOM2， 我们缩减一下结构，从up三层down三层改成up/down两层
        # 先不改，我改一下batch size试一下
        # 改了batch还是不行，这里改channel数，是原代码的1/4
        # 改了channel还是不行，改multihead之前sa5报OOM，改了之后一直都是sa6报，因此结构改成两层
        # 再改batch到4 可以了，因此unet改回来3层
        super().__init__()
        self.device = device
        self.sa_pre = SelfAttention_pre(1, 4, 128) #[channels, top_num, resolution]
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 16)  #16 或者 8  
        self.down1 = Down(16, 32) 
        self.sa1 = SelfAttention(32, 64) 
        self.down2 = Down(32, 64) 
        self.sa2 = SelfAttention(64, 32) 
        self.down3 = Down(64, 64) # 注意这一步 的in/out是一样的 不然x3 x2 concate之后不是2^n
        self.sa3 = SelfAttention(64, 16)
      
        self.bot1 = DoubleConv(64, 128)
        self.bot2 = DoubleConv(128, 128)
        self.bot3 = DoubleConv(128, 64)
        
        # 这里Up(512,)是因为concate了x4和x3（其实就是说Up的第一个输入是bot3最后一个输入的两倍）
        # up了之后 是 x_t + x_{t+1} 因为concate了两步到一起做平滑
        self.up1 = Up(128, 32)  # up1(1) = bot3(2)+sa2(1), up1(2) = up1(1)/4
        self.sa4 = SelfAttention(32, 32) # sa4(2) = sa3(2)*2
        self.up2 = Up(64, 16) # up2(1) = up1(2)+sa1(1), up2(2) = up2(1)/4
        self.sa5 = SelfAttention(16, 64) 
        self.up3 = Up(32, 8) # up3(1) = up2(2)+inc(2), up3(2) = up3(1)/4
        self.sa6 = SelfAttention(8, 128) 
        self.outc = nn.Conv2d(8, c_out, kernel_size=1)
        
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc


    def forward(self, x, lr, t):
        t = t.unsqueeze(-1).type(torch.float)
        # print('t',t.shape)
        t = self.pos_encoding(t, self.time_dim)
        lr = self.sa_pre(lr) # [5, 1, 128, 128] -> [1, 1, 128, 128]
        #lr = lr[0,:,:,:].unsqueeze(dim = 0)
        #！！！！！！！！！！
        x1 = self.inc(x)  # [1, 16, 128, 128]
        lr = self.inc(lr) # [1, 16, 128, 128]
        
        #x1 = torch.cat([x1, lr], dim=1) # [N, 32, 128, 128]
        x2, lr = self.down1(x1, lr, t) # [N, 32, 64, 64] [N, 32, 64, 64]
        x2 = self.sa1(x2, lr)
        x3, lr = self.down2(x2, lr, t)
        x3 = self.sa2(x3,lr)
        
        x4, lr = self.down3(x3, lr, t)
        x4 = self.sa3(x4,lr)
        
        # 如果将来要改回来，这里的x4 x3要注意改啊
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        
        x, lr = self.up1(x4, x3, lr, t)
        x = self.sa4(x,lr)
        x, lr = self.up2(x, x2, lr, t)
        x = self.sa5(x,lr)
        x, lr = self.up3(x, x1, lr, t)
        x = self.sa6(x,lr)
        output = self.outc(x)
        
        return output


