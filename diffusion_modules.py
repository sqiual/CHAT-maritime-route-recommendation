import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import time


def sr_images(xt, lr, model, epoch, mode):
    noise_steps = 100

    beta_start = -6
    beta_end = 6
    betas = torch.linspace(beta_start, beta_end, noise_steps)
    betas = torch.sigmoid(betas) * (0.5e-3 - 1e-6) + 1e-6

    alpha = 1. - betas
    alpha_hat = torch.cumprod(alpha, dim=0)
    # print('alpha_hat',alpha_hat.shape) 1000

    sqrt_alpha_hat = torch.sqrt(alpha_hat)[:, None, None, None].cuda()
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat)[:, None, None, None].cuda()

    cur_x = xt
    #start = time.time()
    for i in reversed(range(noise_steps)):
        cur_x = sr_images_step(model, cur_x, lr, i, betas, sqrt_alpha_hat, sqrt_one_minus_alpha_hat)
        # print('i', i)
        # print('cur_x', cur_x)
    #end = time.time()
    #print('reconstruct time', end - start)

    ############################
    # 保存 sr 图片
    ############################
    for i in range(cur_x.shape[0]):
        y = cur_x[i, :, :, :]
        transform = T.ToPILImage()
        y = transform(y)
        if mode == 'test':
            if i % 5 == 0:
                y.save("./scp/sr/{}_sr_{}_{}.png".format(mode, epoch, i))
        elif (epoch % 1 == 0) and (i % 3 == 0):
            y.save("./scp/sr/{}_sr_{}_{}.png".format(mode, epoch, i))
        elif (epoch % 100 == 0) and (i % 3 == 0):
            y.save("./scp/sr/{}_sr_{}_{}.png".format(mode, epoch, i))
                
            
    return cur_x


def sr_images_step(model, x, lr, i, betas, sqrt_alpha_hat, sqrt_one_minus_alpha_hat):
    model.eval()
    with torch.no_grad():
        i = torch.tensor([i])
        t = np.repeat(i, x.shape[0]).cuda()#.clone().detach()
        # print('beta', betas.shape)
        # print('sqrt', sqrt_one_minus_alpha_hat.shape)
        coeff = betas[i].cuda() / sqrt_one_minus_alpha_hat[i].cuda()
        # 这一步有问题, 要分t 和 i的区别
        epsilon = model(x = x, lr = lr, t = t)
        mean = (1/sqrt_alpha_hat[i].cuda()) * (x - (coeff.cuda() * epsilon))
        mean = mean.cuda()
        z = torch.randn_like(x).cuda()
        sig_t = torch.sqrt(betas[i]).cuda()
        x = mean + sig_t * z
    model.train()
    return x
    

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        # size: 图片分辨率
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        # 由于报OOM1，这里的multihead由原来的4改成1
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

    def forward(self, x, t):
        # 就是这一步的max pool减了像素，所以后面的SA像素会不断地减半
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        # D_in, H_in, W_in 全部乘以二
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

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb



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
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 8)  #16 或者 8 
        self.down1 = Down(8, 16) # 这里之后加入了 lr 64，因此通道数+16
        self.sa1 = SelfAttention(24, 64) 
        self.down2 = Down(24, 48) 
        self.sa2 = SelfAttention(48, 32) 
        self.down3 = Down(48, 48) # 注意这一步 的in/out是一样的 不然x3 x2 concate之后不是2^n
        self.sa3 = SelfAttention(48, 16)
        # self.sa4 = SelfAttention(32, 32)
        
        self.bot1 = DoubleConv(48, 96)
        self.bot2 = DoubleConv(96, 96)
        self.bot3 = DoubleConv(96, 48)
        
        # 这里Up(512,)是因为concate了x4和x3（其实就是说Up的第一个输入是bot3最后一个输入的两倍）
        # up了之后 是 x_t + x_{t+1} 因为concate了两步到一起做平滑
        self.up1 = Up(96, 24)  # up1(1) = bot3(2)+sa2(1), up1(2) = up1(1)/4
        self.sa4 = SelfAttention(24, 32) # sa4(2) = sa3(2)*2
        self.up2 = Up(48, 12) # up2(1) = up1(2)+sa1(1), up2(2) = up2(1)/4
        self.sa5 = SelfAttention(12, 64) 
        self.up3 = Up(20, 5) # up3(1) = up2(2)+inc(2), up3(2) = up3(1)/4
        self.sa6 = SelfAttention(5, 128) 
        # self.up4 = Up(8, 2) # up4(1) = up3(2)+inc(2)
        # self.sa8 = SelfAttention(2, 512) 
        self.outc = nn.Conv2d(5, c_out, kernel_size=1)
        
        self.noise_steps = 100

        self.beta_start = -6
        self.beta_end = 6
        self.betas = self.prepare_noise_schedule().cuda()

        self.alpha = 1. - self.betas
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        # print('alpha_hat',alpha_hat.shape) 1000

        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)[:, None, None, None].cuda()
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)[:, None, None, None].cuda()
        
    def prepare_noise_schedule(self):
        betas = torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        betas = torch.sigmoid(betas) * (0.5e-3 - 1e-6) + 1e-6
        return betas

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
        t = self.pos_encoding(t, self.time_dim)
        
        lr = self.inc(lr) # [N, 16, 64, 64]
        x1 = self.inc(x)  # [N, 16, 128, 128]
        x2 = self.down1(x1, t) # [N, 32, 64, 64]
        # print('x2', x2.shape)
        # print('lr', lr.shape)
        x2 = torch.cat([x2, lr], dim=1) # [N, 48, 64, 64]
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        # x5 = self.down4(x4, t)
        # x5 = self.sa4(x5)
        
        # 如果将来要改回来，这里的x4 x3要注意改啊
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        
        x = self.up1(x4, x3, t)
        # print('AAA x', x.shape)
        x = self.sa4(x)
        # print('BBB x', x.shape)     
        # print('x3', x3.shape)
        # print('x2', x2.shape)
        # print('x1', x1.shape)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        # x = self.up4(x, x1, t)
        # x = self.sa8(x)
        output = self.outc(x)
        
        return output


class UNet_conditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output

# if __name__ == '__main__':
#     # net = UNet(device="cpu")
#     net = UNet_conditional(num_classes=10, device="cpu")
#     print(sum([p.numel() for p in net.parameters()]))
#     x = torch.randn(3, 3, 64, 64)
#     t = x.new_tensor([500] * x.shape[0]).long()
#     y = x.new_tensor([1] * x.shape[0]).long()
#     print(net(x, t, y).shape)
