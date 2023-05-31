import torch
import torch.nn as nn
import torch.nn.functional as F
from bottleneck import Vit_neck

# Self Attention Class
class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 2, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, int(self.size) * int(self.size)).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, int(self.size), int(self.size))

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
    def __init__(self, in_channels, out_channels, emb_dim=1000):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            nn.Dropout2d(p=0.3),
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
    def __init__(self, in_channels, out_channels, emb_dim=1000):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            nn.Dropout2d(p=0.3),
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


class Attention_UNet(nn.Module):
    def __init__(self, c_in=2, c_out=2, img_size=224, max_ch_deep=512, net_dimension=128, device="cuda"):
        super().__init__()
        self.device = device

        self.inc = DoubleConv(c_in, net_dimension*2)
        self.down1 = Down(net_dimension*2, net_dimension*4)
        self.sa1 = SelfAttention(net_dimension*4, img_size//2)
        self.down2 = Down(net_dimension*4, net_dimension*8)
        self.sa2 = SelfAttention(net_dimension*8, img_size//4)
        self.down3 = Down(net_dimension*8, net_dimension*8)
        self.sa3 = SelfAttention(net_dimension*8, img_size//8)
        
        self.bot1 = DoubleConv(net_dimension*8, max_ch_deep)
        self.bot2 = DoubleConv(max_ch_deep+48, net_dimension*8)

        self.up1 = Up(net_dimension*16, net_dimension*4)
        self.sa4 = SelfAttention(net_dimension*4, img_size//4)
        self.up2 = Up(net_dimension*8, net_dimension*2)
        self.sa5 = SelfAttention(net_dimension*2, img_size//2)
        self.up3 = Up(net_dimension*4, net_dimension*2)
        self.sa6 = SelfAttention(net_dimension*2, img_size)
        self.outc = nn.Sequential(
            nn.Conv2d(net_dimension*2, c_out, kernel_size=1)
        )

    def forward(self, x, color):

        color = color.view(-1,48,28,28)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.sa1(x2)
        x3 = self.down2(x2)
        x3 = self.sa2(x3)
        x4 = self.down3(x3)
        x4 = self.sa3(x4)   

        x4 = self.bot1(x4)
        x4 = self.bot2(torch.cat((x4, color),1))

        x = self.up1(x4, x3)
        x = self.sa4(x)
        x = self.up2(x, x2)
        x = self.sa5(x)
        x = self.up3(x, x1)
        
        output = self.outc(x)
        return output
    
if __name__ == "__main__":
    print("main")

    img_size = 224
    net_dimension = 18

    model = Attention_UNet(img_size=img_size, net_dimension=net_dimension)
    img = torch.zeros((1,2,224,224))

    neck = torch.randn((1, 49, 768))

    out = model(img, neck)
    print(out.shape)