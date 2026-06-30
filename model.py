"""
model.py

Exact AResUNet architecture, extracted from the original training notebook
(completefinal.ipynb). DO NOT modify the architecture below — it must match
the state_dict stored in model/newmodel333 (2).pth exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


# ================== CBAM ==================
class CBAM(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(c, c // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // r, c, 1, bias=False)
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1)
        )

    def forward(self, x):
        ca = self.fc(self.avg(x)) + self.fc(self.max(x))
        x = x * torch.sigmoid(ca)

        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.max(x, dim=1, keepdim=True)[0]
        sa = torch.sigmoid(self.spatial(torch.cat([avg, mx], dim=1)))

        return x * sa


# ================== BoundaryBlock ==================
class BoundaryBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


# ================== AttentionGate ==================
class AttentionGate(nn.Module):
    def __init__(self, f_g, f_x, f_int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(f_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(f_x, f_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(f_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)

        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)

        return x * psi


# ================== ASPP ==================
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.b0 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),                              nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.b1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=6,  dilation=6,  bias=False),    nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12, bias=False),    nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18, bias=False),    nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.proj = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        gap = F.interpolate(self.gap(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        return self.proj(torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x), gap], dim=1))


# ================== SEBlock ==================
class SEBlock(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // r, c, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


# ================== DecoderBlock ==================
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        concat_ch = in_ch // 2 + skip_ch

        self.conv1 = nn.Sequential(
            nn.Conv2d(concat_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.conv_dil = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.se = SEBlock(out_ch)

        self.residual = nn.Sequential(
            nn.Conv2d(concat_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        cat = torch.cat([x, skip], dim=1)

        out = self.conv1(cat)
        out = self.conv2(out)
        out = out + self.conv_dil(out)
        out = self.se(out)

        return self.relu(out + self.residual(cat))


# ================== MSFF ==================
class MSFF(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.s1 = nn.Sequential(
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        )

        self.s2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        )

        self.s3 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(ch * 3, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fuse(torch.cat([self.s1(x), self.s2(x), self.s3(x)], dim=1))


# ================== AResUNet ==================
class AResUNet(nn.Module):
    def __init__(self):
        super().__init__()

        _base = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,   # weights are loaded from checkpoint; avoids needing internet
            in_channels=3,
            classes=1
        )

        self.encoder = _base.encoder

        enc_ch = [64, 64, 128, 256, 512]

        self.aspp = ASPP(enc_ch[4], 256)

        dec_ch = [256, 128, 64, 64]

        self.ag3 = AttentionGate(256,       enc_ch[3], 128)
        self.ag2 = AttentionGate(dec_ch[0], enc_ch[2], 64)
        self.ag1 = AttentionGate(dec_ch[1], enc_ch[1], 32)
        self.ag0 = AttentionGate(dec_ch[2], enc_ch[0], 32)

        self.d3 = DecoderBlock(256,       enc_ch[3], dec_ch[0])
        self.d2 = DecoderBlock(dec_ch[0], enc_ch[2], dec_ch[1])
        self.d1 = DecoderBlock(dec_ch[1], enc_ch[1], dec_ch[2])
        self.d0 = DecoderBlock(dec_ch[2], enc_ch[0], dec_ch[3])

        # all refinement at half resolution to save memory
        self.boundary = BoundaryBlock(64)
        self.msff = MSFF(64)
        self.cbam = CBAM(64)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

        self.ds3 = nn.Conv2d(dec_ch[0], 1, kernel_size=1)
        self.ds2 = nn.Conv2d(dec_ch[1], 1, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        features = self.encoder(x)
        e0, e1, e2, e3, bot = features[1], features[2], features[3], features[4], features[5]

        bot = self.aspp(bot)

        d3 = self.d3(bot, self.ag3(bot, e3))
        d2 = self.d2(d3,  self.ag2(d3,  e2))
        d1 = self.d1(d2,  self.ag1(d2,  e1))
        d0 = self.d0(d1,  self.ag0(d1,  e0))

        # refinement at half resolution (d0 is 256x256) — avoids OOM
        d0 = self.boundary(d0)
        d0 = self.msff(d0)
        d0 = self.cbam(d0)

        # upsample only at the end
        out = F.interpolate(d0, size=(H, W), mode='bilinear', align_corners=False)

        logits = self.head(out)

        if self.training:
            ds3 = F.interpolate(self.ds3(d3), size=(H, W), mode='bilinear', align_corners=False)
            ds2 = F.interpolate(self.ds2(d2), size=(H, W), mode='bilinear', align_corners=False)
            return logits, ds3, ds2

        return logits
