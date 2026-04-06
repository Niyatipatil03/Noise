"""
PyTorch CNN14-BSR definition — mirrors src/models/cnn14_bsr.py.
Used by convert_pth_to_tflite.py to reconstruct the model before loading weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        r = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, r, bias=False),
            nn.ReLU(),
            nn.Linear(r, channels, bias=False),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        avg = x.mean(dim=[2, 3])
        mx  = x.flatten(2).max(dim=2).values
        scale = torch.sigmoid(self.fc(avg) + self.fc(mx)).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        scale = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * scale


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_attention=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.cbam  = CBAM(out_ch) if use_attention else nn.Identity()
        self.pool  = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.cbam(x)
        return self.pool(x)


class CNN14_BSR(nn.Module):
    """CNN14 with CBAM attention for BSR noise detection."""

    def __init__(self, input_shape=(1, 128, 300), num_noise_types=8,
                 embedding_dim=512, dropout_rate=0.4, use_attention=True):
        super().__init__()
        self.input_bn = nn.BatchNorm2d(input_shape[0])

        self.block1 = ConvBlock(input_shape[0], 64,   use_attention)
        self.block2 = ConvBlock(64,  128,  use_attention)
        self.block3 = ConvBlock(128, 256,  use_attention)
        self.block4 = ConvBlock(256, 512,  use_attention)
        self.block5 = ConvBlock(512, 1024, use_attention)

        # Final block (no pool)
        self.conv6a = nn.Conv2d(1024, 2048, 3, padding=1, bias=False)
        self.bn6a   = nn.BatchNorm2d(2048)
        self.conv6b = nn.Conv2d(2048, 2048, 3, padding=1, bias=False)
        self.bn6b   = nn.BatchNorm2d(2048)
        self.cbam6  = CBAM(2048) if use_attention else nn.Identity()

        self.fc1     = nn.Linear(2048 * 2, embedding_dim, bias=False)  # avg+max
        self.bn_fc1  = nn.BatchNorm1d(embedding_dim)
        self.drop1   = nn.Dropout(dropout_rate)

        self.fc2     = nn.Linear(embedding_dim, embedding_dim // 2, bias=False)
        self.bn_fc2  = nn.BatchNorm1d(embedding_dim // 2)
        self.drop2   = nn.Dropout(dropout_rate * 0.75)

        self.binary_head = nn.Linear(embedding_dim // 2, 1)
        self.type_head   = nn.Linear(embedding_dim // 2, num_noise_types)

    def forward(self, x):
        # x: (B, 1, n_mels, T)
        x = self.input_bn(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = F.relu(self.bn6a(self.conv6a(x)))
        x = F.relu(self.bn6b(self.conv6b(x)))
        x = self.cbam6(x)

        avg = x.mean(dim=[2, 3])
        mx  = x.flatten(2).max(dim=2).values
        x   = torch.cat([avg, mx], dim=1)

        x = self.drop1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn_fc2(self.fc2(x))))

        binary = torch.sigmoid(self.binary_head(x))
        ntype  = torch.softmax(self.type_head(x), dim=-1)
        return binary, ntype
