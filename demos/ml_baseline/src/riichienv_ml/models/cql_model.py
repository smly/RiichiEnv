import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels, bias=True),
        )
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.constant_(mod.bias, 0)

    def forward(self, x):
        avg_out = self.shared_mlp(x.mean(-1))
        max_out = self.shared_mlp(x.amax(-1))
        weight = (avg_out + max_out).sigmoid()
        return weight.unsqueeze(-1) * x


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.ca = ChannelAttention(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out)
        out += residual
        out = self.relu(out)
        return out


class ResNetBackbone(nn.Module):
    """
    Shared CNN backbone: Conv1d projection -> N ResBlocks -> Flatten -> FC.
    Accepts any number of input channels.

    Input:  (B, in_channels, 34)
    Output: (B, fc_dim)
    """
    def __init__(self, in_channels: int = 74, conv_channels: int = 64,
                 num_blocks: int = 3, fc_dim: int = 256):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm1d(conv_channels)
        self.relu = nn.ReLU(inplace=True)
        self.res_blocks = nn.ModuleList([ResBlock(conv_channels) for _ in range(num_blocks)])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(conv_channels * 34, fc_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            out = block(out)
        out = self.flatten(out)
        out = self.relu(self.fc(out))
        return out


class QNetwork(nn.Module):
    """
    Q-Network for both offline CQL and online DQN training.
    Takes (B, C, 34) tensor input with any number of channels.

    Input:  (B, in_channels, 34)
    Output: (B, num_actions)
    """
    def __init__(self, in_channels: int = 74, num_actions: int = 82,
                 conv_channels: int = 64, num_blocks: int = 3, fc_dim: int = 256):
        super().__init__()
        self.backbone = ResNetBackbone(in_channels, conv_channels, num_blocks, fc_dim)
        self.head = nn.Linear(fc_dim, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)
