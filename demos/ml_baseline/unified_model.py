import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class UnifiedNetwork(nn.Module):
    """
    Unified Network with CNN backbone for actor-critic architecture.

    Legacy mode (default): Uses 46-channel spatial features (46, 34)
    New mode (dict input): Uses 110-channel spatial + 3025 non-spatial features

    This model supports both modes for backwards compatibility.
    """
    def __init__(self, in_channels=46, num_actions=82, filters=64, blocks=3):
        super().__init__()
        # Process spatial features (legacy: 46 channels)
        self.conv_in = nn.Conv1d(in_channels, filters, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm1d(filters)
        self.relu = nn.ReLU(inplace=True)

        self.res_blocks = nn.ModuleList([ResBlock(filters) for _ in range(blocks)])

        self.flatten = nn.Flatten()

        # Legacy mode: Match old QNetwork architecture exactly
        # Old QNetwork: fc1(2176->256) -> fc2(256->82)
        combined_dim_legacy = filters * 34  # 64 * 34 = 2176

        # Match old QNetwork architecture for compatibility
        self.fc_shared = nn.Linear(combined_dim_legacy, 256)

        # Dual Heads (actor and critic share same fc1, but have separate output heads)
        self.actor_head = nn.Linear(256, num_actions)
        self.critic_head = nn.Linear(256, num_actions)

    def forward(self, x):
        """
        Args:
            x: Either a tensor (B, 46, 34) for legacy mode, or dict with keys for new mode:
                'standard': (B, 74, 34) - spatial features
                'decay': (B, 4, 34) - discard decay
                'yaku': (B, 4, 21, 2) - yaku possibility
                'furiten': (B, 4, 21) - furiten detection
                'shanten': (B, 4, 4) - shanten & efficiency
                'kawa': (B, 4, 7, 34) - discard pile overview
                'fuuro': (B, 4, 4, 5, 34) - meld details
                'ankan': (B, 4, 34) - concealed kan
                'action': (B, 11) - action availability
                'riichi_sutehais': (B, 3, 3) - riichi discards
                'last_tedashis': (B, 3, 3) - last hand discards
                'pass_context': (B, 3) - current offer
                'discard_candidates': (B, 5) - discard detail
        """
        if isinstance(x, dict):
            raise NotImplementedError("Dict input mode is deprecated. Use legacy tensor input (B, 46, 34).")

        # Legacy mode: tensor input (B, 46, 34)
        spatial = x

        # Process spatial features with CNN
        out = self.conv_in(spatial)
        out = self.bn_in(out)
        out = self.relu(out)
        for block in self.res_blocks:
            out = block(out)

        out = self.flatten(out)  # (B, filters*34 = 2176)

        # Shared layer (match old QNetwork: fc1)
        features = self.relu(self.fc_shared(out))  # (B, 256)

        # Dual heads
        logits = self.actor_head(features)   # (B, 82)
        q_values = self.critic_head(features)  # (B, 82)

        # Returns (logits, q_values)
        return logits, q_values
