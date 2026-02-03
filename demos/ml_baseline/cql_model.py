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


class QNetwork(nn.Module):
    """
    CNN-based Q-Network that processes spatial tile features.
    Additional non-spatial features are processed separately and concatenated.
    Total features: 6765 (110 spatial channels × 34 + 3025 non-spatial)
    """
    def __init__(self, in_channels=74, num_actions=82, filters=64, blocks=3):
        super().__init__()
        # Process spatial features: standard (74, 34) + decay (4, 34) + kawa (28, 34) + ankan (4, 34) = (110, 34)
        total_spatial_channels = 110
        self.conv_in = nn.Conv1d(total_spatial_channels, filters, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm1d(filters)
        self.relu = nn.ReLU(inplace=True)

        self.res_blocks = nn.ModuleList([ResBlock(filters) for _ in range(blocks)])

        self.flatten = nn.Flatten()

        # Non-spatial features: yaku (168) + furiten (84) + shanten (16) + fuuro (2720) + action (11)
        #                       + riichi_sutehais (9) + last_tedashis (9) + pass_context (3) + discard_candidates (5) = 3025
        non_spatial_dim = 168 + 84 + 16 + 2720 + 11 + 9 + 9 + 3 + 5

        # Combine spatial and non-spatial features
        # Input to FC: filters * 34 + non_spatial_dim
        combined_dim = filters * 34 + non_spatial_dim
        self.ln1 = nn.LayerNorm(combined_dim)  # Stabilize large input
        self.fc1 = nn.Linear(combined_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_actions)

    def forward(self, x):
        """
        Args:
            x: dict or tensor
                If dict: {'standard', 'decay', 'yaku', 'furiten', 'shanten', 'kawa', 'fuuro', 'ankan', 'action',
                          'riichi_sutehais', 'last_tedashis', 'pass_context', 'discard_candidates'}
                If tensor: (B, 110, 34) - concatenated spatial features
        """
        if isinstance(x, dict):
            # Extract features from dict
            standard = x['standard']  # (B, 74, 34)
            decay = x['decay']        # (B, 4, 34)
            yaku = x['yaku']          # (B, 4, 21, 2)
            furiten = x['furiten']    # (B, 4, 21)
            shanten = x['shanten']    # (B, 4, 4)
            kawa = x['kawa']          # (B, 4, 7, 34)
            fuuro = x['fuuro']        # (B, 4, 4, 5, 34)
            ankan = x['ankan']        # (B, 4, 34)
            action = x['action']      # (B, 11)
            riichi_sutehais = x['riichi_sutehais']  # (B, 3, 3)
            last_tedashis = x['last_tedashis']      # (B, 3, 3)
            pass_context = x['pass_context']        # (B, 3)
            discard_candidates = x['discard_candidates']  # (B, 5)

            # Reshape kawa from (B, 4, 7, 34) to (B, 28, 34)
            B = kawa.shape[0]
            kawa_reshaped = kawa.view(B, 4 * 7, 34)  # (B, 28, 34)

            # Concatenate spatial features
            spatial = torch.cat([standard, decay, kawa_reshaped, ankan], dim=1)  # (B, 110, 34)

            # Flatten non-spatial features
            yaku_flat = yaku.flatten(1)      # (B, 168)
            furiten_flat = furiten.flatten(1)  # (B, 84)
            shanten_flat = shanten.flatten(1)  # (B, 16)
            fuuro_flat = fuuro.flatten(1)    # (B, 2720)
            action_flat = action.flatten(1)  # (B, 11)
            riichi_sutehais_flat = riichi_sutehais.flatten(1)  # (B, 9)
            last_tedashis_flat = last_tedashis.flatten(1)      # (B, 9)
            pass_context_flat = pass_context.flatten(1)        # (B, 3)
            discard_candidates_flat = discard_candidates.flatten(1)  # (B, 5)
            non_spatial = torch.cat([
                yaku_flat, furiten_flat, shanten_flat, fuuro_flat, action_flat,
                riichi_sutehais_flat, last_tedashis_flat, pass_context_flat, discard_candidates_flat
            ], dim=1)  # (B, 3025)
        else:
            # Assume already concatenated (legacy support)
            spatial = x
            non_spatial = None

        # Process spatial features with CNN
        out = self.conv_in(spatial)
        out = self.bn_in(out)
        out = self.relu(out)
        for block in self.res_blocks:
            out = block(out)

        out = self.flatten(out)  # (B, filters*34)

        # Concatenate with non-spatial features if available
        if non_spatial is not None:
            out = torch.cat([out, non_spatial], dim=1)

        # MLP head with LayerNorm for stability
        out = self.ln1(out)  # Normalize large combined input
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)

        # Returns unnormalized logits (Q-values)
        return out


class QNetworkMLP(nn.Module):
    """
    MLP-based Q-Network that processes flattened features (6765 dimensions).
    Simpler architecture, may be faster to train but loses spatial structure.
    """
    def __init__(self, input_dim=6765, num_actions=82, hidden_dims=[1024, 512, 256]):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_actions))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 2920) - flattened features
        return self.mlp(x)
