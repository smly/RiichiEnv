import torch
import torch.nn as nn

from riichienv_ml.models.cql_model import ResNetBackbone


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network with shared ResNet backbone.

    - Actor head: outputs action logits (B, num_actions)
    - Critic head: outputs scalar state value (B, 1)

    Input:  (B, in_channels, 34)
    Output: (logits, value) tuple
    """
    def __init__(self, in_channels: int = 74, num_actions: int = 82,
                 conv_channels: int = 128, num_blocks: int = 8, fc_dim: int = 256):
        super().__init__()
        self.backbone = ResNetBackbone(in_channels, conv_channels, num_blocks, fc_dim)
        self.actor_head = nn.Linear(fc_dim, num_actions)
        self.critic_head = nn.Linear(fc_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value.squeeze(-1)
