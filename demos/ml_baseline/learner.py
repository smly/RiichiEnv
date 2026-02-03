import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from unified_model import UnifiedNetwork


class MahjongLearner:
    def __init__(self,
                 device: str = "cuda",
                 actor_lr: float = 1e-4,
                 critic_lr: float = 3e-4,
                 alpha_cql_init: float = 1.0,
                 alpha_cql_final: float = 0.1,
                 gamma: float = 0.99,
                 awac_beta: float = 0.3,
                 awac_max_weight: float = 20.0):

        self.device = torch.device(device)
        self.alpha_cql_init = alpha_cql_init
        self.alpha_cql_final = alpha_cql_final
        self.gamma = gamma
        self.awac_beta = awac_beta
        self.awac_max_weight = awac_max_weight

        # Model with legacy 46-channel features
        self.model = UnifiedNetwork(num_actions=82).to(self.device)

        # Unified optimizer with parameter groups
        # Actor head gets actor_lr, critic head gets critic_lr, shared backbone gets average
        shared_lr = (actor_lr + critic_lr) / 2.0

        self.optimizer = optim.Adam([
            {'params': self.model.actor_head.parameters(), 'lr': actor_lr},
            {'params': self.model.critic_head.parameters(), 'lr': critic_lr},
            {'params': self.model.conv_in.parameters(), 'lr': shared_lr},
            {'params': self.model.bn_in.parameters(), 'lr': shared_lr},
            {'params': [p for block in self.model.res_blocks for p in block.parameters()], 'lr': shared_lr},
            {'params': self.model.fc_shared.parameters(), 'lr': shared_lr},
        ])

        self.mse_loss = nn.MSELoss(reduction='none')
        self.total_steps = 0

    def get_weights(self):
        return self.model.state_dict()

    def load_cql_weights(self, path: str):
        """
        Load weights from an offline CQL model (QNetwork).
        Old QNetwork architecture: CNN -> fc1(2176->256) -> fc2(256->82)
        New UnifiedNetwork: CNN -> fc_shared(2176->256) -> actor_head/critic_head(256->82)

        Mapping:
        - CNN backbone: direct copy
        - fc1 -> fc_shared
        - fc2 -> both actor_head and critic_head (initialize with same weights)
        """
        cql_state = torch.load(path, map_location=self.device)
        new_state = {}

        for k, v in cql_state.items():
            if k.startswith("fc1"):
                # fc1 -> fc_shared
                new_key = k.replace("fc1", "fc_shared")
                new_state[new_key] = v
            elif k.startswith("fc2"):
                # fc2 -> both actor_head and critic_head
                actor_key = k.replace("fc2", "actor_head")
                critic_key = k.replace("fc2", "critic_head")
                new_state[actor_key] = v
                new_state[critic_key] = v.clone()  # Clone for critic
            else:
                # Copy backbone weights directly (conv_in, bn_in, res_blocks)
                new_state[k] = v

        missing, unexpected = self.model.load_state_dict(new_state, strict=False)
        print(f"Loaded CQL weights from {path}")
        print(f"Note: actor_head and critic_head initialized with same fc2 weights")
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")


    def update_critic(self, batch, max_steps=100000):
        """
        CQL Update using a batch from Critic Buffer (Historical/Offline + Online).
        Batch keys: features, action, reward, mask, next_features (if valid), done, index, _weight
        """
        # Get features as tensor (B, 46, 34)
        features = batch["features"].to(self.device)

        actions = batch["action"].long().to(self.device) # (B) or (B,1)
        targets = batch["reward"].float().to(self.device) # G_t
        masks = batch["mask"].float().to(self.device)
        weights = batch["_weight"].float().to(self.device)
        indices = batch["index"].long().to(self.device)

        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        if targets.dim() > 1:
            targets = targets.squeeze()

        # Dynamic CQL alpha scheduling: decrease as online learning progresses
        progress = min(1.0, self.total_steps / max_steps)
        alpha_cql = self.alpha_cql_init + progress * (self.alpha_cql_final - self.alpha_cql_init)

        # Clear gradients before forward pass
        self.optimizer.zero_grad()

        # 1. Forward Pass
        # We only care about Q-values here, but forward returns both
        _, q_values = self.model(features) # (B, 82)

        q_data = q_values.gather(1, actions).squeeze(1) # (B)

        # 2. CQL Loss: logsumexp(Q) - Q_data
        invalid_mask = (masks == 0)
        q_masked = q_values.clone()
        q_masked = q_masked.masked_fill(invalid_mask, -1e9)
        logsumexp_q = torch.logsumexp(q_masked, dim=1) # (B)

        cql_term = (logsumexp_q - q_data) # (B)

        # 3. Bellman/MC Error
        # Loss = (Q - G_t)^2
        mse_term = self.mse_loss(q_data, targets) # (B)

        # 4. Total Loss & IS Weights
        # Loss = mean( (MSE + alpha*CQL) * weight )
        raw_loss = mse_term + alpha_cql * cql_term
        loss = (raw_loss * weights).mean()

        # Check for NaN in loss
        if torch.isnan(loss):
            print(f"WARNING: NaN detected in critic loss at step {self.total_steps}")
            print(f"  q_data: min={q_data.min().item()}, max={q_data.max().item()}, mean={q_data.mean().item()}")
            print(f"  targets: min={targets.min().item()}, max={targets.max().item()}, mean={targets.mean().item()}")
            print(f"  mse_term: min={mse_term.min().item()}, max={mse_term.max().item()}, mean={mse_term.mean().item()}")
            print(f"  cql_term: min={cql_term.min().item()}, max={cql_term.max().item()}, mean={cql_term.mean().item()}")
            # Skip this update
            return {
                "critic/loss": 0.0,
                "critic/cql": 0.0,
                "critic/cql_alpha": alpha_cql,
                "critic/mse": 0.0,
                "critic/q_mean": 0.0,
                "critic/mc_error": 0.0,
                "critic/advantage": 0.0,
            }, indices, torch.ones_like(targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.total_steps += 1
        
        # 5. Priority Calculation
        with torch.no_grad():
            # MC Error = |Q - G_t|
            mc_error = torch.abs(q_data - targets)

            # Advantage = |G_t - V(s)| approx |G_t - max Q|
            # Note: q_masked handles invalid actions
            max_q, _ = q_masked.max(dim=1)
            advantage = torch.abs(targets - max_q)

            # Priority = MC_Error + Advantage + epsilon
            new_priorities = mc_error + advantage + 1e-6

        return {
            "critic/loss": loss.item(),
            "critic/cql": (cql_term * weights).mean().item(),
            "critic/cql_alpha": alpha_cql,
            "critic/mse": (mse_term * weights).mean().item(),
            "critic/q_mean": q_data.mean().item(),
            "critic/mc_error": mc_error.mean().item(),
            "critic/advantage": advantage.mean().item(),
        }, indices, new_priorities

    def update_actor(self, batch):
        """
        AWAC (Advantage Weighted Actor-Critic) Update - Off-Policy Learning.
        Batch contains: features, action, reward (G_t), mask.
        No policy version checking needed (off-policy).
        """
        # Get features as tensor (B, 46, 34)
        features = batch["features"].to(self.device)

        actions = batch["action"].long().to(self.device)
        returns = batch["reward"].float().to(self.device) # G_t
        masks = batch["mask"].float().to(self.device)

        if actions.dim() > 1: actions = actions.squeeze(1)
        if returns.dim() > 1: returns = returns.squeeze(1)

        # Advantage Estimation (raw, not normalized for AWAC)
        with torch.no_grad():
            _, q_values = self.model(features)
            q_values = q_values.masked_fill(masks == 0, -1e9)
            values, _ = q_values.max(dim=1) # (B)
            advantages = returns - values

        # Clear gradients before forward pass
        self.optimizer.zero_grad()

        # AWAC: Advantage-weighted policy gradient
        logits, _ = self.model(features)
        logits = logits.masked_fill(masks == 0, -1e9)
        dist = Categorical(logits=logits)

        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # AWAC weight: exp(advantage / beta), clamped
        weights = torch.exp(advantages / self.awac_beta)
        weights = torch.clamp(weights, max=self.awac_max_weight)

        # AWAC loss: weighted negative log likelihood
        actor_loss = -(new_log_probs * weights.detach()).mean()
        loss = actor_loss - 0.01 * entropy

        # Check for NaN in loss
        if torch.isnan(loss):
            print(f"WARNING: NaN detected in actor loss at step {self.total_steps}")
            print(f"  logits: min={logits.min().item()}, max={logits.max().item()}, mean={logits.mean().item()}")
            print(f"  new_log_probs: min={new_log_probs.min().item()}, max={new_log_probs.max().item()}")
            print(f"  advantages: min={advantages.min().item()}, max={advantages.max().item()}")
            print(f"  weights: min={weights.min().item()}, max={weights.max().item()}")
            # Skip this update
            return {
                "actor/loss": 0.0,
                "actor/entropy": 0.0,
                "actor/advantage": 0.0,
                "actor/awac_weight_mean": 0.0,
            }

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        return {
            "actor/loss": loss.item(),
            "actor/entropy": entropy.item(),
            "actor/advantage": advantages.mean().item(),
            "actor/awac_weight_mean": weights.mean().item(),
        }