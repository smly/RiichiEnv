import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from unified_model import UnifiedNetwork


class MahjongLearner:
    def __init__(self, 
                 device: str = "cuda",
                 lr: float = 1e-4,
                 alpha_cql: float = 1.0,
                 gamma: float = 0.99,
                 ppo_clip: float = 0.2,
                 ppo_epochs: int = 4):
        
        self.device = torch.device(device)
        self.alpha_cql = alpha_cql
        self.gamma = gamma
        self.ppo_clip = ppo_clip
        self.ppo_epochs = ppo_epochs
        
        self.model = UnifiedNetwork(in_channels=46, num_actions=82).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss(reduction='none')

        self.policy_version = 0

    def get_weights(self):
        return self.model.state_dict()

    def load_cql_weights(self, path: str):
        """
        Load weights from an offline CQL model (QNetwork).
        Mapping:
        - backbone -> backbone
        - fc1 -> fc_shared
        - fc2 -> critic_head
        - fc2 -> actor_head (Initialize Actor with Q-values -> Boltzmann Policy)
        """
        cql_state = torch.load(path, map_location=self.device)
        new_state = {}
        
        for k, v in cql_state.items():
            if k.startswith("fc1"):
                new_key = k.replace("fc1", "fc_shared")
                new_state[new_key] = v
            elif k.startswith("fc2"):
                critic_key = k.replace("fc2", "critic_head")
                new_state[critic_key] = v
                
                actor_key = k.replace("fc2", "actor_head")
                new_state[actor_key] = v
            else:
                new_state[k] = v

        self.model.load_state_dict(new_state, strict=True)


    def update_critic(self, batch):
        """
        CQL Update using a batch from Critic Buffer (Historical/Offline).
        Batch keys: features, action, reward, mask, next_features (if valid), done, index, _weight
        """
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
        raw_loss = mse_term + self.alpha_cql * cql_term
        loss = (raw_loss * weights).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
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
            "critic/mse": (mse_term * weights).mean().item(),
            "critic/q_mean": q_data.mean().item(),
            "critic/mc_error": mc_error.mean().item(),
            "critic/advantage": advantage.mean().item(),
        }, indices, new_priorities

    def update_actor(self, batch):
        """
        PPO Update using batch from Actor Buffer (Recent).
        Batch contains: features, action, log_prob (old), reward (G_t), mask.
        """
        features = batch["features"].to(self.device)
        actions = batch["action"].long().to(self.device)
        old_log_probs = batch["log_prob"].float().to(self.device)
        returns = batch["reward"].float().to(self.device) # G_t
        masks = batch["mask"].float().to(self.device)
        
        if actions.dim() > 1: actions = actions.squeeze(1)
        if old_log_probs.dim() > 1: old_log_probs = old_log_probs.squeeze(1)
        if returns.dim() > 1: returns = returns.squeeze(1)

        # Advantage Estimation
        with torch.no_grad():
            _, q_values = self.model(features)
            q_values = q_values.masked_fill(masks == 0, -1e9)
            values, _ = q_values.max(dim=1) # (B)
            
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        # PPO Step
        logits, _ = self.model(features)
        logits = logits.masked_fill(masks == 0, -1e9)
        dist = Categorical(logits=logits)
        
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advantages
        
        actor_loss = -torch.min(surr1, surr2).mean()
        loss = actor_loss - 0.01 * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.policy_version += 1
        
        return {
            "actor/loss": loss.item(),
            "actor/entropy": entropy.item(),
            "actor/advantage": advantages.mean().item()
        }