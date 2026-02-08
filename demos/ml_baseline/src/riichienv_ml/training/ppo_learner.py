import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

from riichienv_ml.config import import_class


class PPOLearner:
    def __init__(self,
                 device: str = "cuda",
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 ppo_clip: float = 0.2,
                 ppo_epochs: int = 4,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 model_config: dict | None = None,
                 model_class: str = "riichienv_ml.models.actor_critic.ActorCriticNetwork"):

        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_clip = ppo_clip
        self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        mc = model_config or {}
        ModelClass = import_class(model_class)
        self.model = ModelClass(**mc).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.total_steps = 0

    def get_weights(self):
        return self.model.state_dict()

    def load_weights(self, path: str):
        """Load weights with backward compatibility for QNetwork checkpoints."""
        state = torch.load(path, map_location=self.device)

        has_actor = any(k.startswith("actor_head.") for k in state.keys())
        has_critic = any(k.startswith("critic_head.") for k in state.keys())

        if has_actor and has_critic:
            # Native ActorCriticNetwork checkpoint
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            logger.info(f"Loaded ActorCriticNetwork weights from {path}")
        elif any(k.startswith("head.") for k in state.keys()):
            # QNetwork checkpoint: map head -> actor_head, init critic_head randomly
            new_state = {}
            for k, v in state.items():
                if k.startswith("head."):
                    new_state[k.replace("head.", "actor_head.")] = v
                else:
                    new_state[k] = v
            missing, unexpected = self.model.load_state_dict(new_state, strict=False)
            logger.info(f"Loaded QNetwork weights from {path} (head -> actor_head, critic_head initialized randomly)")
        else:
            # Old format
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            logger.info(f"Loaded weights from {path} (best effort)")

        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")

    def update(self, rollout_batch: dict) -> dict:
        """
        PPO update over a batch of on-policy trajectory data.

        rollout_batch contains:
            features: (N, C, 34) observation features
            masks: (N, num_actions) legal action masks
            actions: (N,) actions taken
            old_log_probs: (N,) log-prob under old policy
            advantages: (N,) GAE advantages
            returns: (N,) discounted returns (advantage + value)
        """
        features = rollout_batch["features"].to(self.device)
        masks = rollout_batch["masks"].to(self.device)
        actions = rollout_batch["actions"].long().to(self.device)
        old_log_probs = rollout_batch["old_log_probs"].to(self.device)
        advantages = rollout_batch["advantages"].to(self.device)
        returns = rollout_batch["returns"].to(self.device)

        # Pre-normalization advantage/return stats (for diagnostics)
        adv_raw_mean = advantages.mean().item()
        adv_raw_std = advantages.std().item()
        return_mean = returns.mean().item()
        return_std = returns.std().item()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "loss": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
            "ratio/mean": 0.0,
            "ratio/std": 0.0,
            "ratio/max": 0.0,
            "kl/max": 0.0,
            "value/predicted_mean": 0.0,
            "grad_norm": 0.0,
        }

        N = features.shape[0]

        for epoch in range(self.ppo_epochs):
            # Shuffle indices for mini-batching within epoch
            perm = torch.randperm(N, device=self.device)

            for start in range(0, N, 128):
                end = min(start + 128, N)
                idx = perm[start:end]

                batch_features = features[idx]
                batch_masks = masks[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]

                logits, values = self.model(batch_features)

                # Mask invalid actions
                mask_bool = batch_masks.bool()
                logits = logits.masked_fill(~mask_bool, -1e9)

                # Action log-probs and entropy
                log_probs_all = torch.log_softmax(logits, dim=-1)
                log_probs = log_probs_all.gather(1, batch_actions.unsqueeze(1)).squeeze(1)

                # Entropy: -sum(p * log p) over valid actions only
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * log_probs_all).sum(dim=-1)
                # Zero out contribution from invalid actions (already -1e9 so probs ~ 0)
                entropy = entropy.mean()

                # Policy loss (PPO clipping)
                ratio = (log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = ratio.clamp(1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, batch_returns)

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                if torch.isnan(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Accumulate metrics
                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - log_probs).mean().item()
                    kl_max = (batch_old_log_probs - log_probs).max().item()
                    clip_frac = ((ratio - 1.0).abs() > self.ppo_clip).float().mean().item()

                total_metrics["policy_loss"] += policy_loss.item()
                total_metrics["value_loss"] += value_loss.item()
                total_metrics["entropy"] += entropy.item()
                total_metrics["loss"] += loss.item()
                total_metrics["approx_kl"] += approx_kl
                total_metrics["clip_frac"] += clip_frac
                total_metrics["ratio/mean"] += ratio.mean().item()
                total_metrics["ratio/std"] += ratio.std().item()
                total_metrics["ratio/max"] = max(total_metrics["ratio/max"], ratio.max().item())
                total_metrics["kl/max"] = max(total_metrics["kl/max"], kl_max)
                total_metrics["value/predicted_mean"] += values.mean().item()
                total_metrics["grad_norm"] += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

        # Average across all mini-batches
        num_batches = self.ppo_epochs * max(1, (N + 127) // 128)
        avg_keys = ["policy_loss", "value_loss", "entropy", "loss", "approx_kl",
                     "clip_frac", "ratio/mean", "ratio/std", "value/predicted_mean", "grad_norm"]
        for k in avg_keys:
            total_metrics[k] /= num_batches

        # Add rollout-level stats (not averaged over mini-batches)
        total_metrics["adv/raw_mean"] = adv_raw_mean
        total_metrics["adv/raw_std"] = adv_raw_std
        total_metrics["return/mean"] = return_mean
        total_metrics["return/std"] = return_std
        total_metrics["return/target_mean"] = returns.mean().item()

        # Explained variance: how well value predictions explain return variance
        with torch.no_grad():
            logits_all, values_all = self.model(features)
            v_pred = values_all.detach()
            var_returns = returns.var()
            if var_returns < 1e-8:
                total_metrics["explained_variance"] = 0.0
            else:
                total_metrics["explained_variance"] = (1.0 - (returns - v_pred).var() / var_returns).item()

        self.total_steps += 1
        return total_metrics
