import torch
import torch.nn as nn
import torch.optim as optim

from riichienv_ml.config import import_class


class MahjongLearner:
    def __init__(self,
                 device: str = "cuda",
                 lr: float = 1e-4,
                 alpha_cql_init: float = 1.0,
                 alpha_cql_final: float = 0.1,
                 gamma: float = 0.99,
                 model_config: dict | None = None,
                 model_class: str = "riichienv_ml.models.cql_model.QNetwork"):

        self.device = torch.device(device)
        self.alpha_cql_init = alpha_cql_init
        self.alpha_cql_final = alpha_cql_final
        self.gamma = gamma

        mc = model_config or {}
        ModelClass = import_class(model_class)
        self.model = ModelClass(**mc).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss(reduction='none')
        self.total_steps = 0

    def get_weights(self):
        return self.model.state_dict()

    def load_cql_weights(self, path: str):
        """Load weights from a QNetwork checkpoint."""
        cql_state = torch.load(path, map_location=self.device)

        has_backbone = any(k.startswith("backbone.") for k in cql_state.keys())
        has_head = any(k.startswith("head.") for k in cql_state.keys())

        if has_backbone and has_head:
            # New QNetwork checkpoint -> direct load
            missing, unexpected = self.model.load_state_dict(cql_state, strict=False)
            print(f"Loaded QNetwork weights from {path}")
        elif has_backbone:
            # UnifiedNetwork checkpoint -> extract backbone + map actor_head/critic_head -> head
            new_state = {}
            for k, v in cql_state.items():
                if k.startswith("actor_head.") or k.startswith("critic_head."):
                    head_key = k.replace("actor_head.", "head.").replace("critic_head.", "head.")
                    if head_key not in new_state:
                        new_state[head_key] = v
                else:
                    new_state[k] = v
            missing, unexpected = self.model.load_state_dict(new_state, strict=False)
            print(f"Loaded UnifiedNetwork weights from {path} (mapped to QNetwork)")
        else:
            # Old format (no backbone. prefix)
            new_state = {}
            for k, v in cql_state.items():
                if k.startswith("fc1.") or k.startswith("fc_shared."):
                    suffix = k.split(".", 1)[1]
                    new_state[f"backbone.fc.{suffix}"] = v
                elif k.startswith("fc2."):
                    suffix = k.split(".", 1)[1]
                    new_state[f"head.{suffix}"] = v
                elif k.startswith("actor_head."):
                    suffix = k.split(".", 1)[1]
                    new_state[f"head.{suffix}"] = v
                else:
                    new_state[f"backbone.{k}"] = v
            missing, unexpected = self.model.load_state_dict(new_state, strict=False)
            print(f"Loaded old-format weights from {path} (converted to QNetwork)")

        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")

    def update(self, batch, max_steps=100000):
        """DQN + CQL update using a batch from the replay buffer."""
        features = batch["features"].to(self.device)
        actions = batch["action"].long().to(self.device)
        targets = batch["reward"].float().to(self.device)
        masks = batch["mask"].float().to(self.device)

        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        if targets.dim() > 1:
            targets = targets.squeeze()

        # Dynamic CQL alpha scheduling
        progress = min(1.0, self.total_steps / max_steps)
        alpha_cql = self.alpha_cql_init + progress * (self.alpha_cql_final - self.alpha_cql_init)

        self.optimizer.zero_grad()

        q_values = self.model(features)
        q_data = q_values.gather(1, actions).squeeze(1)

        # CQL Loss: logsumexp(Q) - Q_data
        invalid_mask = (masks == 0)
        q_masked = q_values.clone()
        q_masked = q_masked.masked_fill(invalid_mask, -1e9)
        logsumexp_q = torch.logsumexp(q_masked, dim=1)
        cql_term = logsumexp_q - q_data

        # MSE Loss
        mse_term = self.mse_loss(q_data, targets)

        # Total Loss
        loss = (mse_term + alpha_cql * cql_term).mean()

        if torch.isnan(loss):
            return {
                "loss": 0.0,
                "cql": 0.0,
                "cql_alpha": alpha_cql,
                "mse": 0.0,
                "q_mean": 0.0,
            }

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.total_steps += 1

        return {
            "loss": loss.item(),
            "cql": cql_term.mean().item(),
            "cql_alpha": alpha_cql,
            "mse": mse_term.mean().item(),
            "q_mean": q_data.mean().item(),
        }
