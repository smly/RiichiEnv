import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from riichienv_ml.config import import_class


class MahjongLearner:
    def __init__(self,
                 device: str = "cuda",
                 lr: float = 1e-4,
                 lr_min: float = 1e-6,
                 num_steps: int = 1000000,
                 max_grad_norm: float = 1.0,
                 alpha_cql_init: float = 1.0,
                 alpha_cql_final: float = 0.1,
                 gamma: float = 0.99,
                 weight_decay: float = 0.0,
                 aux_weight: float = 0.0,
                 model_config: dict | None = None,
                 model_class: str = "riichienv_ml.models.cql_model.QNetwork"):

        self.device = torch.device(device)
        self.alpha_cql_init = alpha_cql_init
        self.alpha_cql_final = alpha_cql_final
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.aux_weight = aux_weight

        mc = model_config or {}
        ModelClass = import_class(model_class)
        self.model = ModelClass(**mc).to(self.device)
        self.has_aux = hasattr(self.model, 'aux_head') and self.model.aux_head is not None

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_steps, eta_min=lr_min)
        self.mse_loss = nn.MSELoss(reduction='none')
        self.total_steps = 0

    def get_weights(self):
        return self.model.state_dict()

    def load_cql_weights(self, path: str):
        """Load weights from a QNetwork checkpoint (handles old/new/dueling formats)."""
        cql_state = torch.load(path, map_location=self.device)

        has_backbone = any(k.startswith("backbone.") for k in cql_state.keys())
        has_v_head = any(k.startswith("v_head.") for k in cql_state.keys())
        has_a_head = any(k.startswith("a_head.") for k in cql_state.keys())
        has_head = any(k.startswith("head.") for k in cql_state.keys())

        if has_backbone and has_v_head and has_a_head:
            # New dueling QNetwork checkpoint -> direct load
            missing, unexpected = self.model.load_state_dict(cql_state, strict=False)
            print(f"Loaded dueling QNetwork weights from {path}")
        elif has_backbone and has_head:
            # Old non-dueling QNetwork: head.* -> a_head.*, v_head init random
            new_state = {}
            for k, v in cql_state.items():
                if k.startswith("head."):
                    new_state[k.replace("head.", "a_head.")] = v
                else:
                    new_state[k] = v
            missing, unexpected = self.model.load_state_dict(new_state, strict=False)
            print(f"Loaded old QNetwork weights from {path} (head -> a_head, v_head init random)")
        elif has_backbone:
            # UnifiedNetwork checkpoint -> extract backbone + map actor_head/critic_head
            new_state = {}
            for k, v in cql_state.items():
                if k.startswith("actor_head."):
                    new_state[k.replace("actor_head.", "a_head.")] = v
                elif k.startswith("critic_head."):
                    new_state[k.replace("critic_head.", "v_head.")] = v
                else:
                    new_state[k] = v
            missing, unexpected = self.model.load_state_dict(new_state, strict=False)
            print(f"Loaded UnifiedNetwork weights from {path} (mapped to dueling QNetwork)")
        else:
            # Old format (no backbone. prefix)
            new_state = {}
            for k, v in cql_state.items():
                if k.startswith("fc1.") or k.startswith("fc_shared."):
                    suffix = k.split(".", 1)[1]
                    new_state[f"backbone.fc.{suffix}"] = v
                elif k.startswith("fc2."):
                    suffix = k.split(".", 1)[1]
                    new_state[f"a_head.{suffix}"] = v
                elif k.startswith("actor_head."):
                    suffix = k.split(".", 1)[1]
                    new_state[f"a_head.{suffix}"] = v
                else:
                    new_state[f"backbone.{k}"] = v
            missing, unexpected = self.model.load_state_dict(new_state, strict=False)
            print(f"Loaded old-format weights from {path} (converted to dueling QNetwork)")

        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")

    def update(self, batch, max_steps=100000):
        """DQN + CQL update using a batch from the replay buffer."""
        # Ensure training mode with frozen BatchNorm stats (from CQL pretraining)
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.eval()

        features = batch["features"].to(self.device)
        actions = batch["action"].long().to(self.device)
        targets = batch["reward"].float().to(self.device)
        masks = batch["mask"].float().to(self.device)
        ranks = batch["rank"].long().to(self.device) if "rank" in batch.keys() else None

        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        if targets.dim() > 1:
            targets = targets.squeeze()

        # Dynamic CQL alpha scheduling
        progress = min(1.0, self.total_steps / max_steps)
        alpha_cql = self.alpha_cql_init + progress * (self.alpha_cql_final - self.alpha_cql_init)

        self.optimizer.zero_grad()

        # Forward pass (with aux if available)
        if self.has_aux and ranks is not None:
            q_values, aux_logits = self.model.forward_with_aux(features)
        else:
            q_values = self.model(features)
            aux_logits = None

        q_data = q_values.gather(1, actions).squeeze(1)

        # CQL Loss: logsumexp(Q) - Q_data
        invalid_mask = (masks == 0)
        q_masked = q_values.clone()
        q_masked = q_masked.masked_fill(invalid_mask, -1e9)
        logsumexp_q = torch.logsumexp(q_masked, dim=1)
        cql_term = logsumexp_q - q_data

        # MSE Loss
        mse_term = self.mse_loss(q_data, targets)

        # Auxiliary Loss (rank prediction)
        aux_loss_val = 0.0
        if aux_logits is not None and ranks is not None:
            aux_loss = F.cross_entropy(aux_logits, ranks)
            aux_loss_val = aux_loss.item()
        else:
            aux_loss = 0.0

        # Total Loss
        loss = (mse_term + alpha_cql * cql_term).mean() + self.aux_weight * aux_loss

        if torch.isnan(loss):
            return {
                "loss": 0.0,
                "cql": 0.0,
                "cql_alpha": alpha_cql,
                "mse": 0.0,
                "q_mean": 0.0,
            }

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        self.total_steps += 1

        # Q-value distribution metrics
        with torch.no_grad():
            q_probs = torch.softmax(q_masked, dim=1)
            q_log_probs = torch.log_softmax(q_masked, dim=1)
            q_entropy = -(q_probs * q_log_probs).sum(dim=1).mean().item()

        return {
            "loss": loss.item(),
            "cql": cql_term.mean().item(),
            "cql_alpha": alpha_cql,
            "mse": mse_term.mean().item(),
            "aux": aux_loss_val,
            "q_mean": q_data.mean().item(),
            "q_std": q_data.std().item(),
            "q_min": q_data.min().item(),
            "q_max": q_data.max().item(),
            "q_entropy": q_entropy,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "lr": self.scheduler.get_last_lr()[0],
        }
