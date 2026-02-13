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
                 alpha_kl: float = 0.0,
                 gamma: float = 0.99,
                 weight_decay: float = 0.0,
                 aux_weight: float = 0.0,
                 entropy_coef: float = 0.0,
                 model_config: dict | None = None,
                 model_class: str = "riichienv_ml.models.cql_model.QNetwork"):

        self.device = torch.device(device)
        self.alpha_cql_init = alpha_cql_init
        self.alpha_cql_final = alpha_cql_final
        self.alpha_kl = alpha_kl
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.aux_weight = aux_weight
        self.entropy_coef = entropy_coef

        mc = model_config or {}
        ModelClass = import_class(model_class)
        self.model = ModelClass(**mc).to(self.device)
        self.has_aux = hasattr(self.model, 'aux_head') and self.model.aux_head is not None

        # Frozen reference model for KL regularization (loaded in load_cql_weights)
        self.ref_model = None
        if alpha_kl > 0:
            self.ref_model = ModelClass(**mc).to(self.device)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

        # Mortal-style selective weight decay: only Linear.weight and Conv1d.weight
        decay_params = []
        no_decay_params = []
        for mod_name, mod in self.model.named_modules():
            for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)
        param_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        self.optimizer = optim.AdamW(param_groups, lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_steps, eta_min=lr_min)
        self.td_loss = nn.SmoothL1Loss(reduction='none')
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

        # Copy loaded weights to frozen reference model for KL regularization
        if self.ref_model is not None:
            self.ref_model.load_state_dict(self.model.state_dict())
            print(f"Loaded reference model for KL regularization (frozen)")

    def update(self, batch, max_steps=100000):
        """DQN + KL/CQL update using a batch from the on-policy buffer."""
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

        # Forward pass (with aux if available), passing mask for Mortal-style
        # Dueling DQN (advantage mean over legal actions only)
        if self.has_aux and ranks is not None:
            q_values, aux_logits = self.model.forward_with_aux(features, mask=masks)
        else:
            q_values = self.model(features, mask=masks)
            aux_logits = None

        q_data = q_values.gather(1, actions).squeeze(1)

        invalid_mask = (masks == 0)
        q_masked = q_values.clone()
        q_masked = q_masked.masked_fill(invalid_mask, -1e9)

        # CQL Loss: logsumexp(Q) - Q_data
        logsumexp_q = torch.logsumexp(q_masked, dim=1)
        cql_term = logsumexp_q - q_data

        # KL Regularization: KL(π_current || π_pretrained) over legal actions
        # Prevents policy drift without reinforcing current greedy selections
        kl_val = 0.0
        if self.ref_model is not None and self.alpha_kl > 0:
            with torch.no_grad():
                ref_q = self.ref_model(features, mask=masks)
            ref_q_masked = ref_q.masked_fill(invalid_mask, -1e9)
            # KL(current || reference) = Σ P_cur * (log P_cur - log P_ref)
            cur_log_probs = F.log_softmax(q_masked, dim=1)
            ref_log_probs = F.log_softmax(ref_q_masked, dim=1)
            cur_probs = cur_log_probs.exp()
            # Mask illegal actions to avoid NaN from 0 * -inf
            kl_per_action = cur_probs * (cur_log_probs - ref_log_probs)
            kl_per_action = kl_per_action.masked_fill(invalid_mask, 0.0)
            kl_term = kl_per_action.sum(dim=1).mean()
            kl_val = kl_term.item()
        else:
            kl_term = 0.0

        # TD Loss (Huber / Smooth L1 — less sensitive to outliers)
        td_term = self.td_loss(q_data, targets)

        # Auxiliary Loss (rank prediction)
        aux_loss_val = 0.0
        if aux_logits is not None and ranks is not None:
            aux_loss = F.cross_entropy(aux_logits, ranks)
            aux_loss_val = aux_loss.item()
        else:
            aux_loss = 0.0

        # Entropy regularization to prevent Q-value / advantage collapse
        if self.entropy_coef > 0:
            q_probs_ent = torch.softmax(q_masked, dim=1)
            q_log_probs_ent = torch.log_softmax(q_masked, dim=1)
            batch_entropy = -(q_probs_ent * q_log_probs_ent).sum(dim=1).mean()
            entropy_loss = -self.entropy_coef * batch_entropy
        else:
            batch_entropy = None
            entropy_loss = 0.0

        # Total Loss: TD + CQL + KL + aux + entropy
        loss = (td_term + alpha_cql * cql_term).mean() \
            + self.alpha_kl * kl_term \
            + self.aux_weight * aux_loss \
            + entropy_loss

        if torch.isnan(loss):
            return {
                "loss": 0.0,
                "cql": 0.0,
                "cql_alpha": alpha_cql,
                "kl": 0.0,
                "td": 0.0,
                "q_mean": 0.0,
            }

        loss.backward()
        if self.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_grad_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=float('inf'))
        self.optimizer.step()
        self.scheduler.step()

        self.total_steps += 1

        # Q-value distribution metrics
        with torch.no_grad():
            if batch_entropy is not None:
                q_entropy = batch_entropy.item()
            else:
                q_probs = torch.softmax(q_masked, dim=1)
                q_log_probs = torch.log_softmax(q_masked, dim=1)
                q_entropy = -(q_probs * q_log_probs).sum(dim=1).mean().item()
            # Advantage head std (diagnostic for Dueling DQN health)
            a_values = q_values - q_values.masked_fill(invalid_mask, 0.0).sum(
                dim=-1, keepdim=True) / masks.sum(dim=-1, keepdim=True).clamp(min=1)
            adv_std = a_values.masked_select(masks.bool()).std().item()

        return {
            "loss": loss.item(),
            "cql": cql_term.mean().item(),
            "cql_alpha": alpha_cql,
            "kl": kl_val,
            "td": td_term.mean().item(),
            "aux": aux_loss_val,
            "q_mean": q_data.mean().item(),
            "q_std": q_data.std().item(),
            "q_min": q_data.min().item(),
            "q_max": q_data.max().item(),
            "q_entropy": q_entropy,
            "q_target_mean": targets.mean().item(),
            "q_target_std": targets.std().item(),
            "advantage_std": adv_std,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "lr": self.scheduler.get_last_lr()[0],
        }
