"""
Offline CQL Trainer.
"""
import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import wandb

from riichienv_ml.config import import_class
from riichienv_ml.models.grp_model import RewardPredictor
from riichienv_ml.utils import AverageMeter


def cql_loss(q_values: torch.Tensor, current_actions: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    Computes CQL Regularization Term: logsumexp(Q(s, a_all)) - Q(s, a_data)
    """
    # q_values: (B, NumActions)
    # current_actions: (B) index

    # 1. Q(s, a_data)
    q_data = q_values.gather(1, current_actions.unsqueeze(1)).squeeze(1)

    # 2. logsumexp(Q(s, .))
    invalid_mask = (masks == 0)
    q_masked = q_values.clone()
    q_masked = q_masked.masked_fill(invalid_mask, float("-inf"))
    logsumexp_q = torch.logsumexp(q_masked, dim=1)

    cql_term = (logsumexp_q - q_data).mean()
    return cql_term, q_data


class Trainer:
    def __init__(
        self,
        grp_model_path: str,
        pts_weight: list,
        data_glob: str,
        device_str: str = "cuda",
        gamma: float = 0.99,
        batch_size: int = 32,
        lr: float = 1e-4,
        alpha: float = 1.0,
        limit: int = 1000000,
        num_epochs: int = 10,
        num_workers: int = 8,
        weight_decay: float = 0.0,
        aux_weight: float = 0.0,
        wandb_entity: str = "smly",
        wandb_project: str = "riichienv-offline",
        model_config: dict | None = None,
        model_class: str = "riichienv_ml.models.cql_model.QNetwork",
        dataset_class: str = "riichienv_ml.data.cql_dataset.MCDataset",
    ):
        self.grp_model_path = grp_model_path
        self.pts_weight = pts_weight
        self.data_glob = data_glob
        self.device_str = device_str
        self.device = torch.device(device_str)
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        self.alpha = alpha
        self.limit = int(limit)
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.weight_decay = weight_decay
        self.aux_weight = aux_weight
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project
        self.model_config = model_config or {}
        self.model_class = model_class
        self.dataset_class = dataset_class

    def train(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Initialize Reward Predictor
        reward_predictor = RewardPredictor(self.grp_model_path, self.pts_weight, device=self.device_str, input_dim=20)

        # Dataset
        data_files = glob.glob(self.data_glob)
        assert data_files, f"No data found at {self.data_glob}"

        print(f"Found {len(data_files)} data files.")

        DatasetClass = import_class(self.dataset_class)
        dataset = DatasetClass(data_files, reward_predictor, gamma=self.gamma)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        )

        ModelClass = import_class(self.model_class)
        model = ModelClass(**self.model_config).to(self.device)
        has_aux = hasattr(model, 'aux_head') and model.aux_head is not None

        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.limit, eta_min=1e-7)
        mse_criterion = nn.MSELoss()
        model.train()

        step = 0
        run = wandb.init(
            entity=self.wandb_entity,
            project=self.wandb_project,
            config={
                "learning_rate": self.lr,
                "batch_size": self.batch_size,
                "gamma": self.gamma,
                "alpha": self.alpha,
                "weight_decay": self.weight_decay,
                "aux_weight": self.aux_weight,
                "dataset": self.data_glob,
                **self.model_config,
            },
        )

        loss_meter = AverageMeter(name="loss")
        cql_meter = AverageMeter(name="cql")
        mse_meter = AverageMeter(name="mse")
        aux_meter = AverageMeter(name="aux")

        for epoch in range(self.num_epochs):
            for i, batch in enumerate(dataloader):
                # (feat, act, return, mask, rank)
                if len(batch) == 5:
                    features, actions, targets, masks, ranks = batch
                    ranks = ranks.long().to(self.device, non_blocking=True)
                else:
                    features, actions, targets, masks = batch
                    ranks = None

                features = features.to(self.device, non_blocking=True)

                actions = actions.long().to(self.device, non_blocking=True)
                targets = targets.float().to(self.device, non_blocking=True)
                masks = masks.float().to(self.device, non_blocking=True)

                optimizer.zero_grad()

                # Forward pass
                if has_aux and ranks is not None:
                    q_values, aux_logits = model.forward_with_aux(features)
                else:
                    q_values = model(features)
                    aux_logits = None

                # 2. CQL Loss
                cql_term, q_data = cql_loss(q_values, actions, masks)

                # 3. Bellman Error
                if targets.dim() > 1:
                    targets = targets.squeeze(-1)
                mse_term = mse_criterion(q_data, targets)

                # 4. Auxiliary Loss (rank prediction)
                aux_loss_val = 0.0
                if aux_logits is not None and ranks is not None:
                    aux_loss = F.cross_entropy(aux_logits, ranks)
                    aux_loss_val = aux_loss.item()
                else:
                    aux_loss = 0.0

                # Total Loss
                loss = mse_term + self.alpha * cql_term + self.aux_weight * aux_loss

                loss.backward()
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

                loss_meter.update(loss.item())
                cql_meter.update(cql_term.item())
                mse_meter.update(mse_term.item())
                aux_meter.update(aux_loss_val)

                if step % 100 == 0:
                    log_msg = (f"Epoch {epoch}, Step {step}, Loss: {loss_meter.avg:.4f} "
                               f"(MSE: {mse_meter.avg:.4f}, CQL: {cql_meter.avg:.4f}")
                    log_dict = {
                        "epoch": epoch,
                        "loss": loss_meter.avg,
                        "mse": mse_meter.avg,
                        "cql": cql_meter.avg,
                    }
                    if has_aux:
                        log_msg += f", AUX: {aux_meter.avg:.4f}"
                        log_dict["aux"] = aux_meter.avg
                    log_msg += ")"
                    print(log_msg)
                    run.log(log_dict, step=step)

                step += 1
                scheduler.step()
                if step >= self.limit:
                    break

            loss_meter.reset()
            cql_meter.reset()
            mse_meter.reset()
            aux_meter.reset()

            torch.save(model.state_dict(), output_path)
            print(f"Saved model to {output_path}")
            if step >= self.limit:
                break

        run.finish()
