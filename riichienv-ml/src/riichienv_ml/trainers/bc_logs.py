"""Offline BC/CQL Trainer on mjai log data."""
import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from loguru import logger

import wandb

from riichienv_ml.config import import_class
from riichienv_ml.models.grp_model import RewardPredictor
from riichienv_ml.utils import AverageMeter


def _create_evaluator(cfg_kwargs: dict, model_config: dict):
    """Create third-party evaluator if configured and 4P mode. Returns None otherwise."""
    model_path = cfg_kwargs.get("model_path")
    n_players = cfg_kwargs.get("n_players", 4)
    if model_path is None or n_players != 4:
        return None

    from riichienv_ml.evaluator import load_evaluator

    return load_evaluator(
        evaluator_name="mortal",
        model_path=model_path,
        model_class=cfg_kwargs.get("model_class"),
        model_config=model_config,
        encoder_class=cfg_kwargs.get("encoder_class"),
        tile_dim=cfg_kwargs.get("tile_dim", 34),
        device=cfg_kwargs.get("device_str", "cuda"),
        eval_device=cfg_kwargs.get("eval_device", "cpu"),
    )


def cql_loss(q_values: torch.Tensor, current_actions: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Computes CQL Regularization Term: logsumexp(Q(s, a_all)) - Q(s, a_data)"""
    q_data = q_values.gather(1, current_actions.unsqueeze(1)).squeeze(1)

    invalid_mask = (masks == 0)
    q_masked = q_values.clone()
    q_masked = q_masked.masked_fill(invalid_mask, -1e9)
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
        model_config: dict | None = None,
        model_class: str = "riichienv_ml.models.q_network.QNetwork",
        dataset_class: str = "riichienv_ml.datasets.mjai_logs.MCDataset",
        encoder_class: str = "riichienv_ml.features.feat_v1.ObservationEncoder",
        n_players: int = 4,
        replay_rule: str = "mjsoul",
        tile_dim: int = 34,
        evaluator_config=None,
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
        self.model_config = model_config or {}
        self.model_class = model_class
        self.dataset_class = dataset_class
        self.encoder_class = encoder_class
        self.n_players = n_players
        self.replay_rule = replay_rule
        self.tile_dim = tile_dim

        if evaluator_config is None:
            from riichienv_ml.config import EvaluatorConfig
            evaluator_config = EvaluatorConfig()
        self.evaluator_config = evaluator_config

        self.tp_evaluator = _create_evaluator(
            cfg_kwargs=dict(
                model_path=evaluator_config.model_path,
                eval_device=evaluator_config.eval_device,
                model_class=model_class,
                encoder_class=encoder_class,
                tile_dim=tile_dim,
                device_str=device_str,
                n_players=n_players,
            ),
            model_config=self.model_config,
        )

    def train(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Initialize Reward Predictor
        input_dim = self.n_players * 4 + 4
        reward_predictor = RewardPredictor(
            self.grp_model_path, self.pts_weight,
            n_players=self.n_players, input_dim=input_dim,
            device=self.device_str,
        )

        # Instantiate encoder
        EncoderClass = import_class(self.encoder_class)
        encoder = EncoderClass(tile_dim=self.tile_dim)

        # Dataset
        data_files = glob.glob(self.data_glob, recursive=True)
        assert data_files, f"No data found at {self.data_glob}"

        print(f"Found {len(data_files)} data files.")

        DatasetClass = import_class(self.dataset_class)
        dataset = DatasetClass(
            data_files, reward_predictor, gamma=self.gamma,
            n_players=self.n_players, replay_rule=self.replay_rule,
            encoder=encoder,
        )
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
        run = wandb.run  # Initialized by init_wandb() in the script

        loss_meter = AverageMeter(name="loss")
        cql_meter = AverageMeter(name="cql")
        mse_meter = AverageMeter(name="mse")
        aux_meter = AverageMeter(name="aux")

        for epoch in range(self.num_epochs):
            for i, batch in enumerate(dataloader):
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

                if has_aux and ranks is not None:
                    q_values, aux_logits = model.forward_with_aux(features)
                else:
                    out = model(features)
                    if isinstance(out, tuple):
                        q_values = out[0]
                    else:
                        q_values = out
                    aux_logits = None

                cql_term, q_data = cql_loss(q_values, actions, masks)

                if targets.dim() > 1:
                    targets = targets.squeeze(-1)
                mse_term = mse_criterion(q_data, targets)

                aux_loss_val = 0.0
                if aux_logits is not None and ranks is not None:
                    aux_loss = F.cross_entropy(aux_logits, ranks)
                    aux_loss_val = aux_loss.item()
                else:
                    aux_loss = 0.0

                loss = mse_term + self.alpha * cql_term + self.aux_weight * aux_loss

                loss.backward()
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
                    wandb.log(log_dict, step=step)

                # Periodic Mortal evaluation
                if (self.tp_evaluator is not None
                        and step > 0
                        and step % self.evaluator_config.eval_interval == 0):
                    try:
                        ckpt_path = output_path.replace(".pth", f"_step{step}.pth")
                        torch.save(model.state_dict(), ckpt_path)
                        logger.info(f"Saved checkpoint to {ckpt_path}")

                        hw = {k: v.cpu() for k, v in model.state_dict().items()}
                        model.eval()
                        metrics = self.tp_evaluator.evaluate(
                            hw, num_episodes=self.evaluator_config.eval_episodes)
                        model.train()
                        logline = self.tp_evaluator.metrics_to_logline(metrics)
                        logger.info(f"Eval @ step {step}: {logline}")
                        wandb.log(metrics, step=step)
                    except Exception as e:
                        logger.error(f"Mortal evaluation failed at step {step}: {e}")

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

        # Final Mortal evaluation
        if self.tp_evaluator is not None:
            try:
                hw = {k: v.cpu() for k, v in model.state_dict().items()}
                model.eval()
                metrics = self.tp_evaluator.evaluate(
                    hw, num_episodes=self.evaluator_config.eval_episodes)
                logline = self.tp_evaluator.metrics_to_logline(metrics)
                logger.info(f"Final Eval: {logline}")
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.error(f"Final Mortal evaluation failed: {e}")

        wandb.finish()
