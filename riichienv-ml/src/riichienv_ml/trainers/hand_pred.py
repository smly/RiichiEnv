"""Hand Prediction Trainer — supervised learning for opponent hand estimation."""

import glob as glob_mod
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import wandb
from loguru import logger
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from riichienv_ml.config import HandPredConfig, import_class
from riichienv_ml.datasets.hand_pred import HandPredDataset
from riichienv_ml.utils import AverageMeter


class Trainer:
    def __init__(self, cfg: HandPredConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.n_train_files = 0

        # Instantiate encoder
        EncoderCls = import_class(cfg.encoder_class)
        game = cfg.game
        self.encoder = EncoderCls(tile_dim=game.tile_dim)

        # Datasets & dataloaders
        if cfg.data_glob:
            self.n_train_files = len(glob_mod.glob(cfg.data_glob, recursive=True))
            self.train_dataset = HandPredDataset(
                data_glob=cfg.data_glob,
                n_players=game.n_players,
                tile_dim=game.tile_dim,
                replay_rule=game.replay_rule,
                is_train=True,
                encoder=self.encoder,
            )
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )
        if cfg.val_data_glob:
            self.val_dataset = HandPredDataset(
                data_glob=cfg.val_data_glob,
                n_players=game.n_players,
                tile_dim=game.tile_dim,
                replay_rule=game.replay_rule,
                is_train=False,
                encoder=self.encoder,
            )
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )

    def _estimate_steps_per_epoch(self) -> int:
        total_samples = self.n_train_files * self.cfg.samples_per_file
        return max(total_samples // self.cfg.batch_size, 1)

    def _build_model(self) -> nn.Module:
        ModelCls = import_class(self.cfg.model_class)
        model_kwargs = self.cfg.model.model_dump()
        model_kwargs["tile_dim"] = self.cfg.game.tile_dim
        return ModelCls(**model_kwargs)

    def _build_criterion(self) -> nn.Module:
        if self.cfg.loss_type == "smooth_l1":
            return nn.SmoothL1Loss()
        elif self.cfg.loss_type == "mse":
            return nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss_type: {self.cfg.loss_type}")

    def train(self) -> None:
        cfg = self.cfg
        output_path = cfg.output
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        model = self._build_model().to(self.device)
        optimizer = optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        t_max = cfg.num_epochs * self._estimate_steps_per_epoch()
        logger.info(
            f"CosineAnnealingLR: T_max={t_max} "
            f"(files={self.n_train_files}, samples_per_file={cfg.samples_per_file}, "
            f"batch_size={cfg.batch_size})")
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=cfg.lr_eta_min)
        criterion = self._build_criterion()

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model: {cfg.model_class} ({param_count:,} params)")
        logger.info(f"sum_constraint_weight: {cfg.sum_constraint_weight}")

        for epoch in range(cfg.num_epochs):
            train_metrics = self._train_epoch(
                epoch, model, optimizer, scheduler, criterion)
            val_metrics = self._val_epoch(epoch, model, criterion)
            torch.save(model.state_dict(), output_path)
            logger.info(f"Saved checkpoint: {output_path}")

            log_dict = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            for k, v in train_metrics.items():
                log_dict[f"train/{k}"] = v
            for k, v in val_metrics.items():
                log_dict[f"val/{k}"] = v
            wandb.log(log_dict)

    def _sum_constraint_loss(
        self, pred: torch.Tensor, concealed_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Smooth L1 loss between predicted total and known concealed tile counts.

        pred:             (B, 3, tile_dim)
        concealed_counts: (B, 3)
        """
        pred_sums = pred.sum(dim=-1)  # (B, 3)
        return nn.functional.smooth_l1_loss(pred_sums, concealed_counts)

    def _train_epoch(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        criterion: nn.Module,
    ) -> dict[str, float]:
        loss_meter = AverageMeter("loss", ":.4e")
        sum_loss_meter = AverageMeter("sum_loss", ":.4e")
        mae_meter = AverageMeter("mae", ":.4f")
        tile_acc_meter = AverageMeter("tile_acc", ":.4f")

        estimated_steps = self._estimate_steps_per_epoch()
        model.train()
        pbar = tqdm.tqdm(
            enumerate(self.train_dataloader),
            desc=f"train {epoch:d}",
            total=estimated_steps,
            mininterval=1.0,
            ncols=120,
        )
        for _idx, (x, y, cc) in pbar:
            x = x.to(self.device)
            y = y.to(self.device)
            cc = cc.to(self.device)

            optimizer.zero_grad()
            pred = model(x)                       # (B, 3, tile_dim)
            main_loss = criterion(pred, y)
            sum_loss = self._sum_constraint_loss(pred, cc)
            loss = main_loss + self.cfg.sum_constraint_weight * sum_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()

            bs = x.size(0)
            loss_meter.update(main_loss.item(), bs)
            sum_loss_meter.update(sum_loss.item(), bs)
            with torch.no_grad():
                mae_meter.update((pred - y).abs().mean().item(), bs)
                tile_acc = (pred.round() == y).float().mean().item()
                tile_acc_meter.update(tile_acc, bs)
            pbar.set_postfix(
                loss=f"{loss_meter.avg:.4e}",
                mae=f"{mae_meter.avg:.4f}",
                acc=f"{tile_acc_meter.avg:.4f}",
            )

        logger.info(
            f"(train) epoch {epoch:d} loss={loss_meter.avg:.4e} "
            f"sum_loss={sum_loss_meter.avg:.4e} "
            f"mae={mae_meter.avg:.4f} tile_acc={tile_acc_meter.avg:.4f}")
        return {
            "loss": loss_meter.avg,
            "sum_loss": sum_loss_meter.avg,
            "mae": mae_meter.avg,
            "tile_acc": tile_acc_meter.avg,
        }

    @torch.no_grad()
    def _val_epoch(
        self, epoch: int, model: nn.Module, criterion: nn.Module,
    ) -> dict[str, float]:
        loss_meter = AverageMeter("loss", ":.4e")
        sum_loss_meter = AverageMeter("sum_loss", ":.4e")
        mae_meter = AverageMeter("mae", ":.4f")
        tile_acc_meter = AverageMeter("tile_acc", ":.4f")

        model.eval()
        pbar = tqdm.tqdm(
            enumerate(self.val_dataloader),
            desc=f"val   {epoch:d}",
            mininterval=1.0,
            ncols=120,
        )
        for _idx, (x, y, cc) in pbar:
            x = x.to(self.device)
            y = y.to(self.device)
            cc = cc.to(self.device)

            pred = model(x)
            main_loss = criterion(pred, y)
            sum_loss = self._sum_constraint_loss(pred, cc)

            bs = x.size(0)
            loss_meter.update(main_loss.item(), bs)
            sum_loss_meter.update(sum_loss.item(), bs)
            mae_meter.update((pred - y).abs().mean().item(), bs)
            tile_acc = (pred.round() == y).float().mean().item()
            tile_acc_meter.update(tile_acc, bs)
            pbar.set_postfix(
                loss=f"{loss_meter.avg:.4e}",
                mae=f"{mae_meter.avg:.4f}",
                acc=f"{tile_acc_meter.avg:.4f}",
            )

        logger.info(
            f"(val) epoch {epoch:d} loss={loss_meter.avg:.4e} "
            f"sum_loss={sum_loss_meter.avg:.4e} "
            f"mae={mae_meter.avg:.4f} tile_acc={tile_acc_meter.avg:.4f}")
        return {
            "loss": loss_meter.avg,
            "sum_loss": sum_loss_meter.avg,
            "mae": mae_meter.avg,
            "tile_acc": tile_acc_meter.avg,
        }
