"""Global Reward Predictor Trainer."""
import glob as glob_mod
from pathlib import Path

import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from loguru import logger

from riichienv_ml.datasets.grp_dataset import GrpReplayDataset
from riichienv_ml.models.grp_model import RankPredictor
from riichienv_ml.utils import AverageMeter


class Trainer:
    def __init__(
        self,
        device_str: str = "cuda",
        data_glob: str = "",
        val_data_glob: str = "",
        batch_size: int = 128,
        num_workers: int = 12,
        lr: float = 5e-4,
        lr_eta_min: float = 1e-7,
        samples_per_file: int = 32,
        n_players: int = 4,
        replay_rule: str = "mjsoul",
    ):
        self.device_str = device_str
        self.device = torch.device(device_str)
        self.lr = lr
        self.lr_eta_min = lr_eta_min
        self.batch_size = batch_size
        self.samples_per_file = samples_per_file
        self.n_players = n_players
        self.input_dim = n_players * 4 + 4
        self.n_train_files = 0

        if data_glob:
            self.n_train_files = len(glob_mod.glob(data_glob, recursive=True))
            self.train_dataset = GrpReplayDataset(
                data_glob=data_glob,
                n_players=n_players,
                replay_rule=replay_rule,
                is_train=True,
            )
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
            )
        if val_data_glob:
            self.val_dataset = GrpReplayDataset(
                data_glob=val_data_glob,
                n_players=n_players,
                replay_rule=replay_rule,
                is_train=False,
            )
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
            )

    def _estimate_steps_per_epoch(self) -> int:
        """Estimate total optimizer steps per epoch from file count."""
        total_samples = self.n_train_files * self.samples_per_file
        return max(total_samples // self.batch_size, 1)

    def train(self, output_path: str, n_epochs: int = 10) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        model = RankPredictor(input_dim=self.input_dim, n_players=self.n_players).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        t_max = n_epochs * self._estimate_steps_per_epoch()
        logger.info(f"CosineAnnealingLR: T_max={t_max} (files={self.n_train_files}, "
              f"samples_per_file={self.samples_per_file}, batch_size={self.batch_size})")
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=self.lr_eta_min)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(n_epochs):
            self._train_epoch(epoch, model, optimizer, scheduler, criterion)
            self._val_epoch(epoch, model, criterion)
            torch.save(model.state_dict(), output_path)

    def _train_epoch(self, epoch: int, model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler, criterion: nn.Module) -> None:
        loss_meter = AverageMeter("loss", ":.4e")
        acc_meter = AverageMeter("acc", ":.4f")

        model = model.train()
        for idx, (x, y) in tqdm.tqdm(enumerate(self.train_dataloader), desc=f"epoch {epoch:d}", mininterval=1.0, ncols=100):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_meter.update(loss.item(), x.size(0))
            y_pred_cls = torch.argmax(y_pred, dim=1)
            y_true_cls = torch.argmax(y, dim=1)
            acc = (y_pred_cls == y_true_cls).sum().item() / x.size(0)
            acc_meter.update(acc, x.size(0))
            if idx > 0 and idx % 10000 == 0:
                logger.info(f"(train) - {idx:d} loss: {loss_meter.avg:.4e} acc: {acc_meter.avg:.4f}")

        logger.info(f"(train) epoch {epoch:d} loss: {loss_meter.avg:.4e} acc: {acc_meter.avg:.4f}")

    def _val_epoch(self, epoch, model: nn.Module, criterion: nn.Module) -> None:
        loss_meter = AverageMeter("loss", ":.4e")
        acc_meter = AverageMeter("acc", ":.4f")

        model = model.eval()
        for idx, (x, y) in tqdm.tqdm(enumerate(self.val_dataloader), desc=f"epoch {epoch:d}", mininterval=1.0, ncols=100):
            x, y = x.to(self.device), y.to(self.device)
            y_pred = model(x)
            loss = criterion(y_pred, y)

            loss_meter.update(loss.item(), x.size(0))
            y_pred_cls = torch.argmax(y_pred, dim=1)
            y_true_cls = torch.argmax(y, dim=1)
            acc = (y_pred_cls == y_true_cls).sum().item() / x.size(0)
            acc_meter.update(acc, x.size(0))

        logger.info(f"(val) epoch {epoch:d} loss: {loss_meter.avg:.4e} acc: {acc_meter.avg:.4f}")
