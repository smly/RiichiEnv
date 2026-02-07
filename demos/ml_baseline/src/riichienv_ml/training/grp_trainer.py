"""
Global Reward Predictor Trainer.
"""
import tqdm
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from riichienv_ml.data.grp_dataset import RankPredictorDataset
from riichienv_ml.models.grp_model import RankPredictor
from riichienv_ml.utils import AverageMeter


class Trainer:
    def __init__(
        self,
        device_str: str = "cuda",
        train_dataframe: pl.DataFrame = None,
        val_dataframe: pl.DataFrame = None,
        batch_size: int = 128,
        num_workers: int = 12,
        lr: float = 5e-4,
        input_dim: int = 20,
    ):
        self.device_str = device_str
        self.device = torch.device(device_str)
        self.lr = lr
        self.input_dim = input_dim

        if train_dataframe is not None:
            self.train_dataset = RankPredictorDataset(train_dataframe)
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        if val_dataframe is not None:
            self.val_dataset = RankPredictorDataset(val_dataframe)
            self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    def train(self, output_path: str, n_epochs: int = 10) -> None:
        model = RankPredictor(input_dim=self.input_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs * len(self.train_dataloader), eta_min=1e-7)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(n_epochs):
            self._train_epoch(epoch, model, optimizer, scheduler, criterion)
            self._val_epoch(epoch, model, criterion)
            torch.save(model.state_dict(), output_path)

    def _train_epoch(self, epoch: int, model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler, criterion: nn.Module) -> None:
        loss_meter = AverageMeter("loss", ":.4e")
        acc_meter = AverageMeter("acc", ":.4f")

        model = model.train()
        for idx, (x, y) in tqdm.tqdm(enumerate(self.train_dataloader), desc=f"epoch {epoch:d}", total=len(self.train_dataloader), mininterval=1.0, ncols=100):
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
                print(f"(train) - {idx:d} loss: {loss_meter.avg:.4e} acc: {acc_meter.avg:.4f}")

        print(f"(train) epoch {epoch:d} loss: {loss_meter.avg:.4e} acc: {acc_meter.avg:.4f}")

    def _val_epoch(self, epoch, model: nn.Module, criterion: nn.Module) -> None:
        loss_meter = AverageMeter("loss", ":.4e")
        acc_meter = AverageMeter("acc", ":.4f")

        model = model.eval()
        for idx, (x, y) in tqdm.tqdm(enumerate(self.val_dataloader), desc=f"epoch {epoch:d}", total=len(self.val_dataloader), mininterval=1.0, ncols=100):
            x, y = x.to(self.device), y.to(self.device)
            y_pred = model(x)
            loss = criterion(y_pred, y)

            loss_meter.update(loss.item(), x.size(0))
            y_pred_cls = torch.argmax(y_pred, dim=1)
            y_true_cls = torch.argmax(y, dim=1)
            acc = (y_pred_cls == y_true_cls).sum().item() / x.size(0)
            acc_meter.update(acc, x.size(0))

        print(f"(val) epoch {epoch:d} loss: {loss_meter.avg:.4e} acc: {acc_meter.avg:.4f}")
