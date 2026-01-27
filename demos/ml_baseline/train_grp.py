"""
Global Reward Predictor

This script trains a global reward predictor to predict the reward of each player in a game.

The script will load the training data from the parquet files in the data directory.

Usage:
    python train_grp.py
"""
import tqdm
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from grp_dataset import RankPredictorDataset
from grp_model import RankPredictor, RewardPredictor
from utils import AverageMeter


class Trainer:
    def __init__(
        self,
        device_str: str = "cuda",
        train_dataframe: pl.DataFrame = None,
        val_dataframe: pl.DataFrame = None,
    ):
        self.device_str = device_str
        self.device = torch.device(device_str)

        if train_dataframe is not None:
            self.train_dataset = RankPredictorDataset(train_dataframe)
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=128, shuffle=True, num_workers=12, pin_memory=True)
        if val_dataframe is not None:
            self.val_dataset = RankPredictorDataset(val_dataframe)
            self.val_dataloader = DataLoader(self.val_dataset, batch_size=128, shuffle=False, num_workers=12, pin_memory=True)

    def train(self, n_epochs: int = 10) -> None:
        model = RankPredictor(input_dim=20).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs * len(self.train_dataloader), eta_min=1e-7)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(n_epochs):
            self._train_epoch(epoch, model, optimizer, scheduler, criterion)
            self._val_epoch(epoch, model, criterion)
            torch.save(model.state_dict(), "grp_model.pth")

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


def main():
    device_str = "cuda"

    df_trn = pl.concat([
        pl.read_parquet("/data/train_grp.pq"),
        pl.read_parquet("/data/train_grp_2024.pq"),
    ])
    df_val = pl.read_parquet("/data/val_grp.pq")

    trainer = Trainer(device_str, df_trn, df_val)
    trainer.train()


def check_reward_predictions() -> None:
    kyoku_df = pl.read_parquet("/data/train_grp.pq", n_rows=10)
    kyoku_df = kyoku_df.drop(["p0_hand", "p1_hand", "p2_hand", "p3_hand", "p0_dora_marker"])

    kyoku_features = kyoku_df.select([
        # scores at the start of the kyoku
        "p0_init_score",
        "p1_init_score",
        "p2_init_score",
        "p3_init_score",
        # scores at the end of the kyoku
        "p0_end_score",
        "p1_end_score",
        "p2_end_score",
        "p3_end_score",
        # delta scores
        "p0_delta_score",
        "p1_delta_score",
        "p2_delta_score",
        "p3_delta_score",
        # ba, kyoku, honba, riichi sticks
        "chang",
        "ju",
        "ben",
        "liqibang",
    ]).to_dicts()

    point_weights = [100, 40, -40, -100]
    player_idx = 0

    device_str = "cuda"
    device = torch.device(device_str)

    rp = RewardPredictor("grp_model.pth", point_weights, device=device_str, input_dim=20)
    _, kyoku_rewards = rp.calc_pts_rewards(kyoku_features, player_idx)
    print(kyoku_rewards)


if __name__ == "__main__":
    main()
    # check_reward_predictions()