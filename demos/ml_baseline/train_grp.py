"""
Global Reward Predictor
"""
import tqdm
import polars as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from grp_dataloader import RankPredictorDataset
from grp_model import RankPredictor, RewardPredictor
from utils import AverageMeter


def main():
    device_str = "cuda"
    device = torch.device(device_str)
    trn_dataset = RankPredictorDataset("/data/train_grp.pq")
    trn_dataloader = DataLoader(trn_dataset, batch_size=128, shuffle=True, num_workers=12, pin_memory=True)
    val_dataset = RankPredictorDataset("/data/val_grp.pq")
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=12, pin_memory=True)
    model = RankPredictor().to(device)

    n_epochs = 10
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs * len(trn_dataloader), eta_min=1e-7)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        loss_meter = AverageMeter("loss", ":.4e")
        acc_meter = AverageMeter("acc", ":.4f")

        model = model.train()
        for idx, (x, y) in tqdm.tqdm(enumerate(trn_dataloader), desc=f"epoch {epoch:d}", total=len(trn_dataloader), mininterval=1.0, ncols=100):
            x, y = x.to(device), y.to(device)
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

        loss_meter = AverageMeter("loss", ":.4e")
        acc_meter = AverageMeter("acc", ":.4f")

        model = model.eval()
        for idx, (x, y) in tqdm.tqdm(enumerate(val_dataloader), desc=f"epoch {epoch:d}", total=len(val_dataloader), mininterval=1.0, ncols=100):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)

            loss_meter.update(loss.item(), x.size(0))
            y_pred_cls = torch.argmax(y_pred, dim=1)
            y_true_cls = torch.argmax(y, dim=1)
            acc = (y_pred_cls == y_true_cls).sum().item() / x.size(0)
            acc_meter.update(acc, x.size(0))

        print(f"(val) epoch {epoch:d} loss: {loss_meter.avg:.4e} acc: {acc_meter.avg:.4f}")
        torch.save(model.state_dict(), "grp_model.pth")


def check_gpr_output() -> None:
    kyoku_df = pl.read_parquet("/data/train_grp.pq", n_rows=10)
    kyoku_df = kyoku_df.drop(["p0_hand", "p1_hand", "p2_hand", "p3_hand", "p0_dora_marker"])
    kyoku_features = kyoku_df.to_dicts()

    point_weights = [100, 40, -40, -100]
    player_idx = 0

    device_str = "cuda"
    device = torch.device(device_str)

    rp = RewardPredictor("grp_model.pth", point_weights, device=device_str)
    _, kyoku_rewards = rp.calc_pts_rewards(kyoku_features, player_idx)
    print(kyoku_rewards)


if __name__ == "__main__":
    main()
    check_gpr_output()