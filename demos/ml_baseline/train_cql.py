import argparse
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import wandb
from dotenv import load_dotenv
load_dotenv()

from cql_dataset import MCDataset
from cql_model import QNetwork
from grp_model import RewardPredictor
from utils import AverageMeter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_glob", type=str, required=True, help="Glob path for training data (.xz)")
    parser.add_argument("--grp_model", type=str, default="./grp_model.pth", help="Path to reward model")
    parser.add_argument("--output", type=str, default="cql_model.pth", help="Output model path")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1.0, help="CQL Scale")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=3e6)

    args = parser.parse_args()
    return args


def cql_loss(q_values: torch.Tensor, current_actions: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    Computes CQL Regularization Term: logsumexp(Q(s, a_all)) - Q(s, a_data)
    """
    # q_values: (B, NumActions)
    # current_actions: (B) index

    # 1. Q(s, a_data)
    # actions is (B), unsqueeze to (B,1), gather, squeeze -> (B)
    q_data = q_values.gather(1, current_actions.unsqueeze(1)).squeeze(1)

    # 2. logsumexp(Q(s, .))
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
        pts_weight: list[float],
        data_glob: str,
        device_str: str = "cuda",
        gamma: float = 0.99,
        batch_size: int = 32,
        lr: float = 1e-4,
        alpha: float = 1.0,
        limit: int = 1e6,
        num_epochs: int = 10,
        num_workers: int = 8,
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
        self.limit = limit
        self.num_epochs = num_epochs
        self.num_workers = num_workers

    def train(self, output_path: str) -> None:
        # Initialize Reward Predictor
        reward_predictor = RewardPredictor(self.grp_model_path, self.pts_weight, device=self.device_str, input_dim=20)

        # Dataset
        data_files = glob.glob(self.data_glob)
        assert data_files, f"No data found at {self.data_glob}"

        print(f"Found {len(data_files)} data files.")    
        dataset = MCDataset(data_files, reward_predictor, gamma=self.gamma)        
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)

        # Model
        model = QNetwork(in_channels=46, num_actions=82).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.limit, eta_min=1e-7)
        mse_criterion = nn.MSELoss()
        model.train()
        
        step = 0
        run = wandb.init(
            entity="smly",
            project="riichienv-mc-cql",
            config={
                "learning_rate": self.lr,
                "batch_size": self.batch_size,
                "gamma": self.gamma,
                "alpha": self.alpha,
                "dataset": self.data_glob,
            },
        )

        loss_meter = AverageMeter(name="loss")
        cql_meter = AverageMeter(name="cql")
        mse_meter = AverageMeter(name="mse")

        for epoch in range(self.num_epochs):
            for i, batch in enumerate(dataloader):
                # (feat, act, return, mask)
                features, actions, targets, masks = batch
                # targets is G_t

                features = features.to(self.device)
                actions = actions.long().to(self.device)
                targets = targets.float().to(self.device)
                masks = masks.float().to(self.device)
                
                optimizer.zero_grad()
                
                # 1. Compute Q(s, a)
                q_values = model(features)
                
                # 2. CQL Loss
                cql_term, q_data = cql_loss(q_values, actions, masks)
                
                # 3. Bellman Error
                # Regression to G_t
                # targets is (B, 1), q_data is (B). Squeeze targets.
                mse_term = mse_criterion(q_data, targets.squeeze(-1))

                # Total Loss
                loss = mse_term + self.alpha * cql_term
                
                loss.backward()
                optimizer.step()

                loss_meter.update(loss.item())
                cql_meter.update(cql_term.item())
                mse_meter.update(mse_term.item())

                if step % 100 == 0:
                    print(f"Epoch {epoch}, Step {step}, Loss: {loss_meter.avg:.4f} (MSE: {mse_meter.avg:.4f}, CQL: {cql_meter.avg:.4f})")
                    run.log({
                        "epoch": epoch,
                        "loss": loss_meter.avg,
                        "mse": mse_meter.avg,
                        "cql": cql_meter.avg,
                    }, step=step)

                step += 1
                scheduler.step()
                if step >= self.limit:
                    break

            loss_meter.reset()
            cql_meter.reset()
            mse_meter.reset()

            torch.save(model.state_dict(), output_path)
            print(f"Saved model to {output_path}")
            if step >= self.limit:
                break

        run.finish()


def train(args: argparse.Namespace):
    device_str = "cuda"
    pts_weight = [10.0, 4.0, -4.0, -10.0]

    trainer = Trainer(
        args.grp_model,
        pts_weight,
        args.data_glob,
        device_str=device_str,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
        gamma=args.gamma,
        limit=args.limit,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
    )
    trainer.train(args.output)


if __name__ == "__main__":
    train(parse_args())