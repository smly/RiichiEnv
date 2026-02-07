"""
Train Offline CQL model.

Usage:
    uv run python scripts/train_cql.py -c configs/baseline.yml
    uv run python scripts/train_cql.py -c configs/baseline.yml --lr 1e-4
"""
import argparse

import torch.multiprocessing

from dotenv import load_dotenv
load_dotenv() # Used for wandb API key

from riichienv_ml.config import load_config
from riichienv_ml.training.cql_trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Offline CQL model")
    parser.add_argument("-c", "--config", type=str, default="configs/baseline.yml", help="Path to config YAML")
    parser.add_argument("--data_glob", type=str, default=None, help="Glob path for training data (.xz)")
    parser.add_argument("--grp_model", type=str, default=None, help="Path to reward model")
    parser.add_argument("--output", type=str, default=None, help="Output model path")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None, help="CQL Scale")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    # Model architecture overrides
    parser.add_argument("--num_blocks", type=int, default=None, help="Number of residual blocks")
    parser.add_argument("--conv_channels", type=int, default=None, help="Conv hidden channels")
    parser.add_argument("--fc_dim", type=int, default=None, help="FC hidden dimension")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config).cql

    # Override config with CLI args
    overrides = {}
    for field in ["data_glob", "grp_model", "output", "batch_size", "lr", "alpha",
                  "gamma", "num_epochs", "num_workers", "limit"]:
        val = getattr(args, field, None)
        if val is not None:
            overrides[field] = val
    if overrides:
        cfg = cfg.model_copy(update=overrides)

    # Override model config with CLI args
    model_overrides = {}
    for field in ["num_blocks", "conv_channels", "fc_dim"]:
        val = getattr(args, field, None)
        if val is not None:
            model_overrides[field] = val
    if model_overrides:
        cfg = cfg.model_copy(update={"model": cfg.model.model_copy(update=model_overrides)})

    trainer = Trainer(
        grp_model_path=cfg.grp_model,
        pts_weight=cfg.pts_weight,
        data_glob=cfg.data_glob,
        device_str=cfg.device,
        gamma=cfg.gamma,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        alpha=cfg.alpha,
        limit=cfg.limit,
        num_epochs=cfg.num_epochs,
        num_workers=cfg.num_workers,
        wandb_entity=cfg.wandb_entity,
        wandb_project=cfg.wandb_project,
        model_config=cfg.model.model_dump(),
        model_class=cfg.model_class,
        dataset_class=cfg.dataset_class,
    )
    trainer.train(cfg.output)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
