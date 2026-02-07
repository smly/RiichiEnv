"""
Train Global Reward Predictor (GRP).

Usage:
    uv run python scripts/train_grp.py -c configs/baseline.yml
"""
import argparse

import polars as pl

from riichienv_ml.config import load_config
from riichienv_ml.training.grp_trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Global Reward Predictor")
    parser.add_argument("-c", "--config", type=str, default="configs/baseline.yml", help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config).grp

    # Override config with CLI args
    overrides = {}
    if args.device is not None:
        overrides["device"] = args.device
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers
    if args.num_epochs is not None:
        overrides["num_epochs"] = args.num_epochs
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.output is not None:
        overrides["output"] = args.output
    if overrides:
        cfg = cfg.model_copy(update=overrides)

    df_trn = pl.concat([pl.read_parquet(p) for p in cfg.train_data])
    df_val = pl.read_parquet(cfg.val_data)

    trainer = Trainer(
        device_str=cfg.device,
        train_dataframe=df_trn,
        val_dataframe=df_val,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        lr=cfg.lr,
        input_dim=cfg.input_dim,
    )
    trainer.train(output_path=cfg.output, n_epochs=cfg.num_epochs)


if __name__ == "__main__":
    main()
