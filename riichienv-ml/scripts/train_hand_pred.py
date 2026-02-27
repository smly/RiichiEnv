"""Train hand prediction model (opponent hand estimation).

Usage:
    uv run python scripts/train_hand_pred.py -c src/riichienv_ml/configs/4p/hand_pred_cnn.yml
    uv run python scripts/train_hand_pred.py -c src/riichienv_ml/configs/4p/hand_pred_seq.yml
"""

import argparse
from pathlib import Path

from riichienv_ml.config import load_config
from riichienv_ml.trainers.hand_pred import Trainer
from riichienv_ml.utils import init_wandb, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hand prediction model")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config).hand_pred

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

    log_dir = str(Path(cfg.output).parent)
    setup_logging(log_dir, "train_hand_pred")
    init_wandb(cfg, config_path=args.config)

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
