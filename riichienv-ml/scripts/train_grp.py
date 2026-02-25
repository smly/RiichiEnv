"""Train Global Reward Predictor (GRP).

Usage:
    uv run python scripts/train_grp.py -c src/riichienv_ml/configs/4p/grp.yml
"""
import argparse
from pathlib import Path

from riichienv_ml.config import load_config
from riichienv_ml.utils import setup_logging
from riichienv_ml.trainers.grp import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Global Reward Predictor")
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

    log_dir = str(Path(cfg.output).parent)
    setup_logging(log_dir, "train_grp")

    game = cfg.game
    trainer = Trainer(
        device_str=cfg.device,
        data_glob=cfg.data_glob,
        val_data_glob=cfg.val_data_glob,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        lr=cfg.lr,
        lr_eta_min=cfg.lr_eta_min,
        samples_per_file=cfg.samples_per_file,
        n_players=game.n_players,
        replay_rule=game.replay_rule,
    )
    trainer.train(output_path=cfg.output, n_epochs=cfg.num_epochs)


if __name__ == "__main__":
    main()
