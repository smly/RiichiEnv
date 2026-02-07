"""
Train Online DQN + CQL with Ray distributed workers.

Usage:
    uv run python scripts/train_online.py -c configs/baseline.yml
    uv run python scripts/train_online.py -c configs/baseline.yml --load_model cql_model.pth --num_workers 12
"""
import argparse

from dotenv import load_dotenv
load_dotenv() # Used for wandb API key

from riichienv_ml.config import load_config
from riichienv_ml.training.online_trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Online DQN model")
    parser.add_argument("-c", "--config", type=str, default="configs/baseline.yml", help="Path to config YAML")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--load_model", type=str, default=None, help="Path to offline CQL model (cql_model.pth)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--alpha_cql_init", type=float, default=None, help="Initial CQL alpha")
    parser.add_argument("--alpha_cql_final", type=float, default=None, help="Final CQL alpha")
    parser.add_argument("--exploration", type=str, default=None, choices=["epsilon_greedy", "boltzmann"],
                        help="Exploration strategy")
    parser.add_argument("--epsilon_start", type=float, default=None, help="Initial epsilon (epsilon-greedy)")
    parser.add_argument("--epsilon_final", type=float, default=None, help="Final epsilon (epsilon-greedy)")
    parser.add_argument("--boltzmann_epsilon", type=float, default=None, help="Prob of Boltzmann sampling vs greedy")
    parser.add_argument("--boltzmann_temp_start", type=float, default=None, help="Initial Boltzmann temperature")
    parser.add_argument("--boltzmann_temp_final", type=float, default=None, help="Final Boltzmann temperature")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p nucleus sampling threshold")
    parser.add_argument("--capacity", type=int, default=None, help="Replay buffer capacity")
    parser.add_argument("--eval_interval", type=int, default=None, help="Evaluation interval")
    parser.add_argument("--weight_sync_freq", type=int, default=None, help="Sync weights every N steps")
    parser.add_argument("--worker_device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--gpu_per_worker", type=float, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--encoder_class", type=str, default=None, help="Encoder class dotted path")
    # Model architecture overrides
    parser.add_argument("--num_blocks", type=int, default=None, help="Number of residual blocks")
    parser.add_argument("--conv_channels", type=int, default=None, help="Conv hidden channels")
    parser.add_argument("--fc_dim", type=int, default=None, help="FC hidden dimension")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config).online

    # Override config with CLI args
    overrides = {}
    for field in ["num_workers", "num_steps", "batch_size", "device", "load_model",
                  "lr", "alpha_cql_init", "alpha_cql_final",
                  "exploration", "epsilon_start", "epsilon_final",
                  "boltzmann_epsilon", "boltzmann_temp_start", "boltzmann_temp_final", "top_p",
                  "capacity",
                  "eval_interval", "weight_sync_freq", "worker_device", "gpu_per_worker",
                  "checkpoint_dir", "encoder_class"]:
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

    run_training(cfg)


if __name__ == "__main__":
    main()
