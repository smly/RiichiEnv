"""Train Online RL with Ray distributed workers.

Supports two algorithms:
  - dqn: DQN + CQL (value-based, on-policy buffer)
  - ppo: PPO (actor-critic, on-policy)

Usage:
    uv run python scripts/train_ppo.py -c src/riichienv_ml/configs/4p/ppo.yml
    uv run python scripts/train_ppo.py -c src/riichienv_ml/configs/4p/ppo.yml --algorithm ppo
"""
import argparse

from dotenv import load_dotenv
load_dotenv()

from riichienv_ml.config import load_config
from riichienv_ml.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Online RL model")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--algorithm", type=str, default=None, choices=["dqn", "ppo"])
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    # DQN-specific
    parser.add_argument("--alpha_cql_init", type=float, default=None)
    parser.add_argument("--alpha_cql_final", type=float, default=None)
    parser.add_argument("--exploration", type=str, default=None, choices=["epsilon_greedy", "boltzmann"])
    parser.add_argument("--epsilon_start", type=float, default=None)
    parser.add_argument("--epsilon_final", type=float, default=None)
    parser.add_argument("--boltzmann_epsilon", type=float, default=None)
    parser.add_argument("--boltzmann_temp_start", type=float, default=None)
    parser.add_argument("--boltzmann_temp_final", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--capacity", type=int, default=None)
    # PPO-specific
    parser.add_argument("--ppo_clip", type=float, default=None)
    parser.add_argument("--ppo_epochs", type=int, default=None)
    parser.add_argument("--gae_lambda", type=float, default=None)
    parser.add_argument("--entropy_coef", type=float, default=None)
    parser.add_argument("--value_coef", type=float, default=None)
    # Common
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--weight_sync_freq", type=int, default=None)
    parser.add_argument("--worker_device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--gpu_per_worker", type=float, default=None)
    parser.add_argument("--num_envs_per_worker", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--encoder_class", type=str, default=None)
    # Model architecture overrides
    parser.add_argument("--num_blocks", type=int, default=None)
    parser.add_argument("--conv_channels", type=int, default=None)
    parser.add_argument("--fc_dim", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config).ppo

    # Override config with CLI args
    overrides = {}
    for field in ["algorithm", "num_workers", "num_steps", "batch_size", "device", "load_model",
                  "lr", "alpha_cql_init", "alpha_cql_final",
                  "exploration", "epsilon_start", "epsilon_final",
                  "boltzmann_epsilon", "boltzmann_temp_start", "boltzmann_temp_final", "top_p",
                  "capacity",
                  "ppo_clip", "ppo_epochs", "gae_lambda", "entropy_coef", "value_coef",
                  "eval_interval", "weight_sync_freq", "worker_device", "gpu_per_worker",
                  "num_envs_per_worker",
                  "checkpoint_dir", "encoder_class"]:
        val = getattr(args, field, None)
        if val is not None:
            overrides[field] = val
    if overrides:
        cfg = cfg.model_copy(update=overrides)

    model_overrides = {}
    for field in ["num_blocks", "conv_channels", "fc_dim"]:
        val = getattr(args, field, None)
        if val is not None:
            model_overrides[field] = val
    if model_overrides:
        cfg = cfg.model_copy(update={"model": cfg.model.model_copy(update=model_overrides)})

    setup_logging(cfg.checkpoint_dir, "train_ppo")

    if cfg.algorithm == "ppo":
        from riichienv_ml.trainers.ppo import run_ppo_training
        run_ppo_training(cfg)
    else:
        from riichienv_ml.trainers._dqn_online import run_training
        run_training(cfg)


if __name__ == "__main__":
    main()
