"""
Train Online RL with Ray distributed workers.

Supports two algorithms:
  - dqn: DQN + CQL (value-based, replay buffer)
  - ppo: PPO (actor-critic, on-policy)

Usage:
    uv run python scripts/train_online.py -c configs/baseline.yml
    uv run python scripts/train_online.py -c configs/ppo_baseline.yml --algorithm ppo
    uv run python scripts/train_online.py -c configs/baseline.yml --load_model cql_model.pth --num_workers 12
"""
import argparse

from dotenv import load_dotenv
load_dotenv() # Used for wandb API key

from riichienv_ml.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Online RL model")
    parser.add_argument("-c", "--config", type=str, default="configs/baseline.yml", help="Path to config YAML")
    parser.add_argument("--algorithm", type=str, default=None, choices=["dqn", "ppo"], help="RL algorithm")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--load_model", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    # DQN-specific
    parser.add_argument("--alpha_cql_init", type=float, default=None, help="Initial CQL alpha")
    parser.add_argument("--alpha_cql_final", type=float, default=None, help="Final CQL alpha")
    parser.add_argument("--exploration", type=str, default=None, choices=["epsilon_greedy", "boltzmann"],
                        help="Exploration strategy (DQN)")
    parser.add_argument("--epsilon_start", type=float, default=None, help="Initial epsilon (epsilon-greedy)")
    parser.add_argument("--epsilon_final", type=float, default=None, help="Final epsilon (epsilon-greedy)")
    parser.add_argument("--boltzmann_epsilon", type=float, default=None, help="Prob of Boltzmann sampling vs greedy")
    parser.add_argument("--boltzmann_temp_start", type=float, default=None, help="Initial Boltzmann temperature")
    parser.add_argument("--boltzmann_temp_final", type=float, default=None, help="Final Boltzmann temperature")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p nucleus sampling threshold")
    parser.add_argument("--capacity", type=int, default=None, help="Replay buffer capacity")
    # PPO-specific
    parser.add_argument("--ppo_clip", type=float, default=None, help="PPO clipping epsilon")
    parser.add_argument("--ppo_epochs", type=int, default=None, help="PPO epochs per update")
    parser.add_argument("--gae_lambda", type=float, default=None, help="GAE lambda")
    parser.add_argument("--entropy_coef", type=float, default=None, help="Entropy coefficient")
    parser.add_argument("--value_coef", type=float, default=None, help="Value loss coefficient")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--alpha_kl", type=float, default=None, help="KL regularization coefficient")
    parser.add_argument("--alpha_kl_warmup_steps", type=int, default=None, help="KL warmup steps")
    parser.add_argument("--freeze_backbone", action="store_true", default=None, help="Freeze backbone, only train heads")
    parser.add_argument("--detach_critic", action="store_true", default=None, help="Stop-gradient: prevent value loss from training backbone")
    parser.add_argument("--resume_training_state", type=str, default=None, help="Path to full training state checkpoint")
    parser.add_argument("--value_clip", type=float, default=None, help="Clip critic predictions to [-v, v]")
    parser.add_argument("--lr_min", type=float, default=None, help="Minimum LR for cosine schedule")
    # Common
    parser.add_argument("--eval_interval", type=int, default=None, help="Evaluation interval")
    parser.add_argument("--eval_episodes", type=int, default=None, help="Number of eval episodes")
    parser.add_argument("--weight_sync_freq", type=int, default=None, help="Sync weights every N steps")
    parser.add_argument("--worker_device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--gpu_per_worker", type=float, default=None)
    parser.add_argument("--num_envs_per_worker", type=int, default=None, help="Number of envs per worker for batched rollout")
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
    for field in ["algorithm", "num_workers", "num_steps", "batch_size", "device", "load_model",
                  "lr", "alpha_cql_init", "alpha_cql_final",
                  "exploration", "epsilon_start", "epsilon_final",
                  "boltzmann_epsilon", "boltzmann_temp_start", "boltzmann_temp_final", "top_p",
                  "capacity",
                  "ppo_clip", "ppo_epochs", "gae_lambda", "entropy_coef", "value_coef",
                  "weight_decay", "alpha_kl", "alpha_kl_warmup_steps",
                  "freeze_backbone", "detach_critic", "value_clip", "lr_min",
                  "resume_training_state",
                  "eval_interval", "eval_episodes", "weight_sync_freq", "worker_device",
                  "gpu_per_worker", "num_envs_per_worker",
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

    if cfg.algorithm == "ppo":
        from riichienv_ml.training.ppo_trainer import run_ppo_training
        run_ppo_training(cfg)
    else:
        from riichienv_ml.training.online_trainer import run_training
        run_training(cfg)


if __name__ == "__main__":
    main()
