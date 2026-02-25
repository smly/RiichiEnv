# riichienv-ml

Mahjong RL training pipeline for RiichiEnv. This package implements a 3-stage training approach:

1. **GRP (Global Reward Predictor)** — Trains a reward shaping model that predicts final game rankings from round-level features, providing dense reward signals for downstream RL.
2. **Offline RL (BC or CQL)** — Behavior Cloning or Conservative Q-Learning on human replay data with GRP-shaped rewards.
3. **Online RL (PPO or DQN)** — Fine-tunes the offline model via self-play with epsilon-greedy exploration using Ray-distributed workers.

## Setup

```sh
uv sync

# Stage1
uv run python scripts/train_grp.py -c src/riichienv_ml/configs/4p/grp.yml

# Stage2
uv run python scripts/train_bc.py -c src/riichienv_ml/configs/4p/bc_mortal.yml

# Stage3
uv run python scripts/train_ppo.py -c src/riichienv_ml/configs/4p/ppo.yml
```

## Structures

```
# Configs (4 players)
src/riichienv_ml/configs/4p/grp.yml
src/riichienv_ml/configs/4p/bc_mortal.yml
src/riichienv_ml/configs/4p/bc_logs.yml
src/riichienv_ml/configs/4p/cql.yml
src/riichienv_ml/configs/4p/ppo.yml
src/riichienv_ml/configs/4p/ppo_v2.yml
src/riichienv_ml/configs/4p/ppo_v3.yml

# Configs (3 players)
src/riichienv_ml/configs/3p/grp.yml
src/riichienv_ml/configs/3p/bc_logs.yml
src/riichienv_ml/configs/3p/ppo.yml
src/riichienv_ml/configs/3p/ppo_v2.yml
src/riichienv_ml/configs/3p/ppo_v3.yml

# Dataset
src/riichienv_ml/datasets/mjai_logs.py
src/riichienv_ml/datasets/ppo.py

# Features
src/riichienv_ml/features/feat_v1.py
src/riichienv_ml/features/feat_v2.py  # v1 + discard hisotry decay + shanten efficiency
src/riichienv_ml/features/feat_v3.py  # v2 + other features

# Models
src/riichienv_ml/models/

# Trainers
src/riichienv_ml/trainers/grp.py
src/riichienv_ml/trainers/bc_mortal.py
src/riichienv_ml/trainers/bc_logs.py
src/riichienv_ml/trainers/ppo.py
src/riichienv_ml/trainers/cql.py
```