# riichienv-ml

Mahjong RL training pipeline for RiichiEnv. This package implements a 3-stage training approach:

1. **GRP (Global Reward Predictor)** — Trains a reward shaping model that predicts final game rankings from round-level features, providing dense reward signals for downstream RL.
2. **Offline RL (CQL)** — Conservative Q-Learning on human replay data with GRP-shaped rewards.
3. **Online RL (DQN + CQL)** — Fine-tunes the offline QNetwork via self-play with epsilon-greedy exploration and CQL regularization, using Ray-distributed workers.

## Setup

```sh
uv sync
```

## Usage

All scripts read from a shared config file (`configs/baseline.yml`) by default. CLI arguments override config values.

```sh
# Stage 1: Train GRP reward model
uv run python scripts/train_grp.py -c configs/baseline.yml

# Stage 2: Train offline CQL model
uv run python scripts/train_cql.py -c configs/baseline.yml
uv run python scripts/train_cql.py -c configs/baseline.yml --lr 5e-4 --batch_size 128 --alpha 0.1 --num_blocks 6 --conv_channels 128

# Stage 3: Train online RL model
uv run python scripts/train_online.py -c configs/baseline.yml
uv run python scripts/train_online.py -c configs/baseline.yml --load_model cql_model.pth --num_workers 12 --num_steps 5000000
```

## Project Structure

```
configs/          Config YAML files
scripts/          Training entry points
src/riichienv_ml/ Package source code
  config.py       Pydantic config models + YAML loader
  utils.py        Shared utilities
  models/         Model architectures (ResNetBackbone, QNetwork, GRP, Mortal)
  data/           Dataset classes and observation encoders
  training/       Trainer classes, learner, buffer, Ray workers
```
