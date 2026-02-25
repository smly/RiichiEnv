# riichienv-ml

Mahjong RL training pipeline for RiichiEnv.

## Setup

```sh
uv sync
uv run python scripts/train_grp.py -c src/riichienv_ml/configs/4p/grp.yml

# CQL+PPO
uv run python scripts/train_cql.py -c src/riichienv_ml/configs/4p/cql.yml
uv run python scripts/train_ppo.py -c src/riichienv_ml/configs/4p/ppo.yml

# BC+PPO (requires online teacher, not included in repo)
uv run python scripts/train_bc.py -c src/riichienv_ml/configs/4p/bc_model.yml
uv run python scripts/train_ppo.py -c src/riichienv_ml/configs/4p/bc_ppo.yml
```
