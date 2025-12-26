# RiichiEnv Core

**High-Performance Research Environment for Riichi Mahjong**

`riichienv` は、Rust による高速な麻雀シミュレーションと、Python (Gym) API を提供する研究用ライブラリです。

## ✨ Features

- **高速シミュレーション**: Rust 実装により、非常に高速な状態遷移とロールアウトが可能。
- **並列化 (VecEnv)**: `step_batch` による数千卓規模の並列実行をサポート。
- **柔軟なルールセット**: 4人麻雀/3人麻雀、赤ドラ、ウマ/オカなどのルール設定が可能。
- **Gym 互換 API**: 強化学習の標準的なインターフェース (`reset`, `step`, `step_batch`) を提供。
- **mjai プロトコル**: 学習環境として必要な mjai メッセージの解釈と生成をサポート。

## 📦 Installation

This package requires **Rust** to build the core extension.

```bash
# Using uv (Recommended)
uv sync
# or
uv pip install .

# Using pip
pip install .
```

## 🚀 Usage

```python
from riichienv import RiichiEnv
from riichienv.agents import RandomAgent

agent = RandomAgent()
env = RiichiEnv()
obs_dict = env.reset()
while not env.done():
    actions = {player_id: agent.act(obs)
               for player_id, obs in obs_dict.items()}
    obs_dict = env.step(actions)

returns = env.rewards()
```

## 🛠 Development

- **Python**: 3.13+
- **Rust**: Nightly (recommended)
- **Build System**: `maturin`

```bash
# Developers build
uv run maturin develop
```
