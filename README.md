<div align="center">
<img src="docs/assets/logo.jpg" width="35%">
</div>

[![rustfmt and clippy](https://github.com/smly/RiichiEnv/actions/workflows/rustfmt_clippy.yml/badge.svg?branch=main)](https://github.com/smly/RiichiEnv/actions/workflows/rustfmt_clippy.yml)
[![pytest](https://github.com/smly/RiichiEnv/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/smly/RiichiEnv/actions/workflows/pytest.yml)
[![ruff and ty](https://github.com/smly/RiichiEnv/actions/workflows/ruff_ty.yml/badge.svg?branch=main)](https://github.com/smly/RiichiEnv/actions/workflows/ruff_ty.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/smly/RiichiEnv/riichienv/notebooks/riichienv.ipynb)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/smly/RiichiEnv/riichienv/notebooks/riichienv.ipynb)
![License](https://img.shields.io/crates/l/daberu)

# RiichiEnv

**High-Performance Research Environment for Riichi Mahjong**

注意：現在、まだ安定版のリリースに向けて開発中です。仕様が変更される可能性があります。

- [ ] TODO: Colab badge
- [ ] TODO: Kaggle Notebook badge
- [ ] TODO: Build Status Badge

## ✨ Features

* **High-performance**: Rust implementation for fast state transitions and rollouts
* **Gym-like API**: Design for reinforcement learning
* **Compatible with Mortal**: Easy to connect with Mortal Bot using mjai protocol
* **Various Rules**: Support for various rules. No red dragons, three-player mahjong, etc.
* **Game Replay**: Save and replay the game on Jupyter notebook

## 📊 Performance

- [ ] TODO: Add performance comparison with other packages (`mahjong`, `mjx`, `mahjax`, `mortal`)

## 📦 Installation

For now, this package requires **Rust** to build the package.

- [ ] TODO: Upload the binary wheel packages to PyPI.

```bash
uv add riichienv
# Or
pip install riichienv
```

## 🚀 Usage

- [ ] TODO: Support four-player hanchan game without red dragons
- [ ] TODO: Support three-player game rules
- [ ] TODO: Example codes for reinforcement learning

### Gym-like API

```python
from riichienv import RiichiEnv, GameType
from riichienv.agents import RandomAgent

agent = RandomAgent()
env = RiichiEnv()
obs_dict = env.reset()
while not env.done():
    actions = {player_id: agent.act(obs)
               for player_id, obs in obs_dict.items()}
    obs_dict = env.step(actions)

scores, points, ranks = env.scores(), env.points(), env.ranks()
print(scores, points, ranks)
```

### Various Game Rules

`game_type` キーワード引数にルールセット名を与えることでルールを切り替えることができます。
最終的に12種類のゲームルールをプリセットとして定義して提供する予定です。
将来的には飛び終了や1翻縛り、責任払いの無効など、細かいルールをカスタマイズすることができるようにする予定です。

| Rule | Players | Rounds | Red Dragons | Available |
|------|---------|--------|-------------|-----------|
| `4p-red-single` | 4 | Single | True | ✅️ (Default) |
| `4p-red-half` | 4 | Half | True | ✅️ |
| `4p-red-east` | 4 | East | True | ✅️ |
| `3p-red-single` | 3 | Single | True | not yet |
| `3p-red-half` | 3 | Half | True | not yet |
| `3p-red-east` | 3 | East | True | not yet |

例えば4人半荘赤ドラありのルールの場合、以下のように指定します。

```python
from riichienv import RiichiEnv, GameType
from riichienv.agents import RandomAgent

agent = RandomAgent()
env = RiichiEnv(game_type="4p-red-half")
obs_dict = env.reset()
while not env.done():
    actions = {player_id: agent.act(obs)
               for player_id, obs in obs_dict.items()}
    obs_dict = env.step(actions)

scores, points, ranks = env.scores(), env.points(), env.ranks()
print(scores, points, ranks)
```

### Compatibility with Mortal

Mortal の mjai Bot とイベント処理フローの互換性を持ちます。`obs.new_events()` により、行動可能になるまでの未読の mjai イベントを文字列形式で取得できます。
`Agent` クラスの `act()` メソッドは `riichienv.action.Action` を返す必要があります。`obs.select_action_from_mjai()` メソッドを使うことで、mjai 形式のイベント文字列から選択可能な `Action` オブジェクトを選択することができます。

```python
from riichienv import RiichiEnv
from riichienv.game_mode import GameType
from riichienv.action import Action

from model import load_model

class MortalAgent:
    def __init__(self, player_id: int):
        self.player_id = player_id
        # Load `libriichi.mjai.Bot` instance
        self.model = load_model(player_id, "./mortal_v4.pth")

    def act(self, obs) -> Action:
        resp = None
        for event in obs.new_events():
            resp = self.model.react(event)

        action = obs.select_action_from_mjai(resp)
        assert action is not None, f"No response despite legal actions: {obs.legal_actions()}"
        return action

env = RiichiEnv(game_type="4p-red-half", mjai_mode=True)
agents = {pid: MortalAgent(pid) for pid in range(4)}
obs_dict = env.reset()
while not env.done():
    actions = {pid: agents[pid].act(obs) for pid, obs in obs_dict.items()}
    obs_dict = env.step(actions)

scores, points, ranks = env.scores(), env.points(), env.ranks()
print("FINISHED:", scores, points, ranks)
```

### Agari Calculation

`mahjong` パッケージと互換性を持つインターフェースで役と点数計算をすることができます。

```python
TBD
```

### Tile Conversion

136-tile format, mpsz format, mjai format など、牌の表現方法を変換することができます。

```python
import riichienv.convert as cvt
```

## Rust API

Python interface のオーバーヘッドを避けたい用途に対して、Rust package として利用することもできます。

- [ ] TODO: Upload the binary packages to crates.io.

```rust
cargo add riichienv
```

## 🛠 Development

- **Python**: 3.13+
- **Rust**: Nightly (recommended)
- **Build System**: `maturin`
- **OS**: MacOS, Windows, Linux

See detail in [CONTRIBUTING.md](CONTRIBUTING.md) and [DEVELOPMENT.md](DEVELOPMENT.md).

## LICENSE

Apache License 2.0