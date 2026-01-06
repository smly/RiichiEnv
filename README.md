<div align="center">
<img src="docs/assets/logo.jpg" width="35%">

<br />
<br />

[![CI](https://github.com/smly/RiichiEnv/actions/workflows/ci.yml/badge.svg)](https://github.com/smly/RiichiEnv/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/smly/RiichiEnv/demos/replay_demo.ipynb)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/smly/RiichiEnv/demos/replay_demo.ipynb)
![License](https://img.shields.io/github/license/smly/riichienv)

</div>

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
from riichienv import RiichiEnv
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

`env.reset()` はゲーム状態を初期して、最初の観測情報を返します。この観測情報は、行動可能なプレイヤーごとに `Observation` オブジェクトを格納した `obs_dict: dict[int, Observation]` です。

```python
>>> from riichienv import RiichiEnv
... env = RiichiEnv()
... obs_dict = env.reset()
... obs_dict
{0: <riichienv._riichienv.Observation object at 0x7fae7e52b6e0>}
```

ゲームの終了判定は `env.done()` で行います。

```python
>>> env.done()
False
```

デフォルトは1局の強制終了です。サドンデスルールありの東風や半荘などのゲームルールの場合、1局が終わった後も終了条件を満たすまで続行します。

### Observation

プレイヤーは `Observation` オブジェクトから行動可能なプレイヤーに与えられる観測情報や、選択可能な行動を取得できます。
`obs.new_events() -> list[str]` は、プレイヤーが観測する新しいイベントのリストです。イベント情報は MJAI プロトコルでエンコードされた JSON 文字列です。`obs.events: list[str]` プロパティにこれまでの全てのイベントが格納されています。

```python
>>> obs = obs_dict[0]
<riichienv._riichienv.Observation object at 0x7fae7e52b6e0>

>>> obs.new_events()
['{"id":0,"type":"start_game"}', '{"bakaze":"E","dora_marker":"S","honba":0,"kyoku":1,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"tehais":[["1m","4m","6m","1p","3p","5p","1s","2s","3s","4s","7s","E","W"],["?","?","?","?","?","?","?","?","?","?","?","?","?"],["?","?","?","?","?","?","?","?","?","?","?","?","?"],["?","?","?","?","?","?","?","?","?","?","?","?","?"]],"type":"start_kyoku"}', '{"actor":0,"pai":"6p","type":"tsumo"}']
```

`obs.legal_actions() -> list[Action]` は、プレイヤーが選択可能な行動のリストです。

```python
>>> obs.legal_actions()
[Action(action_type=Discard, tile=Some(1), consume_tiles=[]), Action(action_type=Discard, tile=Some(13), consume_tiles=[]), Action(action_type=Discard, tile=Some(23), consume_tiles=[]), Action(action_type=Discard, tile=Some(37), consume_tiles=[]), Action(action_type=Discard, tile=Some(44), consume_tiles=[]), Action(action_type=Discard, tile=Some(54), consume_tiles=[]), Action(action_type=Discard, tile=Some(57), consume_tiles=[]), Action(action_type=Discard, tile=Some(73), consume_tiles=[]), Action(action_type=Discard, tile=Some(78), consume_tiles=[]), Action(action_type=Discard, tile=Some(82), consume_tiles=[]), Action(action_type=Discard, tile=Some(85), consume_tiles=[]), Action(action_type=Discard, tile=Some(96), consume_tiles=[]), Action(action_type=Discard, tile=Some(108), consume_tiles=[]), Action(action_type=Discard, tile=Some(117), consume_tiles=[])]
```

もしあなたが書いているプログラムが MJAI プロトコルで通信する機能を持っている場合は、MJAI 形式の JSON データに対応する選択可能な Action オブジェクトを簡単に取り出すことができます。

```python
>>> obs.select_action_from_mjai({"type":"dahai","pai":"1m","tsumogiri":False,"actor":0})
Action(action_type=Discard, tile=Some(1), consume_tiles=[])
```

### Various Game Rules

`game_type` キーワード引数にルールセット名を与えることでルールを切り替えることができます。

>NOTE: 最終的に12種類のゲームルールをプリセットとして定義して提供する予定です。
>将来的には飛び終了や1翻縛り、責任払いの無効など、細かいルールをカスタマイズすることも検討します。

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

Mortal の mjai Bot とイベント処理フローの互換性を持ちます。
例えば以下のように実装することで Mortal で実装されたモデルとベンチマークをとることができます。

```python
from riichienv import RiichiEnv, Action

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
print(scores, points, ranks)
```

### Agari Calculation

`mahjong` パッケージと互換性を持つインターフェースで役と点数計算をすることができます。

```python
TBD
```

### Tile Conversion & Hand Parsing

136-tile format, mpsz format, mjai format など、牌の表現方法を変換することができます。

```python
>> import riichienv.convert as cvt
>> cvt.mpsz_to_tid("1z")
108

>> from riichienv import parse_hand
>> parse_hand("123m406m789m777z")
```

詳細については DATA_REPRESENTATION.md を参照ください。

## Rust API

>まだ未整備です

- [ ] TODO: Upload the binary packages to crates.io.

```rust
cargo add riichienv
```

## 🛠 Development

詳細については [CONTRIBUTING.md](CONTRIBUTING.md) と [DEVELOPMENT.md](DEVELOPMENT.md) を参照してください。

## LICENSE

Apache License 2.0