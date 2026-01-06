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

> [!NOTE]
> This project is currently under active development. The API and specifications are subject to change before the stable release.

## ✨ Features

* **High Performance**: Core logic implemented in Rust for lightning-fast state transitions and rollouts.
* **Gym-style API**: Intuitive interface designed specifically for reinforcement learning.
* **Mortal Compatibility**: Seamlessly interface with the Mortal Bot using the standard MJAI protocol.
* **Rule Flexibility**: Support for diverse rule sets, including no-red-dragon variants and three-player mahjong.
* **Game Visualization**: Integrated replay viewer for Jupyter Notebooks.

## 📊 Performance

- [ ] TODO: Add performance benchmarks compared to other packages (`mahjong`, `mjx`, `mahjax`, `mortal`).

## 📦 Installation

Currently, building from source requires the **Rust** toolchain.

- [ ] TODO: Automated binary wheel distribution on PyPI.

```bash
uv add riichienv
# Or
pip install riichienv
```

## 🚀 Usage

- [ ] TODO: Support four-player half-round (hanchan) without red dragons.
- [ ] TODO: Complete three-player rule sets.
- [ ] TODO: Provide reference reinforcement learning examples.

### Gym-style API

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

`env.reset()` initializes the game state and returns the initial observations. The returned `obs_dict` maps each active player ID to their respective `Observation` object.

```python
>>> from riichienv import RiichiEnv
>>> env = RiichiEnv()
>>> obs_dict = env.reset()
>>> obs_dict
{0: <riichienv._riichienv.Observation object at 0x7fae7e52b6e0>}
```

Use `env.done()` to check if the game has concluded.

```python
>>> env.done()
False
```

By default, the environment runs a single round (kyoku). For game rules supporting sudden death or standard match formats like East-only or Half-round, the environment continues until the game-end conditions are met.

### Observation

The `Observation` object provides all relevant information to a player, including the current game state and available legal actions.

`obs.new_events() -> list[str]` returns a list of new events since the last step, encoded as JSON strings in the MJAI protocol. The full history of events is accessible via `obs.events`.

```python
>>> obs = obs_dict[0]
>>> obs.new_events()
['{"id":0,"type":"start_game"}', '{"bakaze":"E","dora_marker":"S", ...}', '{"actor":0,"pai":"6p","type":"tsumo"}']
```

`obs.legal_actions() -> list[Action]` provides the list of all valid moves the player can make.

```python
>>> obs.legal_actions()
[Action(action_type=Discard, tile=Some(1), ...), ...]
```

If your agent communicates via the MJAI protocol, you can easily map an MJAI response to a valid `Action` object using `obs.select_action_from_mjai()`.

```python
>>> obs.select_action_from_mjai({"type":"dahai","pai":"1m","tsumogiri":False,"actor":0})
Action(action_type=Discard, tile=Some(1), consume_tiles=[])
```

### Supported Game Rules

Switch between different rule sets using the `game_type` keyword argument in the constructor.

> [!NOTE]
> We plan to provide 12 standard preset rule sets. In the future, we will also allow granular customization (e.g., enabling/disabling red dragons, sudden death, 1-han minimum, etc.).

| Rule Set | Players | Duration | Red Dragons | Status |
|----------|---------|----------|-------------|--------|
| `4p-red-single` | 4 | 1 Round | Enabled | ✅ Ready (Default) |
| `4p-red-half` | 4 | Half-round | Enabled | ✅ Ready |
| `4p-red-east` | 4 | East-only | Enabled | ✅ Ready |
| `3p-red-single` | 3 | 1 Round | Enabled | 🚧 In progress |
| `3p-red-half` | 3 | Half-round | Enabled | 🚧 In progress |
| `3p-red-east` | 3 | East-only | Enabled | 🚧 In progress |

Example of initializing a four-player half-round game with red dragons:

```python
from riichienv import RiichiEnv
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

RiichiEnv is fully compatible with the Mortal MJAI bot processing flow. You can easily benchmark your models against Mortal using the MJAI event stream.

```python
from riichienv import RiichiEnv, Action
from model import load_model

class MortalAgent:
    def __init__(self, player_id: int):
        self.player_id = player_id
        # Initialize your libriichi.mjai.Bot or equivalent
        self.model = load_model(player_id, "./mortal_v4.pth")

    def act(self, obs) -> Action:
        resp = None
        for event in obs.new_events():
            resp = self.model.react(event)

        action = obs.select_action_from_mjai(resp)
        assert action is not None, "Mortal must return a legal action"
        return action

env = RiichiEnv(game_type="4p-red-half", mjai_mode=True)
agents = {pid: MortalAgent(pid) for pid in range(4)}
obs_dict = env.reset()
while not env.done():
    actions = {pid: agents[pid].act(obs) for pid, obs in obs_dict.items()}
    obs_dict = env.step(actions)

print(env.scores(), env.points(), env.ranks())
```

### Agari Calculation

Calculate hands and scores using an interface compatible with the popular `mahjong` package.

```python
# TBD
```

### Tile Conversion & Hand Parsing

Standardize between various tile formats (136-tile, MPSZ, MJAI) and easily parse hand strings.

```python
>>> import riichienv.convert as cvt
>>> cvt.mpsz_to_tid("1z")
108

>>> from riichienv import parse_hand
>>> parse_hand("123m406m789m777z")
```

See [DATA_REPRESENTATION.md](DATA_REPRESENTATION.md) for more details.

## Rust API

> [!WARNING]
> The Rust API is currently under construction and may be unstable.

- [ ] TODO: Publish crates to crates.io.

```rust
cargo add riichienv
```

## 🛠 Development

For more architectural details and contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md) and [DEVELOPMENT.md](DEVELOPMENT.md).

## 📄 License

Apache License 2.0