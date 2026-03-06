# Demos

## GameViewer

`GameViewer` renders an interactive 3D replay viewer for Jupyter Notebooks. It takes MJAI-format event logs and displays the game state with metadata such as tenpai waits and winning hand details.

```python
from riichienv.visualizer import GameViewer
```

### Creating a viewer

`GameViewer` provides multiple ways to create a viewer instance.

#### `env.get_viewer()` — from a RiichiEnv instance

```python
from riichienv import RiichiEnv
from riichienv.agents import RandomAgent

agent = RandomAgent()
env = RiichiEnv(game_mode="4p-red-half")
obs_dict = env.reset()
while not env.done():
    actions = {pid: agent.act(obs) for pid, obs in obs_dict.items()}
    obs_dict = env.step(actions)

env.get_viewer().show()
```

#### `from_jsonl` — from a JSONL file

```python
viewer = GameViewer.from_jsonl("path/to/game.jsonl")
viewer.show(step=100, perspective=0)
```

#### `from_list` — from a list of event dicts

```python
events = [{"type": "start_game", ...}, ...]
viewer = GameViewer.from_list(events)
viewer.show(step=50, freeze=True)
```

### Displaying the viewer

`show()` accepts optional keyword arguments to control the initial display:

| Parameter | Type | Description |
|---|---|---|
| `step` | `int \| None` | Initial step to display |
| `perspective` | `int \| None` | Player perspective (0–3) |
| `freeze` | `bool` | Freeze the viewer at the given step |

In Jupyter, placing a `GameViewer` at the end of a cell automatically renders the 3D viewer via `_repr_html_`. You can also call `show()` explicitly to get the `HTML` object.

```python
viewer = env.get_viewer()
viewer          # auto-renders in Jupyter via _repr_html_
viewer.show()   # returns IPython.display.HTML
viewer.show(step=100, perspective=0, freeze=True)
```

### Inspecting game data

#### `summary()` — round overview

Returns a `list[dict]` with metadata for each round (kyoku) in the game.

```python
>>> viewer.summary()
[
    {'round_idx': 0, 'bakaze': 'E', 'kyoku': 1, 'honba': 0, 'oya': 0, 'scores': [25000, 25000, 25000, 25000]},
    {'round_idx': 1, 'bakaze': 'E', 'kyoku': 2, 'honba': 0, 'oya': 1, 'scores': [33000, 25000, 17000, 25000]},
    ...
]
```

| Key | Type | Description |
|---|---|---|
| `round_idx` | `int` | 0-indexed round number |
| `bakaze` | `str` | Round wind (`"E"`, `"S"`, `"W"`, `"N"`) |
| `kyoku` | `int` | Kyoku number within the wind |
| `honba` | `int` | Honba (repeat) counter |
| `oya` | `int` | Dealer seat (0–3) |
| `scores` | `list[int]` | Player scores at the start of the round |

#### `get_results(round_idx)` — winning results

Returns a `list[WinResult]` for the specified round. The list may contain multiple entries for double-ron, or be empty for a draw.

```python
>>> results = viewer.get_results(0)
>>> for r in results:
...     print(r)
WinResult(is_win=True, yakuman=False, ron_agari=7700, ...)

>>> results[0].yaku_list()
[Yaku(id=14, name='平和', name_en='Pinfu', ...), ...]
```

## Notebook Demos

| Notebook | Description |
|---|---|
| `replay_demo.ipynb` | Basic viewer usage with `from_jsonl` |
| `replay_debug.ipynb` | Viewer with programmatically constructed events via `from_list` |
