"""Hand prediction dataset for estimating opponent hands.

Iterates MJAI replay files, tracking all players' hands from events,
and yields (features, label) where label is a (3, tile_dim) tensor
of tile-type counts for the 3 opponents (relative order: shimocha, toimen, kamicha).
"""

import glob as glob_mod
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset

from riichienv import MjaiReplay


def _mjai_tile_to_type(tile_str: str) -> int:
    """Convert MJAI tile string (e.g. '1m', '5pr', 'E') to tile type index (0-33).

    Returns -1 for unknown tiles ('?').
    """
    if tile_str == "?":
        return -1
    # Strip red-dora suffix
    s = tile_str.rstrip("r")
    last = s[-1]
    if last in "mps":
        offset = {"m": 0, "p": 9, "s": 18}[last]
        return offset + int(s[:-1]) - 1
    # Honor tiles
    honor = {"E": 27, "S": 28, "W": 29, "N": 30, "P": 31, "F": 32, "C": 33}
    return honor.get(s, -1)


def _build_hand_timeline(events: list[dict], n_players: int, tile_dim: int) -> list[np.ndarray]:
    """Build per-event hand-count snapshots from full MJAI events.

    Returns a list of (n_players, tile_dim) arrays, one per event.
    Each array gives tile-type counts for every player's hand AFTER that event.
    """
    counts = np.zeros((n_players, tile_dim), dtype=np.float32)
    timeline: list[np.ndarray] = []

    for ev in events:
        etype = ev.get("type", "")

        if etype == "start_kyoku":
            counts[:] = 0
            for p, tehai in enumerate(ev.get("tehais", [])):
                if p >= n_players:
                    break
                for t_str in tehai:
                    t = _mjai_tile_to_type(t_str)
                    if t >= 0:
                        counts[p, t] += 1

        elif etype == "tsumo":
            actor = ev.get("actor")
            if actor is not None:
                t = _mjai_tile_to_type(ev.get("pai", "?"))
                if t >= 0:
                    counts[actor, t] += 1

        elif etype == "dahai":
            actor = ev.get("actor")
            if actor is not None:
                t = _mjai_tile_to_type(ev.get("pai", "?"))
                if t >= 0:
                    counts[actor, t] -= 1

        elif etype in ("chi", "pon", "daiminkan"):
            actor = ev.get("actor")
            if actor is not None:
                for t_str in ev.get("consumed", []):
                    t = _mjai_tile_to_type(t_str)
                    if t >= 0:
                        counts[actor, t] -= 1

        elif etype == "ankan":
            actor = ev.get("actor")
            if actor is not None:
                for t_str in ev.get("consumed", []):
                    t = _mjai_tile_to_type(t_str)
                    if t >= 0:
                        counts[actor, t] -= 1

        elif etype == "kakan":
            actor = ev.get("actor")
            if actor is not None:
                t = _mjai_tile_to_type(ev.get("pai", "?"))
                if t >= 0:
                    counts[actor, t] -= 1

        timeline.append(counts.copy())

    return timeline


class HandPredDataset(IterableDataset):
    """Iterable dataset for hand prediction from MJAI replay files.

    For each decision point in a replay, yields:
        features:         Tensor from encoder.encode(obs)
        label:            (3, tile_dim) float32 — tile-type counts for 3 opponents
                          in relative order (shimocha, toimen, kamicha)
        concealed_counts: (3,) float32 — total concealed tiles per opponent
                          (= label.sum(dim=-1), provided for sum-constraint loss)
    """

    def __init__(
        self,
        data_glob: str,
        n_players: int = 4,
        tile_dim: int = 34,
        replay_rule: str = "mjsoul",
        is_train: bool = True,
        encoder=None,
    ):
        self.data_glob = data_glob
        self.n_players = n_players
        self.tile_dim = tile_dim
        self.replay_rule = replay_rule
        self.is_train = is_train
        self.encoder = encoder

    def _get_files(self) -> list[str]:
        return sorted(glob_mod.glob(self.data_glob, recursive=True))

    def __iter__(self):
        files = self._get_files()
        if self.is_train:
            random.shuffle(files)

        # Shard files across DataLoader workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            files = files[worker_info.id::worker_info.num_workers]

        for file_path in files:
            try:
                replay = MjaiReplay.from_jsonl(file_path, rule=self.replay_rule)
                buffer: list[tuple[torch.Tensor, torch.Tensor]] = []

                for kyoku in replay.take_kyokus():
                    # Build hand timeline from full (unmasked) events
                    timeline = _build_hand_timeline(
                        kyoku.events, self.n_players, self.tile_dim)

                    if not timeline:
                        continue

                    for player_id in range(self.n_players):
                        for obs, _action in kyoku.steps(player_id):
                            # Synchronize: obs.events has events up to this point
                            event_idx = len(obs.events) - 1
                            if event_idx < 0 or event_idx >= len(timeline):
                                continue

                            hand_state = timeline[event_idx]  # (n_players, tile_dim)

                            # Build label: 3 opponents in relative order
                            label = np.zeros(
                                (self.n_players - 1, self.tile_dim), dtype=np.float32)
                            for rel in range(1, self.n_players):
                                abs_id = (player_id + rel) % self.n_players
                                label[rel - 1] = hand_state[abs_id]

                            # concealed_counts: known total tiles per opponent
                            concealed_counts = label.sum(axis=-1)  # (3,)

                            features = self.encoder.encode(obs)
                            buffer.append((
                                features,
                                torch.from_numpy(label),
                                torch.from_numpy(concealed_counts),
                            ))

                # Flush buffer
                if buffer:
                    if self.is_train:
                        random.shuffle(buffer)
                    yield from buffer

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
