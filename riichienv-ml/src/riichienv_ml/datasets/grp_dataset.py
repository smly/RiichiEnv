import glob
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset

from riichienv import MjaiReplay

from riichienv_ml.datasets.mjai_logs import GrpFeatureEncoder, _compute_rank


class GrpReplayDataset(IterableDataset):
    """GRP dataset that reads .jsonl.gz replay files via MjaiReplay.

    For each kyoku in each replay, extracts GRP features and computes
    end-of-kyoku rank for each player. Yields (features_tensor, rank_one_hot).
    """

    def __init__(
        self,
        data_glob: str,
        n_players: int = 4,
        replay_rule: str = "mjsoul",
        is_train: bool = True,
    ):
        self.data_glob = data_glob
        self.n_players = n_players
        self.replay_rule = replay_rule
        self.is_train = is_train

    def _get_files(self) -> list[str]:
        files = sorted(glob.glob(self.data_glob, recursive=True))
        return files

    def _encode_sample(self, grp_features: dict, player_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.n_players

        scores = np.array(
            [grp_features[f"p{i}_init_score"] / 25000.0 for i in range(n)]
            + [grp_features[f"p{i}_end_score"] / 25000.0 for i in range(n)]
            + [grp_features[f"p{i}_delta_score"] / 12000.0 for i in range(n)],
            dtype=np.float32,
        )
        round_meta = np.array([
            grp_features["chang"] / 3.0,
            grp_features["ju"] / 3.0,
            grp_features["ben"] / 4.0,
            grp_features["liqibang"] / 4.0,
        ], dtype=np.float32)
        player = np.zeros(n, dtype=np.float32)
        player[player_idx] = 1.0

        x = np.concatenate([scores, round_meta, player])

        end_scores = [grp_features[f"p{i}_end_score"] for i in range(n)]
        rank = _compute_rank(end_scores, player_idx, n)
        y = np.zeros(n, dtype=np.float32)
        y[rank] = 1.0

        return torch.from_numpy(x), torch.from_numpy(y)

    def __iter__(self):
        files = self._get_files()
        if self.is_train:
            random.shuffle(files)

        # Shard files across DataLoader workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            files = files[worker_info.id::worker_info.num_workers]

        buffer = []
        for file_path in files:
            try:
                replay = MjaiReplay.from_jsonl(file_path, rule=self.replay_rule)
                for kyoku in replay.take_kyokus():
                    grp_features = GrpFeatureEncoder(kyoku, self.n_players).encode()
                    for player_idx in range(self.n_players):
                        sample = self._encode_sample(grp_features, player_idx)
                        buffer.append(sample)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

            # Flush buffer periodically to limit memory usage
            if len(buffer) >= 10000:
                if self.is_train:
                    random.shuffle(buffer)
                yield from buffer
                buffer.clear()

        # Flush remaining
        if buffer:
            if self.is_train:
                random.shuffle(buffer)
            yield from buffer
