import glob
import logging
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)

from riichienv import MjaiReplay


def _compute_rank(end_scores: list, player_id: int, n_players: int) -> int:
    """Compute rank (0=1st, n-1=last) from end-of-kyoku scores."""
    scores = np.array(end_scores[:n_players], dtype=np.float64)
    return int((-scores).argsort(kind='stable').argsort(kind='stable')[player_id])


class GrpFeatureEncoder:
    """Extracts GRP features from a kyoku, parameterized by n_players."""

    def __init__(self, kyoku, n_players: int = 4):
        self.kyoku = kyoku
        self.n_players = n_players

    def encode(self) -> dict:
        feat = self.kyoku.take_grp_features()
        n = self.n_players
        row = {}
        for i in range(n):
            row[f"p{i}_init_score"] = feat["round_initial_scores"][i]
            row[f"p{i}_end_score"] = feat["round_end_scores"][i]
            row[f"p{i}_delta_score"] = feat["round_delta_scores"][i]
        row["chang"] = feat["chang"]
        row["ju"] = feat["ju"]
        row["ben"] = feat["ben"]
        row["liqibang"] = feat["liqibang"]
        return row


class BaseDataset(IterableDataset):
    def __init__(self, data_sources, reward_predictor=None, gamma=0.99,
                 is_train=True, n_players=4, replay_rule="mjsoul", encoder=None):
        self.data_sources = data_sources
        self.reward_predictor = reward_predictor
        self.gamma = gamma
        self.is_train = is_train
        self.n_players = n_players
        self.replay_rule = replay_rule
        self.encoder = encoder

    def _get_files(self):
        if isinstance(self.data_sources, list):
            return self.data_sources
        elif isinstance(self.data_sources, str):
            return glob.glob(self.data_sources, recursive=True)
        return []


class MCDataset(BaseDataset):
    """Yields (features, action_id, return, mask, rank).

    Target: Monte-Carlo Return (G_t), decayed.
    Uses MjaiReplay.from_jsonl() for replay parsing.
    """

    def __iter__(self):
        files = self._get_files()
        if self.is_train:
            random.shuffle(files)

        # Shard files across DataLoader workers to avoid duplicated work
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            files = files[worker_info.id::worker_info.num_workers]

        skipped = 0
        total = len(files)

        for file_path in files:
            try:
                replay = MjaiReplay.from_jsonl(file_path, rule=self.replay_rule)
            except (RuntimeError, ValueError) as e:
                logger.warning("Skipping unparseable replay: %s: %s", file_path, e)
                skipped += 1
                continue

            buffer = []

            try:
                for kyoku in replay.take_kyokus():
                    grp_features = GrpFeatureEncoder(kyoku, self.n_players).encode()

                    assert self.reward_predictor is not None
                    all_rewards = self.reward_predictor.calc_all_player_rewards(grp_features)

                    end_scores = [grp_features[f"p{i}_end_score"] for i in range(self.n_players)]

                    for player_id in range(self.n_players):
                        trajectory = []
                        final_reward = all_rewards[player_id]
                        rank = _compute_rank(end_scores, player_id, self.n_players)

                        for obs, action in kyoku.steps(player_id):
                            features = self.encoder.encode(obs)
                            action_id = action.encode()

                            mask_bytes = obs.mask()
                            mask = np.frombuffer(mask_bytes, dtype=np.uint8).copy()
                            assert 0 <= action_id < mask.shape[0], f"action_id should be in [0, {mask.shape[0]})"
                            assert mask[action_id] == 1, f"action_id {action_id} should be legal"
                            trajectory.append((features, action_id, mask))

                        T = len(trajectory)
                        for t, (feat, act, mask) in enumerate(trajectory):
                            decayed = final_reward * (self.gamma ** (T - t - 1))
                            buffer.append((feat, act, decayed, mask, rank))
            except (RuntimeError, ValueError) as e:
                logger.warning("Skipping replay due to error: %s: %s", file_path, e)
                skipped += 1
                continue

            if self.is_train:
                random.shuffle(buffer)

            yield from buffer

        if skipped > 0:
            logger.warning("Skipped %d / %d replay files due to errors", skipped, total)


class DiscardHistoryDataset(MCDataset):
    """MCDataset with discard history decay features (78 channels)."""
    pass


class DiscardHistoryShantenDataset(MCDataset):
    """MCDataset with discard history + shanten features (94 channels)."""
    pass


class ExtendedDataset(MCDataset):
    """MCDataset with extended features (215 channels)."""
    pass
