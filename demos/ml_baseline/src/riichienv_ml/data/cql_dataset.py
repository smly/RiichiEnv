import glob
import lzma
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset

from riichienv import MjSoulReplay
from mjsoul_parser.parse import MjsoulPaifuParser


class GrpFeatureEncoder:
    def __init__(self, kyoku):
        self.kyoku = kyoku

    def encode(self) -> dict:
        feat = self.kyoku.take_grp_features()
        row = {
            "p0_init_score": feat["round_initial_scores"][0],
            "p1_init_score": feat["round_initial_scores"][1],
            "p2_init_score": feat["round_initial_scores"][2],
            "p3_init_score": feat["round_initial_scores"][3],
            "p0_end_score": feat["round_end_scores"][0],
            "p1_end_score": feat["round_end_scores"][1],
            "p2_end_score": feat["round_end_scores"][2],
            "p3_end_score": feat["round_end_scores"][3],
            "p0_delta_score": feat["round_delta_scores"][0],
            "p1_delta_score": feat["round_delta_scores"][1],
            "p2_delta_score": feat["round_delta_scores"][2],
            "p3_delta_score": feat["round_delta_scores"][3],
            "chang": feat["chang"],
            "ju": feat["ju"],
            "ben": feat["ben"],
            "liqibang": feat["liqibang"],
        }
        return row


class ObservationEncoder:
    """Encodes observation into (74, 34) spatial tensor using obs.encode()."""

    @staticmethod
    def encode(obs) -> torch.Tensor:
        """Returns (74, 34) float32 tensor from the Rust observation encoder."""
        feat_bytes = obs.encode()
        feat_numpy = np.frombuffer(feat_bytes, dtype=np.float32).reshape(74, 34).copy()
        return torch.from_numpy(feat_numpy)


class BaseDataset(IterableDataset):
    def __init__(self, data_sources, reward_predictor=None, gamma=0.99, is_train=True):
        self.data_sources = data_sources
        self.reward_predictor = reward_predictor
        self.gamma = gamma
        self.is_train = is_train

    def _get_files(self):
        if isinstance(self.data_sources, list):
            return self.data_sources
        elif isinstance(self.data_sources, str):
            return glob.glob(self.data_sources)
        return []


class MCDataset(BaseDataset):
    """
    Yields (features, action_id, return, mask)
    Target: Monte-Carlo Return (G_t), decayed.
    """
    encoder = ObservationEncoder

    def __iter__(self):
        files = self._get_files()
        if self.is_train:
            random.shuffle(files)

        # Shard files across DataLoader workers to avoid duplicated work
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            files = files[worker_info.id::worker_info.num_workers]

        for file_path in files:
            with lzma.open(file_path, "rb") as f:
                paifu = MjsoulPaifuParser.to_dict(f.read())

            replay = MjSoulReplay.from_dict(paifu.data)
            buffer = []

            try:
                for kyoku in replay.take_kyokus():
                    # Encode Group Features for Reward Prediction
                    grp_features = GrpFeatureEncoder(kyoku).encode()

                    # Batch all 4 players' reward predictions in one forward pass
                    assert self.reward_predictor is not None
                    all_rewards = self.reward_predictor.calc_all_player_rewards(grp_features)

                    for player_id in range(4):
                        trajectory = []
                        final_reward = all_rewards[player_id]

                        # Collect Trajectory
                        for obs, action in kyoku.steps(player_id):
                            features = self.encoder.encode(obs)
                            action_id = action.encode()

                            mask_bytes = obs.mask()
                            mask = np.frombuffer(mask_bytes, dtype=np.uint8).copy()
                            assert 0 <= action_id < mask.shape[0], f"action_id should be in [0, {mask.shape[0]})"
                            assert mask[action_id] == 1, f"action_id {action_id} should be legal"
                            trajectory.append((features, action_id, mask))

                        # Compute Returns
                        T = len(trajectory)
                        for t, (feat, act, mask) in enumerate(trajectory):
                            # Decayed Reward: R * gamma^(T-t-1)
                            decayed = final_reward * (self.gamma ** (T - t - 1))
                            buffer.append((feat, act, decayed, mask))
            except RuntimeError as e:
                print(f"Error processing paifu: {file_path}")
                raise e

            if self.is_train:
                random.shuffle(buffer)

            yield from buffer


class DiscardHistoryEncoder:
    """Encodes observation into (78, 34) tensor: base 74ch + 4ch discard decay."""

    @staticmethod
    def encode(obs) -> torch.Tensor:
        base = np.frombuffer(obs.encode(), dtype=np.float32).reshape(74, 34).copy()
        decay = np.frombuffer(obs.encode_discard_history_decay(), dtype=np.float32).reshape(4, 34).copy()
        combined = np.concatenate([base, decay], axis=0)
        return torch.from_numpy(combined)


class DiscardHistoryShantenEncoder:
    """Encodes observation into (94, 34) tensor: base 74ch + 4ch discard decay + 16ch shanten."""

    @staticmethod
    def encode(obs) -> torch.Tensor:
        base = np.frombuffer(obs.encode(), dtype=np.float32).reshape(74, 34).copy()
        decay = np.frombuffer(obs.encode_discard_history_decay(), dtype=np.float32).reshape(4, 34).copy()
        shanten = np.frombuffer(obs.encode_shanten_efficiency(), dtype=np.float32).reshape(4, 4).copy()
        shanten_broadcast = np.repeat(shanten.reshape(16, 1), 34, axis=1)
        combined = np.concatenate([base, decay, shanten_broadcast], axis=0)
        return torch.from_numpy(combined)


class DiscardHistoryDataset(MCDataset):
    """MCDataset with discard history decay features (78 channels)."""
    encoder = DiscardHistoryEncoder


class DiscardHistoryShantenDataset(MCDataset):
    """MCDataset with discard history + shanten features (94 channels)."""
    encoder = DiscardHistoryShantenEncoder


class ExtendedEncoder:
    """Encodes observation into (215, 34) tensor using a single consolidated Rust call."""

    @staticmethod
    def encode(obs) -> torch.Tensor:
        raw = obs.encode_extended()
        return torch.from_numpy(
            np.frombuffer(raw, dtype=np.float32).reshape(215, 34).copy()
        )


class ExtendedDataset(MCDataset):
    """MCDataset with extended features (215 channels)."""
    encoder = ExtendedEncoder


class ExtendedSPEncoder:
    """Encodes observation into (338, 34) tensor using a single consolidated Rust call."""

    @staticmethod
    def encode(obs) -> torch.Tensor:
        raw = obs.encode_extended_sp()
        return torch.from_numpy(
            np.frombuffer(raw, dtype=np.float32).reshape(338, 34).copy()
        )


class ExtendedSPDataset(MCDataset):
    """MCDataset with extended + SP features (338 channels)."""
    encoder = ExtendedSPEncoder
