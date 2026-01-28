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

    def encode(self) -> list[int]:
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
    """
    Wraps the Rust Observation.encode() method which returns a numpy array (46, 34).
    """
    @staticmethod
    def encode(obs):
        # obs.encode() returns bytes, convert to numpy
        feat_bytes = obs.encode()
        feat_numpy = np.frombuffer(feat_bytes, dtype=np.float32).reshape(46, 34).copy()
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
    def __iter__(self):
        files = self._get_files()
        if self.is_train:
            random.shuffle(files)
            
        for file_path in files:
            with lzma.open(file_path, "rb") as f:
                paifu = MjsoulPaifuParser.to_dict(f.read())
            
            replay = MjSoulReplay.from_dict(paifu.data)
            buffer = []

            try:
                for kyoku in replay.take_kyokus():
                    # Encode Group Features for Reward Prediction
                    grp_features = GrpFeatureEncoder(kyoku).encode()
                    
                    for player_id in range(4):
                        trajectory = []

                        # Compute Final Reward for this Kyoku
                        assert self.reward_predictor is not None
                        _, final_reward = self.reward_predictor.calc_pts_rewards([grp_features], player_id)

                        # Collect Trajectory
                        for obs, action in kyoku.steps(player_id):
                            features = ObservationEncoder.encode(obs)
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


class TransitionDataset(BaseDataset):
    """
    Yields (features, action_id, reward, next_features, done, mask, next_mask)
    Target: r + gamma * max Q(s', a')
    """
    pass
