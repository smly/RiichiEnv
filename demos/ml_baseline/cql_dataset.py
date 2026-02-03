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
    Encodes observation using all available feature encodings:
    - Standard encoding: (74, 34) - main spatial features
    - Discard history decay: (4, 34) - exponentially weighted discard history
    - Yaku possibility: (4, 21, 2) - rule-based yaku detection
    - Furiten ron possibility: (4, 21) - tsumogiri-based furiten detection
    - Shanten & efficiency: (4, 4) - shanten number and tile efficiency features
    - Kawa overview: (4, 7, 34) - per-player discard pile summary
    - Fuuro overview: (4, 4, 5, 34) - all players' meld details
    - Ankan overview: (4, 34) - concealed kan tracking
    - Action availability: (11,) - legal action flags
    - Riichi sutehais: (3, 3) - riichi discard tiles for opponents
    - Last tedashis: (3, 3) - last hand discards for opponents
    - Pass context: (3,) - current offer tile info
    - Discard candidates: (5,) - tile efficiency detail

    Total features: 2920 + 952 + 2720 + 136 + 11 + 9 + 9 + 3 + 5 = 6765 features
    """
    @staticmethod
    def encode_legacy(obs):
        """
        Legacy encoding using only standard features (46, 34).
        This is the original encoding used before adding new features.
        Note: obs.encode() now returns 74 channels, but we only use the first 46.
        Returns: torch.Tensor of shape (46, 34)
        """
        feat_bytes = obs.encode()
        # obs.encode() now returns 74 channels, extract first 46
        feat_numpy = np.frombuffer(feat_bytes, dtype=np.float32).reshape(74, 34).copy()
        # Take only the first 46 channels
        feat_legacy = feat_numpy[:46, :]
        return torch.from_numpy(feat_legacy)

    @staticmethod
    def encode(obs):
        # 1. Standard encoding (74, 34) = 2516 features
        standard_bytes = obs.encode()
        standard_numpy = np.frombuffer(standard_bytes, dtype=np.float32).reshape(74, 34).copy()

        # 2. Exponential decay discard history (4, 34) = 136 features
        decay_bytes = obs.encode_discard_history_decay()
        decay_numpy = np.frombuffer(decay_bytes, dtype=np.float32).reshape(4, 34).copy()

        # 3. Yaku possibility (4, 21, 2) = 168 features
        yaku_bytes = obs.encode_yaku_possibility()
        yaku_numpy = np.frombuffer(yaku_bytes, dtype=np.float32).reshape(4, 21, 2).copy()

        # 4. Furiten-aware ron possibility (4, 21) = 84 features
        furiten_bytes = obs.encode_furiten_ron_possibility()
        furiten_numpy = np.frombuffer(furiten_bytes, dtype=np.float32).reshape(4, 21).copy()

        # 5. Shanten and efficiency features (4, 4) = 16 features
        shanten_bytes = obs.encode_shanten_efficiency()
        shanten_numpy = np.frombuffer(shanten_bytes, dtype=np.float32).reshape(4, 4).copy()

        # 6. Kawa overview (4, 7, 34) = 952 features
        kawa_bytes = obs.encode_kawa_overview()
        kawa_numpy = np.frombuffer(kawa_bytes, dtype=np.float32).reshape(4, 7, 34).copy()

        # 7. Fuuro overview (4, 4, 5, 34) = 2720 features
        fuuro_bytes = obs.encode_fuuro_overview()
        fuuro_numpy = np.frombuffer(fuuro_bytes, dtype=np.float32).reshape(4, 4, 5, 34).copy()

        # 8. Ankan overview (4, 34) = 136 features
        ankan_bytes = obs.encode_ankan_overview()
        ankan_numpy = np.frombuffer(ankan_bytes, dtype=np.float32).reshape(4, 34).copy()

        # 9. Action availability (11,) = 11 features
        action_bytes = obs.encode_action_availability()
        action_numpy = np.frombuffer(action_bytes, dtype=np.float32).reshape(11).copy()

        # 10. Riichi sutehais (3, 3) = 9 features
        riichi_sutehais_bytes = obs.encode_riichi_sutehais()
        riichi_sutehais_numpy = np.frombuffer(riichi_sutehais_bytes, dtype=np.float32).reshape(3, 3).copy()

        # 11. Last tedashis (3, 3) = 9 features
        last_tedashis_bytes = obs.encode_last_tedashis()
        last_tedashis_numpy = np.frombuffer(last_tedashis_bytes, dtype=np.float32).reshape(3, 3).copy()

        # 12. Pass context (3,) = 3 features
        pass_context_bytes = obs.encode_pass_context()
        pass_context_numpy = np.frombuffer(pass_context_bytes, dtype=np.float32).reshape(3).copy()

        # 13. Discard candidates (5,) = 5 features
        discard_candidates_bytes = obs.encode_discard_candidates()
        discard_candidates_numpy = np.frombuffer(discard_candidates_bytes, dtype=np.float32).reshape(5).copy()

        # Convert to torch tensors
        standard_tensor = torch.from_numpy(standard_numpy)  # (74, 34)
        decay_tensor = torch.from_numpy(decay_numpy)        # (4, 34)
        yaku_tensor = torch.from_numpy(yaku_numpy)          # (4, 21, 2)
        furiten_tensor = torch.from_numpy(furiten_numpy)    # (4, 21)
        shanten_tensor = torch.from_numpy(shanten_numpy)    # (4, 4)
        kawa_tensor = torch.from_numpy(kawa_numpy)          # (4, 7, 34)
        fuuro_tensor = torch.from_numpy(fuuro_numpy)        # (4, 4, 5, 34)
        ankan_tensor = torch.from_numpy(ankan_numpy)        # (4, 34)
        action_tensor = torch.from_numpy(action_numpy)      # (11,)
        riichi_sutehais_tensor = torch.from_numpy(riichi_sutehais_numpy)  # (3, 3)
        last_tedashis_tensor = torch.from_numpy(last_tedashis_numpy)      # (3, 3)
        pass_context_tensor = torch.from_numpy(pass_context_numpy)        # (3,)
        discard_candidates_tensor = torch.from_numpy(discard_candidates_numpy)  # (5,)

        # Return dict to allow flexibility in the model
        return {
            'standard': standard_tensor,      # (74, 34) - spatial features
            'decay': decay_tensor,            # (4, 34) - discard decay
            'yaku': yaku_tensor,              # (4, 21, 2) - yaku possibility
            'furiten': furiten_tensor,        # (4, 21) - furiten detection
            'shanten': shanten_tensor,        # (4, 4) - shanten & efficiency
            'kawa': kawa_tensor,              # (4, 7, 34) - discard overview
            'fuuro': fuuro_tensor,            # (4, 4, 5, 34) - meld details
            'ankan': ankan_tensor,            # (4, 34) - concealed kan
            'action': action_tensor,          # (11,) - action availability
            'riichi_sutehais': riichi_sutehais_tensor,  # (3, 3) - riichi discards
            'last_tedashis': last_tedashis_tensor,      # (3, 3) - last hand discards
            'pass_context': pass_context_tensor,        # (3,) - current offer
            'discard_candidates': discard_candidates_tensor,  # (5,) - discard detail
        }

    @staticmethod
    def encode_flat(obs):
        """
        Returns a flattened 1D feature vector for MLP models.
        Total: 6765 features
        """
        feat_dict = ObservationEncoder.encode(obs)

        # Flatten all tensors
        standard_flat = feat_dict['standard'].flatten()  # 2516
        decay_flat = feat_dict['decay'].flatten()        # 136
        yaku_flat = feat_dict['yaku'].flatten()          # 168
        furiten_flat = feat_dict['furiten'].flatten()    # 84
        shanten_flat = feat_dict['shanten'].flatten()    # 16
        kawa_flat = feat_dict['kawa'].flatten()          # 952
        fuuro_flat = feat_dict['fuuro'].flatten()        # 2720
        ankan_flat = feat_dict['ankan'].flatten()        # 136
        action_flat = feat_dict['action'].flatten()      # 11
        riichi_sutehais_flat = feat_dict['riichi_sutehais'].flatten()  # 9
        last_tedashis_flat = feat_dict['last_tedashis'].flatten()      # 9
        pass_context_flat = feat_dict['pass_context'].flatten()        # 3
        discard_candidates_flat = feat_dict['discard_candidates'].flatten()  # 5

        # Concatenate into single vector
        return torch.cat([
            standard_flat, decay_flat, yaku_flat, furiten_flat, shanten_flat,
            kawa_flat, fuuro_flat, ankan_flat, action_flat,
            riichi_sutehais_flat, last_tedashis_flat, pass_context_flat, discard_candidates_flat
        ])  # 6765


class BaseDataset(IterableDataset):
    def __init__(self, data_sources, reward_predictor=None, gamma=0.99, is_train=True, use_flat=True):
        self.data_sources = data_sources
        self.reward_predictor = reward_predictor
        self.gamma = gamma
        self.is_train = is_train
        self.use_flat = use_flat  # If True, use flat encoding (for MLP); else use dict (for CNN)
        
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
                            # Encode features (flat or structured based on config)
                            if self.use_flat:
                                features = ObservationEncoder.encode_flat(obs)
                            else:
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
