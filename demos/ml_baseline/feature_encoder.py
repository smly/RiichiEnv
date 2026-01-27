import torch
import numpy as np


class FeatureEncoder:
    """
    Encodes Riichi Mahjong game state into a tensor representation using Observation.

    NOTE: This might be a good candidate for porting to Rust as a baseline

    Output shape: (channels, 34)
    Channels:
    0-3: Own hand count (1, 2, 3, 4)
    4-7: Own Melds (Chi, Pon, Kan, Ankan)
    8-11: Discards p0-p3 (Frequency/Normalized)
    12: Dora indicator count
    13-16: Riichi status (p0-p3)
    17-20: Wind (p0-p3)
    21: Self Wind
    22: Round Wind
    """
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.num_channels = 23
        self.channels = 34
    
    def encode(self, obs) -> torch.Tensor:
        """
        Generate tensor from Observation object.
        obs: RiichiEnv Observation (has hands, melds, discards, etc.)
        """
        features = torch.zeros((self.num_channels, self.channels), dtype=torch.float32)
        
        # 0-3: Hand
        # obs.hands[self.player_id] is a list of tile IDs (0-135)
        my_hand = obs.hands[self.player_id]
        counts = np.zeros(34, dtype=np.int32)
        for t in my_hand:
            # Convert 136-tile to 34
            counts[t // 4] += 1
            
        for t in range(34):
            cnt = counts[t]
            if cnt >= 1: features[0, t] = 1
            if cnt >= 2: features[1, t] = 1
            if cnt >= 3: features[2, t] = 1
            if cnt >= 4: features[3, t] = 1
            
        # 4-7: Melds
        # obs.melds[p] is list of Meld objects.
        # 4=Chi, 5=Pon, 6=Kan, 7=Ankan
        my_melds = obs.melds[self.player_id]
        for m in my_melds:
            m_type = m.meld_type # enum int: Chi=0, Peng=1, Gang=2, Angang=3, Addgang=4
            channel = -1
            if m_type == 0: channel = 4
            elif m_type == 1: channel = 5
            elif m_type == 2 or m_type == 4: channel = 6 
            elif m_type == 3: channel = 7
            
            if channel != -1:
                # Mark tiles involved
                for t in m.tiles:
                    features[channel, t // 4] = 1

        # 8-11: Discards
        for p in range(4):
            d_counts = np.zeros(34, dtype=np.int32)
            for t in obs.discards[p]:
                d_counts[t // 4] += 1
            features[8+p] = torch.from_numpy(d_counts).float() / 4.0
            
        # 12: Dora Indicator
        for d in obs.dora_indicators:
            features[12, d // 4] += 1
            
        # 13-16: Riichi
        for p in range(4):
            if obs.riichi_declared[p]:
                features[13+p, :] = 1
                
        # 17-20: Seat Wind
        for p in range(4):
            wind = (p - obs.oya + 4) % 4
            features[17+p, :] = float(wind) / 4.0

        # 21: Self Wind
        self_wind = (self.player_id - obs.oya + 4) % 4
        features[21, :] = float(self_wind) / 4.0
        
        # 22: Round Wind
        features[22, :] = float(obs.round_wind) / 4.0

        return features

    def get_mask(self, obs) -> torch.Tensor:
        """
        Generate mask from legal actions in Observation.
        """
        mask = torch.zeros(40, dtype=torch.bool)
        
        # Map actions to mask indices
        # 0-33: Discard
        # 34: Chi (generic)
        # 35: Pon
        # 36: Kan
        # 37: Riichi
        # 38: Agari (Ron/Tsumo)
        # 39: Pass
        
        # We need to check obs.legal_actions
        # ActionType: Discard=0, Chi=1, Pon=2, Kan=3...
        # Action struct has action_type and tile.
        
        for act in obs.legal_actions:
            atype_int = act.type # pyo3 exposed as `type` property or `action_type` field?
            # Rust `Action`: `pub action_type: ActionType`. PyO3 getter usually `action_type`.
            # But `to_dict` used `type`.
            # If we access python object attributes:
            # act.action_type (int enum value if eq_int)
            
            # Since `Action` is a class, we access properties.
            # In Rust `ActionType` is int-enum.
            
            # Mapping:
            # Discard (0)
            if act.action_type == 0 and act.tile is not None:
                # act.tile is 0-135.
                mask[act.tile // 4] = True
            
            # Chi (1), Pon (2), Daiminkan (3), Ankan (8), Kakan (9)
            elif act.action_type == 1: mask[34] = True
            elif act.action_type == 2: mask[35] = True
            elif act.action_type == 3 or act.action_type == 8 or act.action_type == 9: mask[36] = True # Kan
            
            # Riichi (5)
            elif act.action_type == 5: mask[37] = True
            
            # Tsumo (6), Ron (4)
            elif act.action_type == 6 or act.action_type == 4: mask[38] = True
            
            # Pass (7)
            elif act.action_type == 7: mask[39] = True
            
        return mask
