import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class RankPredictor(nn.Module):
    def __init__(self, input_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = torch.relu(x)
        x = self.fc2(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


class RewardPredictor:
    def __init__(self, model_path: str, pts_weight: list[float], input_dim: int = 20, device: str = "cuda"):
        self.device: str = device
        self.model: RankPredictor = RankPredictor(input_dim)
        self.pts_weight: list[float] = pts_weight

        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(torch.device(device))
        self.model = self.model.eval()

    def _calc_pts(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return torch.softmax(self.model(x), dim=1) @ torch.tensor(self.pts_weight, device=self.device).float()

    def calc_pts_reward(self, row: dict, player_idx: int) -> np.array:
        scores = np.array([
            row["p0_init_score"],
            row["p1_init_score"],
            row["p2_init_score"],
            row["p3_init_score"],
            row["p0_end_score"],
            row["p1_end_score"],
            row["p2_end_score"],
            row["p3_end_score"],
        ])
        delta_scores = np.array([
            row["p0_delta_score"],
            row["p1_delta_score"],
            row["p2_delta_score"],
            row["p3_delta_score"],
        ])
        scores = scores / 25000.0
        delta_scores = delta_scores / 12000.0
        round_meta = np.array([
            row["chang"] / 3.0, row["ju"] / 3.0, row["ben"] / 4.0, row["liqibang"] / 4.0
        ])
        player = np.zeros(4)
        player[player_idx] = 1.0

        x = np.concatenate([scores, delta_scores, round_meta, player])
        return x

    def calc_all_player_rewards(self, grp_features: dict) -> list[float]:
        """Compute final rewards for all 4 players in one batched forward pass."""
        scores = np.array([
            grp_features["p0_init_score"], grp_features["p1_init_score"],
            grp_features["p2_init_score"], grp_features["p3_init_score"],
            grp_features["p0_end_score"], grp_features["p1_end_score"],
            grp_features["p2_end_score"], grp_features["p3_end_score"],
        ]) / 25000.0
        delta_scores = np.array([
            grp_features["p0_delta_score"], grp_features["p1_delta_score"],
            grp_features["p2_delta_score"], grp_features["p3_delta_score"],
        ]) / 12000.0
        round_meta = np.array([
            grp_features["chang"] / 3.0, grp_features["ju"] / 3.0,
            grp_features["ben"] / 4.0, grp_features["liqibang"] / 4.0,
        ])
        base = np.concatenate([scores, delta_scores, round_meta])  # (16,)

        # Build batch of 4 inputs: same base, different player one-hot
        xs = np.zeros((4, 20), dtype=np.float32)
        for pid in range(4):
            xs[pid, :16] = base
            xs[pid, 16 + pid] = 1.0

        xs_t = torch.from_numpy(xs).to(self.device)
        mean_pts = float(np.mean(self.pts_weight))
        with torch.inference_mode():
            pts_w = torch.tensor(self.pts_weight, device=self.device).float()
            pts = torch.softmax(self.model(xs_t), dim=1) @ pts_w  # (4,)
        rewards = (pts - mean_pts).cpu().tolist()
        return rewards

    def calc_pts_rewards(self, kyoku_features: list[dict], player_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        xs = []
        for row in kyoku_features:
            x = self.calc_pts_reward(row, player_idx)
            xs.append(x)

        xs = torch.from_numpy(np.array(xs)).float().to(self.device)
        pts = torch.concat([
            torch.tensor([np.mean(self.pts_weight)], device=self.device).float(),
            self._calc_pts(xs)
        ], dim=0)

        rewards = pts[1:] - pts[:-1]
        return pts[1:], rewards
