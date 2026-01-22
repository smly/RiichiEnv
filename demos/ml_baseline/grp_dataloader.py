import polars as pl
import numpy as np

import torch
from torch.utils.data import Dataset


class RankPredictorDataset(Dataset):
    def __init__(self, dataframe: pl.DataFrame):
        self.df = dataframe.with_columns(
            (pl.col("chang") / 3).cast(pl.Float32).alias("chang"),
            (pl.col("ju") / 3).cast(pl.Float32).alias("ju"),
            (pl.col("ben") / 4).cast(pl.Float32).alias("ben"),
            (pl.col("liqibang") / 4).cast(pl.Float32).alias("liqibang"),
            (pl.col("p0_init_score") / 25000.0).cast(pl.Float32).alias("p0_init_score"),
            (pl.col("p1_init_score") / 25000.0).cast(pl.Float32).alias("p1_init_score"),
            (pl.col("p2_init_score") / 25000.0).cast(pl.Float32).alias("p2_init_score"),
            (pl.col("p3_init_score") / 25000.0).cast(pl.Float32).alias("p3_init_score"),
            (pl.col("p0_end_score") / 25000.0).cast(pl.Float32).alias("p0_end_score"),
            (pl.col("p1_end_score") / 25000.0).cast(pl.Float32).alias("p1_end_score"),
            (pl.col("p2_end_score") / 25000.0).cast(pl.Float32).alias("p2_end_score"),
            (pl.col("p3_end_score") / 25000.0).cast(pl.Float32).alias("p3_end_score"),
            (pl.col("p0_delta_score") / 12000.0).cast(pl.Float32).alias("p0_delta_score"),
            (pl.col("p1_delta_score") / 12000.0).cast(pl.Float32).alias("p1_delta_score"),
            (pl.col("p2_delta_score") / 12000.0).cast(pl.Float32).alias("p2_delta_score"),
            (pl.col("p3_delta_score") / 12000.0).cast(pl.Float32).alias("p3_delta_score"),
        )

    def __len__(self):
        return len(self.df) * 4

    def __getitem__(self, idx):
        row = self.df.row(idx // 4, named=True)
        player_idx = idx % 4
        scores = np.array([
            row["p0_init_score"],
            row["p1_init_score"],
            row["p2_init_score"],
            row["p3_init_score"],
            row["p0_end_score"],
            row["p1_end_score"],
            row["p2_end_score"],
            row["p3_end_score"],
            row["p0_delta_score"],
            row["p1_delta_score"],
            row["p2_delta_score"],
            row["p3_delta_score"],
        ])
        round_meta = np.array([
            row["chang"], row["ju"], row["ben"], row["liqibang"]
        ])
        player = np.zeros(4)
        player[player_idx] = 1.0
        x = np.concatenate([scores, round_meta, player])
        y = np.zeros(4)
        y[row[f"p{player_idx:d}_rank"]] = 1.0

        return torch.Tensor(x), torch.Tensor(y)
