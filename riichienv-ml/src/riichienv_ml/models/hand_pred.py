"""Hand prediction models for estimating opponent hands.

Predicts (3, tile_dim) tensor — expected tile counts per opponent.
Opponent order: relative (shimocha, toimen, kamicha).
"""

import math

import torch
import torch.nn as nn

from riichienv_ml.features.sequence_features import SequenceFeatureEncoder
from riichienv_ml.models.backbone import ResNetBackbone


class HandPredCNN(nn.Module):
    """CNN-based hand prediction using ResNetBackbone.

    Input:  (B, in_channels, tile_dim) from ObservationEncoder
    Output: (B, 3, tile_dim)  predicted tile counts per opponent
    """

    def __init__(
        self,
        in_channels: int = 74,
        conv_channels: int = 192,
        num_blocks: int = 12,
        fc_dim: int = 512,
        tile_dim: int = 34,
        **kwargs,
    ):
        super().__init__()
        self.tile_dim = tile_dim
        self.backbone = ResNetBackbone(
            in_channels=in_channels,
            conv_channels=conv_channels,
            num_blocks=num_blocks,
            fc_dim=fc_dim,
            tile_dim=tile_dim,
        )
        self.head = nn.Sequential(
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, 3 * tile_dim),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)          # (B, fc_dim)
        out = self.head(features)            # (B, 3 * tile_dim)
        out = out.reshape(-1, 3, self.tile_dim)  # (B, 3, tile_dim)
        out = self.relu(out)                 # non-negative constraint
        return out


class HandPredTransformer(nn.Module):
    """Transformer-based hand prediction over packed sequence features.

    Input:  (B, PACKED_SIZE) float32 from SequenceFeaturePackedEncoder
    Output: (B, 3, tile_dim)  predicted tile counts per opponent
    """

    def __init__(
        self,
        d_model: int = 384,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        tile_dim: int = 34,
        # Embedding sub-dimensions (asymmetric)
        d_sub: int | None = None,
        d_type: int = 96,
        d_other: int = 32,
        # Sequence lengths (must match encoder)
        max_prog_len: int = 256,
        max_cand_len: int = 32,
        # Vocab sizes (from SequenceFeatureEncoder)
        sparse_vocab: int = SequenceFeatureEncoder.SPARSE_VOCAB_SIZE,
        sparse_pad: int = SequenceFeatureEncoder.SPARSE_PAD,
        prog_dims: tuple = SequenceFeatureEncoder.PROG_DIMS,
        cand_dims: tuple = SequenceFeatureEncoder.CAND_DIMS,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.tile_dim = tile_dim

        if d_sub is not None:
            d_type = d_sub
            d_other = d_sub

        # Packed layout constants
        self._S = SequenceFeatureEncoder.MAX_SPARSE_LEN   # 25
        self._N = SequenceFeatureEncoder.NUM_NUMERIC       # 12
        self._P = max_prog_len
        self._C = max_cand_len

        # --- Embedding layers ---
        self.sparse_embed = nn.Embedding(
            sparse_vocab, d_model, padding_idx=sparse_pad)

        self.numeric_proj = nn.Sequential(
            nn.Linear(self._N, d_model),
            nn.LayerNorm(d_model),
        )

        # Progression: embed each of 5 fields -> concat -> project
        prog_sub_dims = [d_other if i != 1 else d_type for i in range(len(prog_dims))]
        self.prog_embeds = nn.ModuleList([
            nn.Embedding(dim, d_s) for dim, d_s in zip(prog_dims, prog_sub_dims)
        ])
        prog_cat_dim = sum(prog_sub_dims)
        self.prog_proj = nn.Sequential(
            nn.Linear(prog_cat_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Candidates: embed each of 4 fields -> concat -> project
        cand_sub_dims = [d_other if i != 0 else d_type for i in range(len(cand_dims))]
        self.cand_embeds = nn.ModuleList([
            nn.Embedding(dim, d_s) for dim, d_s in zip(cand_dims, cand_sub_dims)
        ])
        cand_cat_dim = sum(cand_sub_dims)
        self.cand_proj = nn.Sequential(
            nn.Linear(cand_cat_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # --- CLS token ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # --- Segment embeddings (4 groups: sparse / numeric / prog / cand) ---
        self.segment_embed = nn.Embedding(4, d_model)

        # --- Positional encoding (sinusoidal) ---
        max_seq = 1 + self._S + 1 + self._P + self._C
        self.register_buffer("pos_enc", self._sinusoidal_pe(max_seq, d_model))

        # --- Transformer encoder (pre-LN for stability) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(d_model)

        # --- Hand prediction head ---
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3 * tile_dim),
        )
        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    @staticmethod
    def _sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _unpack(self, x: torch.Tensor):
        """Unpack flat (B, PACKED_SIZE) tensor into components."""
        o = 0
        sparse = x[:, o:o + self._S].long();                        o += self._S
        numeric = x[:, o:o + self._N];                               o += self._N
        prog = x[:, o:o + self._P * 5].reshape(-1, self._P, 5).long()
        o += self._P * 5
        cand = x[:, o:o + self._C * 4].reshape(-1, self._C, 4).long()
        o += self._C * 4
        sparse_mask = x[:, o:o + self._S].bool();                   o += self._S
        prog_mask = x[:, o:o + self._P].bool();                     o += self._P
        cand_mask = x[:, o:o + self._C].bool()
        return sparse, numeric, prog, cand, sparse_mask, prog_mask, cand_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        sparse, numeric, prog, cand, sparse_mask, prog_mask, cand_mask = \
            self._unpack(x)

        # Embed sparse tokens: (B, 25, d)
        sparse_emb = self.sparse_embed(sparse)

        # Project numeric: (B, 1, d)
        numeric_emb = self.numeric_proj(numeric).unsqueeze(1)

        # Embed progression 5-tuples: (B, P, d)
        prog_parts = [emb(prog[:, :, i]) for i, emb in enumerate(self.prog_embeds)]
        prog_emb = self.prog_proj(torch.cat(prog_parts, dim=-1))

        # Embed candidate 4-tuples: (B, C, d)
        cand_parts = [emb(cand[:, :, i]) for i, emb in enumerate(self.cand_embeds)]
        cand_emb = self.cand_proj(torch.cat(cand_parts, dim=-1))

        # CLS token: (B, 1, d)
        cls = self.cls_token.expand(B, -1, -1)

        # Concatenate: [CLS, sparse(S), numeric(1), prog(P), cand(C)]
        tokens = torch.cat([cls, sparse_emb, numeric_emb, prog_emb, cand_emb], dim=1)

        # Add segment embeddings
        seg_ids = torch.cat([
            torch.zeros(B, 1 + self._S, dtype=torch.long, device=x.device),
            torch.ones(B, 1, dtype=torch.long, device=x.device),
            torch.full((B, self._P), 2, dtype=torch.long, device=x.device),
            torch.full((B, self._C), 3, dtype=torch.long, device=x.device),
        ], dim=1)
        tokens = tokens + self.segment_embed(seg_ids)

        # Add positional encoding
        tokens = tokens + self.pos_enc[:, :tokens.shape[1]]

        # Build padding mask: True = ignore (PyTorch convention)
        cls_valid = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        numeric_valid = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        pad_mask = torch.cat([
            cls_valid,
            ~sparse_mask,
            numeric_valid,
            ~prog_mask,
            ~cand_mask,
        ], dim=1)

        # Transformer
        output = self.transformer(tokens, src_key_padding_mask=pad_mask)
        output = self.final_norm(output)

        # CLS output -> hand prediction head
        cls_out = output[:, 0]               # (B, d_model)
        out = self.head(cls_out)             # (B, 3 * tile_dim)
        out = out.reshape(-1, 3, self.tile_dim)  # (B, 3, tile_dim)
        out = self.relu(out)                 # non-negative constraint
        return out
