"""Validate SP feature encodings on MJAI jsonl/jsonl.gz logs.

The validator is intended for large 4P and 3P MjSoul dumps such as
``data-mjsoul-{4p,3p}-2026-01``. It checks structural invariants that should hold
for SP features used as ML inputs, and writes compact JSONL diagnostics for
any observation that violates them.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from riichienv import MjaiReplay, calculate_shanten
from riichienv._riichienv import ActionType, Observation, Observation3P

# Pull the canonical feature-block sizes from the Rust binding when available.
# The fallback values are last-known counts; the binding constants are the
# single source of truth and should win whenever the wheel was rebuilt after
# any feature-space change.
try:
    from riichienv._riichienv import (
        DREV_CHANNELS,
        OBS_EXTENDED_CHANNELS_3P,
        SP_CHANNELS,
        TILE_TYPES,
        TILE_TYPES_3P,
    )
    from riichienv._riichienv import (
        OBS_EXTENDED_CHANNELS as EXTENDED_CHANNELS,
    )
except ImportError:
    TILE_TYPES = 34
    TILE_TYPES_3P = 27
    EXTENDED_CHANNELS = 215
    OBS_EXTENDED_CHANNELS_3P = 215
    SP_CHANNELS = 178
    DREV_CHANNELS = 9

SP_CANDIDATE_TYPES = 34
CANONICAL_TILES_4P = tuple(range(34))
CANONICAL_TILES_3P = (0, 8, *range(9, 34))

EPS = 1e-5


@dataclass(frozen=True)
class FeatureLayout:
    name: str
    tile_types: int
    extended_channels: int
    compact_to_canonical: tuple[int, ...]

    @property
    def canonical_to_compact(self) -> dict[int, int]:
        return {tile: column for column, tile in enumerate(self.compact_to_canonical)}

    @property
    def sp_floats(self) -> int:
        return SP_CHANNELS * self.tile_types

    @property
    def drev_floats(self) -> int:
        return DREV_CHANNELS * self.tile_types

    @property
    def extended_floats(self) -> int:
        return self.extended_channels * self.tile_types

    @property
    def combined_floats(self) -> int:
        return (self.extended_channels + SP_CHANNELS + DREV_CHANNELS) * self.tile_types


LAYOUT_4P = FeatureLayout("4p", TILE_TYPES, EXTENDED_CHANNELS, CANONICAL_TILES_4P)
LAYOUT_3P = FeatureLayout(
    "3p", TILE_TYPES_3P, OBS_EXTENDED_CHANNELS_3P, CANONICAL_TILES_3P
)
# Backward-compatible aliases for callers that validate synthetic 4P arrays.
SP_FLOATS = LAYOUT_4P.sp_floats
DREV_FLOATS = LAYOUT_4P.drev_floats
EXTENDED_FLOATS = LAYOUT_4P.extended_floats
EXTENDED_WITH_SP_FLOATS = LAYOUT_4P.combined_floats


@dataclass
class ValidationIssue:
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservationMetrics:
    encode_sp_ns: int = 0
    encode_extended_with_sp_ns: int = 0
    legal_discard_count: int = 0
    candidate_count: int = 0
    best_ev_100k: float = 0.0
    best_ev_30k: float = 0.0
    is_menzen: bool = True
    riichi_assumed: bool = False


@dataclass
class ValidationContext:
    file: str
    kyoku: int
    step: int
    seat: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "kyoku": self.kyoku,
            "step": self.step,
            "seat": self.seat,
        }


@dataclass
class ValidationStats:
    files_seen: int = 0
    files_validated: int = 0
    kyokus_seen: int = 0
    observations_seen: int = 0
    observations_validated: int = 0
    observations_validated_4p: int = 0
    observations_validated_3p: int = 0
    observations_with_issues: int = 0
    issues: int = 0
    closed_riichi_assumed: int = 0
    open_hand: int = 0
    no_legal_discard: int = 0
    total_candidates: int = 0
    max_candidates: int = 0
    encode_sp_total_ns: int = 0
    encode_sp_max_ns: int = 0
    encode_extended_with_sp_total_ns: int = 0
    encode_extended_with_sp_max_ns: int = 0
    max_best_ev_100k: float = 0.0
    max_best_ev_30k: float = 0.0

    def add_metrics(self, metrics: ObservationMetrics, layout: FeatureLayout) -> None:
        self.observations_validated += 1
        if layout is LAYOUT_3P:
            self.observations_validated_3p += 1
        else:
            self.observations_validated_4p += 1
        self.total_candidates += metrics.candidate_count
        self.max_candidates = max(self.max_candidates, metrics.candidate_count)
        self.encode_sp_total_ns += metrics.encode_sp_ns
        self.encode_sp_max_ns = max(self.encode_sp_max_ns, metrics.encode_sp_ns)
        self.encode_extended_with_sp_total_ns += metrics.encode_extended_with_sp_ns
        self.encode_extended_with_sp_max_ns = max(
            self.encode_extended_with_sp_max_ns,
            metrics.encode_extended_with_sp_ns,
        )
        self.max_best_ev_100k = max(self.max_best_ev_100k, metrics.best_ev_100k)
        self.max_best_ev_30k = max(self.max_best_ev_30k, metrics.best_ev_30k)
        if metrics.riichi_assumed:
            self.closed_riichi_assumed += 1
        if not metrics.is_menzen:
            self.open_hand += 1
        if metrics.legal_discard_count == 0:
            self.no_legal_discard += 1

    def summary(self) -> dict[str, Any]:
        data = asdict(self)
        n = max(1, self.observations_validated)
        data["avg_candidates"] = self.total_candidates / n
        data["encode_sp_avg_ms"] = self.encode_sp_total_ns / n / 1_000_000.0
        data["encode_sp_max_ms"] = self.encode_sp_max_ns / 1_000_000.0
        data["encode_extended_with_sp_avg_ms"] = (
            self.encode_extended_with_sp_total_ns / n / 1_000_000.0
        )
        data["encode_extended_with_sp_max_ms"] = self.encode_extended_with_sp_max_ns / 1_000_000.0
        return data


def _idx(layout: FeatureLayout, channel: int, column: int) -> int:
    return channel * layout.tile_types + column


def _decode_float32(buf: bytes, expected: int, name: str) -> tuple[array.array | None, list[ValidationIssue]]:
    values = array.array("f")
    values.frombytes(buf)
    if sys.byteorder != "little":
        values.byteswap()
    if len(values) != expected:
        return None, [
            ValidationIssue(
                "shape",
                f"{name} has {len(values)} float32 values, expected {expected}",
                {"actual": len(values), "expected": expected},
            )
        ]
    return values, []


def _range_issue(
    layout: FeatureLayout,
    name: str,
    values: Sequence[float],
    start_channel: int,
    end_channel: int,
    lo: float,
    hi: float,
) -> ValidationIssue | None:
    start = start_channel * layout.tile_types
    end = end_channel * layout.tile_types
    subset = values[start:end]
    if not subset:
        return None
    amin = min(subset)
    amax = max(subset)
    if amin < lo - EPS or amax > hi + EPS:
        return ValidationIssue(
            "range",
            f"{name} has values outside [{lo}, {hi}]",
            {"min": float(amin), "max": float(amax)},
        )
    return None


def _binary_issue(
    layout: FeatureLayout,
    name: str,
    values: Sequence[float],
    start_channel: int,
    end_channel: int,
) -> ValidationIssue | None:
    start = start_channel * layout.tile_types
    end = end_channel * layout.tile_types
    for offset, value in enumerate(values[start:end]):
        if abs(value) > EPS and abs(value - 1.0) > EPS:
            flat_index = start + offset
            channel, tile = divmod(flat_index, layout.tile_types)
            return ValidationIssue(
                "binary",
                f"{name} contains non-binary values",
                {"channel": channel, "tile": tile, "value": float(value)},
            )
    return None


def _allclose(a: Sequence[float], b: Sequence[float]) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= EPS for x, y in zip(a, b, strict=True))


def _is_discard_action(action: Any) -> bool:
    try:
        return int(action.action_type) == int(ActionType.DISCARD)
    except Exception:
        return False


def legal_discard_types(obs: Observation | Observation3P) -> set[int]:
    types: set[int] = set()
    for action in obs.legal_actions():
        if _is_discard_action(action) and action.tile is not None:
            types.add(int(action.tile) // 4)
    return types


def observation_is_menzen(obs: Observation | Observation3P) -> bool:
    player = int(obs.player_id)
    melds = list(obs.melds[player])
    return not any(bool(getattr(meld, "opened", True)) for meld in melds)


def observation_riichi_assumed(obs: Observation | Observation3P) -> bool:
    player = int(obs.player_id)
    return observation_is_menzen(obs) and int(obs.scores[player]) >= 1000


def _channel_has_value(
    layout: FeatureLayout, values: Sequence[float], channel: int
) -> bool:
    start = channel * layout.tile_types
    end = start + layout.tile_types
    return any(abs(value) > EPS for value in values[start:end])


def _column_has_value(
    layout: FeatureLayout,
    values: Sequence[float],
    start_channel: int,
    end_channel: int,
    canonical_tile: int,
) -> bool:
    column = layout.canonical_to_compact.get(canonical_tile)
    return column is not None and any(
        abs(values[_idx(layout, channel, column)]) > EPS
        for channel in range(start_channel, end_channel)
    )


def candidate_discard_types(layout: FeatureLayout, sp: Sequence[float]) -> set[int]:
    candidates: set[int] = set()
    for tile in range(SP_CANDIDATE_TYPES):
        if _channel_has_value(layout, sp, 2 + tile):
            candidates.add(tile)
        if _channel_has_value(layout, sp, 2 + SP_CANDIDATE_TYPES + tile):
            candidates.add(tile)
        if _column_has_value(layout, sp, 72, 72 + 17, tile):
            candidates.add(tile)
        if _column_has_value(layout, sp, 72 + 17, 72 + 34, tile):
            candidates.add(tile)
        if _column_has_value(layout, sp, 72 + 34, 72 + 51, tile):
            candidates.add(tile)
        if _channel_has_value(layout, sp, 138 + tile):
            candidates.add(tile)
        if _column_has_value(layout, sp, 123, 138, tile):
            candidates.add(tile)
        if _column_has_value(layout, sp, 172, 178, tile):
            candidates.add(tile)

    for marker_channel in (70, 71):
        for column, tile in enumerate(layout.compact_to_canonical):
            if abs(sp[_idx(layout, marker_channel, column)]) > EPS:
                candidates.add(tile)
    return candidates


def _check_win_le_tenpai(
    layout: FeatureLayout, sp: Sequence[float], candidates: set[int]
) -> list[ValidationIssue]:
    """Per turn: win_prob[t] should not exceed tenpai_prob[t]+EPS.

    A hand cannot be in agari at turn t without being in tenpai at turn t.
    """
    issues: list[ValidationIssue] = []
    for tile in sorted(candidates):
        column = layout.canonical_to_compact.get(tile)
        if column is None:
            continue
        for turn in range(17):
            t = float(sp[_idx(layout, 72 + turn, column)])
            w = float(sp[_idx(layout, 89 + turn, column)])
            if w > t + EPS:
                issues.append(
                    ValidationIssue(
                        "win_gt_tenpai",
                        f"win_prob exceeds tenpai_prob for discard tile {tile}",
                        {"tile": tile, "turn": turn, "tenpai": t, "win": w},
                    )
                )
                break
    return issues


def _hand_counts_to_136(hand_counts: Sequence[int]) -> list[int]:
    """Convert per-type counts (length 34) to a 136-tile-id list usable with calculate_shanten."""
    tiles: list[int] = []
    for tile, count in enumerate(hand_counts):
        for k in range(count):
            tiles.append(tile * 4 + k)
    return tiles


def _check_shanten_consistency(
    layout: FeatureLayout,
    sp: Sequence[float],
    candidates: set[int],
    hand_counts: Sequence[int],
) -> list[ValidationIssue]:
    """For each candidate discard, the post-discard shanten lower-bounds the win turn.

    A hand at shanten s requires at least s+1 self-draws to reach agari, so
    win_prob[turn] must be ~0 for turn < s.  (turn here is 0-indexed = "after
    one draw", so s draws are needed to win which means win_prob[0..s-1] = 0.)
    """
    issues: list[ValidationIssue] = []
    for tile in sorted(candidates):
        column = layout.canonical_to_compact.get(tile)
        if column is None or tile >= len(hand_counts) or hand_counts[tile] == 0:
            continue
        post = list(hand_counts)
        post[tile] -= 1
        post_total = sum(post)
        if post_total % 3 != 1:
            # Open-meld observations may produce non-3n+1 hands; skip rather
            # than confuse calculate_shanten.
            continue
        try:
            shanten = int(calculate_shanten(_hand_counts_to_136(post)))
        except Exception:
            continue
        if shanten < 0:
            continue
        for turn in range(min(shanten, 17)):
            w = float(sp[_idx(layout, 89 + turn, column)])
            if w > EPS:
                issues.append(
                    ValidationIssue(
                        "shanten_win_too_early",
                        f"win_prob > 0 at turn {turn} for shanten-{shanten} state after discarding {tile}",
                        {
                            "tile": tile,
                            "turn": turn,
                            "post_discard_shanten": shanten,
                            "win": w,
                        },
                    )
                )
                break
        if shanten == 0:
            # tenpai_prob は horizon (残り巡目) 内では 1.0、horizon 外は 0 で埋まる。
            # そのため最初の 0 で停止する。
            for turn in range(17):
                tp = float(sp[_idx(layout, 72 + turn, column)])
                if tp < EPS:
                    break
                if tp < 1.0 - EPS:
                    issues.append(
                        ValidationIssue(
                            "tenpai_state_not_full",
                            f"tenpai_prob < 1 at turn {turn} for already-tenpai state after discarding {tile}",
                            {"tile": tile, "turn": turn, "tenpai": tp},
                        )
                    )
                    break
    return issues


def _check_required_drawability(
    layout: FeatureLayout,
    sp: Sequence[float],
    candidates: set[int],
    remaining: Sequence[int],
) -> list[ValidationIssue]:
    """Tiles flagged as required for a candidate must still be drawable from the wall."""
    issues: list[ValidationIssue] = []
    for d in sorted(candidates):
        for column, t in enumerate(layout.compact_to_canonical):
            if float(sp[_idx(layout, 2 + d, column)]) > EPS and remaining[t] == 0:
                issues.append(
                    ValidationIssue(
                        "required_unreachable",
                        f"required tile {t} for discard {d} has 0 remaining in the wall",
                        {"discard": d, "required_tile": t},
                    )
                )
                return issues
            if (
                float(
                    sp[
                        _idx(
                            layout,
                            2 + SP_CANDIDATE_TYPES + d,
                            column,
                        )
                    ]
                )
                > EPS
                and remaining[t] == 0
            ):
                issues.append(
                    ValidationIssue(
                        "yaku_progress_unreachable",
                        f"yaku-progress tile {t} for discard {d} has 0 remaining in the wall",
                        {"discard": d, "yaku_tile": t},
                    )
                )
                return issues
    return issues


def _check_best_marker_in_candidates(
    layout: FeatureLayout,
    sp: Sequence[float],
    non_marker_candidates: set[int],
) -> list[ValidationIssue]:
    """Channel-70/71 one-hot markers must point to a tile that the SP encoder
    independently chose as a candidate (i.e. that has a non-zero presence in the
    required-tile / yaku-progress / series columns, not merely in the marker itself)."""
    issues: list[ValidationIssue] = []
    if not non_marker_candidates:
        return issues
    for ch, name in ((70, "best_required"), (71, "best_yaku_progress")):
        for column, tile in enumerate(layout.compact_to_canonical):
            if (
                float(sp[_idx(layout, ch, column)]) > EPS
                and tile not in non_marker_candidates
            ):
                issues.append(
                    ValidationIssue(
                        "best_marker_outside_candidates",
                        f"{name} marker points at tile {tile} which is not a SP candidate",
                        {"channel": ch, "tile": tile},
                    )
                )
                break
    return issues


def _candidates_excluding_markers(
    layout: FeatureLayout, sp: Sequence[float]
) -> set[int]:
    """Candidate set determined only by required/yaku/series channels — markers excluded."""
    candidates: set[int] = set()
    for tile in range(SP_CANDIDATE_TYPES):
        if _channel_has_value(layout, sp, 2 + tile):
            candidates.add(tile)
        if _channel_has_value(layout, sp, 2 + SP_CANDIDATE_TYPES + tile):
            candidates.add(tile)
        if _column_has_value(layout, sp, 72, 72 + 17, tile):
            candidates.add(tile)
        if _column_has_value(layout, sp, 72 + 17, 72 + 34, tile):
            candidates.add(tile)
        if _column_has_value(layout, sp, 72 + 34, 72 + 51, tile):
            candidates.add(tile)
        if _channel_has_value(layout, sp, 138 + tile):
            candidates.add(tile)
        if _column_has_value(layout, sp, 123, 138, tile):
            candidates.add(tile)
        if _column_has_value(layout, sp, 172, 178, tile):
            candidates.add(tile)
    return candidates


def _compute_remaining_and_hand(
    obs: Observation | Observation3P, layout: FeatureLayout
) -> tuple[list[int], list[int]]:
    """Return (remaining[34], hand_counts[34]) computed the same way as SpInput::from_observation."""
    d = obs.to_dict()
    player = int(d["player_id"])
    tiles_seen = [0] * SP_CANDIDATE_TYPES
    hand_counts = [0] * SP_CANDIDATE_TYPES
    for tile in d["hands"][player]:
        ttype = int(tile) // 4
        if 0 <= ttype < SP_CANDIDATE_TYPES:
            tiles_seen[ttype] += 1
            hand_counts[ttype] += 1
    for player_melds in d["melds"]:
        for meld in player_melds:
            for tile in meld.tiles:
                ttype = int(tile) // 4
                if 0 <= ttype < SP_CANDIDATE_TYPES:
                    tiles_seen[ttype] += 1
    for player_discards in d["discards"]:
        for tile in player_discards:
            ttype = int(tile) // 4
            if 0 <= ttype < SP_CANDIDATE_TYPES:
                tiles_seen[ttype] += 1
    for tile in d["dora_indicators"]:
        ttype = int(tile) // 4
        if 0 <= ttype < SP_CANDIDATE_TYPES:
            tiles_seen[ttype] += 1
    for count in d.get("kita_counts", []):
        tiles_seen[30] += int(count)
    remaining = [max(0, 4 - min(4, c)) for c in tiles_seen]
    if layout is LAYOUT_3P:
        for tile in range(1, 8):
            remaining[tile] = 0
    return remaining, hand_counts


def _check_series_monotonic(
    layout: FeatureLayout, sp: Sequence[float], candidates: set[int]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for base, name in ((72, "tenpai_prob"), (89, "win_prob"), (106, "ev_ratio")):
        for tile in sorted(candidates):
            column = layout.canonical_to_compact.get(tile)
            if column is None:
                continue
            series = [sp[_idx(layout, base + turn, column)] for turn in range(17)]
            positive = [i for i, value in enumerate(series) if value > EPS]
            if not positive:
                continue
            active = series[: positive[-1] + 1]
            for idx, (prev, nxt) in enumerate(zip(active, active[1:], strict=False)):
                if nxt < prev - EPS:
                    issues.append(
                        ValidationIssue(
                            "monotonic",
                            f"{name} decreased for discard tile {tile}",
                            {
                                "tile": tile,
                                "turn": idx + 1,
                                "prev": float(prev),
                                "next": float(nxt),
                            },
                        )
                    )
                    break
    return issues


def _check_point_progress(
    layout: FeatureLayout, sp: Sequence[float], candidates: set[int]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for tile in sorted(candidates):
        column = layout.canonical_to_compact.get(tile)
        if column is None:
            continue
        minimum = float(sp[_idx(layout, 135, column)])
        mean = float(sp[_idx(layout, 136, column)])
        maximum = float(sp[_idx(layout, 137, column)])
        # Pre-tenpai candidates may expose only the rough mean-point estimate;
        # min/max are exact wait-set statistics and stay zero until available.
        if (minimum > EPS or maximum > EPS) and (
            minimum > mean + EPS or mean > maximum + EPS
        ):
            issues.append(
                ValidationIssue(
                    "point_order",
                    f"min/mean/max point progress is not ordered for discard tile {tile}",
                    {"tile": tile, "min": minimum, "mean": mean, "max": maximum},
                )
            )
        targets = [float(sp[_idx(layout, channel, column)]) for channel in range(172, 178)]
        if any(next_value > previous + EPS for previous, next_value in zip(targets, targets[1:])):
            issues.append(
                ValidationIssue(
                    "point_target_monotonic",
                    f"point target probabilities increase for discard tile {tile}",
                    {"tile": tile, "probabilities": targets},
                )
            )
    return issues


def validate_sp_arrays(  # noqa: PLR0915
    layout: FeatureLayout,
    sp: Sequence[float],
    *,
    legal_discards: set[int] | None = None,
    extended: Sequence[float] | None = None,
    drev: Sequence[float] | None = None,
    extended_with_sp: Sequence[float] | None = None,
    hand_counts: Sequence[int] | None = None,
    remaining: Sequence[int] | None = None,
) -> tuple[list[ValidationIssue], ObservationMetrics]:
    issues: list[ValidationIssue] = []
    metrics = ObservationMetrics()

    if len(sp) != layout.sp_floats:
        issues.append(
            ValidationIssue(
                "shape",
                "SP feature array has unexpected length",
                {"actual": len(sp), "expected": layout.sp_floats, "variant": layout.name},
            )
        )
        return issues, metrics

    if not all(math.isfinite(value) for value in sp):
        issues.append(ValidationIssue("finite", "SP feature array contains non-finite values"))

    for name, start, end, lo, hi in (
        ("max_ev_broadcast", 0, 2, 0.0, 1.0),
        ("required_tile_map", 2, 36, 0.0, 1.0),
        ("yaku_progress_map", 36, 70, 0.0, 1.0),
        ("best_markers", 70, 72, 0.0, 1.0),
        ("series", 72, 123, 0.0, 1.0),
        ("yaku_path_mask", 123, 135, 0.0, 1.0),
        ("point_progress", 135, 138, 0.0, 1.0),
        ("future_wait_map", 138, 172, 0.0, 1.0),
        ("point_target_probability", 172, 178, 0.0, 1.0),
    ):
        issue = _range_issue(layout, name, sp, start, end, lo, hi)
        if issue is not None:
            issues.append(issue)

    for name, start, end in (
        ("required_tile_map", 2, 36),
        ("yaku_progress_map", 36, 70),
        ("best_markers", 70, 72),
        ("yaku_path_mask", 123, 135),
        ("future_wait_map", 138, 172),
    ):
        issue = _binary_issue(layout, name, sp, start, end)
        if issue is not None:
            issues.append(issue)

    for channel in (0, 1):
        base = sp[_idx(layout, channel, 0)]
        if (
            max(
                abs(sp[_idx(layout, channel, column)] - base)
                for column in range(layout.tile_types)
            )
            > EPS
        ):
            issues.append(
                ValidationIssue(
                    "broadcast",
                    f"channel {channel} is not broadcast across all tile columns",
                    {"channel": channel},
                )
            )

    if (
        float(sp[_idx(layout, 1, 0)]) < 1.0 - EPS
        and float(sp[_idx(layout, 0, 0)]) < 1.0 - EPS
    ):
        ev_from_100k = float(sp[_idx(layout, 0, 0)]) * 100_000.0
        ev_from_30k = float(sp[_idx(layout, 1, 0)]) * 30_000.0
        if abs(ev_from_100k - ev_from_30k) > 2.0:
            issues.append(
                ValidationIssue(
                    "ev_scale",
                    "max EV broadcast channels disagree",
                    {"ev_from_100k": ev_from_100k, "ev_from_30k": ev_from_30k},
                )
            )

    candidates = candidate_discard_types(layout, sp)
    metrics.candidate_count = len(candidates)
    metrics.best_ev_100k = float(sp[_idx(layout, 0, 0)])
    metrics.best_ev_30k = float(sp[_idx(layout, 1, 0)])

    if legal_discards is not None:
        metrics.legal_discard_count = len(legal_discards)
        # Only enforce candidate-set ⊆ legal_discards at observations where the
        # player is actually choosing a discard. Otherwise (calls, draws, etc.)
        # the SP encoder still produces hypothetical-discard features that are
        # legitimately not in legal_actions.
        if legal_discards:
            extra = candidates - legal_discards
            if extra:
                issues.append(
                    ValidationIssue(
                        "illegal_candidate",
                        "SP has non-zero candidate features for non-discard legal actions",
                        {"tiles": sorted(extra), "legal_discards": sorted(legal_discards)},
                    )
                )

    if float(sp[_idx(layout, 0, 0)]) > EPS and candidates:
        first_turn_max = max(
            float(sp[_idx(layout, 106, column)])
            for column in range(layout.tile_types)
        )
        if first_turn_max < 1.0 - EPS:
            issues.append(
                ValidationIssue(
                    "ev_ratio",
                    "first-turn EV ratio does not identify a best discard with value 1.0",
                    {"max_first_turn_ev_ratio": first_turn_max},
                )
            )

    marker_count = sum(
        1
        for column in range(layout.tile_types)
        if abs(sp[_idx(layout, 70, column)]) > EPS
    )
    if marker_count > 1:
        issues.append(
            ValidationIssue("marker", "best required-tile marker is not one-hot", {"count": marker_count})
        )
    yaku_marker_count = sum(
        1
        for column in range(layout.tile_types)
        if abs(sp[_idx(layout, 71, column)]) > EPS
    )
    if yaku_marker_count > 1:
        issues.append(
            ValidationIssue("marker", "best yaku-progress marker is not one-hot", {"count": yaku_marker_count})
        )

    issues.extend(_check_series_monotonic(layout, sp, candidates))
    issues.extend(_check_win_le_tenpai(layout, sp, candidates))
    issues.extend(_check_point_progress(layout, sp, candidates))
    issues.extend(
        _check_best_marker_in_candidates(
            layout, sp, _candidates_excluding_markers(layout, sp)
        )
    )

    if remaining is not None:
        issues.extend(_check_required_drawability(layout, sp, candidates, remaining))
    if hand_counts is not None:
        issues.extend(_check_shanten_consistency(layout, sp, candidates, hand_counts))

    if drev is not None:
        if len(drev) != layout.drev_floats:
            issues.append(
                ValidationIssue(
                    "shape",
                    "DREV feature array has unexpected length",
                    {"actual": len(drev), "expected": layout.drev_floats},
                )
            )
        elif not all(math.isfinite(value) and -EPS <= value <= 1.0 + EPS for value in drev):
            issues.append(
                ValidationIssue(
                    "drev_range",
                    "DREV features must be finite values in [0, 1]",
                    {"min": float(min(drev)), "max": float(max(drev))},
                )
            )

    if extended is not None and extended_with_sp is not None:
        if len(extended) != layout.extended_floats:
            issues.append(
                ValidationIssue(
                    "shape",
                    "extended feature array has unexpected length",
                    {"actual": len(extended), "expected": layout.extended_floats},
                )
            )
        elif len(extended_with_sp) != layout.combined_floats:
            issues.append(
                ValidationIssue(
                    "shape",
                    "extended_with_sp feature array has unexpected length",
                    {"actual": len(extended_with_sp), "expected": layout.combined_floats},
                )
            )
        else:
            if not _allclose(extended_with_sp[: layout.extended_floats], extended):
                issues.append(
                    ValidationIssue("extended_prefix", "extended_with_sp prefix differs from encode_extended")
                )
            sp_end = layout.extended_floats + layout.sp_floats
            if not _allclose(extended_with_sp[layout.extended_floats:sp_end], sp):
                issues.append(
                    ValidationIssue(
                        "extended_tail",
                        "extended_with_sp SP slice differs from encode_sp",
                        {"block": "sp"},
                    )
                )
            drev_slice = extended_with_sp[sp_end:]
            if any(not math.isfinite(value) for value in drev_slice):
                issues.append(ValidationIssue("drev_finite", "extended_with_sp DREV tail contains non-finite values"))
            if drev is not None and not _allclose(drev_slice, drev):
                issues.append(
                    ValidationIssue(
                        "extended_tail",
                        "extended_with_sp DREV slice differs from encode_drev",
                        {"block": "drev"},
                    )
                )

    return issues, metrics


def validate_sp_observation(
    obs: Observation | Observation3P,
    *,
    check_extended: bool = True,
) -> tuple[list[ValidationIssue], ObservationMetrics]:
    layout = LAYOUT_3P if isinstance(obs, Observation3P) else LAYOUT_4P
    start = time.perf_counter_ns()
    sp_buf = obs.encode_sp()
    sp_ns = time.perf_counter_ns() - start

    sp, issues = _decode_float32(sp_buf, layout.sp_floats, "encode_sp")
    metrics = ObservationMetrics(encode_sp_ns=sp_ns)
    metrics.is_menzen = observation_is_menzen(obs)
    metrics.riichi_assumed = observation_riichi_assumed(obs)
    if sp is None:
        return issues, metrics

    extended = None
    drev = None
    extended_with_sp = None
    if check_extended:
        extended, ext_issues = _decode_float32(
            obs.encode_extended(), layout.extended_floats, "encode_extended"
        )
        issues.extend(ext_issues)
        drev, drev_issues = _decode_float32(
            obs.encode_drev(), layout.drev_floats, "encode_drev"
        )
        issues.extend(drev_issues)

        start = time.perf_counter_ns()
        ext_sp_buf = obs.encode_extended_with_sp()
        ext_sp_ns = time.perf_counter_ns() - start
        metrics.encode_extended_with_sp_ns = ext_sp_ns

        extended_with_sp, ext_sp_issues = _decode_float32(
            ext_sp_buf,
            layout.combined_floats,
            "encode_extended_with_sp",
        )
        issues.extend(ext_sp_issues)

    remaining, hand_counts = _compute_remaining_and_hand(obs, layout)
    array_issues, array_metrics = validate_sp_arrays(
        layout,
        sp,
        legal_discards=legal_discard_types(obs),
        extended=extended,
        drev=drev,
        extended_with_sp=extended_with_sp,
        hand_counts=hand_counts,
        remaining=remaining,
    )
    issues.extend(array_issues)
    metrics.legal_discard_count = array_metrics.legal_discard_count
    metrics.candidate_count = array_metrics.candidate_count
    metrics.best_ev_100k = array_metrics.best_ev_100k
    metrics.best_ev_30k = array_metrics.best_ev_30k
    return issues, metrics


def iter_log_files(paths: list[Path], patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
            continue
        if path.is_dir():
            for pattern in patterns:
                files.extend(sorted(path.rglob(pattern)))
            continue
        print(f"Warning: {path} not found, skipping")
    return sorted(dict.fromkeys(files))


def _write_issues(
    diagnostics,
    *,
    ctx: ValidationContext,
    issues: list[ValidationIssue],
    metrics: ObservationMetrics,
) -> None:
    if diagnostics is None:
        return
    for issue in issues:
        diagnostics.write(
            json.dumps(
                {
                    **ctx.as_dict(),
                    "issue": asdict(issue),
                    "metrics": asdict(metrics),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def validate_log_file(
    path: Path,
    *,
    rule: str,
    seat: int | None,
    check_extended: bool,
    sample_every: int,
    max_observations: int,
    max_issues: int,
    stats: ValidationStats,
    diagnostics,
) -> bool:
    replay = MjaiReplay.from_jsonl(str(path), rule=rule)
    stats.files_seen += 1
    validated_any = False

    for kyoku_idx, kyoku in enumerate(replay.take_kyokus()):
        stats.kyokus_seen += 1
        for step_idx, step in enumerate(kyoku.steps(seat=seat, skip_single_action=False)):
            if seat is None:
                player, obs, _action = step
            else:
                obs, _action = step
                player = seat

            stats.observations_seen += 1
            if sample_every > 1 and (stats.observations_seen - 1) % sample_every != 0:
                continue
            if max_observations > 0 and stats.observations_validated >= max_observations:
                return validated_any

            if not isinstance(obs, (Observation, Observation3P)):
                stats.observations_with_issues += 1
                stats.issues += 1
                ctx = ValidationContext(str(path), kyoku_idx, step_idx, int(player))
                issue = ValidationIssue(
                    "observation_type",
                    "expected Observation or Observation3P",
                    {"type": type(obs).__name__},
                )
                _write_issues(diagnostics, ctx=ctx, issues=[issue], metrics=ObservationMetrics())
                continue

            ctx = ValidationContext(str(path), kyoku_idx, step_idx, int(player))
            issues, metrics = validate_sp_observation(obs, check_extended=check_extended)
            layout = LAYOUT_3P if isinstance(obs, Observation3P) else LAYOUT_4P
            stats.add_metrics(metrics, layout)
            validated_any = True

            if issues:
                stats.observations_with_issues += 1
                stats.issues += len(issues)
                _write_issues(diagnostics, ctx=ctx, issues=issues, metrics=metrics)
                if max_issues > 0 and stats.issues >= max_issues:
                    return validated_any

    return validated_any


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SP feature encodings on MJAI jsonl/jsonl.gz logs")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["data-mjsoul-4p-2026-01", "data-mjsoul-3p-2026-01"],
        help="MJAI log files or directories. Default: bundled 4P and 3P MjSoul corpus paths",
    )
    parser.add_argument("--rule", default="mjsoul", choices=["tenhou", "mjsoul"], help="Replay rule")
    parser.add_argument("--glob", action="append", default=[], help="Recursive glob pattern. Can be repeated")
    parser.add_argument("--max-files", type=int, default=0, help="Max files to validate (0=all)")
    parser.add_argument("--max-observations", type=int, default=0, help="Max observations to validate (0=all)")
    parser.add_argument("--max-issues", type=int, default=20, help="Stop after this many issues (0=no limit)")
    parser.add_argument("--sample-every", type=int, default=1, help="Validate every Nth observation")
    parser.add_argument("--seat", type=int, default=None, choices=[0, 1, 2, 3], help="Restrict validation to one seat")
    parser.add_argument("--no-extended", action="store_true", help="Skip encode_extended_with_sp consistency checks")
    parser.add_argument("--diagnostics-out", type=Path, default=None, help="Write issue diagnostics as JSONL")
    args = parser.parse_args()

    patterns = args.glob or ["*.jsonl.gz", "*.jsonl"]
    paths = [Path(p) for p in args.paths]
    log_files = iter_log_files(paths, patterns)
    if args.max_files > 0:
        log_files = log_files[: args.max_files]

    if not log_files:
        raise SystemExit("No MJAI jsonl/jsonl.gz files found")
    if args.sample_every < 1:
        raise SystemExit("--sample-every must be >= 1")

    stats = ValidationStats()
    diagnostics = None
    if args.diagnostics_out is not None:
        args.diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
        diagnostics = args.diagnostics_out.open("w", encoding="utf-8")

    try:
        for index, path in enumerate(log_files, start=1):
            print(f"[{index}/{len(log_files)}] {path}")
            try:
                validated = validate_log_file(
                    path,
                    rule=args.rule,
                    seat=args.seat,
                    check_extended=not args.no_extended,
                    sample_every=args.sample_every,
                    max_observations=args.max_observations,
                    max_issues=args.max_issues,
                    stats=stats,
                    diagnostics=diagnostics,
                )
                if validated:
                    stats.files_validated += 1
            except Exception as exc:
                stats.issues += 1
                stats.observations_with_issues += 1
                if diagnostics is not None:
                    diagnostics.write(
                        json.dumps(
                            {
                                "file": str(path),
                                "issue": {
                                    "code": "exception",
                                    "message": str(exc),
                                    "details": {"type": type(exc).__name__},
                                },
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                else:
                    print(f"  FAILED: {type(exc).__name__}: {exc}")

            if args.max_observations > 0 and stats.observations_validated >= args.max_observations:
                break
            if args.max_issues > 0 and stats.issues >= args.max_issues:
                break
    finally:
        if diagnostics is not None:
            diagnostics.close()

    print(json.dumps(stats.summary(), ensure_ascii=False, indent=2))
    if stats.issues > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
