"""Validate SP feature encodings on MJAI jsonl/jsonl.gz logs.

The validator is intended for large MjSoul dumps such as
``data-mjsoul-4p-2026-01``. It checks structural invariants that should hold
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
        SP_CHANNELS,
        TILE_TYPES,
    )
    from riichienv._riichienv import (
        OBS_EXTENDED_CHANNELS as EXTENDED_CHANNELS,
    )
except ImportError:
    TILE_TYPES = 34
    EXTENDED_CHANNELS = 215
    SP_CHANNELS = 178
    DREV_CHANNELS = 9

EXTENDED_WITH_SP_CHANNELS = EXTENDED_CHANNELS + SP_CHANNELS + DREV_CHANNELS

SP_FLOATS = SP_CHANNELS * TILE_TYPES
DREV_FLOATS = DREV_CHANNELS * TILE_TYPES
EXTENDED_FLOATS = EXTENDED_CHANNELS * TILE_TYPES
EXTENDED_WITH_SP_FLOATS = EXTENDED_WITH_SP_CHANNELS * TILE_TYPES

EPS = 1e-5


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
    observations_skipped_3p: int = 0
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

    def add_metrics(self, metrics: ObservationMetrics) -> None:
        self.observations_validated += 1
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


def _idx(channel: int, tile: int) -> int:
    return channel * TILE_TYPES + tile


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
    name: str,
    values: Sequence[float],
    start_channel: int,
    end_channel: int,
    lo: float,
    hi: float,
) -> ValidationIssue | None:
    start = start_channel * TILE_TYPES
    end = end_channel * TILE_TYPES
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
    name: str,
    values: Sequence[float],
    start_channel: int,
    end_channel: int,
) -> ValidationIssue | None:
    start = start_channel * TILE_TYPES
    end = end_channel * TILE_TYPES
    for offset, value in enumerate(values[start:end]):
        if abs(value) > EPS and abs(value - 1.0) > EPS:
            flat_index = start + offset
            channel, tile = divmod(flat_index, TILE_TYPES)
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


def legal_discard_types(obs: Observation) -> set[int]:
    types: set[int] = set()
    for action in obs.legal_actions():
        if _is_discard_action(action) and action.tile is not None:
            types.add(int(action.tile) // 4)
    return types


def observation_is_menzen(obs: Observation) -> bool:
    player = int(obs.player_id)
    melds = list(obs.melds[player])
    return not any(bool(getattr(meld, "opened", True)) for meld in melds)


def observation_riichi_assumed(obs: Observation) -> bool:
    player = int(obs.player_id)
    return observation_is_menzen(obs) and int(obs.scores[player]) >= 1000


def _channel_has_value(values: Sequence[float], channel: int) -> bool:
    start = channel * TILE_TYPES
    end = start + TILE_TYPES
    return any(abs(value) > EPS for value in values[start:end])


def _column_has_value(values: Sequence[float], start_channel: int, end_channel: int, tile: int) -> bool:
    return any(abs(values[_idx(channel, tile)]) > EPS for channel in range(start_channel, end_channel))


def candidate_discard_types(sp: Sequence[float]) -> set[int]:
    candidates: set[int] = set()
    for tile in range(TILE_TYPES):
        if _channel_has_value(sp, 2 + tile):
            candidates.add(tile)
        if _channel_has_value(sp, 2 + TILE_TYPES + tile):
            candidates.add(tile)
        if _column_has_value(sp, 72, 72 + 17, tile):
            candidates.add(tile)
        if _column_has_value(sp, 72 + 17, 72 + 34, tile):
            candidates.add(tile)
        if _column_has_value(sp, 72 + 34, 72 + 51, tile):
            candidates.add(tile)

    for marker_channel in (70, 71):
        for tile in range(TILE_TYPES):
            if abs(sp[_idx(marker_channel, tile)]) > EPS:
                candidates.add(tile)
    return candidates


def _check_win_le_tenpai(
    sp: Sequence[float], candidates: set[int]
) -> list[ValidationIssue]:
    """Per turn: win_prob[t] should not exceed tenpai_prob[t]+EPS.

    A hand cannot be in agari at turn t without being in tenpai at turn t.
    """
    issues: list[ValidationIssue] = []
    for tile in sorted(candidates):
        for turn in range(17):
            t = float(sp[_idx(72 + turn, tile)])
            w = float(sp[_idx(89 + turn, tile)])
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
        if tile >= TILE_TYPES or hand_counts[tile] == 0:
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
            w = float(sp[_idx(89 + turn, tile)])
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
                tp = float(sp[_idx(72 + turn, tile)])
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
    sp: Sequence[float],
    candidates: set[int],
    remaining: Sequence[int],
) -> list[ValidationIssue]:
    """Tiles flagged as required for a candidate must still be drawable from the wall."""
    issues: list[ValidationIssue] = []
    for d in sorted(candidates):
        for t in range(TILE_TYPES):
            if float(sp[_idx(2 + d, t)]) > EPS and remaining[t] == 0:
                issues.append(
                    ValidationIssue(
                        "required_unreachable",
                        f"required tile {t} for discard {d} has 0 remaining in the wall",
                        {"discard": d, "required_tile": t},
                    )
                )
                return issues
            if float(sp[_idx(2 + TILE_TYPES + d, t)]) > EPS and remaining[t] == 0:
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
    sp: Sequence[float], non_marker_candidates: set[int]
) -> list[ValidationIssue]:
    """Channel-70/71 one-hot markers must point to a tile that the SP encoder
    independently chose as a candidate (i.e. that has a non-zero presence in the
    required-tile / yaku-progress / series columns, not merely in the marker itself)."""
    issues: list[ValidationIssue] = []
    if not non_marker_candidates:
        return issues
    for ch, name in ((70, "best_required"), (71, "best_yaku_progress")):
        for tile in range(TILE_TYPES):
            if (
                float(sp[_idx(ch, tile)]) > EPS
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


def _candidates_excluding_markers(sp: Sequence[float]) -> set[int]:
    """Candidate set determined only by required/yaku/series channels — markers excluded."""
    candidates: set[int] = set()
    for tile in range(TILE_TYPES):
        if _channel_has_value(sp, 2 + tile):
            candidates.add(tile)
        if _channel_has_value(sp, 2 + TILE_TYPES + tile):
            candidates.add(tile)
        if _column_has_value(sp, 72, 72 + 17, tile):
            candidates.add(tile)
        if _column_has_value(sp, 72 + 17, 72 + 34, tile):
            candidates.add(tile)
        if _column_has_value(sp, 72 + 34, 72 + 51, tile):
            candidates.add(tile)
    return candidates


def _compute_remaining_and_hand(obs: Observation) -> tuple[list[int], list[int]]:
    """Return (remaining[34], hand_counts[34]) computed the same way as SpInput::from_observation."""
    d = obs.to_dict()
    player = int(d["player_id"])
    tiles_seen = [0] * TILE_TYPES
    hand_counts = [0] * TILE_TYPES
    for tile in d["hands"][player]:
        ttype = int(tile) // 4
        if 0 <= ttype < TILE_TYPES:
            tiles_seen[ttype] += 1
            hand_counts[ttype] += 1
    for player_melds in d["melds"]:
        for meld in player_melds:
            for tile in meld.tiles:
                ttype = int(tile) // 4
                if 0 <= ttype < TILE_TYPES:
                    tiles_seen[ttype] += 1
    for player_discards in d["discards"]:
        for tile in player_discards:
            ttype = int(tile) // 4
            if 0 <= ttype < TILE_TYPES:
                tiles_seen[ttype] += 1
    for tile in d["dora_indicators"]:
        ttype = int(tile) // 4
        if 0 <= ttype < TILE_TYPES:
            tiles_seen[ttype] += 1
    remaining = [max(0, 4 - min(4, c)) for c in tiles_seen]
    return remaining, hand_counts


def _check_series_monotonic(sp: Sequence[float], candidates: set[int]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for base, name in ((72, "tenpai_prob"), (89, "win_prob"), (106, "ev_ratio")):
        for tile in sorted(candidates):
            series = [sp[_idx(base + turn, tile)] for turn in range(17)]
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


def validate_sp_arrays(  # noqa: PLR0915
    sp: Sequence[float],
    *,
    legal_discards: set[int] | None = None,
    extended: Sequence[float] | None = None,
    extended_with_sp: Sequence[float] | None = None,
    hand_counts: Sequence[int] | None = None,
    remaining: Sequence[int] | None = None,
) -> tuple[list[ValidationIssue], ObservationMetrics]:
    issues: list[ValidationIssue] = []
    metrics = ObservationMetrics()

    if len(sp) != SP_FLOATS:
        issues.append(
            ValidationIssue(
                "shape",
                "SP feature array has unexpected length",
                {"actual": len(sp), "expected": SP_FLOATS},
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
    ):
        issue = _range_issue(name, sp, start, end, lo, hi)
        if issue is not None:
            issues.append(issue)

    for name, start, end in (
        ("required_tile_map", 2, 36),
        ("yaku_progress_map", 36, 70),
        ("best_markers", 70, 72),
    ):
        issue = _binary_issue(name, sp, start, end)
        if issue is not None:
            issues.append(issue)

    for channel in (0, 1):
        base = sp[_idx(channel, 0)]
        if max(abs(sp[_idx(channel, tile)] - base) for tile in range(TILE_TYPES)) > EPS:
            issues.append(
                ValidationIssue(
                    "broadcast",
                    f"channel {channel} is not broadcast across all tile columns",
                    {"channel": channel},
                )
            )

    if float(sp[_idx(1, 0)]) < 1.0 - EPS and float(sp[_idx(0, 0)]) < 1.0 - EPS:
        ev_from_100k = float(sp[_idx(0, 0)]) * 100_000.0
        ev_from_30k = float(sp[_idx(1, 0)]) * 30_000.0
        if abs(ev_from_100k - ev_from_30k) > 2.0:
            issues.append(
                ValidationIssue(
                    "ev_scale",
                    "max EV broadcast channels disagree",
                    {"ev_from_100k": ev_from_100k, "ev_from_30k": ev_from_30k},
                )
            )

    candidates = candidate_discard_types(sp)
    metrics.candidate_count = len(candidates)
    metrics.best_ev_100k = float(sp[_idx(0, 0)])
    metrics.best_ev_30k = float(sp[_idx(1, 0)])

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

    if float(sp[_idx(0, 0)]) > EPS and candidates:
        first_turn_max = max(float(sp[_idx(106, tile)]) for tile in range(TILE_TYPES))
        if first_turn_max < 1.0 - EPS:
            issues.append(
                ValidationIssue(
                    "ev_ratio",
                    "first-turn EV ratio does not identify a best discard with value 1.0",
                    {"max_first_turn_ev_ratio": first_turn_max},
                )
            )

    marker_count = sum(1 for tile in range(TILE_TYPES) if abs(sp[_idx(70, tile)]) > EPS)
    if marker_count > 1:
        issues.append(
            ValidationIssue("marker", "best required-tile marker is not one-hot", {"count": marker_count})
        )
    yaku_marker_count = sum(1 for tile in range(TILE_TYPES) if abs(sp[_idx(71, tile)]) > EPS)
    if yaku_marker_count > 1:
        issues.append(
            ValidationIssue("marker", "best yaku-progress marker is not one-hot", {"count": yaku_marker_count})
        )

    issues.extend(_check_series_monotonic(sp, candidates))
    issues.extend(_check_win_le_tenpai(sp, candidates))
    issues.extend(
        _check_best_marker_in_candidates(sp, _candidates_excluding_markers(sp))
    )

    if remaining is not None:
        issues.extend(_check_required_drawability(sp, candidates, remaining))
    if hand_counts is not None:
        issues.extend(_check_shanten_consistency(sp, candidates, hand_counts))

    if extended is not None and extended_with_sp is not None:
        if len(extended) != EXTENDED_FLOATS:
            issues.append(
                ValidationIssue(
                    "shape",
                    "extended feature array has unexpected length",
                    {"actual": len(extended), "expected": EXTENDED_FLOATS},
                )
            )
        elif len(extended_with_sp) != EXTENDED_WITH_SP_FLOATS:
            issues.append(
                ValidationIssue(
                    "shape",
                    "extended_with_sp feature array has unexpected length",
                    {"actual": len(extended_with_sp), "expected": EXTENDED_WITH_SP_FLOATS},
                )
            )
        else:
            if not _allclose(extended_with_sp[:EXTENDED_FLOATS], extended):
                issues.append(
                    ValidationIssue("extended_prefix", "extended_with_sp prefix differs from encode_extended")
                )
            sp_end = EXTENDED_FLOATS + SP_FLOATS
            if not _allclose(extended_with_sp[EXTENDED_FLOATS:sp_end], sp):
                issues.append(
                    ValidationIssue(
                        "extended_tail",
                        "extended_with_sp SP slice differs from encode_sp",
                        {"block": "sp"},
                    )
                )
            # The trailing DREV block should be a deterministic function of the
            # observation; we don't validate its shape here (no standalone
            # encode_drev() comparison) but do check it's finite.
            drev_slice = extended_with_sp[sp_end:]
            if any(not math.isfinite(value) for value in drev_slice):
                issues.append(ValidationIssue("drev_finite", "extended_with_sp DREV tail contains non-finite values"))

    return issues, metrics


def validate_sp_observation(
    obs: Observation,
    *,
    check_extended: bool = True,
) -> tuple[list[ValidationIssue], ObservationMetrics]:
    start = time.perf_counter_ns()
    sp_buf = obs.encode_sp()
    sp_ns = time.perf_counter_ns() - start

    sp, issues = _decode_float32(sp_buf, SP_FLOATS, "encode_sp")
    metrics = ObservationMetrics(encode_sp_ns=sp_ns)
    metrics.is_menzen = observation_is_menzen(obs)
    metrics.riichi_assumed = observation_riichi_assumed(obs)
    if sp is None:
        return issues, metrics

    extended = None
    extended_with_sp = None
    if check_extended:
        extended, ext_issues = _decode_float32(obs.encode_extended(), EXTENDED_FLOATS, "encode_extended")
        issues.extend(ext_issues)

        start = time.perf_counter_ns()
        ext_sp_buf = obs.encode_extended_with_sp()
        ext_sp_ns = time.perf_counter_ns() - start
        metrics.encode_extended_with_sp_ns = ext_sp_ns

        extended_with_sp, ext_sp_issues = _decode_float32(
            ext_sp_buf,
            EXTENDED_WITH_SP_FLOATS,
            "encode_extended_with_sp",
        )
        issues.extend(ext_sp_issues)

    remaining, hand_counts = _compute_remaining_and_hand(obs)
    array_issues, array_metrics = validate_sp_arrays(
        sp,
        legal_discards=legal_discard_types(obs),
        extended=extended,
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

            if isinstance(obs, Observation3P):
                stats.observations_skipped_3p += 1
                continue
            if not isinstance(obs, Observation):
                stats.observations_with_issues += 1
                stats.issues += 1
                ctx = ValidationContext(str(path), kyoku_idx, step_idx, int(player))
                issue = ValidationIssue(
                    "observation_type",
                    "expected 4-player Observation",
                    {"type": type(obs).__name__},
                )
                _write_issues(diagnostics, ctx=ctx, issues=[issue], metrics=ObservationMetrics())
                continue

            ctx = ValidationContext(str(path), kyoku_idx, step_idx, int(player))
            issues, metrics = validate_sp_observation(obs, check_extended=check_extended)
            stats.add_metrics(metrics)
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
        default=["data-mjsoul-4p-2026-01"],
        help="MJAI log files or directories. Default: data-mjsoul-4p-2026-01",
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
