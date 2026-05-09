from __future__ import annotations

import gzip
import importlib.util
import sys
from pathlib import Path


def _load_validator():
    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_sp_features.py"
    spec = importlib.util.spec_from_file_location("validate_sp_features", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sp_validation_accepts_gzipped_mjai_sample(tmp_path):
    validator = _load_validator()
    source = Path(__file__).resolve().parent / "data" / "126_204_0_mjai.jsonl"
    gz_path = tmp_path / "sample.jsonl.gz"
    with source.open("rt", encoding="utf-8") as src, gzip.open(gz_path, "wt", encoding="utf-8") as dst:
        dst.write(src.read())

    stats = validator.ValidationStats()
    validated = validator.validate_log_file(
        gz_path,
        rule="tenhou",
        seat=0,
        check_extended=True,
        sample_every=1,
        max_observations=3,
        max_issues=0,
        stats=stats,
        diagnostics=None,
    )

    assert validated
    assert stats.observations_validated == 3
    assert stats.issues == 0
    assert stats.encode_sp_total_ns > 0


def test_sp_validation_detects_extended_tail_mismatch():
    validator = _load_validator()
    sp = [0.0] * validator.SP_FLOATS
    extended = [0.0] * validator.EXTENDED_FLOATS
    extended_with_sp = [0.0] * validator.EXTENDED_WITH_SP_FLOATS
    extended_with_sp[validator.EXTENDED_FLOATS] = 1.0

    issues, _metrics = validator.validate_sp_arrays(
        sp,
        legal_discards=set(),
        extended=extended,
        extended_with_sp=extended_with_sp,
    )

    assert "extended_tail" in {issue.code for issue in issues}


def test_sp_validation_detects_nonmonotonic_probability_series():
    validator = _load_validator()
    sp = [0.0] * validator.SP_FLOATS
    for tile in range(validator.TILE_TYPES):
        sp[tile] = 0.01
        sp[validator.TILE_TYPES + tile] = 1000.0 / 30_000.0
    sp[72 * validator.TILE_TYPES] = 0.8
    sp[73 * validator.TILE_TYPES] = 0.7
    sp[106 * validator.TILE_TYPES] = 1.0

    issues, metrics = validator.validate_sp_arrays(sp, legal_discards={0})

    assert metrics.candidate_count == 1
    assert "monotonic" in {issue.code for issue in issues}


def test_sp_validation_detects_candidate_outside_legal_discards():
    validator = _load_validator()
    sp = [0.0] * validator.SP_FLOATS
    sp[72 * validator.TILE_TYPES + 5] = 0.25

    issues, _metrics = validator.validate_sp_arrays(sp, legal_discards={1})

    assert "illegal_candidate" in {issue.code for issue in issues}


def test_sp_validation_does_not_flag_illegal_candidate_when_no_discards_legal():
    """Non-discard observations (calls, draws) leave legal_discards empty but the SP
    encoder still produces hypothetical-discard features; that should not be flagged."""
    validator = _load_validator()
    sp = [0.0] * validator.SP_FLOATS
    sp[72 * validator.TILE_TYPES + 5] = 0.5
    sp[(72 + 1) * validator.TILE_TYPES + 5] = 0.6

    issues, _metrics = validator.validate_sp_arrays(sp, legal_discards=set())

    assert "illegal_candidate" not in {issue.code for issue in issues}


def test_sp_validation_detects_win_exceeding_tenpai():
    validator = _load_validator()
    sp = [0.0] * validator.SP_FLOATS
    # Make tile 5 a candidate via the required-tile column.
    sp[(2 + 5) * validator.TILE_TYPES + 11] = 1.0
    # tenpai_prob[turn=0] = 0.3, win_prob[turn=0] = 0.6 (impossible).
    sp[72 * validator.TILE_TYPES + 5] = 0.3
    sp[89 * validator.TILE_TYPES + 5] = 0.6

    issues, _metrics = validator.validate_sp_arrays(sp, legal_discards={5})

    assert "win_gt_tenpai" in {issue.code for issue in issues}


def test_sp_validation_detects_required_tile_not_in_wall():
    validator = _load_validator()
    sp = [0.0] * validator.SP_FLOATS
    # Discard tile 5: required tile 11 marked, even though 11 is fully visible.
    sp[(2 + 5) * validator.TILE_TYPES + 11] = 1.0
    remaining = [4] * validator.TILE_TYPES
    remaining[11] = 0  # all 4 copies already visible

    issues, _metrics = validator.validate_sp_arrays(
        sp, legal_discards={5}, remaining=remaining
    )

    assert "required_unreachable" in {issue.code for issue in issues}


def test_sp_validation_detects_shanten_winning_too_early():
    validator = _load_validator()
    sp = [0.0] * validator.SP_FLOATS
    # Discard tile 0: post-discard hand is 2-shanten (build a real 14-tile hand).
    # Tehai with 14 tiles where dropping tile 0 leaves a 2-shanten hand:
    # 0-tile + a clearly 2-shanten 13-tile residual. Use a non-pair 13-tile junk hand.
    # 14 tiles: 0,2,5,8,11,14,17,20,23,26,29,30,31,32 → discard 0 → 13 disjoint tiles, ~6-shanten.
    # But a 6-shanten state still satisfies "win at turn 0 must be 0".
    hand_counts = [0] * validator.TILE_TYPES
    for t in (0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 30, 31, 32):
        hand_counts[t] += 1
    # Mark tile 0 as a candidate via required-tile column.
    sp[(2 + 0) * validator.TILE_TYPES + 11] = 1.0
    # Inject an impossibly early win.
    sp[89 * validator.TILE_TYPES + 0] = 0.5
    sp[72 * validator.TILE_TYPES + 0] = 0.6  # avoid tripping win_gt_tenpai

    issues, _metrics = validator.validate_sp_arrays(
        sp, legal_discards={0}, hand_counts=hand_counts
    )

    assert "shanten_win_too_early" in {issue.code for issue in issues}


def test_sp_validation_detects_best_marker_outside_candidates():
    validator = _load_validator()
    sp = [0.0] * validator.SP_FLOATS
    # Tile 5 is a candidate, but the best-required marker points at tile 12.
    sp[(2 + 5) * validator.TILE_TYPES + 11] = 1.0
    sp[70 * validator.TILE_TYPES + 12] = 1.0

    issues, _metrics = validator.validate_sp_arrays(sp, legal_discards={5})

    assert "best_marker_outside_candidates" in {issue.code for issue in issues}
