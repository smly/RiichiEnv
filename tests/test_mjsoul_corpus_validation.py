"""Mutation tests ensure the corpus checker detects independent oracle errors."""

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("corpus_validator", ROOT / "scripts/validate_mjsoul_corpus.py")
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


@pytest.fixture
def record():
    return json.loads((ROOT / "tests/data/mjsoul_corpus_validation.json").read_text())


def validate(record):
    validator.validate_round(record["events"], record["next_start"], record["mode"], Counter())


def test_raw_record_matches_simulator(record):
    validate(record)


def test_heavenly_hand_allows_an_alternative_named_win_tile():
    record = json.loads((ROOT / "tests/data/mjsoul_heavenly_hand.json").read_text())
    validate(record)
    record["events"][-1]["data"]["hules"][0]["hand"][0] = "7z"
    with pytest.raises(validator.ReplayMismatchError, match="winning_hand"):
        validate(record)


def test_incorrect_score_is_detected(record):
    record["events"][-1]["data"]["scores"][0] += 100
    with pytest.raises(validator.ReplayMismatchError, match="end_scores"):
        validate(record)


def test_incorrect_draw_is_detected(record):
    draw = next(e["data"] for e in record["events"] if e["name"] == "DealTile")
    draw["tile"] = "1z" if draw["tile"] != "1z" else "2z"
    with pytest.raises(validator.ReplayMismatchError, match="drawn_tile"):
        validate(record)


def test_incorrect_next_round_is_detected(record):
    record["next_start"]["ben"] += 1
    with pytest.raises(validator.ReplayMismatchError, match="next_round"):
        validate(record)


def test_premature_match_end_is_detected(record):
    record["next_start"] = None
    with pytest.raises(validator.ReplayMismatchError, match="match_finished"):
        validate(record)


def test_red_tile_mismatch_is_detected(record):
    nr = record["events"][0]["data"]
    for seat in range(4):
        hand = nr[f"tiles{seat}"]
        for index, tile in enumerate(hand):
            if tile in ("5m", "5p", "5s"):
                hand[index] = "0" + tile[1]
                with pytest.raises(validator.ReplayMismatchError, match="initial_hand"):
                    validate(record)
                return
    pytest.fail("Fixture must contain a normal five")


def test_empty_exhaustive_draw_deltas_mean_no_payment():
    events = [{"name": "NoTile", "data": {"scores": [{"old_scores": [25000] * 4, "delta_scores": []}]}}]
    assert validator.expected_end_scores(events) == [25000] * 4
