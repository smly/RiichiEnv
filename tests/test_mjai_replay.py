import gzip
import json
import os

import pytest

from riichienv import MjaiReplay


@pytest.fixture
def sample_mjai_data():
    return [
        {"type": "start_game", "names": ["A", "B", "C", "D"], "id": "test_game"},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyoutaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": [
                ["1s", "1s", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "9s", "9s"],
                ["1s", "1s", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "9s", "9s"],
                ["1s", "1s", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "9s", "9s"],
                ["1s", "1s", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "9s", "9s"],
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "2m"},
        {"type": "dahai", "actor": 0, "pai": "2m", "tsumogiri": True},
        {"type": "ryukyoku", "reason": "test"},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]


def test_mjai_replay_jsonl_plain(tmp_path, sample_mjai_data):
    file_path = tmp_path / "test.jsonl"
    with open(file_path, "w") as f:
        for event in sample_mjai_data:
            f.write(json.dumps(event) + "\n")

    replay = MjaiReplay.from_jsonl(str(file_path))
    assert replay.num_rounds() == 1

    kyokus = replay.take_kyokus()
    kyoku_list = list(kyokus)
    assert len(kyoku_list) == 1

    kyoku = kyoku_list[0]
    events = kyoku.events()
    assert len(events) == 4
    assert events[0]["name"] == "NewRound"

    features = kyoku.grp_features()
    for key in ["chang", "ju", "ben", "liqibang", "scores", "end_scores", "delta_scores", "wliqi"]:
        assert key in features

    assert features["scores"] == [25000, 25000, 25000, 25000]
    assert features["end_scores"] == [25000, 25000, 25000, 25000]
    assert features["delta_scores"] == [0, 0, 0, 0]


def test_mjai_replay_jsonl_gzip(tmp_path, sample_mjai_data):
    file_path = tmp_path / "test.jsonl.gz"
    with gzip.open(file_path, "wt") as f:
        for event in sample_mjai_data:
            f.write(json.dumps(event) + "\n")

    replay = MjaiReplay.from_jsonl(str(file_path))
    assert replay.num_rounds() == 1
    kyokus = list(replay.take_kyokus())
    assert len(kyokus) == 1
    assert len(kyokus[0].events()) == 4


def test_mjai_replay_real_file():
    file_path = os.path.join(os.path.dirname(__file__), "data", "126_204_0_mjai.jsonl")

    if not os.path.exists(file_path):
        pytest.skip(f"Test file not found: {file_path}")

    replay = MjaiReplay.from_jsonl(file_path)
    assert replay.num_rounds() == 12

    kyokus = list(replay.take_kyokus())
    assert len(kyokus) == replay.num_rounds()

    features = kyokus[0].grp_features()
    assert "scores" in features
    assert "end_scores" in features
    assert "wliqi" in features
