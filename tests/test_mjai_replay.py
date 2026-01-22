import gzip
import json

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
            "oyaxd": 0,
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
    # We can't easily check len(kyokus) directly as it might be an iterator or similar,
    # but we can listify it.
    kyoku_list = list(kyokus)
    assert len(kyoku_list) == 1

    kyoku = kyoku_list[0]
    events = kyoku.events()
    # start_kyoku -> NewRound (0)
    # tsumo -> DealTile (1)
    # dahai -> DiscardTile (2)
    # ryukyoku -> NoTile (3)
    assert len(events) == 4
    assert events[0]["name"] == "NewRound"
    assert events[1]["name"] == "DealTile"
    assert events[2]["name"] == "DiscardTile"
    assert events[3]["name"] == "NoTile"


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
