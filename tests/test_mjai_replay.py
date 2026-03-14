import gzip
import json
import os

import pytest

from riichienv import ActionType, MjaiReplay


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

    # Verify grp features
    assert kyokus[0].grp_features()["scores"] == [25000, 25000, 25000, 25000]
    assert kyokus[0].grp_features()["end_scores"] == [21000, 22000, 23000, 34000]
    assert kyokus[0].grp_features()["delta_scores"] == [-4000, -3000, -2000, 9000]
    for idx in range(len(kyokus) - 1):
        assert kyokus[idx + 1].grp_features()["scores"] == kyokus[idx].grp_features()["end_scores"]


def test_mjai_replay_3p_reach_discard_observation_is_not_duplicated_state(tmp_path):
    data = [
        {"type": "start_game", "names": ["A", "B", "C"], "id": "test_3p_reach"},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyoutaku": 0,
            "oya": 0,
            "scores": [35000, 35000, 35000],
            "dora_marker": "1p",
            "tehais": [
                ["1p", "1p", "2p", "2p", "2p", "3p", "3p", "3p", "4p", "4p", "4p", "5z", "5z"],
                ["1s", "1s", "1s", "2s", "2s", "2s", "3s", "3s", "3s", "4s", "4s", "4s", "6z"],
                ["1z", "1z", "2z", "2z", "3z", "3z", "4z", "4z", "5z", "5z", "6z", "6z", "7z"],
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "1p"},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "1p", "tsumogiri": True},
        {"type": "ryukyoku", "reason": "test"},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]

    file_path = tmp_path / "test_3p_reach.jsonl"
    with open(file_path, "w") as f:
        for event in data:
            f.write(json.dumps(event) + "\n")

    replay = MjaiReplay.from_jsonl(str(file_path), rule="mjsoul")
    kyoku = list(replay.take_kyokus())[0]
    steps = list(kyoku.steps(0, skip_single_action=False))

    assert len(steps) >= 2
    riichi_obs, riichi_act = steps[0]
    discard_obs, discard_act = steps[1]

    assert riichi_act.action_type == ActionType.RIICHI
    assert discard_act.action_type == ActionType.DISCARD

    riichi_legals = [a.action_type for a in riichi_obs.legal_actions()]
    discard_legals = [a.action_type for a in discard_obs.legal_actions()]
    assert ActionType.RIICHI in riichi_legals
    assert ActionType.RIICHI not in discard_legals


def test_mjai_replay_mjsoul_nonfinal_end_scores_fallback_to_next_round_scores(tmp_path):
    data = [
        {"type": "start_game", "names": ["A", "B", "C"], "id": "test_3p_scores"},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyoutaku": 0,
            "oya": 0,
            "scores": [35000, 35000, 35000],
            "dora_marker": "1p",
            "tehais": [
                ["1p", "1p", "2p", "2p", "2p", "3p", "3p", "3p", "4p", "4p", "4p", "5z", "5z"],
                ["1s", "1s", "1s", "2s", "2s", "2s", "3s", "3s", "3s", "4s", "4s", "4s", "6z"],
                ["1z", "1z", "2z", "2z", "3z", "3z", "4z", "4z", "5z", "5z", "6z", "6z", "7z"],
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "1p"},
        {"type": "hora", "actor": 0, "target": 0, "pai": "1p"},
        {"type": "end_kyoku"},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 2,
            "honba": 0,
            "kyoutaku": 0,
            "oya": 1,
            "scores": [39000, 33000, 33000],
            "dora_marker": "2p",
            "tehais": [
                ["1m", "2m", "3m", "4m", "5p", "6p", "7p", "1s", "2s", "3s", "E", "S", "W"],
                ["2m", "3m", "4m", "5m", "6p", "7p", "8p", "2s", "3s", "4s", "S", "W", "N"],
                ["3m", "4m", "5m", "6m", "7p", "8p", "9p", "3s", "4s", "5s", "W", "N", "P"],
            ],
        },
        {"type": "tsumo", "actor": 1, "pai": "2p"},
        {"type": "dahai", "actor": 1, "pai": "2p", "tsumogiri": True},
        {"type": "ryukyoku", "reason": "test"},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]

    file_path = tmp_path / "test_3p_scores.jsonl"
    with open(file_path, "w") as f:
        for event in data:
            f.write(json.dumps(event) + "\n")

    replay = MjaiReplay.from_jsonl(str(file_path), rule="mjsoul")
    kyokus = list(replay.take_kyokus())

    assert len(kyokus) == 2
    assert kyokus[0].grp_features()["scores"] == [35000, 35000, 35000]
    assert kyokus[0].grp_features()["end_scores"] == [39000, 33000, 33000]
    assert kyokus[0].grp_features()["delta_scores"] == [4000, -2000, -2000]
    assert kyokus[1].grp_features()["scores"] == kyokus[0].grp_features()["end_scores"]


def test_mjai_replay_4p_reach_discard_observation_is_not_duplicated_state(tmp_path):
    data = [
        {"type": "start_game", "names": ["A", "B", "C", "D"], "id": "test_4p_reach"},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyoutaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1p",
            "tehais": [
                ["1p", "1p", "2p", "2p", "2p", "3p", "3p", "3p", "4p", "4p", "4p", "5z", "5z"],
                ["1s", "1s", "1s", "2s", "2s", "2s", "3s", "3s", "3s", "4s", "4s", "4s", "6z"],
                ["1z", "1z", "2z", "2z", "3z", "3z", "4z", "4z", "5z", "5z", "6z", "6z", "7z"],
                ["5m", "5m", "6m", "6m", "7m", "7m", "8m", "8m", "9m", "9m", "1m", "1m", "2m"],
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "1p"},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "1p", "tsumogiri": True},
        {"type": "ryukyoku", "reason": "test"},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]

    file_path = tmp_path / "test_4p_reach.jsonl"
    with open(file_path, "w") as f:
        for event in data:
            f.write(json.dumps(event) + "\n")

    replay = MjaiReplay.from_jsonl(str(file_path))
    kyoku = list(replay.take_kyokus())[0]
    steps = list(kyoku.steps(0, skip_single_action=False))

    assert len(steps) >= 2
    riichi_obs, riichi_act = steps[0]
    discard_obs, discard_act = steps[1]

    assert riichi_act.action_type == ActionType.RIICHI
    assert discard_act.action_type == ActionType.DISCARD

    riichi_legals = [a.action_type for a in riichi_obs.legal_actions()]
    discard_legals = [a.action_type for a in discard_obs.legal_actions()]
    assert ActionType.RIICHI in riichi_legals
    assert ActionType.RIICHI not in discard_legals
