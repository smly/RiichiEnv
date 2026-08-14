from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from riichienv import ActionType, MjaiReplay, MjSoulReplay

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "scripts" / "validate_external_replays.py"
MANIFEST_PATH = Path(__file__).parent / "data" / "external_replay" / "manifest.json"
TENHOU_FIXTURE = MANIFEST_PATH.parent / "tenhou_4p_ranked_excerpt.jsonl"
MJSOUL_FIXTURE = MANIFEST_PATH.parent / "mjsoul_3p_double_ron_excerpt.jsonl"
TRACKED_RIICHI_FIXTURE = Path(__file__).parent / "data" / "126_204_0_mjai.jsonl"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_external_replays", VALIDATOR_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_committed_external_replay_manifest_matches_calculator_and_replay():
    validator = _load_validator()
    assert validator.validate_manifest(MANIFEST_PATH) == 4


def test_external_validator_composes_accepted_riichi_costs_and_sticks():
    validator = _load_validator()
    observed = validator.observe_fixture(TRACKED_RIICHI_FIXTURE, "tenhou", [0])[0]

    # The hora delta is [-4000, -2000, -2000, +10000]. Players 1 and 3
    # each paid a separately emitted accepted-riichi deposit.
    assert observed["delta"] == [-4000, -3000, -2000, 9000]
    assert observed["end_scores"] == [21000, 22000, 23000, 34000]


def test_tenhou_fixture_does_not_double_decrement_wall_or_count_kan_dora_twice():
    kyoku = list(MjaiReplay.from_jsonl(str(TENHOU_FIXTURE), rule="tenhou").take_kyokus())[0]
    context = list(kyoku.take_win_result_contexts())[0]

    assert context.conditions.houtei is False
    assert context.actual.han == 4
    assert context.actual.fu == 30
    assert sorted(context.actual.yaku) == [8, 27, 32]
    assert context.actual.ron_agari == 7700


def test_mjsoul_3p_fixture_uses_sanma_evaluator_and_preserves_fan_values():
    kyoku = list(MjaiReplay.from_jsonl(str(MJSOUL_FIXTURE), rule="mjsoul").take_kyokus())[0]
    contexts = list(kyoku.take_win_result_contexts())

    assert len(kyoku.hands) == 3
    assert len(contexts) == 2
    assert all(context.conditions.is_sanma for context in contexts)
    assert all(context.conditions.num_players == 3 for context in contexts)
    assert contexts[0].expected_yaku_values == [(21, 2), (31, 3)]
    assert contexts[1].expected_yaku_values == [(20, 2), (7, 1), (31, 5)]
    assert [(context.actual.han, context.actual.ron_agari) for context in contexts] == [(5, 12000), (8, 16000)]

    ron_seats = [
        player_id
        for player_id, _, action in kyoku.steps(seat=None, skip_single_action=False)
        if action.action_type == ActionType.RON
    ]
    assert ron_seats == [0, 1]

    with pytest.raises(ValueError, match="4-player evaluator for a 3-player context"):
        contexts[0].create_calculator()
    calculator = contexts[0].create_calculator_3p()
    recalculated = contexts[0].calculate_3p(calculator)
    assert (recalculated.han, recalculated.fu, recalculated.ron_agari) == (5, 40, 12000)


def test_native_mjsoul_hule_keeps_terminal_scores_fan_values_and_split_payments():
    replay = MjSoulReplay.from_dict(
        [
            [
                {
                    "name": "NewRound",
                    "data": {
                        "scores": [35000, 35000, 35000],
                        "tiles0": [],
                        "tiles1": [],
                        "tiles2": [],
                        "tiles3": [],
                        "chang": 0,
                        "ju": 0,
                        "liqibang": 0,
                    },
                },
                {
                    "name": "Hule",
                    "data": {
                        "old_scores": [35000, 35000, 35000],
                        "delta_scores": [8000, -4000, -4000],
                        "hules": [
                            {
                                "seat": 0,
                                "hu_tile": "1p",
                                "zimo": True,
                                "count": 3,
                                "fu": 30,
                                "fans": [{"id": 1, "val": 1}, {"id": 31, "val": 2}],
                                "hand": [],
                                "yiman": False,
                                "point_rong": 0,
                                "point_zimo_qin": 4000,
                                "point_zimo_xian": 2000,
                            }
                        ],
                    },
                },
            ]
        ]
    )
    kyoku = list(replay.take_kyokus())[0]
    hule = next(event for event in kyoku.events() if event["name"] == "Hule")["data"]["hules"][0]

    assert len(kyoku.hands) == 3
    assert kyoku.grp_features()["end_scores"] == [43000, 31000, 31000]
    assert kyoku.grp_features()["delta_scores"] == [8000, -4000, -4000]
    assert hule["fans"] == [{"id": 1, "val": 1}, {"id": 31, "val": 2}]
    assert hule["point_rong"] == 0
    assert hule["point_zimo_qin"] == 4000
    assert hule["point_zimo_xian"] == 2000


def _native_new_round(scores: list[int], tiles0: list[str] | None = None):
    return {
        "name": "NewRound",
        "data": {
            "scores": scores,
            "tiles0": tiles0 or [],
            "tiles1": [],
            "tiles2": [],
            "tiles3": [],
            "chang": 0,
            "ju": 0,
            "liqibang": 0,
        },
    }


def _native_hule(*, seat: int, tile: str, zimo: bool, ura: list[str] | None = None):
    return {
        "name": "Hule",
        "data": {
            "hules": [
                {
                    "seat": seat,
                    "hu_tile": tile,
                    "zimo": zimo,
                    "count": 2,
                    "fu": 30,
                    "fans": [{"id": 21, "val": 2}],
                    "hand": [],
                    "ura_dora_indicators": ura,
                    "yiman": False,
                    "point_rong": 2000 if not zimo else 0,
                    "point_zimo_qin": 1000 if zimo else 0,
                    "point_zimo_xian": 500 if zimo else 0,
                }
            ]
        },
    }


def test_native_mjsoul_riichi_stick_is_accepted_after_the_declaration_tile_survives():
    header = _native_new_round([25000, 25000, 25000, 25000])
    declaration = {
        "name": "DiscardTile",
        "data": {"seat": 1, "tile": "1m", "is_liqi": True},
    }
    ron = _native_hule(seat=0, tile="1m", zimo=False)

    immediate = MjSoulReplay.from_dict([[header, declaration, ron]])
    immediate_context = list(list(immediate.take_kyokus())[0].take_win_result_contexts())[0]
    assert immediate_context.conditions.riichi_sticks == 0

    accepted = MjSoulReplay.from_dict(
        [
            [
                header,
                declaration,
                {"name": "DealTile", "data": {"seat": 2, "tile": "2m"}},
                ron,
            ]
        ]
    )
    accepted_context = list(list(accepted.take_kyokus())[0].take_win_result_contexts())[0]
    assert accepted_context.conditions.riichi_sticks == 1


@pytest.mark.parametrize("scores", [[25000] * 4, [35000] * 3])
def test_native_mjsoul_exclusive_double_riichi_is_replayed_and_keeps_ippatsu_ura(scores):
    tenpai_hand = [
        "1p",
        "1p",
        "2p",
        "2p",
        "2p",
        "3p",
        "3p",
        "3p",
        "4p",
        "4p",
        "4p",
        "5z",
        "5z",
    ]
    replay = MjSoulReplay.from_dict(
        [
            [
                _native_new_round(scores, tenpai_hand),
                {"name": "DealTile", "data": {"seat": 0, "tile": "1p"}},
                {
                    "name": "DiscardTile",
                    "data": {"seat": 0, "tile": "1p", "is_wliqi": True},
                },
                {"name": "DealTile", "data": {"seat": 0, "tile": "1p"}},
                _native_hule(seat=0, tile="1p", zimo=True, ura=["3p"]),
            ]
        ]
    )
    kyoku = list(replay.take_kyokus())[0]

    steps = iter(kyoku.steps(0, skip_single_action=False))
    _, riichi_action = next(steps)
    _, discard_action = next(steps)
    assert riichi_action.action_type == ActionType.RIICHI
    assert discard_action.action_type == ActionType.DISCARD

    context = list(kyoku.take_win_result_contexts())[0]
    assert context.conditions.riichi is True
    assert context.conditions.double_riichi is True
    assert context.conditions.ippatsu is True
    assert context.conditions.riichi_sticks == 1
    assert context.ura_indicators


def test_native_mjsoul_three_player_default_wall_count_reaches_haitei_at_55_draws():
    actions = [_native_new_round([35000, 35000, 35000])]
    for index in range(55):
        actions.append({"name": "DealTile", "data": {"seat": 0, "tile": "1p"}})
        if index < 54:
            actions.append({"name": "DiscardTile", "data": {"seat": 0, "tile": "1p"}})
    actions.append(_native_hule(seat=0, tile="1p", zimo=True))

    kyoku = list(MjSoulReplay.from_dict([actions]).take_kyokus())[0]
    assert kyoku.left_tile_count == 55
    assert list(kyoku.take_win_result_contexts())[0].conditions.haitei is True


def test_native_mjsoul_kita_ron_is_not_scored_as_tsumo():
    waiting_on_north = [
        "1p",
        "1p",
        "1p",
        "2p",
        "2p",
        "2p",
        "3p",
        "3p",
        "3p",
        "4p",
        "4p",
        "4p",
        "4z",
    ]
    header = _native_new_round([35000, 35000, 35000], waiting_on_north)
    header["data"]["tiles2"] = ["4z"]
    replay = MjSoulReplay.from_dict(
        [
            [
                header,
                {"name": "BaBei", "data": {"seat": 2}},
                _native_hule(seat=0, tile="4z", zimo=True),
            ]
        ]
    )
    kyoku = list(replay.take_kyokus())[0]
    context = list(kyoku.take_win_result_contexts())[0]

    assert context.conditions.tsumo is False
    assert context.conditions.chankan is False
    assert context.actual.ron_agari > 0


def test_native_mjsoul_rejects_malformed_rounds_and_out_of_range_seats():
    with pytest.raises(ValueError, match="round contains no actions"):
        MjSoulReplay.from_dict([[]])

    with pytest.raises(ValueError, match="round must start with NewRound"):
        MjSoulReplay.from_dict([[{"name": "DiscardTile", "data": {"seat": 0, "tile": "1m"}}]])

    with pytest.raises(ValueError, match="scores must contain 3 or 4 players"):
        MjSoulReplay.from_dict([[_native_new_round([25000, 25000])]])

    with pytest.raises(ValueError, match="seat 4 is out of range for 4 players"):
        MjSoulReplay.from_dict(
            [
                [
                    _native_new_round([25000] * 4),
                    {
                        "name": "DiscardTile",
                        "data": {"seat": 4, "tile": "1m", "is_wliqi": True},
                    },
                ]
            ]
        )

    with pytest.raises(ValueError, match="requires 3 tiles/froms, got 3/2"):
        MjSoulReplay.from_dict(
            [
                [
                    _native_new_round([25000] * 4),
                    {
                        "name": "ChiPengGang",
                        "data": {
                            "seat": 0,
                            "type": 0,
                            "tiles": ["1m", "2m", "3m"],
                            "froms": [1, 0],
                        },
                    },
                ]
            ]
        )

    with pytest.raises(ValueError, match="contains invalid tile"):
        MjSoulReplay.from_dict(
            [
                [
                    _native_new_round([25000] * 4),
                    {"name": "DealTile", "data": {"seat": 0, "tile": "🀄"}},
                ]
            ]
        )

    with pytest.raises(ValueError, match="player count changed from 4 to 3"):
        MjSoulReplay.from_dict(
            [
                [_native_new_round([25000] * 4)],
                [_native_new_round([35000] * 3)],
            ]
        )

    with pytest.raises(ValueError, match="Hule must contain at least one winner"):
        MjSoulReplay.from_dict(
            [
                [
                    _native_new_round([25000] * 4),
                    {"name": "Hule", "data": {"hules": []}},
                ]
            ]
        )

    invalid_ju = _native_new_round([35000] * 3)
    invalid_ju["data"]["ju"] = 3
    with pytest.raises(ValueError, match="ju 3 is out of range for 3 players"):
        MjSoulReplay.from_dict([[invalid_ju]])

    invalid_chang = _native_new_round([25000] * 4)
    invalid_chang["data"]["chang"] = 4
    with pytest.raises(ValueError, match="chang must be between 0 and 3"):
        MjSoulReplay.from_dict([[invalid_chang]])


def test_mjai_rejects_round_indices_outside_the_variant():
    log = "\n".join(
        [
            '{"type":"start_game"}',
            '{"type":"start_kyoku","bakaze":"E","kyoku":4,"honba":0,"kyotaku":0,'
            '"oya":0,"scores":[35000,35000,35000],"dora_marker":"1p",'
            '"tehais":[[],[],[]]}',
            '{"type":"end_kyoku"}',
        ]
    )
    with pytest.raises(ValueError, match="kyoku 4 is out of range for 3 players"):
        MjaiReplay.from_jsonl_text(log, rule="mjsoul")

    with pytest.raises(ValueError, match="bakaze must be E, S, W, or N"):
        MjaiReplay.from_jsonl_text(log.replace('"bakaze":"E"', '"bakaze":"X"'), rule="mjsoul")


def test_native_mjsoul_result_rejects_wrong_player_score_vectors():
    round_header = {
        "name": "NewRound",
        "data": {
            "scores": [35000, 35000, 35000],
            "tiles0": [],
            "tiles1": [],
            "tiles2": [],
            "tiles3": [],
            "chang": 0,
            "ju": 0,
            "liqibang": 0,
        },
    }
    malformed_result = _native_hule(seat=0, tile="1p", zimo=True)
    malformed_result["data"].update(
        {
            "old_scores": [35000, 35000, 35000],
            "delta_scores": [8000, -4000],
        }
    )

    with pytest.raises(ValueError, match="result score vectors must contain 3 players"):
        MjSoulReplay.from_dict([[round_header, malformed_result]])

    overflowing_result = _native_hule(seat=0, tile="1p", zimo=True)
    overflowing_result["data"].update(
        {
            "old_scores": [2_147_483_647, 0, 0],
            "delta_scores": [1, 0, 0],
        }
    )
    with pytest.raises(ValueError, match="result score overflows i32"):
        MjSoulReplay.from_dict([[round_header, overflowing_result]])


def test_terminal_mjai_delta_is_kept_without_a_following_round_header():
    log = "\n".join(
        [
            '{"type":"start_game"}',
            '{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,'
            '"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m",'
            '"tehais":[[],[],[],[]]}',
            '{"type":"hora","actor":0,"target":1,"pai":"1m","deltas":[8000,-8000,0,0]}',
            '{"type":"end_kyoku"}',
            '{"type":"end_game"}',
        ]
    )

    kyoku = list(MjaiReplay.from_jsonl_text(log, rule="tenhou").take_kyokus())[0]
    assert kyoku.grp_features()["delta_scores"] == [8000, -8000, 0, 0]
    assert kyoku.grp_features()["end_scores"] == [33000, 17000, 25000, 25000]


def test_win_context_counts_only_accepted_riichi_sticks():
    declaration_ron = "\n".join(
        [
            '{"type":"start_game"}',
            '{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":1,'
            '"oya":0,"scores":[24000,25000,25000,25000],"dora_marker":"1m",'
            '"tehais":[[],[],[],[]]}',
            '{"type":"reach","actor":1}',
            '{"type":"dahai","actor":1,"pai":"1m","tsumogiri":false}',
            '{"type":"hora","actor":0,"target":1,"pai":"1m"}',
            '{"type":"end_kyoku"}',
        ]
    )

    kyoku = list(MjaiReplay.from_jsonl_text(declaration_ron, rule="tenhou").take_kyokus())[0]
    assert list(kyoku.take_win_result_contexts())[0].conditions.riichi_sticks == 1

    accepted = declaration_ron.replace(
        '{"type":"hora","actor":0,"target":1,"pai":"1m"}',
        '{"type":"reach_accepted","actor":1}\n{"type":"hora","actor":0,"target":1,"pai":"1m"}',
    )
    kyoku = list(MjaiReplay.from_jsonl_text(accepted, rule="tenhou").take_kyokus())[0]
    assert list(kyoku.take_win_result_contexts())[0].conditions.riichi_sticks == 2


@pytest.mark.parametrize(
    ("acceptance_event", "expected_scores"),
    [
        (None, [25000, 25000, 25000, 25000]),
        ('{"type":"reach_accepted","actor":1}', [25000, 24000, 25000, 25000]),
    ],
)
def test_mjai_ryukyoku_charges_only_accepted_riichi_deposits(acceptance_event, expected_scores):
    events = [
        '{"type":"start_game"}',
        '{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,'
        '"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m",'
        '"tehais":[[],[],[],[]]}',
        '{"type":"reach","actor":1}',
        '{"type":"dahai","actor":1,"pai":"1m","tsumogiri":false}',
    ]
    if acceptance_event is not None:
        events.append(acceptance_event)
    events.extend(
        [
            '{"type":"ryukyoku","reason":"test","deltas":[0,0,0,0]}',
            '{"type":"end_kyoku"}',
        ]
    )

    kyoku = list(MjaiReplay.from_jsonl_text("\n".join(events), rule="tenhou").take_kyokus())[0]
    assert kyoku.grp_features()["end_scores"] == expected_scores


def test_non_one_manzu_ankan_keeps_rinshan_through_replacement_draw():
    log = "\n".join(
        [
            '{"type":"start_game"}',
            '{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,'
            '"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m",'
            '"tehais":[["1m","2m","3m","1p","2p","3p","5p","5p","5p",'
            '"1s","2s","3s","E"],[],[],[]]}',
            '{"type":"tsumo","actor":0,"pai":"5p"}',
            '{"type":"ankan","actor":0,"consumed":["5p","5p","5p","5p"]}',
            '{"type":"tsumo","actor":0,"pai":"E"}',
            '{"type":"hora","actor":0,"target":0,"pai":"E"}',
            '{"type":"end_kyoku"}',
        ]
    )

    kyoku = list(MjaiReplay.from_jsonl_text(log, rule="tenhou").take_kyokus())[0]
    context = list(kyoku.take_win_result_contexts())[0]
    assert context.conditions.rinshan is True
    assert context.conditions.haitei is False
    assert context.melds[0].tiles == [52, 53, 54, 55]
    assert 4 in context.actual.yaku

    malformed = log.replace(
        '["5p","5p","5p","5p"]',
        '["5p","5p","5p","6p"]',
    )
    with pytest.raises(ValueError, match="ankan consumed tiles must all have the same tile type"):
        MjaiReplay.from_jsonl_text(malformed, rule="tenhou")


def test_external_validator_rejects_unknown_gameplay_events():
    validator = _load_validator()
    with pytest.raises(validator.ExternalReplayError, match="unsupported raw replay event"):
        validator._normalize_raw_actions([{"type": "future_gameplay_event"}])


def test_unknown_external_yaku_is_rejected_instead_of_silently_dropped():
    log = "\n".join(
        [
            '{"type":"start_game"}',
            '{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,'
            '"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m",'
            '"tehais":[[],[],[],[]]}',
            '{"type":"hora","actor":0,"target":0,"pai":"1m","yaku":[["UNKNOWN EXTERNAL YAKU",1]],"han":1,"fu":30}',
            '{"type":"end_kyoku"}',
        ]
    )

    with pytest.raises(ValueError, match="unknown MJAI yaku label"):
        MjaiReplay.from_jsonl_text(log, rule="tenhou")

    validator = _load_validator()
    with pytest.raises(validator.ExternalReplayError, match="unknown external yaku label"):
        validator._yaku_id("Seat Wind dragon")
