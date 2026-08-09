from __future__ import annotations

import json
import sys

import pytest

from riichienv import (
    ACTION_SPACE_3P_V0,
    ACTION_SPACE_3P_V1,
    ACTION_SPACE_4P_V0,
    ACTION_SPACE_4P_V1,
    Action,
    Action3P,
    ActionType,
    BatchGameEngine,
    GameEngine,
    Observation3P,
)


def test_game_engine_has_strict_modes_and_structured_step_outcomes():
    with pytest.raises(ValueError, match="unknown game mode"):
        GameEngine("default")

    engine = GameEngine("4p-red-single", seed=42)
    observations = engine.decisions()
    assert list(observations) == [0]

    player_id, observation = next(iter(observations.items()))
    action = observation.legal_actions()[0]
    outcome = engine.step({player_id: action})

    assert outcome["error"] is None
    assert outcome["snapshot"]["num_players"] == 4
    assert outcome["snapshot"]["mode"] == "4p-red-single"
    assert outcome["events"]
    assert engine.event_journal.revision == len(engine.mjai_log)


def test_game_engine_reports_typed_legacy_illegal_action():
    engine = GameEngine(seed=42)
    player_id = next(iter(engine.decisions()))
    outcome = engine.step({player_id: Action(ActionType.PASS)})

    assert outcome["error"] == {
        "kind": "illegal_action",
        "player_id": player_id,
        "message": f"Error: Illegal Action by Player {player_id}",
    }

    # The facade clears the previous legacy error before advancing again.
    assert engine.step({})["error"] is None


def test_game_engine_dispatches_three_player_observations():
    engine = GameEngine("3p-red-single", seed=42, event_log=False)
    observation = next(iter(engine.decisions().values()))

    assert isinstance(observation, Observation3P)
    assert engine.num_players == 3
    assert engine.mjai_log == []


def test_batch_game_engine_flattens_decisions_and_steps_all_environments():
    batch = BatchGameEngine(3, seed=100)
    decisions = batch.decisions()
    assert [environment for environment, _player, _observation in decisions] == [0, 1, 2]

    actions: list[dict[int, Action | Action3P]] = [{} for _ in range(len(batch))]
    for environment, player, observation in decisions:
        actions[environment][player] = observation.legal_actions()[0]

    outcome = batch.step(actions)
    assert len(outcome["snapshots"]) == 3
    assert outcome["errors"] == [None, None, None]


def test_batch_game_engine_validates_action_batch_length():
    batch = BatchGameEngine(2, seed=100)
    with pytest.raises(ValueError, match="action maps"):
        batch.step([])


def test_batch_game_engine_rejects_impossible_capacity_without_panicking():
    with pytest.raises(ValueError, match="cannot allocate batch"):
        BatchGameEngine(sys.maxsize * 2 + 1)


@pytest.mark.parametrize("game_mode", ["4p-red-single", "3p-red-single"])
def test_reset_validates_before_mutation_and_reseeds_deterministically(game_mode):
    fresh = GameEngine(game_mode, seed=42)
    expected = next(iter(fresh.decisions().values())).serialize_to_base64()

    engine = GameEngine(game_mode, seed=999)
    before = engine.snapshot
    with pytest.raises(ValueError, match="oya"):
        engine.reset(oya=engine.num_players)
    assert engine.snapshot == before

    with pytest.raises(ValueError, match="wall length"):
        engine.reset(wall=[])
    assert engine.snapshot == before

    with pytest.raises(ValueError, match="safe engine range"):
        engine.reset(scores=[2**31 - 1] * engine.num_players)
    assert engine.snapshot == before

    with pytest.raises(ValueError, match="riichi_sticks"):
        engine.reset(riichi_sticks=2**32 - 1)
    assert engine.snapshot == before

    actual = next(iter(engine.reset(seed=42).values())).serialize_to_base64()
    assert actual == expected
    repeated = next(iter(engine.reset(seed=42).values())).serialize_to_base64()
    assert repeated == expected


def test_snapshot_is_json_serializable_and_has_list_active_players():
    snapshot = GameEngine(seed=42).snapshot

    assert snapshot["phase"] == "WaitAct"
    assert snapshot["active_players"] == [0]
    assert snapshot["wall_tiles_remaining"] == 69
    json.dumps(snapshot)


def test_action_for_non_pending_player_is_rejected_without_mutation():
    engine = GameEngine("3p-red-single", seed=42)
    before = engine.snapshot
    outcome = engine.step({3: Action(ActionType.PASS)})

    assert outcome["error"]["kind"] == "state"
    assert "no pending decision" in outcome["error"]["message"]
    assert engine.snapshot == before


def test_missing_pending_action_is_rejected_without_implicit_pass():
    engine = GameEngine(seed=42)
    before = engine.snapshot
    outcome = engine.step({})

    assert outcome["error"]["kind"] == "state"
    assert "missing action" in outcome["error"]["message"]
    assert engine.snapshot == before


def test_red_aware_v1_action_ids_preserve_the_legacy_contract():
    red = Action(ActionType.DISCARD, tile=16)
    normal = Action(ActionType.DISCARD, tile=17)

    assert red.encode() == normal.encode()
    assert red.encode_v1() == normal.encode_v1() + 1
    assert (ACTION_SPACE_4P_V0, ACTION_SPACE_4P_V1) == (82, 164)
    assert (ACTION_SPACE_3P_V0, ACTION_SPACE_3P_V1) == (60, 120)


@pytest.mark.parametrize(
    ("game_mode", "v0_size", "v1_size"),
    [("4p-red-single", 82, 164), ("3p-red-single", 60, 120)],
)
def test_observation_exposes_versioned_action_masks(game_mode, v0_size, v1_size):
    observation = next(iter(GameEngine(game_mode, seed=42).decisions().values()))

    assert observation.action_space_size == v0_size
    assert observation.action_space_size_v1 == v1_size
    assert len(observation.mask()) == v0_size
    assert len(observation.mask_v1()) == v1_size
    for action in observation.legal_actions():
        action_id = action.encode_v1()
        assert observation.mask_v1()[action_id] == 1
        assert observation.find_action_v1(action_id) is not None
