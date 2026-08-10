from __future__ import annotations

import pytest

from riichienv import (
    Action,
    Action3P,
    ActionType,
    EventJournal,
    GameEngine,
    MjaiReplay,
    Observation,
    Observation3P,
)


def _choose_action(observation: Observation | Observation3P) -> Action | Action3P:
    legal = observation.legal_actions()
    for action_type in (
        ActionType.TSUMO,
        ActionType.RON,
        ActionType.DISCARD,
        ActionType.PASS,
        ActionType.KYUSHU_KYUHAI,
        ActionType.KITA,
        ActionType.ANKAN,
        ActionType.KAKAN,
        ActionType.DAIMINKAN,
        ActionType.PON,
        ActionType.CHI,
        ActionType.RIICHI,
    ):
        candidates = [action for action in legal if action.action_type == action_type]
        if candidates:
            return min(
                candidates,
                key=lambda action: (
                    action.tile is None,
                    action.tile if action.tile is not None else 255,
                    tuple(action.consume_tiles),
                ),
            )
    raise AssertionError("pending observation has no selectable action")


@pytest.mark.parametrize(
    ("game_mode", "seed", "minimum_rounds"),
    [
        ("4p-red-east", 0x4E01, 4),
        ("3p-red-east", 0x3E01, 3),
    ],
)
def test_python_engine_full_game_round_trips_through_replay(game_mode: str, seed: int, minimum_rounds: int) -> None:
    engine = GameEngine(game_mode, seed=seed)
    observations = engine.decisions()
    steps = 0

    while not engine.is_done:
        steps += 1
        assert steps < 20_000
        assert list(observations) == engine.snapshot["active_players"]

        actions: dict[int, Action | Action3P] = {}
        for player_id, observation in observations.items():
            mask = observation.mask()
            mask_v1 = observation.mask_v1()
            assert len(mask) == observation.action_space_size
            assert len(mask_v1) == observation.action_space_size_v1
            for legal in observation.legal_actions():
                assert mask_v1[legal.encode_v1()] == 1
                assert observation.find_action_v1(legal.encode_v1()) is not None
            actions[player_id] = _choose_action(observation)

        previous_log = engine.mjai_log
        outcome = engine.step(actions)
        assert outcome["error"] is None
        assert outcome["events"] == engine.mjai_log[len(previous_log) :]
        assert outcome["snapshot"] == engine.snapshot
        observations = outcome["observations"]

    assert observations == {}
    events = engine.mjai_log
    journal = EventJournal.from_events(events)
    replay = MjaiReplay.from_events(events)
    kyokus = list(replay.take_kyokus())

    assert journal.is_complete
    assert journal.events == events
    assert len(journal.completed_kyokus) == replay.num_rounds()
    assert len(kyokus) >= minimum_rounds
    assert kyokus[-1].grp_features()["end_scores"] == engine.snapshot["scores"]
    for previous, following in zip(kyokus, kyokus[1:]):
        assert previous.grp_features()["end_scores"] == following.grp_features()["scores"]
