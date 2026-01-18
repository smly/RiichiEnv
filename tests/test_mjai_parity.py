import json

import pytest

from riichienv import Action, ActionType, RiichiEnv


def test_action_to_mjai_red_fives():
    # Test red fives mapping to 5mr, 5pr, 5sr
    act_m = Action(ActionType.Discard, tile=16)
    assert '"pai":"5mr"' in act_m.to_mjai()

    act_p = Action(ActionType.Discard, tile=52)
    assert '"pai":"5pr"' in act_p.to_mjai()

    act_s = Action(ActionType.Discard, tile=88)
    assert '"pai":"5sr"' in act_s.to_mjai()


def test_select_action_from_mjai_discard():
    env = RiichiEnv(seed=42)
    obs_dict = env.reset()
    obs = obs_dict[0]

    # Get a legal discard
    legal_discards = [a for a in obs.legal_actions() if a.action_type == ActionType.Discard]
    target_act = legal_discards[0]
    mjai_resp = json.loads(target_act.to_mjai())

    # Select from MJAI
    selected = obs.select_action_from_mjai(mjai_resp)
    assert selected is not None
    assert selected.action_type == ActionType.Discard
    assert selected.tile == target_act.tile


@pytest.mark.skip(reason="See Issue #32")
def test_select_action_from_mjai_chi():
    env = RiichiEnv(seed=42)
    env.reset()

    # Setup for Chi
    # P0 discards 3m. In seed 42, P0 has tile 9 (3m).

    # Manually trigger discard and claim update
    # In a real scenario, we'd use env.step, but for unit test:
    env.current_player = 0
    obs_dict = env.step({0: Action(ActionType.Discard, tile=9)})

    obs1 = obs_dict[1]
    chi_acts = [a for a in obs1.legal_actions() if a.action_type == ActionType.Chi]
    assert len(chi_acts) > 0

    target_act = chi_acts[0]

    mjai_resp = json.loads(target_act.to_mjai())

    selected = obs1.select_action_from_mjai(mjai_resp)
    assert selected is not None
    assert selected.action_type == ActionType.Chi
    assert set(selected.consume_tiles) == set(target_act.consume_tiles)


@pytest.mark.skip(reason="See Issue #32")
def test_select_action_from_mjai_none():
    env = RiichiEnv(seed=42)
    env.reset()
    # tile 9 is in P0 hand (3m)
    obs_dict = env.step({0: Action(ActionType.Discard, tile=9)})
    obs1 = obs_dict[1]

    selected = obs1.select_action_from_mjai({"type": "none"})
    assert selected is not None
    assert selected.action_type == ActionType.Pass
