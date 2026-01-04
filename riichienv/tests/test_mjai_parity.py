import pytest
from riichienv import RiichiEnv, Action, ActionType


def test_action_to_mjai_red_fives():
    # Test red fives mapping to 5mr, 5pr, 5sr
    act_m = Action(ActionType.DISCARD, tile=16)
    assert '"pai":"5mr"' in act_m.to_mjai()

    act_p = Action(ActionType.DISCARD, tile=52)
    assert '"pai":"5pr"' in act_p.to_mjai()

    act_s = Action(ActionType.DISCARD, tile=88)
    assert '"pai":"5sr"' in act_s.to_mjai()


def test_select_action_from_mjai_discard():
    env = RiichiEnv(seed=42)
    obs_dict = env.reset()
    obs = obs_dict[0]

    # Get a legal discard
    legal_discards = [a for a in obs.legal_actions() if a.type == ActionType.DISCARD]
    target_act = legal_discards[0]
    import json

    mjai_resp = json.loads(target_act.to_mjai())

    # Select from MJAI
    selected = obs.select_action_from_mjai(mjai_resp)
    assert selected is not None
    assert selected.type == ActionType.DISCARD
    assert selected.tile == target_act.tile


def test_select_action_from_mjai_chi():
    env = RiichiEnv(seed=42)
    env.reset()

    # Setup for Chi
    # P0 discards 3m (tile_id=8..11). Let's use tid=8.
    # P1 has 1m, 2m in hand.
    env.hands[1] = [0, 4, 100, 104, 108, 112, 116, 120, 124, 128, 132, 133, 134]

    # Manually trigger discard and claim update
    # In a real scenario, we'd use env.step, but for unit test:
    env.current_player = 0
    obs_dict = env.step({0: Action(ActionType.DISCARD, tile=8)})

    obs1 = obs_dict[1]
    chi_acts = [a for a in obs1.legal_actions() if a.type == ActionType.CHI]
    assert len(chi_acts) > 0

    target_act = chi_acts[0]
    import json

    mjai_resp = json.loads(target_act.to_mjai())

    selected = obs1.select_action_from_mjai(mjai_resp)
    assert selected is not None
    assert selected.type == ActionType.CHI
    assert set(selected.consume_tiles) == set(target_act.consume_tiles)


def test_select_action_from_mjai_none():
    env = RiichiEnv(seed=42)
    env.reset()
    obs_dict = env.step({0: Action(ActionType.DISCARD, tile=8)})
    obs1 = obs_dict[1]

    selected = obs1.select_action_from_mjai({"type": "none"})
    assert selected is not None
    assert selected.type == ActionType.PASS
