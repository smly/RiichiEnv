import json

from riichienv import Observation
from riichienv.action import Action, ActionType


def test_action_to_mjai_dahai():
    # 5p (tid 53)
    act = Action(ActionType.DISCARD, tile=53)
    mjai = act.to_mjai()
    # "tsumogiri" is not included by default in internal action?
    # Our impl: {"type": "dahai", "pai": "5p"}
    assert json.loads(mjai) == {"type": "dahai", "pai": "5p"}


def test_action_to_mjai_chi():
    # Chi 5p with 4p,6p consumed
    # 5p (53), 4p (49), 6p (57)
    act = Action(ActionType.CHI, tile=53, consume_tiles=[49, 57])
    mjai = act.to_mjai()
    expected = {"type": "chi", "pai": "5p", "consumed": ["4p", "6p"]}
    assert json.loads(mjai) == expected


def test_action_to_mjai_reach():
    act = Action(ActionType.RIICHI)
    mjai = act.to_mjai()
    assert json.loads(mjai) == {"type": "reach"}


def test_select_action_from_mjai():
    legal_actions = [
        Action(ActionType.DISCARD, tile=53),  # 5p
        Action(ActionType.RIICHI),
        Action(ActionType.TSUMO, tile=53),
    ]

    # Rust calls: Observation.__init__(player_id, hand, events_json, prev_events_size, legal_actions)
    # events_json must be list of strings.
    obs = Observation(0, [], [], 0, legal_actions)

    # Text Search Discard 5p
    # Rust API expects Dict, not Str
    mjai_dict = {"type": "dahai", "pai": "5p"}
    selected = obs.select_action_from_mjai(mjai_dict)
    assert selected is not None
    assert selected.action_type == ActionType.DISCARD
    assert selected.tile == 53

    # JSON with spaces -> parsed to dict
    mjai_dict_spaces = {"type": "dahai", "pai": "5p"}
    selected = obs.select_action_from_mjai(mjai_dict_spaces)
    assert selected is not None
    assert selected.action_type == ActionType.DISCARD

    # Riichi
    selected = obs.select_action_from_mjai({"type": "reach"})
    assert selected is not None
    assert selected.action_type == ActionType.RIICHI

    # Non-existent action
    selected = obs.select_action_from_mjai({"type": "dahai", "pai": "1z"})
    assert selected is None

    # Invalid JSON string test removed as we pass dicts now.
    # If invalid dict struct, Rust might error or return None.

    # Loose matching (extra fields)
    mjai_loose = {"type": "dahai", "pai": "5p", "tsumogiri": True, "meta": {"foo": "bar"}}
    selected = obs.select_action_from_mjai(mjai_loose)
    assert selected is not None
    assert selected.action_type == ActionType.DISCARD
    assert selected.tile == 53
