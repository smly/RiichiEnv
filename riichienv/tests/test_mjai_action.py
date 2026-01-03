import json

from riichienv.action import Action, ActionType
from riichienv.env import Observation


def test_action_to_mjai_dahai():
    # 5p (tid 53)
    act = Action(type=ActionType.DISCARD, tile=53)
    mjai = act.to_mjai()
    # "tsumogiri" is not included by default in internal action?
    # Our impl: {"type": "dahai", "pai": "5p"}
    assert json.loads(mjai) == {"type": "dahai", "pai": "5p"}


def test_action_to_mjai_chi():
    # Chi 5p with 4p,6p consumed
    # 5p (53), 4p (49), 6p (57)
    act = Action(type=ActionType.CHI, tile=53, consume_tiles=[49, 57])
    mjai = act.to_mjai()
    expected = {"type": "chi", "pai": "5p", "consumed": ["4p", "6p"]}
    assert json.loads(mjai) == expected


def test_action_to_mjai_reach():
    act = Action(type=ActionType.RIICHI)
    mjai = act.to_mjai()
    assert json.loads(mjai) == {"type": "reach"}


def test_select_action_from_mjai():
    legal_actions = [
        Action(type=ActionType.DISCARD, tile=53),  # 5p
        Action(type=ActionType.RIICHI),
        Action(type=ActionType.TSUMO, tile=53),  # Tsumo 5p ? Usually Tsumo doesn't have tile in internal Action?
        # Internal action TSUMO sometimes has tile checking.
    ]

    obs = Observation(player_id=0, hand=[], events=[], _legal_actions=legal_actions)

    # Text Search Discard 5p
    mjai_str = '{"type": "dahai", "pai": "5p"}'
    selected = obs.select_action_from_mjai(mjai_str)
    assert selected is not None
    assert selected.type == ActionType.DISCARD
    assert selected.tile == 53

    # JSON with spaces
    mjai_str_spaces = ' { "type": 	"dahai", "pai": "5p" } '
    selected = obs.select_action_from_mjai(mjai_str_spaces)
    assert selected is not None
    assert selected.type == ActionType.DISCARD

    # Riichi
    selected = obs.select_action_from_mjai('{"type": "reach"}')
    assert selected is not None
    assert selected.type == ActionType.RIICHI

    # Non-existent action
    selected = obs.select_action_from_mjai('{"type": "dahai", "pai": "1z"}')
    assert selected is None

    # Invalid JSON
    selected = obs.select_action_from_mjai("invalid json")
    assert selected is None

    # Loose matching (extra fields)
    mjai_loose = '{"type": "dahai", "pai": "5p", "tsumogiri": true, "meta": {"foo": "bar"}}'
    selected = obs.select_action_from_mjai(mjai_loose)
    assert selected is not None
    assert selected.type == ActionType.DISCARD
    assert selected.tile == 53
