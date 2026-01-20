import json

from riichienv import Action, ActionType, Observation


class TestActionToMjaiFormat:
    def test_action_to_mjai_dahai(self):
        # TODO: Impl tsumogiri flag
        act = Action(ActionType.Discard, tile=53)
        mjai = act.to_mjai()
        assert json.loads(mjai) == {"type": "dahai", "pai": "5p"}

    def test_action_to_mjai_chi(self):
        # Chi 5p with 4p,6p consumed
        # 5p (53), 4p (49), 6p (57)
        act = Action(ActionType.Chi, tile=53, consume_tiles=[49, 57])
        mjai = act.to_mjai()
        expected = {"type": "chi", "pai": "5p", "consumed": ["4p", "6p"]}
        assert json.loads(mjai) == expected

    def test_action_to_mjai_reach(self):
        act = Action(ActionType.Riichi)
        mjai = act.to_mjai()
        assert json.loads(mjai) == {"type": "reach"}

    def test_select_action_from_mjai(self):
        legal_actions = [
            Action(ActionType.Discard, tile=53),  # 5p
            Action(ActionType.Riichi),
        ]

        player_id = 0
        hand = []
        events_json = []
        prev_events_size = 0
        obs = Observation(player_id, hand, events_json, prev_events_size, legal_actions)

        mjai_dict = {"type": "dahai", "pai": "5p"}
        selected = obs.select_action_from_mjai(mjai_dict)
        assert selected is not None
        assert selected.action_type == ActionType.Discard
        assert selected.tile == 53

        # Riichi
        selected = obs.select_action_from_mjai({"type": "reach"})
        assert selected is not None
        assert selected.action_type == ActionType.Riichi

        # Non-existent action
        selected = obs.select_action_from_mjai({"type": "dahai", "pai": "1z"})
        assert selected is None

        # Loose matching (extra fields)
        mjai_loose = {"type": "dahai", "pai": "5p", "tsumogiri": True, "meta": {"foo": "bar"}}
        selected = obs.select_action_from_mjai(mjai_loose)
        assert selected is not None
        assert selected.action_type == ActionType.Discard
        assert selected.tile == 53
