import json

from riichienv import Action, ActionType, Observation


class TestActionToMjaiFormat:
    def test_action_to_mjai_dahai(self):
        act = Action(ActionType.DISCARD, tile=53)
        mjai = act.to_mjai()
        assert json.loads(mjai) == {"type": "dahai", "pai": "5p"}

    def test_action_to_mjai_chi(self):
        # Chi 5p with 4p,6p consumed
        # 5p (53), 4p (49), 6p (57)
        act = Action(ActionType.CHI, tile=53, consume_tiles=[49, 57])
        mjai = act.to_mjai()
        expected = {"type": "chi", "pai": "5p", "consumed": ["4p", "6p"]}
        assert json.loads(mjai) == expected

    def test_action_to_mjai_reach(self):
        act = Action(ActionType.RIICHI)
        mjai = act.to_mjai()
        assert json.loads(mjai) == {"type": "reach"}

    def test_select_action_from_mjai(self):
        legal_actions = [
            Action(ActionType.DISCARD, tile=53),  # 5p
            Action(ActionType.RIICHI),
        ]

        player_id = 0
        hands = [[] for _ in range(4)]
        melds = [[] for _ in range(4)]
        discards = [[] for _ in range(4)]
        dora_indicators = []
        scores = [25000] * 4
        riichi_declared = [False] * 4
        events = []
        honba = 0
        riichi_sticks = 0
        round_wind = 0
        oya = 0

        obs = Observation(
            player_id,
            hands,
            melds,
            discards,
            dora_indicators,
            scores,
            riichi_declared,
            legal_actions,
            events,
            honba,
            riichi_sticks,
            round_wind,
            oya,
            0,  # kyoku_index
            [],  # waits
            False,  # is_tenpai
            [None] * 4,  # riichi_sutehais
            [None] * 4,  # last_tedashis
            None,  # last_discard
        )

        mjai_dict = {"type": "dahai", "pai": "5p"}
        selected = obs.select_action_from_mjai(mjai_dict)
        assert selected is not None
        assert selected.action_type == ActionType.DISCARD
        assert selected.tile == 53

        # Riichi
        selected = obs.select_action_from_mjai({"type": "reach"})
        assert selected is not None
        assert selected.action_type == ActionType.RIICHI

        # Non-existent action
        selected = obs.select_action_from_mjai({"type": "dahai", "pai": "1z"})
        assert selected is None

        # Loose matching (extra fields)
        mjai_loose = {"type": "dahai", "pai": "5p", "tsumogiri": True, "meta": {"foo": "bar"}}
        selected = obs.select_action_from_mjai(mjai_loose)
        assert selected is not None
        assert selected.action_type == ActionType.DISCARD
        assert selected.tile == 53
