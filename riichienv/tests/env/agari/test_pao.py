import pytest

from riichienv import Meld, MeldType, Phase, RiichiEnv
from riichienv.action import Action, ActionType


class TestPao:
    @pytest.mark.skip(reason="Pao logic not implemented in Rust yet")
    def test_daisangen_pao_tsumo(self):
        """
        Verify Daisangen Pao for Tsumo.
        P0 has 2 dragon melds, calls 3rd from P1, then Tsumos.
        """
        env = RiichiEnv(seed=42)
        env.reset(oya=0)

        # Setup P0 hand: 2x Red Dragon (132, 133), 3x 1m (0, 1, 2), 2x 2m (4, 5)
        # Melds: 31 (White), 32 (Green)
        # 31: 124, 125, 126. 32: 128, 129, 130.
        env.melds[0] = [
            Meld(MeldType.Peng, [124, 125, 126], True),
            Meld(MeldType.Peng, [128, 129, 130], True),
        ]
        hands = env.hands
        hands[0] = [132, 133, 0, 1, 2, 4, 5]  # 7 tiles + 2 melds (6 tiles) = 13 tiles
        env.hands = hands

        # P1 discards 3rd Red Dragon (134)
        env.current_player = 1
        env.phase = Phase.WaitResponse
        env.last_discard = (1, 134)
        env.active_players = [0]
        env.current_claims = {0: [Action(ActionType.PON, tile=134, consume_tiles=[132, 133])]}

        # P0 calls Pon
        env.step({0: Action(ActionType.PON, tile=134, consume_tiles=[132, 133])})

        # Verify Pao is established
        assert env.pao[0].get(37) == 1

        # P0 draws completing tile for 2m pair (6 is 2m)
        env.current_player = 0
        env.phase = Phase.WaitAct
        env.active_players = [0]
        env.drawn_tile = 6

        # Tsumo
        env.step({0: Action(ActionType.TSUMO)})

        # Verify scoring
        agari_res = env.agari_results[0]
        assert agari_res.agari
        assert 37 in agari_res.yaku
        assert 39 not in agari_res.yaku  # Not Tsuiso (39 is Tsuiso)

        hora = next(m for m in reversed(env.mjai_log) if m["type"] == "hora")
        deltas = hora["deltas"]

        # Oya Yakuman Tsumo: 48000 total.
        # P1 (Pao) should pay 48000.
        assert deltas[0] == 48000
        assert deltas[1] == -48000
        assert deltas[2] == 0
        assert deltas[3] == 0

    @pytest.mark.skip(reason="Pao logic not implemented in Rust yet")
    def test_daisangen_pao_ron(self):
        """
        Verify Daisangen Pao for Ron from a 3rd party.
        """
        env = RiichiEnv(seed=42)
        env.reset(oya=1)  # P1 is Oya

        # Establish Pao
        hands = env.hands
        hands[0] = [132, 133, 0, 1, 2, 4, 5]
        env.hands = hands
        env.melds[0] = [
            Meld(MeldType.Peng, [124, 125, 126], True),
            Meld(MeldType.Peng, [128, 129, 130], True),
        ]

        env.current_player = 1
        env.phase = Phase.WaitResponse
        env.last_discard = (1, 134)
        env.active_players = [0]
        env.current_claims = {0: [Action(ActionType.PON, tile=134, consume_tiles=[132, 133])]}

        env.step({0: Action(ActionType.PON, tile=134, consume_tiles=[132, 133])})

        assert env.pao[0].get(37) == 1

        # P2 discards winning tile (6 for 2m triplet)
        env.current_player = 2
        env.phase = Phase.WaitResponse
        env.last_discard = (2, 6)
        env.active_players = [0]
        env.current_claims = {0: [Action(ActionType.RON, tile=6)]}

        # P0 hand after Pon: 3x 1m (0-2), 2x 2m (4-5)
        # Win on 2m (6)
        env.step({0: Action(ActionType.RON)})

        # Ko Yakuman Ron: 32000 points.
        hora = next(m for m in reversed(env.mjai_log) if m["type"] == "hora")
        deltas = hora["deltas"]

        assert deltas[0] == 32000
        assert deltas[1] == -16000  # Pao (Oya)
        assert deltas[2] == -16000  # Discarder (Ko)
        assert deltas[3] == 0
