from riichienv import Meld, MeldType, Phase, RiichiEnv
from riichienv.action import Action, ActionType


class TestPaoHonba:
    def test_daisangen_pao_ron_honba(self):
        """
        Verify Daisangen Pao for Ron from a 3rd party with Honba.
        Log context: 251214-37c7d4e3-4b6f-4cd1-a70c-ba2b27aaf1a8.json.gz, 2.
        """
        env = RiichiEnv(seed=42)
        # Seat 3 (West) is winner, Seat 0 (East) is Pao, Seat 2 (South) is Discarder.
        # Honba: 2.
        env.reset(oya=0, honba=2)

        # Setup Winner (Seat 3) hand
        # Needs to be close to agari
        env.hands[3] = [0, 4, 8, 12, 16, 20, 24, 28, 32, 120, 121, 122, 123]  # Junk
        # Actually let's use the log details or just a simple Yakuman setup.
        # Seat 3 hand: 3x 1m, 3x 2m, 3x 3m, 3x 4m, 2x 5m
        env.hands[3] = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16]
        # Establish Pao for Seat 3 (Winner) with Seat 0 (Pao)
        # Winner (Seat 3) has 2 dragon melds: White (124-126), Green (128-130)
        melds = env.melds
        melds[3] = [
            Meld(MeldType.Peng, [124, 125, 126], True),
            Meld(MeldType.Peng, [128, 129, 130], True),
        ]
        env.melds = melds
        # Hand (7 tiles): 3x 1m (0,1,2), 1x 2m (4), 2x Red (132,133), 1x Junk (99)
        hands = env.hands
        hands[3] = [0, 1, 2, 4, 99, 132, 133]
        env.hands = hands

        # Pao transition: Seat 3 calls Pon from Seat 0
        env.current_player = 0
        env.phase = Phase.WaitResponse
        env.last_discard = (0, 134)  # 3rd Red
        env.active_players = [3]
        env.current_claims = {3: [Action(ActionType.Pon, tile=134, consume_tiles=[132, 133])]}

        env.step({3: Action(ActionType.Pon, tile=134, consume_tiles=[132, 133])})

        assert env.pao[3].get(37) == 0  # Seat 3 won, Seat 0 responsible

        # After PON, P3 must discard. Hand: [0, 1, 2, 4, 5].
        # After PON, P3 must discard. Hand: [0, 1, 2, 4, 99].
        # Discard 99. Hand: [0, 1, 2, 4]. Triplet 1m + Single 2m. Wait on 2m.
        # Total tiles: 4 (hand) + 9 (3 melds) = 13.
        env.step({3: Action(ActionType.Discard, tile=99)})

        # Now Seat 2 discards winning tile (6 for 2m triplet)
        env.current_player = 2
        env.phase = Phase.WaitResponse
        env.last_discard = (2, 6)
        env.active_players = [3]
        env.current_claims = {3: [Action(ActionType.Ron, tile=6)]}

        # Result:
        # Winner: Seat 3 (Ko)
        # Discarder: Seat 2 (Ko)
        # Pao: Seat 0 (Oya)
        # Yakuman (32000) + 2 Honba (600) = 32600.
        # Discarder pays 16000.
        # Pao pays 16600.

        env.step({3: Action(ActionType.Ron, tile=6)})

        hora = next(m for m in reversed(env.mjai_log) if m["type"] == "hora")
        deltas = hora["deltas"]

        assert deltas[3] == 32600
        assert deltas[2] == -16000  # Discarder
        assert deltas[0] == -16600  # Pao. NOTE: In Tenhou, the Pao player pays for the honba.
        assert deltas[1] == 0
