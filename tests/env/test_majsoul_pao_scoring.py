import pytest

from riichienv import Action, ActionType, GameRule, Meld, MeldType, Phase, RiichiEnv


class TestPaoCompositeYakuman:
    def test_majsoul_pao_tsumo_composite(self) -> None:
        # Setup: Majsoul Rules
        rule = GameRule.default_mjsoul()
        env = RiichiEnv(rule=rule, game_mode="4p-red-single")
        env.reset()

        # Manually Override State for Test
        env.oya = 0
        env.set_scores([25000, 25000, 25000, 25000])

        # Daisangen + Tsuuiisou (All Honors)
        # Hand: Hatsu (Pon) + Chun (Pon) + East (Pon) + South (Pair)
        # Meld: Haku (Pon)
        # Total 14 tiles.

        p0_hand = [
            128,
            129,
            130,  # Hatsu
            132,
            133,
            134,  # Chun
            108,
            109,
            110,  # East
            112,  # South (Wait)
        ]

        current_hands = env.hands
        current_hands[0] = p0_hand
        env.hands = current_hands

        # Open Meld for Haku (124, 125, 126)
        m = Meld(MeldType.Peng, [124, 125, 126], True, 1)  # Called from P1

        current_melds = env.melds
        current_melds[0] = [m]
        env.melds = current_melds

        # Inject Pao: Player 2 liable for Daisangen (ID 37)
        id_daisangen = 37
        current_pao = env.pao
        current_pao[0] = {id_daisangen: 2}
        env.pao = current_pao

        # Set correct turn to P0
        env.current_player = 0
        # Winning tile (South - 113)
        env.drawn_tile = 113
        env.needs_tsumo = False
        env.phase = Phase.WaitAct

        # Execute Tsumo
        action = Action(ActionType.Tsumo)
        env.step({0: action})
        assert env.agari_results[0].yakuman

        # Expected: Double Yakuman (Daisangen + Tsuuiisou).
        # Dealer Double Yakuman = 96000.
        # Unit = 48000.

        # Majsoul Rule:
        # Pao (Daisangen 1x) = 48000. Paid by Pao Player (P2).
        # Normal (Tsuuiisou 1x) = 48000. Split normally (Dealer Tsumo -> All pay 16000).

        # P2 pays: 48000 (Pao) + 16000 (Normal Share) = 64000.
        # P1 pays: 16000.
        # P3 pays: 16000.
        # Winner (P0) gets: 96000.

        # NOTE: Tenhou Rule would be P2 pays ALL 96000.
        # verify difference: 64000 (Majsoul) != 96000 (Tenhou)

        assert env.score_deltas[0] == 96000
        assert env.score_deltas[1] == -16000
        assert env.score_deltas[2] == -64000
        assert env.score_deltas[3] == -16000

    def test_majsoul_pao_ron_composite(self) -> None:
        # Setup: Majsoul Rules
        rule = GameRule.default_mjsoul()
        env = RiichiEnv(rule=rule, game_mode="4p-red-single")
        env.reset()

        env.oya = 0
        env.set_scores([25000, 25000, 25000, 25000])

        # Player 0 Hand: Daisuushi (Big Four Winds) + Tsuuiisou (All Honors)
        # Total 3x Yakuman.
        # Pao on Daisuushi (2x) by Player 2.
        # Deal-in by Player 1.

        # Hand (P0):
        # East (Pon), South (Pon), West (Pon)
        # Wait on North (Pon) + Haku (Pair)

        p0_hand = [
            108,
            109,
            110,  # East
            112,
            113,
            114,  # South
            116,
            117,
            118,  # West
            124,  # Haku (Pair)
        ]

        current_hands = env.hands
        current_hands[0] = p0_hand
        env.hands = current_hands

        melds_p0 = [
            Meld(MeldType.Peng, [108, 109, 110], True, 1),  # East
            Meld(MeldType.Peng, [112, 113, 114], True, 1),  # South
            Meld(MeldType.Peng, [116, 117, 118], True, 1),  # West
        ]

        current_melds = env.melds
        current_melds[0] = melds_p0
        env.melds = current_melds

        p0_hand_reduced = [120, 121, 124, 124]  # North, North, Haku, Haku
        current_hands = env.hands
        current_hands[0] = p0_hand_reduced
        env.hands = current_hands

        id_daisuushi = 50
        current_pao = env.pao
        current_pao[0] = {id_daisuushi: 2}  # P2 liable for Daisuushi
        env.pao = current_pao

        # P1 discards North (122)
        env.current_player = 1

        # P1 setup
        hands_p1 = [122, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        current_hands = env.hands
        current_hands[1] = hands_p1
        env.hands = current_hands

        env.drawn_tile = 122
        env.needs_tsumo = False
        env.phase = Phase.WaitAct

        # Discard 122
        action = Action(ActionType.Discard, 122, [])
        env.step({1: action})

        if env.phase == Phase.WaitResponse:
            assert env.phase == Phase.WaitResponse

            action_ron = Action(ActionType.Ron, 122, [])
            env.step({0: action_ron})

            print(f"Debug: Winner Yaku: {env.agari_results[0].yaku}")
            print(f"Scores deltas: {env.score_deltas}")

            # Expected: Triple Yakuman (Daisuushi 2x + Tsuuiisou 1x) = 144000.

            # Majsoul Rule:
            # Pao Part (Daisuushi 2x = 96000): Split 50/50.
            # Pao pays 48000. Deal-in pays 48000.

            # Normal Part (Tsuuiisou 1x = 48000): Paid by Deal-in.
            # Deal-in pays 48000.

            # Total:
            # Pao (P2) pays: 48000.
            # Deal-in (P1) pays: 48000 + 48000 = 96000.

            # Difference from Tenhou:
            # Tenhou Composite (3x = 144000) -> Split 50/50 -> Each pays 72000.
            # Majsoul -> P2: 48000, P1: 96000. (Distinct).

            assert env.score_deltas[0] == 144000
            assert env.score_deltas[1] == -96000  # Deal-in (48k split + 48k normal)
            assert env.score_deltas[2] == -48000  # Pao (48k split)
            assert env.score_deltas[3] == 0
        else:
            pytest.fail(f"Did not enter WaitResponse phase. Current phase: {env.phase}")

    def test_majsoul_pao_ron_single(self) -> None:
        # Setup: Majsoul Rules
        rule = GameRule.default_mjsoul()
        env = RiichiEnv(rule=rule, game_mode="4p-red-single")
        env.reset()

        env.oya = 0
        env.set_scores([25000, 25000, 25000, 25000])

        # Simplify:
        # Hatsu Pon (Open), Chun Pon (Open), Haku Pon (Open)
        # Hand: 123m + 55m. Ron 5m.

        # Melds
        melds_p0 = [
            Meld(MeldType.Peng, [128, 129, 130], True, 1),  # Hatsu
            Meld(MeldType.Peng, [132, 133, 134], True, 1),  # Chun
            Meld(MeldType.Peng, [124, 125, 126], True, 1),  # Haku
        ]
        current_melds = env.melds
        current_melds[0] = melds_p0
        env.melds = current_melds

        # Hand
        p0_hand_reduced = [0, 4, 8, 16]  # 1m, 2m, 3m, 5m (Wait)
        current_hands = env.hands
        current_hands[0] = p0_hand_reduced
        env.hands = current_hands

        # Pao Daisangen
        id_daisangen = 37
        current_pao = env.pao
        current_pao[0] = {id_daisangen: 2}  # P2 liable
        env.pao = current_pao

        # P1 discards 5m (17)
        env.current_player = 1
        env.drawn_tile = 17
        env.phase = Phase.WaitAct

        # Discard 17
        action = Action(ActionType.Discard, 17, [])
        env.step({1: action})

        if env.phase == Phase.WaitResponse:
            action_ron = Action(ActionType.Ron, 17, [])
            env.step({0: action_ron})

            # Expected: Single Yakuman (48000).
            # Majsoul Single Pao Ron = Same as Tenhou.
            # Split 50/50.
            # Pao pays 24000. Deal-in pays 24000.

            assert env.score_deltas[0] == 48000
            assert env.score_deltas[1] == -24000
            assert env.score_deltas[2] == -24000
            assert env.score_deltas[3] == 0
