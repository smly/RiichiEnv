from riichienv import Action, ActionType, GameRule, Meld, MeldType, Phase, RiichiEnv


class TestPaoCompositeYakuman:
    def test_majsoul_pao_tsumo_composite(self) -> None:
        # Setup: Majsoul Rules
        rule = GameRule.default_mjsoul()
        env = RiichiEnv(seed=1, rule=rule, game_mode="4p-red-single")
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
        m = Meld(MeldType.Pon, [124, 125, 126], True, 1)  # Called from P1

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
        assert env.win_results[0].yakuman

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
        env = RiichiEnv(seed=1, rule=rule, game_mode="4p-red-single")
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
            Meld(MeldType.Pon, [108, 109, 110], True, 1),  # East
            Meld(MeldType.Pon, [112, 113, 114], True, 1),  # South
            Meld(MeldType.Pon, [116, 117, 118], True, 1),  # West
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

        assert env.phase == Phase.WaitResponse

        action_ron = Action(ActionType.Ron, 122, [])
        env.step({0: action_ron})

        # Expected: Triple Yakuman (Daisuushi 2x + Tsuuiisou 1x) = 144000.

        # 4P MjSoul Ron PAO: PAO portion only split 50/50.
        # PAO portion = Daisuushi 2x * 48000 = 96000. Split 50/50 = 48000.
        # Deal-in (P1) pays: 144000 - 48000 = 96000.
        # Pao (P2) pays: 48000.

        assert env.score_deltas[0] == 144000
        assert env.score_deltas[1] == -96000  # Deal-in (total - pao_half)
        assert env.score_deltas[2] == -48000  # Pao (half of PAO portion)
        assert env.score_deltas[3] == 0

    def test_majsoul_pao_ron_single(self) -> None:
        # Setup: Majsoul Rules
        rule = GameRule.default_mjsoul()
        env = RiichiEnv(seed=1, rule=rule, game_mode="4p-red-single")
        env.reset()

        env.oya = 0
        env.set_scores([25000, 25000, 25000, 25000])

        # Simplify:
        # Hatsu Pon (Open), Chun Pon (Open), Haku Pon (Open)
        # Hand: 123m + 55m. Ron 5m.

        # Melds
        melds_p0 = [
            Meld(MeldType.Pon, [128, 129, 130], True, 1),  # Hatsu
            Meld(MeldType.Pon, [132, 133, 134], True, 1),  # Chun
            Meld(MeldType.Pon, [124, 125, 126], True, 1),  # Haku
        ]
        current_melds = env.melds
        current_melds[0] = melds_p0
        env.melds = current_melds

        # Hand (Shanpon Wait: 1m Pair + 2m Pair)
        # Wait on 2m (4)
        p0_hand_reduced = [0, 1, 4, 4]  # 1m, 1m, 2m, 2m
        current_hands = env.hands
        current_hands[0] = p0_hand_reduced
        env.hands = current_hands

        # Pao Daisangen
        id_daisangen = 37
        current_pao = env.pao
        current_pao[0] = {id_daisangen: 2}  # P2 liable
        env.pao = current_pao

        # P1 discards 2m (4)
        env.current_player = 1

        # Ensure P1 has the tile in hand (required for legal discard check)
        hands_p1 = [4, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        current_hands = env.hands
        current_hands[1] = hands_p1
        env.hands = current_hands

        env.drawn_tile = 4
        env.needs_tsumo = False
        env.phase = Phase.WaitAct

        # Discard 4
        action = Action(ActionType.Discard, 4, [])
        env.step({1: action})

        assert env.phase == Phase.WaitResponse
        action_ron = Action(ActionType.Ron, 4, [])
        env.step({0: action_ron})

        # Expected: Single Yakuman (48000).
        # Majsoul Single Pao Ron = Same as Tenhou.
        # Split 50/50.
        # Pao pays 24000. Deal-in pays 24000.

        assert env.score_deltas[0] == 48000
        assert env.score_deltas[1] == -24000
        assert env.score_deltas[2] == -24000
        assert env.score_deltas[3] == 0

    def test_mjsoul_4p_ron_pao_real_record(self) -> None:
        """Real MjSoul game 251122-9051f1e9, round 11.

        P2 (ko) wins by ron on P0's discard of 2z (South).
        Hand: [2z], Melds: [kezi(5z), kezi(3z), kezi(6z), kezi(7z)]
        Yakuman: Daisangen (PAO→P3) + Tsuuiisou = double yakuman = 64000.
        PAO-portion-only 50/50: P0 (deal-in) pays 48000, P3 (PAO) pays 16000.
        """
        rule = GameRule.default_mjsoul()
        env = RiichiEnv(seed=1, rule=rule, game_mode="4p-red-single")
        env.reset()

        env.oya = 0
        env.set_scores([25000, 25000, 25000, 25000])

        # P2 hand: just the pair wait tile (2z = South)
        # Tile IDs: South = 112-115
        p2_hand = [112]

        current_hands = env.hands
        current_hands[2] = p2_hand
        env.hands = current_hands

        # P2 melds: 4 open Pon (all honor tiles)
        melds_p2 = [
            Meld(MeldType.Pon, [124, 125, 126], True, 3),  # 5z Haku - called from P3
            Meld(MeldType.Pon, [116, 117, 118], True, 1),  # 3z West
            Meld(MeldType.Pon, [128, 129, 130], True, 1),  # 6z Hatsu - called from P3
            Meld(MeldType.Pon, [132, 133, 134], True, 3),  # 7z Chun - called from P3
        ]

        current_melds = env.melds
        current_melds[2] = melds_p2
        env.melds = current_melds

        # PAO: P3 liable for Daisangen (yaku ID 37) on P2
        id_daisangen = 37
        current_pao = env.pao
        current_pao[2] = {id_daisangen: 3}
        env.pao = current_pao

        # P0 discards 2z (South = 113)
        env.current_player = 0

        hands_p0 = [113, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        current_hands = env.hands
        current_hands[0] = hands_p0
        env.hands = current_hands

        env.drawn_tile = 113
        env.needs_tsumo = False
        env.phase = Phase.WaitAct

        action = Action(ActionType.Discard, 113, [])
        env.step({0: action})

        assert env.phase == Phase.WaitResponse

        action_ron = Action(ActionType.Ron, 113, [])
        env.step({2: action_ron})

        # Double yakuman (Daisangen 1x + Tsuuiisou 1x) = 64000 (ko).
        # MjSoul PAO-portion-only: split_base = 1 * 32000 = 32000.
        # PAO pays 16000 (= 32000/2). Deal-in pays 48000 (= 64000 - 16000).

        assert env.score_deltas[0] == -48000  # Deal-in
        assert env.score_deltas[1] == 0
        assert env.score_deltas[2] == 64000   # Winner
        assert env.score_deltas[3] == -16000  # PAO

    def test_mjsoul_4p_ron_pao_real_record_with_riichi(self) -> None:
        """Same as above but with 1 riichi stick on the table.

        Winner (P2) should also collect the riichi deposit: 64000 + 1000 = 65000.
        """
        rule = GameRule.default_mjsoul()
        env = RiichiEnv(seed=1, rule=rule, game_mode="4p-red-single")
        env.reset()

        env.oya = 0
        env.set_scores([25000, 25000, 25000, 25000])
        env.riichi_sticks = 1

        p2_hand = [112]
        current_hands = env.hands
        current_hands[2] = p2_hand
        env.hands = current_hands

        melds_p2 = [
            Meld(MeldType.Pon, [124, 125, 126], True, 3),
            Meld(MeldType.Pon, [116, 117, 118], True, 1),
            Meld(MeldType.Pon, [128, 129, 130], True, 1),
            Meld(MeldType.Pon, [132, 133, 134], True, 3),
        ]
        current_melds = env.melds
        current_melds[2] = melds_p2
        env.melds = current_melds

        id_daisangen = 37
        current_pao = env.pao
        current_pao[2] = {id_daisangen: 3}
        env.pao = current_pao

        env.current_player = 0
        hands_p0 = [113, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        current_hands = env.hands
        current_hands[0] = hands_p0
        env.hands = current_hands

        env.drawn_tile = 113
        env.needs_tsumo = False
        env.phase = Phase.WaitAct

        action = Action(ActionType.Discard, 113, [])
        env.step({0: action})
        assert env.phase == Phase.WaitResponse

        action_ron = Action(ActionType.Ron, 113, [])
        env.step({2: action_ron})

        # 64000 + 1000 riichi deposit = 65000 for winner.
        # PAO and deal-in amounts unchanged.
        assert env.score_deltas[0] == -48000
        assert env.score_deltas[1] == 0
        assert env.score_deltas[2] == 65000
        assert env.score_deltas[3] == -16000
