from riichienv import Action, ActionType, Meld, MeldType, Phase, RiichiEnv, convert, parse_hand


class TestChankan:
    def test_chankan_ron(self):
        """
        Verify standard Chankan Ron when a player performs KAKAN.
        """
        # Use GameType=0 (Ikkyoku) so game ends after one hand
        env = RiichiEnv(seed=42, game_type=0)
        env.reset()

        # Player 0: Performs KAKAN of 1m
        h = env.hands
        h[0] = list(parse_hand("02346789m01234p")[0])
        env.hands = h

        m = env.melds
        m[0] = [Meld(MeldType.Peng, tiles=[0, 1, 2], opened=True)]
        env.melds = m

        env.drawn_tile = 3  # The 4th 1m

        # Player 1: Waits for 1m, has Red Dragon Pon for Yaku
        # Hand: 1m, 1m (4, 5) ...
        # Wait: 1m (Shanpon wait or similar)
        # Actually let's make it easy: 2m, 3m in hand, wait is 1m, 4m (Ryanmen).
        h = env.hands
        h[1] = list(parse_hand("23m11223344p")[0])
        env.hands = h
        # 10 tiles + 3 in meld = 13.
        m = env.melds
        m[1] = [Meld(MeldType.Peng, tiles=[132, 133, 134], opened=True)]  # Red Dragon
        env.melds = m

        env.current_player = 0
        env.phase = Phase.WaitAct
        env.active_players = [0]

        # Player 0 performs KAKAN
        kakan_action = Action(ActionType.KAKAN, tile=3, consume_tiles=[3])
        obs_dict = env.step({0: kakan_action})

        # Env should transition to WaitResponse for Player 1
        assert env.phase == Phase.WaitResponse, f"Expected WaitResponse, got {env.phase}"
        assert 1 in env.active_players
        assert 1 in obs_dict

        # Player 1 should have RON action
        legal_actions = obs_dict[1].legal_actions()
        ron_actions = [a for a in legal_actions if a.action_type == ActionType.RON]
        assert len(ron_actions) > 0, f"No RON actions found. Legal actions: {legal_actions}"
        assert ron_actions[0].tile == 3

        # Player 1 performs RON
        print("Executing RON action...")
        env.step({1: ron_actions[0]})
        print(f"Post-RON: is_done={env.is_done}, agari_results={env.agari_results.keys()}")

        # Check result
        assert env.is_done, "Env should be done after RON"
        assert 1 in env.agari_results
        res = env.agari_results[1]
        assert res.agari
        # Chankan yaku (ID 3)
        assert 3 in res.yaku

    def test_chankan_pass(self):
        """
        Verify that if Chankan is available but PASSed, the game proceeds with Rinshan draw.
        """
        env = RiichiEnv(seed=42, game_type=1)
        env.reset()

        # Same setup as test_chankan_ron
        m1_tiles = [0, 1, 2]
        h = env.hands
        h[0] = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52]
        env.hands = h
        m = env.melds
        m[0] = [Meld(MeldType.Peng, tiles=m1_tiles, opened=True)]
        env.melds = m
        env.drawn_tile = 3

        h = env.hands
        h[1] = [
            convert.mpsz_to_tid("2m"),
            convert.mpsz_to_tid("3m"),
            convert.mpsz_to_tid("1p"),
            convert.mpsz_to_tid("1p"),
            convert.mpsz_to_tid("2p"),
            convert.mpsz_to_tid("2p"),
            convert.mpsz_to_tid("3p"),
            convert.mpsz_to_tid("3p"),
            convert.mpsz_to_tid("4p"),
            convert.mpsz_to_tid("4p"),
        ]
        env.hands = h
        m = env.melds
        m[1] = [Meld(MeldType.Peng, tiles=[132, 133, 134], opened=True)]
        env.melds = m

        env.current_player = 0
        env.phase = Phase.WaitAct
        env.active_players = [0]

        kakan_action = Action(ActionType.KAKAN, tile=3, consume_tiles=[3])
        env.step({0: kakan_action})

        # Player 1 performs PASS
        env.step({1: Action(ActionType.PASS)})

        # Env should proceed with KAKAN execution and Rinshan Draw for Player 0
        assert env.phase == Phase.WaitAct
        assert env.current_player == 0
        assert len(env.melds[0]) == 1
        assert env.melds[0][0].meld_type == MeldType.Addgang
        assert env.drawn_tile is not None
        assert env.is_rinshan_flag

    def test_kokushi_ankan_ron(self):
        """
        Verify that Kokushi Musou can Ron on an ANKAN.
        """
        env = RiichiEnv(seed=42, game_type=1)
        env.reset()

        # Player 0: Performs ANKAN of 1z (East)
        e_tiles = [108, 109, 110]
        h = env.hands
        h[0] = e_tiles + [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 111]
        env.hands = h
        env.drawn_tile = 111

        # Player 1: Kokushi Musou waiting for 1z
        yaochu_mpsz = ["1m", "9m", "1p", "9p", "1s", "9s", "2z", "3z", "4z", "5z", "6z", "7z"]
        kokushi_tiles = [convert.mpsz_to_tid(m) for m in yaochu_mpsz]
        # Add a pair (e.g., 1m)
        kokushi_tiles.append(convert.mpsz_to_tid("1m") + 1)
        h = env.hands
        h[1] = sorted(kokushi_tiles)  # 13 tiles, waiting for 1z (108-111)
        env.hands = h

        env.current_player = 0
        env.phase = Phase.WaitAct
        env.active_players = [0]

        # Use tile 111 for Ankan
        ankan_action = Action(ActionType.ANKAN, tile=111, consume_tiles=[108, 109, 110, 111])
        obs_dict = env.step({0: ankan_action})

        # Should transition to WaitResponse for Player 1
        assert env.phase == Phase.WaitResponse
        assert 1 in env.active_players

        legal_actions = obs_dict[1].legal_actions()
        ron_actions = [a for a in legal_actions if a.action_type == ActionType.RON]
        assert len(ron_actions) > 0
        assert ron_actions[0].tile == 111

        env.step({1: ron_actions[0]})
        assert env.is_done
        assert env.agari_results[1].agari
        # Kokushi (ID 42)
        assert 42 in env.agari_results[1].yaku

    def test_non_kokushi_ankan_no_ron(self):
        """
        Verify that non-Kokushi hands cannot Ron on an ANKAN.
        """
        env = RiichiEnv(seed=42, game_type=1)
        env.reset()

        # Player 0: Performs ANKAN of 1z
        e_tiles = [108, 109, 110]
        h = env.hands
        h[0] = e_tiles + [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 111]
        env.hands = h
        env.drawn_tile = 111

        # Player 1: Normal hand waiting for 1z (Shanpon), has Red Dragon Pon for Yaku
        h = env.hands
        h[1] = [112, 113, 116, 116, 120, 120, 124, 124, 128, 128]  # Wait is 1z for Shanpon
        env.hands = h
        m = env.melds
        m[1] = [Meld(MeldType.Peng, tiles=[132, 133, 134], opened=True)]
        env.melds = m

        env.current_player = 0
        env.phase = Phase.WaitAct
        env.active_players = [0]

        ankan_action = Action(ActionType.ANKAN, tile=111, consume_tiles=[108, 109, 110, 111])
        env.step({0: ankan_action})

        # Should NOT transition to WaitResponse. Should immediately execute ANKAN.
        assert env.phase == Phase.WaitAct
        assert env.current_player == 0
        assert len(env.melds[0]) == 1
        assert env.melds[0][0].meld_type == MeldType.Angang
        assert env.drawn_tile is not None

    def test_ankan_generation(self):
        """
        Verify that ANKAN action is generated in the legal actions list.
        """
        env = RiichiEnv(seed=42)
        env.reset()

        # Setup: P0 has 3x 1m and draws the 4th
        h = env.hands
        # Hand must include drawn tile (14 tiles total)
        h[0] = [0, 1, 2, 3] + list(range(12, 12 + 10))
        env.hands = h
        env.drawn_tile = 3
        env.active_players = [0]
        env.current_player = 0
        env.phase = Phase.WaitAct

        obs = env.get_observations([0])[0]
        legals = obs.legal_actions()
        ankan = [a for a in legals if a.action_type == ActionType.ANKAN]
        assert len(ankan) > 0
        assert ankan[0].tile in [0, 1, 2, 3]
        assert sorted(ankan[0].consume_tiles) == [0, 1, 2, 3]

    def test_ankan_generation_riichi(self):
        """
        Verify that ANKAN action is generated even after Riichi declaration.
        """
        env = RiichiEnv(seed=42)
        env.reset()

        # Setup: P0 has 3x 1m and draws the 4th
        h = env.hands
        # Hand must include drawn tile (14 tiles total)
        h[0] = [0, 1, 2, 3] + list(range(12, 12 + 10))
        env.hands = h
        env.drawn_tile = 3
        env.active_players = [0]
        env.current_player = 0
        env.phase = Phase.WaitAct
        env.riichi_declared[0] = True

        obs = env.get_observations([0])[0]
        legals = obs.legal_actions()
        ankan = [a for a in legals if a.action_type == ActionType.ANKAN]
        assert len(ankan) > 0
        assert ankan[0].tile == 0  # In Riichi, must be the drawn tile (or equivalent type)
        assert sorted(ankan[0].consume_tiles) == [0, 1, 2, 3]
