from riichienv import Action, ActionType, GameRule, Meld, MeldType, Phase, RiichiEnv, convert, parse_hand

from ..helper import helper_setup_env


class TestChankan:
    def test_chankan_ron(self):
        """
        Verify standard Chankan Ron when a player performs Kakan (Add Kan).
        """
        env = helper_setup_env(
            hands=[
                [3] + list(parse_hand("02346789m01234p")[0]),
                list(parse_hand("23m11p234s456s")[0]),
                [],
                [],
            ],
            melds=[
                [Meld(MeldType.Peng, tiles=[0, 1, 2], opened=True)],
                [Meld(MeldType.Peng, tiles=list(parse_hand("999m")[0]), opened=True)],
                [],
                [],
            ],
            game_mode=0,  # game ends after one hand
            drawn_tile=3,
            current_player=0,
            phase=Phase.WaitAct,
            active_players=[0],
        )

        # Player 0 performs Kakan
        # Pon tiles are [0, 1, 2]
        kakan_action = Action(ActionType.Kakan, tile=3, consume_tiles=[0, 1, 2])
        obs_dict = env.step({0: kakan_action})

        # Env should transition to WaitResponse for Player 1

        assert env.phase == Phase.WaitResponse, f"Expected WaitResponse, got {env.phase}"
        assert env.active_players == [1]
        assert list(obs_dict.keys()) == [1]

        # Player 1 should have Ron action
        legal_actions = obs_dict[1].legal_actions()
        ron_actions = [a for a in legal_actions if a.action_type == ActionType.Ron]
        assert len(ron_actions) > 0, f"No Ron actions found. Legal actions: {legal_actions}"
        assert ron_actions[0].tile == 3

        # Player 1 performs Ron
        env.step({1: ron_actions[0]})

        # Check result
        assert env.done(), "Env should be done after Ron"
        assert 1 in env.agari_results
        res = env.agari_results[1]
        assert res.agari
        # Chankan (ID 3)
        assert res.yaku == [3]

    def test_chankan_pass(self):
        """
        Verify that if Chankan is available but PASSed, the game proceeds with Rinshan draw.
        """
        env = RiichiEnv(seed=42, game_mode=1)
        env.reset()

        # Same setup as test_chankan_ron
        m1_tiles = [0, 1, 2]
        h = env.hands
        h[0] = [3, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52]
        env.hands = h
        m = env.melds
        m[0] = [Meld(MeldType.Peng, tiles=m1_tiles, opened=True)]
        env.melds = m
        env.drawn_tile = 3

        h = env.hands
        h[1] = list(parse_hand("23m11p234s456s")[0])
        env.hands = h
        m = env.melds
        m[1] = [Meld(MeldType.Peng, tiles=[132, 133, 134], opened=True)]
        env.melds = m

        env.current_player = 0
        env.phase = Phase.WaitAct
        env.active_players = [0]

        kakan_action = Action(ActionType.Kakan, tile=3, consume_tiles=[0, 1, 2])
        env.step({0: kakan_action})

        # Player 1 performs PASS
        env.step({1: Action(ActionType.Pass)})

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
        rule = GameRule.default_mjsoul()

        env = RiichiEnv(seed=42, game_mode=1, rule=rule)
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
        ankan_action = Action(ActionType.Ankan, tile=111, consume_tiles=[108, 109, 110, 111])
        obs_dict = env.step({0: ankan_action})

        # Should transition to WaitResponse for Player 1
        assert env.phase == Phase.WaitResponse
        assert 1 in env.active_players

        legal_actions = obs_dict[1].legal_actions()
        ron_actions = [a for a in legal_actions if a.action_type == ActionType.Ron]
        assert len(ron_actions) > 0
        assert ron_actions[0].tile == 111

        env.step({1: ron_actions[0]})
        assert env.is_done
        assert env.agari_results[1].agari
        # Kokushi (ID 42)
        assert 42 in env.agari_results[1].yaku

    def test_kokushi_ankan_ron_tenhou(self):
        """
        Verify that Kokushi Musou cannot Ron on an ANKAN under Tenhou rules.
        """
        rule = GameRule.default_tenhou()

        env = RiichiEnv(seed=42, game_mode=1, rule=rule)
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
        ankan_action = Action(ActionType.Ankan, tile=111, consume_tiles=[108, 109, 110, 111])
        env.step({0: ankan_action})

        assert env.phase == Phase.WaitAct
        assert [0] == env.active_players

    def test_non_kokushi_ankan_no_ron(self):
        """
        Verify that non-Kokushi hands cannot Ron on an ANKAN.
        """
        env = RiichiEnv(seed=42, game_mode=1)
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

        ankan_action = Action(ActionType.Ankan, tile=111, consume_tiles=[108, 109, 110, 111])
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
        ankan = [a for a in legals if a.action_type == ActionType.Ankan]
        assert len(ankan) > 0
        assert ankan[0].tile in [0, 1, 2, 3]
        assert sorted(ankan[0].consume_tiles) == [0, 1, 2, 3]

    def test_ankan_generation_riichi(self) -> None:
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
        ankan = [a for a in legals if a.action_type == ActionType.Ankan]
        assert len(ankan) > 0
        assert ankan[0].tile == 0  # In Riichi, must be the drawn tile (or equivalent type)
        assert sorted(ankan[0].consume_tiles) == [0, 1, 2, 3]

    def test_chankan_stale_claims_repro(self) -> None:
        """
        Reproduce Match 27 Step 267 inconsistency.
        P2 discards 7p (60) -> P0 has Pon offer with 61, 62.
        All pass.
        P3 tsutmos 6p (59), kakans 59. (P3 has Pon meld 56, 57, 58).
        P0 has 6p sequence wait (4p-5p).
        P0 should have Ron (Chankan) offer on 59.
        """
        # 4p: 48, 49, 50, 51
        # 5p: 52, 53, 54, 55
        # 6p: 56, 57, 58, 59
        # 7p: 60, 61, 62, 63
        env = helper_setup_env(
            hands=[
                [4, 5, 6, 8, 9, 10, 12, 13, 14, 61, 62, 49, 53],
                [],
                [63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75],
                [],
            ],
            melds=[
                [],
                [],
                [],
                [Meld(MeldType.Peng, [56, 57, 58], True)],
            ],
            discards=[[60], [], [], []],
            current_player=2,
            phase=Phase.WaitAct,
        )

        # 1. P2 discards 7p (63). (P0 has 61, 62).
        # Wait, P2 discard 7p (63). P0 (61, 62) should have Pon.
        obs = env.step({2: Action(ActionType.Discard, 63)})
        assert 0 in obs

        # 2. All pass.
        # Strict turn validation requires only active players to act.
        env.step({pid: Action(ActionType.Pass) for pid in obs.keys()})
        assert env.current_player == 3

        # 3. P3 draws 6p (59) and kakans.
        env.drawn_tile = 59
        h3 = env.hands
        h3[3] = [59]
        env.hands = h3

        obs = env.step({3: Action(ActionType.Kakan, 59, [56, 57, 58])})

        assert 0 in obs, f"P0 should be active for Chankan. Phase: {env.phase}, Active: {env.active_players}"
        action_types = [a.action_type for a in obs[0].legal_actions()]
        assert ActionType.Ron in action_types, f"P0 should have Ron offered. Actions: {action_types}"
