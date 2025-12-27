from riichienv import RiichiEnv
from riichienv.action import Action, ActionType
from riichienv.agents import RandomAgent


class TestRiichiEnv:
    def test_initialization(self):
        env = RiichiEnv(seed=42)
        obs_dict = env.reset()

        # Only dealer (0) is actionable at start
        assert len(obs_dict) == 1
        assert 0 in obs_dict
        assert env.current_player == 0
        assert not env.done()

        # Check secure wall
        assert len(env.salt) == 16
        assert len(env.wall_digest) == 64  # SHA256 hex digest

        # Check hands
        obs = obs_dict[0]
        assert obs.player_id == 0
        # 13 tiles + 1 drawn for dealer (0)
        assert len(obs.hand) == 14

    def test_step_progression(self):
        env = RiichiEnv(seed=42)
        obs_dict = env.reset()  # Capture initial observations

        # Step 0: Player 0 discards
        # Need a valid action (Action object)
        # P0 has 14 tiles. Discard last one (13th index).
        # We need the TILE ID.
        obs = obs_dict[0]
        tile_to_discard = obs.hand[-1]

        obs_dict = env.step({0: Action(ActionType.DISCARD, tile=tile_to_discard)})

        # Should now be Player 1's turn
        assert env.current_player == 1
        assert not env.done()

        # Player 1 should have 14 tiles (13 + 1 drawn)
        assert len(env.hands[1]) == 13  # Internal hand
        # Drawn tile is separate
        assert env.drawn_tile is not None

        # Obs for P1 should include drawn tile -> 14
        assert len(obs_dict[1].hand) == 14

    def test_mjai_logs(self):
        env = RiichiEnv(seed=42)
        env.reset()

        # Check start events
        assert len(env.mjai_log) >= 3  # start_game, start_kyoku, tsumo
        assert env.mjai_log[0]["type"] == "start_game"
        assert env.mjai_log[1]["type"] == "start_kyoku"
        assert env.mjai_log[2]["type"] == "tsumo"
        assert env.mjai_log[2]["actor"] == 0

    def test_new_events_api(self):
        env = RiichiEnv(seed=42)
        obs_dict = env.reset()

        p0_obs = obs_dict[0]
        # P0 should see start_game, start_kyoku, tsumo(0)
        # tsumo(0) tile is visible to P0
        initial_events = p0_obs.new_events()
        assert len(initial_events) == 3
        # Check masking
        assert initial_events[2]["tile"] != "?"  # tsumo for self is visible

        # NOTE: P1 is not actionable, so not in obs_dict.
        # We cannot check P1's new_events() from obs_dict directly unless we cheat.
        # This test previously checked P1.
        # Now P1 is not in obs_dict.

        # Execute one step (P0 discards)
        # Need tile ID
        p0_tile = p0_obs.hand[0]  # Discard first tile
        obs_dict = env.step({0: Action(ActionType.DISCARD, tile=p0_tile)})

        # Now P1 should be actionable
        assert 1 in obs_dict
        assert 0 not in obs_dict

        p1_obs = obs_dict[1]
        p1_new = p1_obs.new_events()
        # P1 needs to catch up all events from start?
        # If this is the first time P1 received Observation, prev_events_size should be 0?
        # RiichiEnv tracks _player_event_counts.
        # When _get_observations called for P1:
        # P1 has not been called before. prev_size = 0.
        # So it should return ALL events (start_game..tsumo(0)..dahai(0)..tsumo(1)).

        # Let's count:
        # 1. start_game
        # 2. start_kyoku
        # 3. tsumo(0)
        # 4. dahai(0)
        # 5. tsumo(1) OR if P1 can Ron, it might stop at dahai(0) and wait for P1?

        # Check legal actions to understand state
        print(f"P1 Actionable. Legal: {p1_obs.legal_actions()}")

        # If P1 has drawn a tile, there should be a tsumo event.
        # If P1 is responding to a discard, no tsumo event for P1 yet.
        has_drawn = len(p1_obs.hand) % 3 == 2  # 14 tiles

        if has_drawn:
            assert len(p1_new) == 5
            assert p1_new[4]["type"] == "tsumo"
            assert p1_new[4]["actor"] == 1
        else:
            # Waiting for response (Ron)
            assert len(p1_new) == 4
            assert p1_new[-1]["type"] == "dahai"
            assert p1_new[-1]["actor"] == 0
            # Ensure P1 has legal action RON or PASS
            legals = p1_obs.legal_actions()
            assert any(a.type in [ActionType.RON, ActionType.PASS] for a in legals)

    def test_run_full_game_random(self):
        # Deterministic random agents
        env = RiichiEnv(seed=123)
        agent = RandomAgent(seed=123)
        obs_dict = env.reset()

        steps = 0
        while not env.done() and steps < 1000:
            actions = {pid: agent.act(obs) for pid, obs in obs_dict.items()}
            obs_dict = env.step(actions)
            steps += 1

        assert env.done()
        # Either someone won or ryukyoku or aborted
        # Ryukyoku adds event logs
        assert env.mjai_log[-1]["type"] == "end_game"

    def test_pon_claim(self):
        env = RiichiEnv(seed=42)
        env.reset()

        # Setup: P0 discards 1m (tile_id=0..3).
        # P1 needs a pair of 1m to Pon.
        # Let's force hands.

        # P0 hand: [0, 4, 8...] (doesn't matter)
        env.hands[0] = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
        # P1 hand: has 1m pair (1, 2)
        env.hands[1] = [1, 2, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45]
        env.hands[1].sort()
        env.active_players = [0]
        env.current_player = 0

        env.drawn_tile = 52  # 5p (irrelevant)

        # P0 discards 1m (ID 0)
        env.step({0: Action(ActionType.DISCARD, tile=0)})

        # Should be in WAIT_RESPONSE
        # Note: If logic not implemented, this might fail or skip
        # Logic check: 'pon_potential' needs to be implemented.
        assert env.phase == 1  # Phase.WAIT_RESPONSE, currently checks Ron.
        # This assertion expects Pon support.

        assert 1 in env.active_players

        # P1 Legal actions Check
        obs = env._get_observations([1])[1]
        legals = obs.legal_actions()
        pon_actions = [a for a in legals if a.type == ActionType.PON]
        assert len(pon_actions) > 0

        # P1 performs PON
        # Manual: consume [1, 2]
        action = Action(ActionType.PON, tile=0, consume_tiles=[1, 2])
        env.step({1: action})

        # P1 should now be current player (WAIT_ACT)
        assert env.current_player == 1
        assert env.phase == 0  # Phase.WAIT_ACT

        # Check MJAI log
        last_ev = env.mjai_log[-1]
        assert last_ev["type"] == "pon"
        assert last_ev["actor"] == 1
        assert last_ev["target"] == 0  # From P0

    def test_chi_claim(self):
        env = RiichiEnv(seed=42)
        env.reset()

        # Setup: P0 discards 3m (ID 8,9,10,11).
        # P1 (Right/Next) has 4m, 5m.
        # 3m, 4m, 5m sequence.

        tile_3m = 8
        tile_4m = 12
        tile_5m = 16

        env.hands[0] = [tile_3m] + list(range(40, 40 + 12))
        env.hands[1] = [tile_4m, tile_5m] + list(range(60, 60 + 11))
        env.hands[1].sort()

        env.active_players = [0]
        env.current_player = 0

        env.drawn_tile = 100  # irrelevant

        # P0 Discards 3m
        env.step({0: Action(ActionType.DISCARD, tile=tile_3m)})

        # Should be in WAIT_RESPONSE
        # P1 is next player (0->1 is Chi valid)
        assert env.phase == 1
        assert 1 in env.active_players

        # P1 Checks Legal
        obs = env._get_observations([1])[1]
        legals = obs.legal_actions()
        chi_actions = [a for a in legals if a.type == ActionType.CHI]
        assert len(chi_actions) > 0

        # P1 performs CHI
        action = Action(ActionType.CHI, tile=tile_3m, consume_tiles=[tile_4m, tile_5m])
        env.step({1: action})

        # P1 current player
        assert env.current_player == 1
        assert env.phase == 0

        last_ev = env.mjai_log[-1]
        assert last_ev["type"] == "chi"

    def test_ron_claim(self):
        env = RiichiEnv(seed=42)
        env.reset()

        # Setup P0 discards
        # P1 tenpai for 1m

        # 1m,1m,1m (0,1,2), 2m,2m,2m (4,5,6), 3m,3m,3m (8,9,10), 4m,4m,4m (12,13,14), 5m (16)
        # Wait 5m.
        p1_hand = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16]
        p1_hand = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16]
        env.hands[1] = p1_hand

        env.active_players = [0]
        env.current_player = 0

        # P0 Discards 5m (ID 17, matches pair for 5m)
        tile_5m_target = 17
        env.hands[0] = [tile_5m_target] + list(range(40, 40 + 13))
        env.hands[0].sort()

        env.step({0: Action(ActionType.DISCARD, tile=tile_5m_target)})

        assert env.phase == 1
        assert 1 in env.active_players

        # P1 Legal Ron
        obs = env._get_observations([1])[1]
        ron = [a for a in obs.legal_actions() if a.type == ActionType.RON]
        assert len(ron) > 0

        # Execute Ron
        env.step({1: Action(ActionType.RON, tile=tile_5m_target)})

        assert env.done()
