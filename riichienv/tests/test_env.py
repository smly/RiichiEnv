import json

from riichienv import RiichiEnv
from riichienv.action import Action, ActionType
from riichienv.agents import RandomAgent


class TestRiichiEnv:

    def test_new_events_api(self):
        env = RiichiEnv(seed=42)
        obs_dict = env.reset()

        p0_obs = obs_dict[0]
        # P0 should see start_game, start_kyoku, tsumo(0)
        # tsumo(0) tile is visible to P0
        initial_events = p0_obs.new_events()
        assert len(initial_events) == 3
        # Check masking
        assert json.loads(initial_events[2])["pai"] != "?"  # tsumo for self is visible

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
        p1_new = [json.loads(ev) for ev in p1_obs.new_events()]
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
        env = RiichiEnv(seed=123, game_type=1)
        agent = RandomAgent(seed=123)
        obs_dict = env.reset()

        steps = 0
        while not env.done() and steps < 1000:
            actions = {pid: agent.act(obs) for pid, obs in obs_dict.items()}
            obs_dict = env.step(actions)
            steps += 1

        # Either someone won or ryukyoku or aborted
        # Ryukyoku adds event logs
        last_ev = env.mjai_log[-1]["type"]
        assert last_ev in ["end_game", "end_kyoku", "hora"]
        # Note: In single-round mode, end_kyoku or hora are expected.

    def test_pon_claim(self):
        env = RiichiEnv(seed=42)
        env.reset()

        # Setup: P0 discards 1m (tile_id=0..3).
        # P1 needs a pair of 1m to Pon.
        # Let's force hands.

        # P0 hand: [0, 4, 8...] (doesn't matter)
        h = env.hands
        h[0] = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
        # P1 hand: has 1m pair (1, 2)
        h[1] = [1, 2, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45]
        h[1].sort()
        env.hands = h
        env.active_players = [0]
        env.current_player = 0

        env.drawn_tile = 52  # 5p (irrelevant)

        # P0 discards 1m (ID 0)
        env.step({0: Action(ActionType.DISCARD, tile=0)})

        # Should be in WaitResponse
        # Note: If logic not implemented, this might fail or skip
        # Logic check: 'pon_potential' needs to be implemented.
        assert env.phase == 1  # Phase.WaitResponse, currently checks Ron.
        # This assertion expects Pon support.

        assert 1 in env.active_players

        # P1 Legal actions Check
        obs = env.get_observations([1])[1]
        legals = obs.legal_actions()
        pon_actions = [a for a in legals if a.type == ActionType.PON]
        assert len(pon_actions) > 0

        # P1 performs PON
        # Manual: consume [1, 2]
        action = Action(ActionType.PON, tile=0, consume_tiles=[1, 2])
        env.step({1: action})

        # P1 should now be current player (WaitAct)
        assert env.current_player == 1
        assert env.phase == 0  # Phase.WaitAct

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

        h = env.hands
        h[0] = [tile_3m] + list(range(40, 40 + 12))
        h[1] = [tile_4m, tile_5m] + list(range(60, 60 + 11))
        h[1].sort()
        env.hands = h

        env.active_players = [0]
        env.current_player = 0

        env.drawn_tile = 100  # irrelevant

        # P0 Discards 3m
        env.step({0: Action(ActionType.DISCARD, tile=tile_3m)})

        # Should be in WaitResponse
        # P1 is next player (0->1 is Chi valid)
        assert env.phase == 1
        assert 1 in env.active_players

        # P1 Checks Legal
        obs = env.get_observations([1])[1]
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
        h = env.hands
        h[1] = p1_hand
        env.hands = h

        env.active_players = [0]
        env.current_player = 0

        # P0 Discards 5m (ID 17, matches pair for 5m)
        tile_5m_target = 17
        h = env.hands
        h[0] = [tile_5m_target] + list(range(40, 40 + 13))
        h[0].sort()
        env.hands = h

        env.step({0: Action(ActionType.DISCARD, tile=tile_5m_target)})

        assert env.phase == 1
        assert 1 in env.active_players

        # P1 Legal Ron
        obs = env.get_observations([1])[1]
        ron = [a for a in obs.legal_actions() if a.type == ActionType.RON]
        assert len(ron) > 0

        # Execute Ron
        env.step({1: Action(ActionType.RON, tile=tile_5m_target)})

        # Should log Hora (A single Ron doesn't necessarily end the game)
        ev_types = [ev["type"] for ev in env.mjai_log[-3:]]
        assert "hora" in ev_types
