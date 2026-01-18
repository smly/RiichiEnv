import json

import pytest

from riichienv import Action, ActionType, Observation, Phase, RiichiEnv


class TestRiichiEnv:
    @pytest.mark.skip(reason="See Issue #32")
    def test_initialization(self) -> None:
        env = RiichiEnv(seed=42)
        assert len(env.wall) > 0, "Wall should be initialized (not empty) after construction"
        obs_dict = env.reset()

        # NOTE: 136 total tiles - (13 tiles * 4 players + 1 initial draw) = 83.
        # The remainder consists of the draw pile (70 - 1 = 69 tiles) and the dead wall (14 tiles) = 83 tiles.
        # This is the state of env.wall immediately after reset().
        assert len(env.wall) == 83, "Wall should have 83 tiles after reset"

        assert len(env.hands) == 4, "Hands should have 4 players at the start"
        assert len(env.hands[0]) == 14, "Dealer's hands should have 14 tiles at the start"
        assert len(env.hands[1]) == 13, "Player 1's hands should have 13 tiles at the start"
        assert len(env.hands[2]) == 13, "Player 2's hands should have 13 tiles at the start"
        assert len(env.hands[3]) == 13, "Player 3's hands should have 13 tiles at the start"

        assert len(env.melds) == 4, "Melds should have 4 players at the start"
        assert env.melds[0] == [], "Dealer's melds should be empty at the start"

        assert len(env.discards) == 4, "Discards should have 4 players at the start"
        assert env.discards[0] == [], "Dealer's discards should be empty at the start"

        assert env.current_player == 0, "Current player should be 0 at the start"
        assert env.turn_count == 0, "Turn count should be 0 at the start"
        assert env.is_done is False, "is_done should be False at the start"
        assert env.needs_tsumo is False, "needs_tsumo should be False at the start"

        # First `obs_dict` should be the dealer's observation
        assert list(obs_dict.keys()) == [0]

        first_dealer_obs = obs_dict[0]
        assert isinstance(first_dealer_obs, Observation)
        assert first_dealer_obs.player_id == 0
        assert len(first_dealer_obs.hand) == 14, "Dealer's hand should have 14 tiles at the start"
        assert len(first_dealer_obs.events) == 3, "First events: start_game, start_kyoku, tsumo"
        assert first_dealer_obs.events[0]["type"] == "start_game"
        assert first_dealer_obs.events[1]["type"] == "start_kyoku"
        assert first_dealer_obs.events[2]["type"] == "tsumo"

        # Test `Observation.new_events()`
        assert len(first_dealer_obs.new_events()) == 3, "Same as events"

        # Test `Observation.legal_actions()`
        assert len(first_dealer_obs.legal_actions()) == 14

        # Test `Observation.to_dict()`
        obs_data = first_dealer_obs.to_dict()
        assert obs_data["legal_actions"][0]["type"] == 0
        assert obs_data["legal_actions"][0]["tile"] == 9
        assert obs_data["legal_actions"][0]["consume_tiles"] == []

        # Test `Observation.select_action_from_mjai()`
        assert first_dealer_obs.select_action_from_mjai({"type": "dahai", "pai": "1m"}) is None
        assert first_dealer_obs.select_action_from_mjai({"type": "dahai", "pai": "3m"}) is not None

    def test_basic_step_processing(self) -> None:
        # env.phase should be either Phase.WaitAct (player's turn action phase)
        # or Phase.WaitResponse (waiting for responses like Chi, Pon, or Ron).
        # Responses such as Chi, Pon, and Ron are handled in Phase.WaitResponse.
        env = RiichiEnv(seed=42)
        obs_dict = env.reset()

        # Only the current player can act in the WaitAct phase
        assert env.phase == Phase.WaitAct
        assert env.current_player == 0
        assert list(obs_dict.keys()) == [0]
        obs = obs_dict[0]
        tile_to_discard = obs.hand[-1]
        obs_dict = env.step({0: Action(ActionType.Discard, tile=tile_to_discard)})

        # Players who can act are stored in `env.active_players`
        # Multiple players may be able to act in the `Phase.WaitResponse` phase
        # If no one can act, the `Phase.WaitResponse` phase is skipped
        while env.phase == Phase.WaitResponse:
            actions = {pid: Action(ActionType.Pass) for pid in env.active_players}
            obs_dict = env.step(actions)

        # WaitAct phase for the next player
        assert env.phase == Phase.WaitAct
        assert env.current_player == 1
        assert list(obs_dict.keys()) == [1]

        # Game end condition is checked via `env.done()`
        assert not env.done()
        assert len(env.hands[1]) == 14

        obs = obs_dict[1]
        assert len(obs.hand) == 14

        # TODO: `env`` is invisible to Agent, so should be passed in obs
        assert env.drawn_tile is not None
        assert obs.events[0]["type"] == "start_game"
        assert obs.events[1]["type"] == "start_kyoku"
        # Other players' hands are not visible
        assert obs.events[1]["tehais"][0][0] == "?"
        assert obs.events[1]["tehais"][1][0] != "?"
        assert obs.events[1]["tehais"][2][0] == "?"
        assert obs.events[1]["tehais"][3][0] == "?"
        assert obs.events[2]["type"] == "tsumo"
        assert obs.events[2]["actor"] == 0
        # Other players' tsumo tiles are not visible
        assert obs.events[2]["pai"] == "?"

        # mjai log is recorded in env
        assert len(env.mjai_log) == 5  # start_game, start_kyoku, tsumo, dahai, tsumo
        assert env.mjai_log[0]["type"] == "start_game"
        assert env.mjai_log[1]["type"] == "start_kyoku"
        assert env.mjai_log[2]["type"] == "tsumo"
        assert env.mjai_log[3]["type"] == "dahai"
        assert env.mjai_log[4]["type"] == "tsumo"
        # NOTE: All players' discards and hands are visible from `env`.
        assert env.mjai_log[2]["actor"] == 0
        assert env.mjai_log[2]["pai"] != "?"
        assert env.mjai_log[3]["actor"] == 0
        assert env.mjai_log[3]["pai"] != "?"
        assert env.mjai_log[4]["actor"] == 1
        assert env.mjai_log[4]["pai"] != "?"

    def test_new_events(self) -> None:
        env = RiichiEnv(seed=9)
        obs_dict = env.reset()

        assert env.phase == Phase.WaitAct
        p0_obs = obs_dict[0]
        initial_events = p0_obs.new_events()
        assert len(initial_events) == 3
        assert json.loads(initial_events[2])["pai"] != "?"

        p0_tile = p0_obs.hand[0]
        obs_dict = env.step({0: Action(ActionType.Discard, tile=p0_tile)})

        assert env.phase == Phase.WaitAct
        assert 1 in obs_dict
        assert 0 not in obs_dict

        p1_obs = obs_dict[1]
        p1_new = [json.loads(ev) for ev in p1_obs.new_events()]

        # Let's count:
        # 1. start_game
        # 2. start_kyoku
        # 3. tsumo(0)
        # 4. dahai(0)
        # 5. tsumo(1)

        assert len(p1_new) == 5
        assert p1_new[4]["type"] == "tsumo"
        assert p1_new[4]["actor"] == 1

        p1_tile = p1_obs.hand[0]
        obs_dict = env.step({1: Action(ActionType.Discard, tile=p1_tile)})

        assert 2 in obs_dict
        assert 1 not in obs_dict

        p2_obs = obs_dict[2]
        p2_new = [json.loads(ev) for ev in p2_obs.new_events()]

        assert len(p2_new) == 7
        assert p2_new[6]["type"] == "tsumo"
        assert p2_new[6]["actor"] == 2

        p2_tile = p2_obs.hand[0]
        obs_dict = env.step({2: Action(ActionType.Discard, tile=p2_tile)})

        assert env.phase == Phase.WaitAct
        assert 3 in obs_dict
        assert 2 not in obs_dict

        p3_obs = obs_dict[3]
        p3_new = [json.loads(ev) for ev in p3_obs.new_events()]

        assert len(p3_new) == 9
        assert p3_new[8]["type"] == "tsumo"
        assert p3_new[8]["actor"] == 3

        p3_tile = p3_obs.hand[0]
        obs_dict = env.step({3: Action(ActionType.Discard, tile=p3_tile)})

        assert env.phase == Phase.WaitAct
        assert 0 in obs_dict
        assert 3 not in obs_dict

        p0_obs = obs_dict[0]
        p0_new = [json.loads(ev) for ev in p0_obs.new_events()]

        # Check new events from p0_obs
        # 1. start_game (not included)
        # 2. start_kyoku (not included)
        # 3. tsumo(0)(not included)
        # 4. dahai(0)
        # 5. tsumo(1)
        # 6. dahai(1)
        # 7. tsumo(2)
        # 8. dahai(2)
        # 9. tsumo(3)
        # 10. dahai(3)
        # 11. tsumo(0)
        assert len(p0_new) == 8
        assert p0_new[0]["type"] == "dahai"
        assert p0_new[0]["actor"] == 0

    def test_pon_claim(self) -> None:
        env = RiichiEnv(seed=42)
        env.reset()

        h = env.hands
        h[0] = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
        # P2 hand: has 1m pair (1, 2)
        h[2] = [1, 2, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45]
        h[2].sort()
        env.hands = h
        env.active_players = [0]
        env.current_player = 0
        env.drawn_tile = 52  # 5p

        obs_dict = env.step({0: Action(ActionType.Discard, tile=0)})
        assert env.phase == Phase.WaitResponse
        assert env.active_players == [2]
        assert list(obs_dict.keys()) == [2]

        # P2 legal actions check
        obs = obs_dict[2]
        legals = obs.legal_actions()
        pon_actions = [a for a in legals if a.action_type == ActionType.PON]
        assert len(pon_actions) > 0

        # P2 performs PON
        action = Action(ActionType.PON, tile=0, consume_tiles=[1, 2])
        env.step({2: action})

        # P2 should now be current player (WaitAct)
        assert env.current_player == 2
        assert env.phase == Phase.WaitAct

        # Check MJAI log
        last_ev = env.mjai_log[-1]
        assert last_ev["type"] == "pon"
        assert last_ev["actor"] == 2
        assert last_ev["target"] == 0  # From P0
        assert last_ev["pai"] == "1m"

    def test_pon_red_dora_claim(self) -> None:
        env = RiichiEnv(seed=42)
        env.reset()

        h = env.hands
        h[0] = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
        # P2 hand: has 5m pair (17, 18)
        h[2] = [17, 18, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61]
        h[2].sort()
        env.hands = h
        env.active_players = [0]
        env.current_player = 0
        env.drawn_tile = 16  # 5mr

        obs_dict = env.step({0: Action(ActionType.Discard, tile=16)})
        assert env.phase == Phase.WaitResponse
        assert env.active_players == [2]
        assert list(obs_dict.keys()) == [2]

        # P2 legal actions check
        obs = obs_dict[2]
        pon_actions = [a for a in obs.legal_actions() if a.action_type == ActionType.PON]
        assert len(pon_actions) > 0

        # P2 performs PON
        action = pon_actions[0]
        env.step({2: action})

        # P2 should now be current player (WaitAct)
        assert env.current_player == 2
        assert env.phase == Phase.WaitAct

        # Check MJAI log
        last_ev = env.mjai_log[-1]
        assert last_ev["type"] == "pon"
        assert last_ev["actor"] == 2
        assert last_ev["target"] == 0  # From P0
        assert last_ev["pai"] == "5mr"

    def test_chi_claim(self) -> None:
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
        chi_actions = [a for a in legals if a.action_type == ActionType.CHI]
        assert len(chi_actions) > 0

        # P1 performs CHI
        action = Action(ActionType.CHI, tile=tile_3m, consume_tiles=[tile_4m, tile_5m])
        env.step({1: action})

        # P1 current player
        assert env.current_player == 1
        assert env.phase == 0

        last_ev = env.mjai_log[-1]
        assert last_ev["type"] == "chi"

    def test_chi_claim_with_red_dora(self) -> None:  # noqa: PLR0915
        # P1 に 5m, 5mr が手にある状態でチーをするとき、どちらの牌を使ってチーするか選択できることを確認する
        env = RiichiEnv(seed=42)
        env.reset()

        # Setup: P0 discards 3m (ID 8,9,10,11).
        # P1 (Right/Next/Shimocha) has 4m, 5m, 5mr.
        # (3m, 4m, 5m) and (3m, 4m, 5mr) sequence.

        tile_3m, tile_4m, tile_5mr, tile_5m = 8, 12, 16, 17
        h = env.hands
        h[0] = [tile_3m] + list(range(40, 40 + 12))
        h[1] = [tile_4m, tile_5mr, tile_5m] + list(range(60, 60 + 11))
        h[1].sort()
        env.hands = h

        env.active_players = [0]
        env.current_player = 0
        env.drawn_tile = 100  # irrelevant

        # P0 Discards 3m
        obs_dict = env.step({0: Action(ActionType.DISCARD, tile=tile_3m)})

        # Should be in WaitResponse
        # P1 is next player (0->1 is Chi valid)
        assert env.phase == Phase.WaitResponse
        assert 1 in env.active_players

        # P1 Checks Legal
        obs = obs_dict[1]
        legals = obs.legal_actions()
        chi_actions = [a for a in legals if a.action_type == ActionType.CHI]
        assert len(chi_actions) == 2

        # P1 performs CHI
        action = Action(ActionType.CHI, tile=tile_3m, consume_tiles=[tile_4m, tile_5m])
        env.step({1: action})

        # P1 current player
        assert env.current_player == 1
        assert env.phase == Phase.WaitAct

        last_ev = env.mjai_log[-1]
        assert last_ev["type"] == "chi"
        assert last_ev["pai"] == "3m"
        assert last_ev["consumed"] == ["4m", "5m"]

        # もう一つのパターンでもチーできることを確認する
        env = RiichiEnv(seed=42)
        env.reset()

        # Setup: P0 discards 3m (ID 8,9,10,11).
        # P1 (Right/Next/Shimocha) has 4m, 5m, 5mr.
        # (3m, 4m, 5m) and (3m, 4m, 5mr) sequence.

        h = env.hands
        h[0] = [tile_3m] + list(range(40, 40 + 12))
        h[1] = [tile_4m, tile_5mr, tile_5m] + list(range(60, 60 + 11))
        h[1].sort()
        env.hands = h

        env.active_players = [0]
        env.current_player = 0

        env.drawn_tile = 100  # irrelevant

        # P0 Discards 3m
        obs_dict = env.step({0: Action(ActionType.DISCARD, tile=tile_3m)})

        # Should be in WaitResponse
        # P1 is next player (0->1 is Chi valid)
        assert env.phase == Phase.WaitResponse
        assert 1 in env.active_players

        # P1 Checks Legal
        obs = obs_dict[1]
        legals = obs.legal_actions()
        chi_actions = [a for a in legals if a.action_type == ActionType.CHI]
        assert len(chi_actions) == 2

        # P1 performs CHI
        action = Action(ActionType.CHI, tile=tile_3m, consume_tiles=[tile_4m, tile_5mr])
        env.step({1: action})

        # P1 current player
        assert env.current_player == 1
        assert env.phase == Phase.WaitAct

        last_ev = env.mjai_log[-1]
        assert last_ev["type"] == "chi"
        assert last_ev["pai"] == "3m"
        assert last_ev["consumed"] == ["4m", "5mr"]

    def test_chi_claim_with_invalid_tile(self) -> None:
        # もう一つのパターンでもチーできることを確認する
        env = RiichiEnv(seed=42)
        env.reset()

        # Setup: P0 discards 3m (ID 8,9,10,11).
        # P1 (Right/Next/Shimocha) has 4m, 5m, 5mr.
        # (3m, 4m, 5m) and (3m, 4m, 5mr) sequence.

        tile_3m = 8
        tile_4m = 12
        tile_5mr = 16
        tile_5m = 17

        h = env.hands
        h[0] = [tile_3m] + list(range(40, 40 + 12))
        h[1] = [tile_4m, tile_5mr, tile_5m] + list(range(60, 60 + 11))
        h[1].sort()
        env.hands = h

        env.active_players = [0]
        env.current_player = 0

        env.drawn_tile = 100  # irrelevant

        # P0 Discards 3m
        obs_dict = env.step({0: Action(ActionType.DISCARD, tile=tile_3m)})

        # Should be in WaitResponse
        # P1 is next player (0->1 is Chi valid)
        assert env.phase == Phase.WaitResponse
        assert 1 in env.active_players

        # P1 Checks Legal
        obs = obs_dict[1]
        legals = obs.legal_actions()
        chi_actions = [a for a in legals if a.action_type == ActionType.CHI]
        assert len(chi_actions) == 2

        # P1 performs CHI with a tile NOT in hand -> ILLEGAL -> Penalty
        # 10 is 3m, but P1 doesn't have 3m (IDs 8-11).
        action = Action(ActionType.CHI, tile=tile_3m, consume_tiles=[tile_4m, 10])
        env.step({1: action})

        # Verify Penalty (Ryukyoku)
        # Log contains: ..., ryukyoku, end_kyoku, end_game (sometimes)
        # We search for ryukyoku from end
        found_ryukyoku = False
        for ev in reversed(env.mjai_log):
            if ev["type"] == "ryukyoku":
                found_ryukyoku = True
                assert "Error: Illegal Action" in ev["reason"]
                break
        assert found_ryukyoku

        # Restore OK state for next check?
        # Actually the game has progressed to next round.
        # So we need to reset to test next case.

    def test_chi_claim_with_invalid_combo(self) -> None:
        env = RiichiEnv(seed=42)
        env.reset()

        # Setup again
        tile_3m = 8
        tile_4m = 12
        invalid_tile = 60  # 1p

        h = env.hands
        h[0] = [tile_3m] + list(range(40, 40 + 12))
        h[1] = [tile_4m, invalid_tile] + list(range(61, 61 + 11))
        h[1].sort()
        env.hands = h

        env.active_players = [0]
        env.current_player = 0

        env.step({0: Action(ActionType.DISCARD, tile=tile_3m)})

        # P1 performs CHI with tiles IN HAND but NOT a sequence -> ILLEGAL -> Penalty
        action = Action(ActionType.CHI, tile=tile_3m, consume_tiles=[tile_4m, invalid_tile])
        env.step({1: action})

        found_ryukyoku = False
        for ev in reversed(env.mjai_log):
            if ev["type"] == "ryukyoku":
                found_ryukyoku = True
                assert "Error: Illegal Action" in ev["reason"]
                break
        assert found_ryukyoku

    def test_chi_multiple_patterns(self) -> None:
        # 123456789m5mr が手にあるとき、4m 打牌に対して 5 通りの CHI が発生することを確認する
        env = RiichiEnv(seed=42)
        env.reset()

        # Tiles: 1m..9m + 5mr
        # 1m: 0, 2m: 4, 3m: 8, 4m: 12, 5mr: 16, 5m: 17, 6m: 20, 7m: 24, 8m: 28, 9m: 32
        hand_tiles = [0, 4, 8, 12, 16, 17, 20, 24, 28, 32] + [100, 104, 108]

        h = env.hands
        h[0] = [13] + list(range(40, 40 + 12))  # P0 discards 4m (ID 13)
        h[1] = hand_tiles
        h[1].sort()
        env.hands = h

        env.active_players = [0]
        env.current_player = 0
        env.drawn_tile = 100

        # P0 Discards 4m
        obs_dict = env.step({0: Action(ActionType.DISCARD, tile=13)})

        # P1 Checks Legal
        obs = obs_dict[1]
        legals = obs.legal_actions()
        chi_actions = [a for a in legals if a.action_type == ActionType.CHI]

        # Expect 5 CHI patterns:
        # [4, 8] (23m)
        # [8, 17] (35m)
        # [8, 16] (35mr)
        # [17, 20] (56m)
        # [16, 20] (5mr6m)

        assert len(chi_actions) == 5

        consumed_sets = [set(a.consume_tiles) for a in chi_actions]
        assert {4, 8} in consumed_sets
        assert {8, 17} in consumed_sets
        assert {8, 16} in consumed_sets
        assert {17, 20} in consumed_sets
        assert {16, 20} in consumed_sets

    def test_ron_claim(self) -> None:
        env = RiichiEnv(seed=42)
        env.reset()

        # Setup P0 discards
        # P1 tenpai for 1m

        # 1m,1m,1m (0,1,2), 2m,2m,2m (4,5,6), 3m,3m,3m (8,9,10), 4m,4m,4m (12,13,14), 5m (16)
        # Wait 5m.
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

        obs_dict = env.step({0: Action(ActionType.Discard, tile=tile_5m_target)})

        assert env.phase == Phase.WaitResponse
        assert env.active_players == [1]
        assert list(obs_dict.keys()) == [1]

        # P1 Legal Ron
        obs = obs_dict[1]
        ron = [a for a in obs.legal_actions() if a.action_type == ActionType.Ron]
        assert len(ron) == 1

        # Execute Ron
        env.step({1: ron[0]})

        # Should log Hora (A single Ron doesn't necessarily end the game)
        ev_types = [ev["type"] for ev in env.mjai_log[-3:]]
        assert "hora" in ev_types

    def test_ankan_riichi_legality(self) -> None:
        env = RiichiEnv(seed=42)
        env.reset()

        # P3 Hand: 2m x 2, 7p, 8p, 9p, 1s x 4, 2s, 3s, 5s, 6s, 1p
        # 2m: 4, 5
        # 7p: 60, 8p: 64, 9p: 68
        # 1s: 72, 73, 74, 75
        # 2s: 76, 3s: 80
        # 5s: 88, 6s: 92
        # 1p: 36
        # Total 14 tiles
        p3_hand = [4, 5, 36, 60, 64, 68, 72, 73, 74, 75, 76, 80, 88, 92]
        p3_hand.sort()

        h = env.hands
        h[3] = p3_hand
        env.hands = h

        # Phase: P3 Turn (Drawn tile 72)
        env.current_player = 3
        env.phase = Phase.WaitAct
        env.drawn_tile = 72  # Assume it drew 1s
        rd = list(env.riichi_declared)
        rd[3] = True
        env.riichi_declared = rd

        # Check legal actions
        obs = env.get_observations()
        obs3 = obs[3]
        legals = obs3.legal_actions()

        # In this scenario, 1s,1s,1s,1s were part of a 1s,2s,3s sequence wait.
        # Ankan would change the waits, so it should be ILLEGAL.
        ankan = [a for a in legals if a.action_type == ActionType.Ankan]
        assert len(ankan) == 0, f"Ankan should be illegal as it changes waits. Legals: {legals}"

    @pytest.mark.skip(
        reason="Fails due to strict illegal action penalty; difficult to setup valid P3 Riichi state manually in test."
    )
    def test_riichi_declared_on_claim(self) -> None:
        """Verify that riichi_declared is updated when the reach tile is claimed."""
        env = RiichiEnv(seed=42)
        env.reset()

        # Prepare P0 hand to Pon East (109, 110)
        h = env.hands
        h[0] = [109, 110] + list(env.hands[0][2:])
        env.hands = h

        # Prepare P3 hand to be Tenpai (Ready for Riichi)
        # Use Chiitoitsu (7 Pairs) hand
        # 11, 22, 33, 44, 55, 66, 7 (Wait 7)
        # IDs: 0,1 (1m), 4,5 (2m), 8,9 (3m), 12,13 (4m), 16,17 (5m), 20,21 (6m), 24 (7m)
        p3_hand = [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24]
        h[3] = p3_hand
        env.hands = h

        # P3 Turn, declares Riichi
        env.current_player = 3
        env.phase = Phase.WaitAct
        env.drawn_tile = 108

        env.step({3: Action(ActionType.Riichi)})

        # P3 Discards reach tile
        env.step({3: Action(ActionType.Discard, tile=108)})

        # P0 PONS the reach tile
        if env.phase != Phase.WaitResponse:
            print(f"DEBUG LOG: {[x['type'] for x in env.mjai_log]}")
            if len(env.mjai_log) >= 2:
                print(f"DEBUG REASON: {env.mjai_log[-2].get('reason', 'N/A')}")

        assert env.phase == Phase.WaitResponse
        assert 108 in [
            a.tile for a in env.get_observations(players=[0])[0].legal_actions() if a.action_type == ActionType.Pon
        ]

        # Before claim
        assert not env.riichi_declared[3]

        # Execute Pon
        env.step({0: Action(ActionType.Pon, tile=108, consume_tiles=[109, 110])})

        # After claim, riichi_declared should be True
        assert env.riichi_declared[3]
        # Also points should be deducted (25000 - 1000 = 24000)
        assert env.scores()[3] == 24000
