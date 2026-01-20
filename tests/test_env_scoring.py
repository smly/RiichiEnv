import riichienv.convert as cvt
from riichienv import Action, ActionType, AgariCalculator, Conditions, Phase, RiichiEnv, Wind


class TestRiichiScoring:
    def test_ron_deltas(self):
        env = RiichiEnv(seed=42)
        env.reset()

        haku_tiles = [124, 125]
        misc_tiles = [0, 4, 8, 5, 9, 12, 16, 20, 24, 14, 15]

        h = env.hands
        h[0] = sorted(haku_tiles + misc_tiles)  # 13 tiles
        env.hands = h

        env.current_player = 1
        env.phase = Phase.WaitAct
        env.active_players = [1]
        h = env.hands
        h[1].append(126)  # Give P1 the target tile
        env.hands = h

        # P1 discards 126
        obs = env.step({1: Action(ActionType.Discard, tile=126)})

        # P0 should have Ron option
        assert 0 in obs
        actions = obs[0].legal_actions()
        ron_action = next((a for a in actions if a.action_type == ActionType.Ron), None)
        assert ron_action is not None

        # P0 declares Ron
        env.step({0: Action(ActionType.Ron, tile=126)})

        # Check log
        hora_event = next(e for e in reversed(env.mjai_log) if e["type"] == "hora")
        assert "deltas" in hora_event

        deltas = hora_event["deltas"]
        assert deltas[1] < 0
        assert deltas[0] > 0
        assert sum(deltas) == 0  # Zero sum if no riichi sticks

        # Check internal scores updated
        assert env.scores()[0] == 25000 + deltas[0]
        assert env.scores()[1] == 25000 + deltas[1]

    def test_tsumo_deltas(self):
        env = RiichiEnv(seed=42)
        env.reset()

        hand = [124, 125, 126, 0, 4, 8, 5, 9, 13, 16, 20, 24, 12]
        h = env.hands
        h[0] = sorted(hand)
        env.hands = h

        env.drawn_tile = 14
        env.current_player = 0
        env.turn_count = 1  # Avoid Tenhou/Renhou checks
        d = env.discards
        d[0].append(0)
        env.discards = d

        # Verify legal actions
        actions = env._get_legal_actions(0)
        tsumo_act = next((a for a in actions if a.action_type == ActionType.Tsumo), None)
        assert tsumo_act is not None

        # Execute Tsumo
        env.step({0: Action(ActionType.Tsumo)})

        # Check log
        hora_event = next(e for e in reversed(env.mjai_log) if e["type"] == "hora")
        assert hora_event["tsumo"] is True
        assert "deltas" in hora_event

        deltas = hora_event["deltas"]
        assert deltas[0] > 0
        assert deltas[1] < 0
        assert deltas[2] < 0
        assert deltas[3] < 0
        assert sum(deltas) == 0

        assert env.scores()[0] == 25000 + deltas[0]

    def test_ura_markers(self):
        env = RiichiEnv(seed=42)
        env.reset()

        # Force Riichi state
        rd = env.riichi_declared
        rd[0] = True
        env.riichi_declared = rd

        # Force Tsumo win
        hand = [124, 125, 126, 0, 1, 2, 4, 5, 6, 8, 9, 10, 12]
        h = env.hands
        h[0] = sorted(hand)
        env.hands = h
        env.drawn_tile = 13
        env.current_player = 0

        env.step({0: Action(ActionType.Tsumo)})

        hora_event = next(e for e in reversed(env.mjai_log) if e["type"] == "hora")
        assert len(hora_event["ura_markers"]) > 0

    def test_south_round_win(self):
        env = RiichiEnv(seed=42, round_wind=1)
        env.reset()

        # Verify start_kyoku event has bakaze="S"
        start_kyoku = next(e for e in env.mjai_log if e["type"] == "start_kyoku")
        assert start_kyoku["bakaze"] == "S"

        assert cvt.mjai_to_tid("S") == 112
        assert cvt.tid_to_mjai(113) == "S"
        assert cvt.tid_to_mjai(114) == "S"
        south_triplet = [112, 113, 114]

        misc = [0, 4, 8, 5, 9, 12, 16, 20, 24]
        pair_tile = 13
        hand = south_triplet + misc + [pair_tile]

        h = env.hands
        h[0] = sorted(hand)
        env.hands = h

        # Draw pair match
        env.drawn_tile = 14  # 4m
        env.current_player = 0
        env.turn_count = 1  # Avoid Tenhou (Turn check)
        d = env.discards
        d[0].append(0)
        env.discards = d  # Reassign for PyO3

        # Execute Tsumo
        env.step({0: Action(ActionType.Tsumo)})
        hora_event = next(e for e in reversed(env.mjai_log) if e["type"] == "hora")

        deltas = hora_event["deltas"]

        cond = Conditions(tsumo=True, player_wind=Wind.East, round_wind=Wind.South)

        calc = AgariCalculator(env.hands[0], env.melds[0]).calc(env.drawn_tile, conditions=cond)

        # Note: Set comparison is safer for tests.
        # Using 16, 20, 24 sequence -> 16 is Red 5 (Akadora).
        # So Yaku: Tsumo(1), South(11), Honitsu(27), Akadora(32).
        assert set(calc.yaku) == {1, 11, 27, 32}
        assert deltas[0] == 18000
        assert deltas[1] * -1 == calc.tsumo_agari_ko
