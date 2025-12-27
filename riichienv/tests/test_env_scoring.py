from riichienv import RiichiEnv
from riichienv.action import Action, ActionType


class TestRiichiScoring:
    def test_ron_deltas(self):
        env = RiichiEnv(seed=42)
        env.reset()

        # Manually setup a scenario
        # Player 0 needs to win by Ron from Player 1
        # Player 0 hand: Kokushi? Or just simple hand?
        # Let's give P0 a simple Tanyao hand ready.
        # 2m 3m 4m 5p 6p 7p 2s 3s 4s 6z 6z 8p 8p (wait on 6z or 8p?)
        # Let's do partial hand setup or just mock the Agari.

        # Force P0 hand to be tenpai for 1p (Tanyao pinfu like)
        # 234m 234p 234s 66z 88p -> wait?
        # Simple: 234m 234p 234s 88p 7p (wait 6p/9p)
        # Using tiles 0-135 logic is tedious to set manually correctly.
        # Instead, let's rely on random play until something happens? No, too slow.

        # Forced Check:
        # P0 wins on P1's discard.
        # We can mock the AgariCalculator return but that's hard inside `step`.

        # We can construct a known hand.
        # White Dragon (Haku) Pon + ...
        # Easy win: Yakuhai.

        # P0 Hand: [Haku, Haku, Haku, 2m, 3m, 4m, 5s, 6s, 7s, 1p, 1p] (11 tiles)
        # We need 13 tiles.
        # [Haku, Haku], [Haku] discared by P1 -> Ron.

        # Let's set P0 hand: [0, 1, 2, 3, 4, 5, 8, 9, 10, 136-1.., ...]
        # 136-ids.
        # Haku = 31 (start index for type 31 ~ 124)
        # Haku IDs: 124, 125, 126, 127

        haku_tiles = [124, 125]
        # Some misc tiles
        misc_tiles = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13]  # 11 tiles

        env.hands[0] = sorted(haku_tiles + misc_tiles)  # 13 tiles

        # P1 needs to discard Haku (126)
        # P0 current player -> Discard something to pass turn?
        # Hack state: Set env.current_player = 1
        # Set phase WAIT_ACT
        env.current_player = 1
        env.phase = 0  # WAIT_ACT
        env.active_players = [1]
        env.hands[1].append(126)  # Give P1 the target tile

        # P1 discards 126
        obs = env.step({1: Action(ActionType.DISCARD, tile=126)})

        # P0 should have Ron option
        assert 0 in obs
        actions = obs[0].legal_actions()
        ron_action = next((a for a in actions if a.type == ActionType.RON), None)
        assert ron_action is not None

        # P0 declares Ron
        env.step({0: Action(ActionType.RON, tile=126)})

        # Check log
        last_event = env.mjai_log[-1]
        assert last_event["type"] == "end_game"
        assert env.mjai_log[-2]["type"] == "end_kyoku"
        hora_event = env.mjai_log[-3]
        assert hora_event["type"] == "hora"
        assert "deltas" in hora_event

        deltas = hora_event["deltas"]
        # Haku only -> 1 han? Nominal points.
        # P1 (loser) should be negative, P0 (winner) positive.
        assert deltas[1] < 0
        assert deltas[0] > 0
        assert sum(deltas) == 0  # Zero sum if no riichi sticks

        # Check internal scores updated
        assert env.scores[0] == 25000 + deltas[0]
        assert env.scores[1] == 25000 + deltas[1]

    def test_tsumo_deltas(self):
        env = RiichiEnv(seed=42)
        env.reset()

        # P0 Tsumo
        # Hand: Haku (124, 125, 126) + misc pair + ...
        # 13 tiles: [0..10] (9 tiles) + [124, 125] (2 tiles) ?
        # Let's make: 124,125,126 (3) + 0,1,2 (3) + 4,5,6 (3) + 8,9,10 (3) + 12 (1) -> wait on 12 (pair)?

        hand = [124, 125, 126, 0, 1, 2, 4, 5, 6, 8, 9, 10, 12]
        env.hands[0] = sorted(hand)

        # Draw 13 (pair for 12)
        env.drawn_tile = 13
        env.current_player = 0
        env.turn_count = 1  # Avoid Tenhou/Renhou checks

        # Verify legal actions
        actions = env._get_legal_actions(0)
        tsumo_act = next((a for a in actions if a.type == ActionType.TSUMO), None)
        assert tsumo_act is not None

        # Execute Tsumo
        env.step({0: Action(ActionType.TSUMO)})

        # Check log
        hora_event = env.mjai_log[-3]
        assert hora_event["type"] == "hora"
        assert hora_event["tsumo"] is True
        assert "deltas" in hora_event

        deltas = hora_event["deltas"]
        assert deltas[0] > 0
        assert deltas[1] < 0
        assert deltas[2] < 0
        assert deltas[3] < 0
        assert sum(deltas) == 0

        assert env.scores[0] == 25000 + deltas[0]

    def test_ura_markers(self):
        env = RiichiEnv(seed=42)
        env.reset()

        # Force Riichi state
        env.riichi_declared[0] = True

        # Force Tsumo win
        hand = [124, 125, 126, 0, 1, 2, 4, 5, 6, 8, 9, 10, 12]
        env.hands[0] = sorted(hand)
        env.drawn_tile = 13
        env.current_player = 0

        env.step({0: Action(ActionType.TSUMO)})

        hora_event = env.mjai_log[-3]
        assert "ura_markers" in hora_event
        assert isinstance(hora_event["ura_markers"], list)
        assert len(hora_event["ura_markers"]) > 0
