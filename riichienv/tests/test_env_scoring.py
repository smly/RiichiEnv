import riichienv.convert as cvt
from riichienv import AgariCalculator, RiichiEnv, Wind
from riichienv.action import Action, ActionType
from riichienv.hand import Conditions


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
        # Set phase WaitAct
        env.current_player = 1
        env.phase = 0  # WaitAct
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
        assert env.scores()[0] == 25000 + deltas[0]
        assert env.scores()[1] == 25000 + deltas[1]

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

        assert env.scores()[0] == 25000 + deltas[0]

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
        assert len(hora_event["ura_markers"]) > 0

    def test_south_round_win(self):
        # Round wind = 1 (South)
        env = RiichiEnv(seed=42, round_wind=0)  # Rust new() expects Option<u8> or maybe int?
        # Wait, Rust signature: new(..., round_wind=None). Default is 0.
        # But `pyo3(signature = (..., round_wind=None))` expects strict typing if I changed it?
        # Wind is u8. The python binding for new accepts u8?
        # But `Conditions` uses `Wind`.
        # `RiichiEnv` field `round_wind` is `u8` in `env.rs: new()`.
        # BUT the field `round_wind` in struct `RiichiEnv`?
        # Let's check `env.rs` definition of `RiichiEnv`.
        # `pub round_wind: u8` line 125 in env.rs? No.
        # Let's check struct definition.
        # If `RiichiEnv` uses `u8`, then tests passing `1` is OK.
        # But `test_south_round_win` failure said: `AssertionError: assert 'E' == 'S'`.
        # That was logic error.
        # Wait, `test_env_scoring.py` failed with `TypeError` in `test_south_round_win`?
        # `FAILED tests/test_env_scoring.py::TestRiichiScoring::test_south_round_win - AssertionError: assert 'E' == 'S'`
        # This means it ran fine but start kyoku event had wrong bakaze?
        # `assert start_kyoku["bakaze"] == "S"` failed.
        # This implies `RiichiEnv(..., round_wind=1)` worked?

        # But `test_env_scoring.py` line 193:
        # `cond = Conditions(tsumo=True, player_wind=0, round_wind=1)`
        # This line likely failed if Conditions requires Wind.
        # The log showed many failures.
        # Let's assume Conditions needs Wind.

        env = RiichiEnv(seed=42, round_wind=1)  # Keeping 1 for RiichiEnv constructor if it accepts int.
        env.reset()

        # Verify start_kyoku event has bakaze="S"
        start_kyoku = next(e for e in env.mjai_log if e["type"] == "start_kyoku")
        assert start_kyoku["bakaze"] == "S"

        # P0 Wind (East) is not Yaku in South Round (unless East Round too, but this is South)
        # But South Wind IS Yaku (Round Wind)

        # Construct Hand with South Triplet
        # South tiles: 1s=72..107 is Sou, Honors start 108.
        # 1z=East, 2z=South
        # 2z (South) IDs: 112, 113, 114, 115
        assert cvt.mjai_to_tid("S") == 112
        assert cvt.tid_to_mjai(113) == "S"
        assert cvt.tid_to_mjai(114) == "S"
        south_triplet = [112, 113, 114]

        # 111222567m 44m 222z
        misc = [0, 1, 2, 4, 5, 6, 17, 20, 24]  # 3 sets
        pair_tile = 12  # 4m

        hand = south_triplet + misc + [pair_tile]
        # 13 tiles

        env.hands[0] = sorted(hand)

        # Draw pair match
        env.drawn_tile = 13  # 4m
        env.current_player = 0
        env.turn_count = 1  # Avoid Tenhou (Turn check)
        env.discards[0].append(0)  # Avoid Tenhou (Discard check)

        # Execute Tsumo
        env.step({0: Action(ActionType.TSUMO)})

        hora_event = env.mjai_log[-3]
        assert hora_event["type"] == "hora"

        # Check if Yaku includes South (Round Wind)
        deltas = hora_event["deltas"]

        # Manual check needs correct conditions
        # Player 0 (East), Round 1 (South)
        cond = Conditions(tsumo=True, player_wind=Wind.East, round_wind=Wind.South)

        calc = AgariCalculator(env.hands[0], env.melds[0]).calc(env.drawn_tile, conditions=cond)
        # Sort yaku for comparison? AgariCalculator output order might vary?
        # Usually it returns sorted or fixed order.
        # But failing output was [22, 27]. Expected [11, 22, 27].
        # 11 is usually Yakuhai.

        # Note: Set comparison is safer for tests.
        assert set(calc.yaku) == {1, 11, 22, 27}  # Menzen Tsumo, Yakuhai:South, Sanankou, Honitsu
        assert deltas[0] == 18000
        assert deltas[1] * -1 == calc.tsumo_agari_ko
