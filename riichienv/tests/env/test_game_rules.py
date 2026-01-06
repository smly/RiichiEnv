import pytest

from riichienv import AgariCalculator, Conditions, RiichiEnv, Wind
from riichienv.game_mode import GameType


@pytest.mark.skip(reason="Legacy python test - Rust implementation pending or parity missing")
def test_tonpuu_transitions():
    # Test East 1 -> East 2 transition
    env = RiichiEnv(game_type=GameType.YON_TONPUSEN)
    env.reset(oya=0, scores=[25000, 25000, 25000, 25000])

    # Simulate a transition
    # For simplicity, we can just trigger a ryukyoku with Oya not tenpai

    # Let's manually set hands to something non-tenpai and check oya is not tenpai
    for i in range(4):
        # Definitely not tenpai (disconnected tiles)
        env.hands[i] = [0, 4, 8, 12, 36, 40, 44, 48, 72, 76, 80, 108, 112]

    env._trigger_ryukyoku("exhaustive_draw")

    assert env.oya == 1
    assert env._custom_round_wind == int(Wind.East)  # Still East
    assert not env.done()


@pytest.mark.skip(reason="Legacy python test - Rust implementation pending or parity missing")
def test_tonpuu_game_end():
    # Test game ends after East 4 because someone >= 30000
    env = RiichiEnv(game_type=GameType.YON_TONPUSEN)
    # Start at East 4 (Oya = 3)
    env.reset(oya=3, bakaze=int(Wind.East), scores=[40000, 20000, 20000, 20000])

    for i in range(4):
        env.hands[i] = [0, 4, 8, 12, 36, 40, 44, 48, 72, 76, 80, 108, 112]

    # No one tenpai, Ryukyoku
    env._trigger_ryukyoku("exhaustive_draw")

    # Someone (Player 0) is > 30000 and top, so game should end
    assert env.done()


@pytest.mark.skip(reason="Legacy python test - Rust implementation pending or parity missing")
def test_tonpuu_sudden_death():
    # Test South entrance if no one >= 30000 at East 4
    env = RiichiEnv(game_type=GameType.YON_TONPUSEN)
    env.reset(oya=3, bakaze=int(Wind.East), scores=[28000, 24000, 24000, 24000])

    for i in range(4):
        env.hands[i] = [0, 4, 8, 12, 36, 40, 44, 48, 72, 76, 80, 108, 112]

    env._trigger_ryukyoku("exhaustive_draw")

    # No one reached 30000, should enter South 1
    assert not env.done()
    assert env.oya == 0
    assert env._custom_round_wind == int(Wind.South)  # South


@pytest.mark.skip(reason="Legacy python test - Rust implementation pending or parity missing")
def test_tonpuu_v_goal():
    # Test game ends immediately if someone >= 30000 in South (extension)
    env = RiichiEnv(game_type=GameType.YON_TONPUSEN)
    env.reset(oya=0, bakaze=int(Wind.South), scores=[29000, 28000, 22000, 21000])

    # Player 0 (Oya) wins and reaches 30000
    # or just any win during extension
    env._scores = [35000, 22000, 22000, 21000]
    # Simulate a win
    env._end_kyoku(is_renchan=False)  # Not renchan, normally would move to South 2

    assert env.done()


@pytest.mark.skip(reason="Legacy python test - Rust implementation pending or parity missing")
def test_oya_agari_yame():
    # Oya top at East 4 ends game
    env = RiichiEnv(game_type=GameType.YON_TONPUSEN)
    env.reset(oya=3, bakaze=int(Wind.East), scores=[40000, 20000, 20000, 20000])  # Oya is 3, but Player 0 is top
    # Wait, Oya is 3. Let's make Oya 3 top.
    env.reset(oya=3, bakaze=int(Wind.East), scores=[20000, 20000, 20000, 40000])

    # Oya wins/renchans
    env._end_kyoku(is_renchan=True)

    # In last round, if oya is top and >= 30k, end game (Agari-yame)
    assert env.done()


@pytest.mark.skip(reason="Legacy python test - Rust implementation pending or parity missing")
def test_hanchan_transitions():
    # East 4 -> South 1
    env = RiichiEnv(game_type=GameType.YON_HANCHAN)
    env.reset(oya=3, bakaze=int(Wind.East), scores=[25000, 25000, 25000, 25000])

    for i in range(4):
        env.hands[i] = [0, 4, 8, 12, 36, 40, 44, 48, 72, 76, 80, 108, 112]
    env._trigger_ryukyoku("exhaustive_draw")

    assert env.oya == 0
    assert env._custom_round_wind == int(Wind.South)  # South entrance
    assert not env.done()


@pytest.mark.skip(reason="Legacy python test - Rust implementation pending or parity missing")
def test_tobi():
    # Test game ends if someone's score < 0
    env = RiichiEnv(game_type=GameType.YON_TONPUSEN)
    env.reset(oya=0, scores=[25000, 25000, 25000, 25000])

    # Player 1's score becomes negative
    env._scores = [40000, -1000, 31000, 30000]

    # Any round end should trigger game over
    env._end_kyoku(is_renchan=False)
    assert env.done()


@pytest.mark.skip(reason="Legacy python test - Rust implementation pending or parity missing")
def test_one_kyoku_always_ends():
    # Test YON_IKKYOKU always ends after one kyoku
    env = RiichiEnv(game_type=GameType.YON_IKKYOKU)
    env.reset(oya=0, scores=[25000, 25000, 25000, 25000])

    # Even if Oya renchans
    env._end_kyoku(is_renchan=True)
    assert env.done()


@pytest.mark.skip(reason="Legacy python test - Rust implementation pending or parity missing")
def test_kyotaku_carry_over():
    # Test that riichi sticks are carried over after a draw and awarded on win
    env = RiichiEnv(game_type=GameType.YON_TONPUSEN)
    env.reset(oya=0, scores=[25000, 25000, 25000, 25000])

    # Player 0 (Oya) declares riichi
    env.riichi_sticks = 1
    env._scores[0] -= 1000

    # Draw (Ryukyoku)
    for i in range(4):
        env.hands[i] = [0, 4, 8, 12, 36, 40, 44, 48, 72, 76, 80, 108, 112]
    env._trigger_ryukyoku("exhaustive_draw")

    # Next round should have 1 kyotaku
    assert env.riichi_sticks == 1
    assert env.oya == 1

    # Player 2 wins in the next round
    res = AgariCalculator([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16], []).calc(
        17, dora_indicators=[32], conditions=Conditions(tsumo=True, round_wind=Wind.East, player_wind=Wind.South)
    )

    # Winner is player 2 (index 2)
    # _calculate_deltas will award kyotaku to winner
    # We need to make sure the score before win is consistent
    pre_win_score = env.scores()[2]
    deltas = env._calculate_deltas(res, winner=2, is_tsumo=True)

    assert env.scores()[2] == pre_win_score + deltas[2]
    assert deltas[2] >= 1000 + res.tsumo_agari_ko * 3
    assert env.riichi_sticks == 0
