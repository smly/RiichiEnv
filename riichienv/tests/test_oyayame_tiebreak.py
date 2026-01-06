import pytest

from riichienv.rules import StandardRule

from .env.helper import helper_setup_env


@pytest.mark.skip(reason="Legacy python test - Rust implementation pending or parity missing")
def test_oyayame_tiebreak_extension():
    # end_field=1 (South), target_score=30000, max_extension_field=2 (West)
    rule = StandardRule(end_field=1, target_score=30000, max_extension_field=2)

    # --- Scenario 1: Oya (Seat 0) tied for top at West 1 ---
    # At index 0 < index 1, so Seat 0 is Rank 1.
    env = helper_setup_env(
        oya=0,
        round_wind=2,  # West
        points=[35000, 35000, 15000, 15000],
    )
    env.rule = rule

    # Verify rank
    # ranks() returns 1-based ranks. [1, 2, 3, 4] means P0 is 1st, P1 is 2nd, etc.
    assert env.ranks()[0] == 1
    assert env.ranks()[1] == 2

    # is_game_over should be True (Agari-yame/Tenpai-yame)
    assert rule.is_game_over(env, is_renchan=True) is True

    # --- Scenario 2: Oya (Seat 1) tied for top at West 1 ---
    # Seat 0 (index 0) is Rank 1, Oya (Seat 1) is Rank 2.
    env = helper_setup_env(
        oya=1,
        round_wind=2,  # West
        points=[35000, 35000, 15000, 15000],
    )
    env.rule = rule

    # Verify rank
    assert env.ranks()[0] == 1
    assert env.ranks()[1] == 2

    # is_game_over should be False (Game continues because Oya is not 1st)
    assert rule.is_game_over(env, is_renchan=True) is False


@pytest.mark.skip(reason="Legacy python test - Rust implementation pending or parity missing")
def test_oyayame_tiebreak_last_round():
    # Test same logic in South 4 (Last round of standard game)
    rule = StandardRule(target_score=30000, end_field=1, max_extension_field=2)

    # South 4, Oya is Seat 3
    # Scenario: Oya (P3) tied with P0 at 30,000
    env = helper_setup_env(
        oya=3,
        round_wind=1,  # South
        points=[30000, 20000, 20000, 30000],
    )
    env.rule = rule

    assert env.ranks()[0] == 1
    assert env.ranks()[3] == 2

    # Should NOT end because P3 is not Rank 1
    assert rule.is_game_over(env, is_renchan=True) is False

    # Scenario: Oya (P3) is sole top at 30,000
    env = helper_setup_env(
        oya=3,
        round_wind=1,  # South
        points=[29000, 20000, 21000, 30000],
    )
    env.rule = rule

    assert env.ranks()[3] == 1
    assert rule.is_game_over(env, is_renchan=True) is True
