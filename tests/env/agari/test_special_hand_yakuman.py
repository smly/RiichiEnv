"""Exercise special-hand scoring from complete walls using legal actions."""

import pytest

from riichienv import ActionType, GameRule, Phase, RiichiEnv

HANDS = {
    # Single-wait kokushi: pair of 1m, missing White.
    "kokushi": ([0, 1, 32, 36, 68, 72, 104, 108, 112, 116, 120, 128, 132], 124),
    "kokushi_13": ([0, 32, 36, 68, 72, 104, 108, 112, 116, 120, 124, 128, 132], 1),
    "seven_pairs": ([32, 33, 36, 37, 48, 49, 64, 65, 84, 92, 93, 96, 97], 85),
    "all_honors": ([108, 109, 112, 113, 116, 117, 120, 121, 124, 125, 128, 129, 132], 133),
}


def setup_wall(players, winner, kind, first_turn, winning_tile=None):
    hand, default_tile = HANDS[kind]
    if winning_tile is None:
        winning_tile = default_tile
    else:
        hand = hand + [default_tile]
        hand.remove(winning_tile)
    all_tiles = [t for t in range(136) if players == 4 or t < 4 or t >= 32]
    # 6p indicates 7p, absent from every test hand. Keep it in the dead wall.
    indicator = 56
    pool = [t for t in all_tiles if t not in hand and t not in (winning_tile, indicator)]
    hands = []
    for seat in range(players):
        if seat == winner:
            hands.append(hand)
        else:
            hands.append(pool[:13])
            del pool[:13]
    before_win = winner + (0 if first_turn else players)
    draws = [pool.pop() for _ in range(before_win)] + [winning_tile]
    wall = [t for start in (0, 4, 8) for h in hands for t in h[start : start + 4]]
    wall += [h[12] for h in hands] + draws + pool + [indicator]
    index = 99 if players == 3 else len(wall) - 5
    wall[index], wall[-1] = wall[-1], wall[index]
    assert sorted(wall) == all_tiles
    return wall, before_win, all_tiles


@pytest.mark.parametrize("players", [3, 4])
@pytest.mark.parametrize("preset", ["tenhou", "mjsoul"])
@pytest.mark.parametrize("winner", [0, 1], ids=["dealer", "nondealer"])
@pytest.mark.parametrize("kind", HANDS)
@pytest.mark.parametrize("first_turn", [True, False], ids=["first_draw", "later_draw"])
def test_special_hand_yakuman_payments(players, preset, winner, kind, first_turn):
    wall, before_win, all_tiles = setup_wall(players, winner, kind, first_turn)
    rule = getattr(GameRule, f"default_{preset}")()
    env = RiichiEnv(f"{players}p-red-single", rule=rule, seed=42)
    env.reset(wall=wall, oya=0, scores=[100000] * players)

    def step(actions):
        env.step(actions)
        assert not any(str(e.get("reason", "")).startswith("Error:") for e in env.mjai_log)
        tiles = env.wall + [t for h in env.hands for t in h] + [t for d in env.discards for t in d]
        assert sorted(tiles) == all_tiles
        assert sum(env.scores()) + 1000 * env.riichi_sticks == 100000 * players

    for _ in range(before_win):
        seat = env.current_player
        discard = next(
            a
            for a in env.get_observation(seat).legal_actions()
            if a.action_type == ActionType.DISCARD and a.tile == env.drawn_tile
        )
        step({seat: discard})
        if env.phase == Phase.WaitResponse:
            step(
                {
                    p: next(a for a in env.get_observation(p).legal_actions() if a.action_type == ActionType.PASS)
                    for p in env.active_players
                }
            )
    assert env.current_player == winner
    assert env.drawn_tile == HANDS[kind][1]
    assert env.dora_indicators == [56]
    tsumo = next(a for a in env.get_observation(winner).legal_actions() if a.action_type == ActionType.TSUMO)
    step({winner: tsumo})
    assert env.is_done

    units = int(first_turn)
    expected_yaku = [35 if winner == 0 else 36] if first_turn else []
    if kind in ("kokushi", "kokushi_13"):
        thirteen_sided = kind == "kokushi_13" or (first_turn and winner == 0)
        units += 1 + int(thirteen_sided and preset == "mjsoul")
        expected_yaku.append(49 if thirteen_sided else 42)
    elif kind == "all_honors":
        units += 1
        expected_yaku.append(39)
    elif not first_turn:
        expected_yaku = [1, 25]  # Menzen tsumo + seven pairs, 3 han / 25 fu.

    result = env.win_results[winner]
    assert result.yakuman == bool(units)
    assert result.han == (13 * units if units else 3)
    assert result.fu == (0 if units else 25)
    assert sorted(result.yaku) == sorted(expected_yaku)
    base_payment = 8000 * units if units else 800
    deltas = [0] * players
    for seat in range(players):
        if seat != winner:
            deltas[seat] = -base_payment * (2 if winner == 0 or seat == 0 else 1)
    deltas[winner] = -sum(deltas)
    assert env.score_deltas == deltas
    assert env.scores() == [100000 + delta for delta in deltas]


@pytest.mark.parametrize("players", [3, 4])
@pytest.mark.parametrize("double_kokushi", [True, False], ids=["double", "single"])
def test_tenhou_kokushi_score_is_independent_of_initial_deal_order(players, double_kokushi):
    # Observed Mahjong Soul behavior: https://mj-news.net/column/nemata-quiz/20200516145895
    # Every choice of the 14th dealt tile must yield the same highest-scoring hand.
    rule = GameRule.default_mjsoul()
    rule.is_kokushi_musou_13machi_double = double_kokushi
    for tile in HANDS["kokushi"][0] + [HANDS["kokushi"][1]]:
        wall, _, _ = setup_wall(players, 0, "kokushi", True, winning_tile=tile)
        env = RiichiEnv(f"{players}p-red-single", rule=rule, seed=42)
        env.reset(wall=wall, oya=0, scores=[100000] * players)
        assert env.drawn_tile == tile
        tsumo = next(a for a in env.get_observation(0).legal_actions() if a.action_type == ActionType.TSUMO)
        env.step({0: tsumo})
        assert env.is_done
        result = env.win_results[0]
        units = 3 if double_kokushi else 2
        assert result.han == 13 * units
        assert sorted(result.yaku) == [35, 49]
        assert result.tsumo_agari_ko == 16000 * units
        assert env.score_deltas == [16000 * units * (players - 1)] + [-16000 * units] * (players - 1)
