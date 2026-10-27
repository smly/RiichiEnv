"""Issue #245: engine settlement and public viewer APIs after the first round."""

import copy
import json
import random
import re

import pytest

from riichienv import ActionType, Phase, RiichiEnv
from riichienv.visualizer import GameViewer


@pytest.fixture(params=[(3, False), (3, True), (4, False), (4, True)])
def completed_riichi_game(request):
    players, late_reach = request.param
    winner = players - 1
    # 123456p 789s 55s 46s, with a red 5p; closed wait on 5s gives 40 fu on ron.
    hand = [36, 40, 44, 48, 52, 56, 96, 100, 104, 89, 90, 84, 92]
    winning_tile, dora, ura = 91, 132, 49  # C -> P (none), 4p -> red 5p (one).
    tiles = [t for t in range(136) if players == 4 or t < 4 or t >= 32]
    pool = [t for t in tiles if t not in hand and t not in (winning_tile, dora, ura)]
    random.Random(245).shuffle(pool)
    hands = []
    for seat in range(players):
        if seat == winner:
            hands.append(hand)
        else:
            hands.append(pool[:13])
            del pool[:13]

    # Win off the dealer after the winner's third discard, outside ippatsu even
    # when riichi is declared on the second turn (the normal-riichi control).
    before_win = winner + 2 * players + 1
    draws = [pool.pop() for _ in range(before_win)] + [winning_tile]
    wall = [t for start in (0, 4, 8) for h in hands for t in h[start : start + 4]]
    wall += [h[12] for h in hands] + draws + pool + [ura, dora]
    dora_index = 99 if players == 3 else len(wall) - 5
    for tile, index in ((dora, dora_index), (ura, dora_index - 1)):
        old_index = wall.index(tile)
        wall[index], wall[old_index] = wall[old_index], wall[index]
    assert sorted(wall) == tiles

    env = RiichiEnv(f"{players}p-red-single", seed=245)
    env.reset(wall=wall, oya=0, scores=[25000] * players, honba=1)
    assert env.dora_indicators == [dora]
    declaration_turn = winner + (players if late_reach else 0)
    for turn in range(before_win + 1):
        actor = env.current_player
        if turn == before_win:
            assert not env.is_first_turn
            assert len(env.discards[winner]) == 3
        if turn == declaration_turn:
            declare = next(a for a in env.get_observation(actor).legal_actions() if a.action_type == ActionType.RIICHI)
            env.step({actor: declare})
        discard = next(
            a
            for a in env.get_observation(actor).legal_actions()
            if a.action_type == ActionType.DISCARD and a.tile == env.drawn_tile
        )
        env.step({actor: discard})
        if env.phase == Phase.WaitResponse:
            actions = {}
            for seat in env.active_players:
                kind = ActionType.RON if turn == before_win and seat == winner else ActionType.PASS
                actions[seat] = next(a for a in env.get_observation(seat).legal_actions() if a.action_type == kind)
            env.step(actions)

    assert env.is_done
    assert not any(str(e.get("reason", "")).startswith("Error:") for e in env.mjai_log)
    hora = next(e for e in env.mjai_log if e["type"] == "hora")
    assert hora["actor"] == winner
    assert hora["target"] == 0
    assert hora["ura_markers"] == ["4p"]
    return env, winner, late_reach


def test_riichienv_settles_riichi_after_first_round(completed_riichi_game):
    env, winner, late_reach = completed_riichi_game
    result = env.win_results[winner]
    assert set(result.yaku) == {2 if late_reach else 18, 32, 33}
    assert result.han == (3 if late_reach else 4)
    assert result.fu == 40
    players = len(env.scores())
    payment = (5200 if late_reach else 8000) + 100 * (players - 1)
    assert result.ron_agari == payment
    deltas = [0] * players
    deltas[0] = -payment
    deltas[winner] = payment + 1000  # Return the winner's riichi deposit.
    assert env.score_deltas == deltas
    hora = next(e for e in env.mjai_log if e["type"] == "hora")
    assert hora["deltas"] == deltas
    scores = [25000 + delta for delta in deltas]
    scores[winner] -= 1000
    assert env.scores() == scores


@pytest.mark.parametrize("source", ["get_viewer", "from_env", "from_list", "from_jsonl"])
def test_gameviewer_riichi_results_and_html(completed_riichi_game, source, tmp_path):
    env, winner, late_reach = completed_riichi_game
    log = env.mjai_log
    original_log = copy.deepcopy(log)
    original_scores = env.scores()
    if source == "get_viewer":
        viewer = env.get_viewer()
    elif source == "from_env":
        viewer = GameViewer.from_env(env)
    elif source == "from_list":
        viewer = GameViewer.from_list(log)
    else:
        path = tmp_path / "replay.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in log), encoding="utf-8")
        viewer = GameViewer.from_jsonl(str(path))

    results = viewer.get_results(0)
    assert len(results) == 1
    result = results[0]
    assert set(result.yaku) == {2 if late_reach else 18, 32, 33}
    assert result.han == (3 if late_reach else 4)
    assert result.fu == 40
    # GameViewer currently returns base hand points, excluding honba/deposits.
    assert result.ron_agari == (5200 if late_reach else 8000)

    html = viewer.show().data
    assert isinstance(html, str)
    match = re.search(r"const logData = (.*);", html)
    assert match is not None
    rendered_log = json.loads(match[1])
    score = next(e["meta"]["score"] for e in rendered_log if e["type"] == "hora")
    assert score == {"han": result.han, "fu": 40, "points": str(result.ron_agari), "yaku": result.yaku}
    end = next(e for e in rendered_log if e["type"] == "end_kyoku")
    assert end["meta"]["results"] == [{"actor": winner, "target": 0, "score": score}]
    assert log == original_log
    assert env.mjai_log == original_log
    assert env.scores() == original_scores
