"""Mahjong Soul treats the dealer's uninterrupted initial hand as all tedashi."""

import base64
import json
import random

import pytest

from riichienv import Action, ActionType, GameRule, Observation, Observation3P, Phase, RiichiEnv


def setup_env(players, dealer, preset, drawn=53, opening=None, rule=None):
    hand = [36, 37, 38, 44, 45, 46, 52, 53, 54, 60, 61, 62, 108, 109]
    if opening == ActionType.ANKAN:
        hand[12] = 39
    elif opening == ActionType.KITA:
        hand[12] = 120
    tiles = [t for t in range(136) if players == 4 or t < 4 or t >= 32]
    pool = [t for t in tiles if t not in hand and t != 48]
    random.Random(42).shuffle(pool)
    # The wall's dealing order starts at the dealer, regardless of seat number.
    hands = [[t for t in hand if t != drawn]]
    for _ in range(players - 1):
        hands.append(pool[:13])
        del pool[:13]
    deal = [t for start in (0, 4, 8) for h in hands for t in h[start : start + 4]]
    tail = 8 if players == 3 else 4
    wall = deal + [h[12] for h in hands] + [drawn] + pool[:-tail] + [48] + pool[-tail:]
    assert sorted(wall) == tiles
    rule = rule if rule is not None else getattr(GameRule, f"default_{preset}")()
    env = RiichiEnv(f"{players}p-red-single", seed=42, rule=rule)
    observations = env.reset(wall=wall, oya=dealer)
    assert env.drawn_tile == drawn
    assert sorted(env.hands[dealer]) == sorted(hand)
    return env, observations


def discard(env, actor, tile):
    action = next(a for a in env._get_legal_actions(actor) if a.action_type == ActionType.DISCARD and a.tile == tile)
    env.step({actor: action})


def pass_responses(env):
    while env.phase == Phase.WaitResponse:
        env.step({p: Action(ActionType.PASS) for p in env.active_players})


def check_history(env, players, actor, tile, tedashi, last_tedashi):
    assert env.discard_from_hand[actor][-1] is tedashi
    event = next(e for e in reversed(env.mjai_log) if e["type"] == "dahai")
    assert event["actor"] == actor
    assert event["tsumogiri"] is not tedashi
    cls = Observation3P if players == 3 else Observation
    kinds = [0, *range(8, 34)] if players == 3 else list(range(34))
    for observer in range(players):
        obs = env.get_observation(observer)
        restored = cls.deserialize_from_base64(obs.serialize_to_base64())
        for view in (obs, restored):
            assert view.last_discard == tile
            assert view.last_tedashis[actor] == last_tedashi
            if observer == actor:
                continue
            slot = [p for p in range(players) if p != observer].index(actor)
            expected = (
                [
                    kinds.index(last_tedashi // 4) / (len(kinds) - 1),
                    float(last_tedashi in (16, 52, 88)),
                    float(last_tedashi // 4 == 13),
                ]
                if last_tedashi is not None
                else [0.0] * 3
            )
            actual = list(memoryview(view.encode_last_tedashis()).cast("f"))[slot * 3 : (slot + 1) * 3]
            assert actual == pytest.approx(expected)
            extended = memoryview(view.encode_extended()).cast("f")
            for channel, value in enumerate(expected, 197 + slot * 3):
                assert list(extended[channel * len(kinds) : (channel + 1) * len(kinds)]) == pytest.approx(
                    [value] * len(kinds)
                )


@pytest.mark.parametrize("players,dealer", [(p, d) for p in (3, 4) for d in range(p)])
@pytest.mark.parametrize("preset", ["mjsoul", "tenhou"])
@pytest.mark.parametrize("drawn", [52, 53, 109], ids=["red", "normal", "honor"])
@pytest.mark.parametrize("discard_drawn", [False, True])
def test_dealer_initial_discard_metadata(players, dealer, preset, drawn, discard_drawn):
    env, _ = setup_env(players, dealer, preset, drawn)
    tile = drawn if discard_drawn else 44
    discard(env, dealer, tile)
    tedashi = preset == "mjsoul" or not discard_drawn
    check_history(env, players, dealer, tile, tedashi, tile if tedashi else None)


@pytest.mark.parametrize("players", [3, 4])
@pytest.mark.parametrize("preset", ["mjsoul", "tenhou"])
def test_later_draws_remain_tsumogiri(players, preset):
    env, _ = setup_env(players, 0, preset)
    discard(env, 0, 44)
    pass_responses(env)
    # Nondealers' first draws and the dealer's second draw use the usual rule.
    for actor in [*range(1, players), 0]:
        assert env.current_player == actor
        tile = env.drawn_tile
        discard(env, actor, tile)
        check_history(env, players, actor, tile, False, 44 if actor == 0 else None)
        pass_responses(env)
    env.reset(oya=0)
    tile = env.drawn_tile
    discard(env, 0, tile)
    tedashi = preset == "mjsoul"
    check_history(env, players, 0, tile, tedashi, tile if tedashi else None)


@pytest.mark.parametrize("players,opening", [(3, ActionType.ANKAN), (4, ActionType.ANKAN), (3, ActionType.KITA)])
@pytest.mark.parametrize("preset", ["mjsoul", "tenhou"])
def test_initial_replacement_draw_can_be_tsumogiri(players, opening, preset):
    env, _ = setup_env(players, 0, preset, drawn=109, opening=opening)
    action = next(a for a in env._get_legal_actions(0) if a.action_type == opening)
    env.step({0: action})
    pass_responses(env)
    assert env.current_player == 0
    assert not env.is_first_turn
    tile = env.drawn_tile
    discard(env, 0, tile)
    check_history(env, players, 0, tile, False, None)


@pytest.mark.parametrize("players", [3, 4])
@pytest.mark.parametrize("forced", [False, True])
def test_initial_discard_convention_is_configurable(players, forced):
    rule = GameRule(dealer_first_discard_is_tedashi=forced)
    assert rule.dealer_first_discard_is_tedashi is forced
    assert "dealer_first_discard_is_tedashi=" in repr(rule)
    env, _ = setup_env(players, 0, "tenhou", rule=rule)
    discard(env, 0, 53)
    check_history(env, players, 0, 53, forced, 53 if forced else None)


@pytest.mark.parametrize("preset", ["mjsoul", "tenhou"])
def test_initial_discard_sequence_candidates_match_metadata(preset):
    # Use the representative red tile ID so candidate lookup is unambiguous.
    env, observations = setup_env(4, 0, preset, drawn=52)
    obs = observations[0]
    restored = Observation.deserialize_from_base64(obs.serialize_to_base64())
    for view in (obs, restored):
        assert view.drawn_tile == 52
        assert view.forced_tedashi is (preset == "mjsoul")
        candidates = list(memoryview(view.encode_seq_candidates()).cast("H"))
        moqie = 0 if preset == "mjsoul" else 1
        assert [10, moqie, 2, 3] in [candidates[i : i + 4] for i in range(0, len(candidates), 4)]


@pytest.mark.parametrize("players", [3, 4])
def test_observation_without_tedashi_convention_remains_readable(players):
    env, _ = setup_env(players, 0, "mjsoul")
    obs = env.get_observation(0)
    payload = json.loads(base64.b64decode(obs.serialize_to_base64()))
    payload.pop("forced_tedashi", None)
    cls = Observation3P if players == 3 else Observation
    restored = cls.deserialize_from_base64(base64.b64encode(json.dumps(payload).encode()).decode())
    assert restored.drawn_tile == 53
    assert not restored.forced_tedashi
