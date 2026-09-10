"""Physical tile IDs and dora features must agree across observation encoders."""

import pytest

from riichienv import Action, ActionType, GameRule, Observation, Observation3P, Phase, RiichiEnv


def tile_types(players):
    return [0, *range(8, 34)] if players == 3 else list(range(34))


def dora_pairs(players):
    # Explicit cycles keep the expected result independent of the engine helper.
    manzu = [0, 8] if players == 3 else list(range(9))
    cycles = [manzu, list(range(9, 18)), list(range(18, 27)), [27, 28, 29, 30], [31, 32, 33]]
    return [(cycle[i - 1], kind) for cycle in cycles for i, kind in enumerate(cycle)]


def make_observation(players, player_id, indicators, tile, target):
    discards = [None] * players
    discards[target] = tile
    # Self must be excluded from the opponent features even if it has a discard.
    discards[player_id] = 88
    cls = Observation3P if players == 3 else Observation
    hands = [[] for _ in range(players)]
    hands[player_id] = [0, 1, 32, 33, 36, 37, 40, 41, 72, 73, 80, 81, 84]
    return cls(
        player_id=player_id,
        hands=hands,
        melds=[[] for _ in range(players)],
        discards=[[] for _ in range(players)],
        dora_indicators=indicators,
        scores=[35000 if players == 3 else 25000] * players,
        riichi_declared=[False] * players,
        legal_actions=[],
        events=[],
        honba=0,
        riichi_sticks=0,
        round_wind=0,
        oya=0,
        kyoku_index=0,
        waits=[],
        is_tenpai=False,
        riichi_sutehais=discards,
        last_tedashis=discards,
        last_discard=tile,
    )


def assert_tile_features(obs, players, tile, target, is_dora):
    types = tile_types(players)
    expected = (
        [types.index(tile // 4) / (len(types) - 1), float(tile in (16, 52, 88)), float(is_dora)]
        if tile is not None
        else [0.0] * 3
    )
    opponents = [p for p in range(players) if p != obs.player_id]
    opponent_features = [value for p in opponents for value in (expected if p == target else [0.0] * 3)]
    features = [
        (obs.encode_pass_context(), 194, expected),
        (obs.encode_last_tedashis(), 197, opponent_features),
        (obs.encode_riichi_sutehais(), 206, opponent_features),
    ]
    extended = memoryview(obs.encode_extended()).cast("f")
    width = len(types)
    assert len(extended) == 215 * width
    for raw, offset, values in features:
        assert list(memoryview(raw).cast("f")) == pytest.approx(values)
        for channel, value in enumerate(values, offset):
            assert list(extended[channel * width : (channel + 1) * width]) == pytest.approx([value] * width)
    if players == 3:
        # The third opponent's slots stay reserved in the 215-channel layout.
        for start in (203, 212):
            assert list(extended[start * width : (start + 3) * width]) == [0.0] * (3 * width)


@pytest.mark.parametrize("players", [3, 4])
@pytest.mark.parametrize("copy", range(4))
@pytest.mark.parametrize("indicator_copy", range(4))
def test_dora_features_cover_every_tile_kind_and_copy(players, copy, indicator_copy):
    for indicator_kind, dora_kind in dora_pairs(players):
        player_id = indicator_kind % players
        target = (player_id + 1 + copy % (players - 1)) % players
        indicator = indicator_kind * 4 + indicator_copy
        tile = dora_kind * 4 + copy
        obs = make_observation(players, player_id, [indicator], tile, target)
        assert_tile_features(obs, players, tile, target, is_dora=True)

        # An indicator is not itself a dora, including red fives and honors.
        tile = indicator_kind * 4 + copy
        obs = make_observation(players, player_id, [indicator], tile, target)
        assert_tile_features(obs, players, tile, target, is_dora=False)


@pytest.mark.parametrize("players", [3, 4])
@pytest.mark.parametrize("tile", [None, 52, 53, 91])
@pytest.mark.parametrize("indicators", [[], [48, 49, 84]])
def test_dora_features_with_missing_tiles_and_multiple_indicators(players, tile, indicators):
    obs = make_observation(players, 1, indicators, tile, 0)
    # is_dora is a boolean even when two indicators refer to the same kind.
    assert_tile_features(obs, players, tile, 0, is_dora=bool(indicators) and tile is not None)


@pytest.mark.parametrize("players", [3, 4])
@pytest.mark.parametrize("preset", ["mjsoul", "tenhou"])
def test_last_discard_preserves_physical_tile_for_every_observer(players, preset):
    env = RiichiEnv(f"{players}p-red-single", seed=42, rule=getattr(GameRule, f"default_{preset}")())
    cls = Observation3P if players == 3 else Observation
    for observer in range(players):
        assert env.get_observation(observer).last_discard is None

    actors = set()
    for _ in range(players * 2):
        actor = env.current_player
        actors.add(actor)
        discard = next(a for a in env._get_legal_actions(actor) if a.action_type == ActionType.DISCARD and a.tile > 3)
        env.step({actor: discard})
        assert env.last_discard == (actor, discard.tile)
        for observer in range(players):
            obs = env.get_observation(observer)
            assert obs.last_discard == discard.tile
            restored = cls.deserialize_from_base64(obs.serialize_to_base64())
            assert restored.last_discard == discard.tile
            expected_kind = tile_types(players).index(discard.tile // 4) / (len(tile_types(players)) - 1)
            assert memoryview(obs.encode_pass_context()).cast("f")[0] == pytest.approx(expected_kind)
        while env.phase == Phase.WaitResponse:
            env.step({p: Action(ActionType.PASS) for p in env.active_players})
    assert actors == set(range(players))


@pytest.mark.parametrize("players", [3, 4])
@pytest.mark.parametrize("preset", ["mjsoul", "tenhou"])
@pytest.mark.parametrize("tsumogiri", [False, True])
def test_riichi_discard_features_after_separate_declaration(players, preset, tsumogiri):
    # A complete wall deals 111333555777p EE to the dealer. Discarding 5p
    # leaves a shanpon wait; 4p is the indicator, making the non-red 5p dora.
    tile = 53
    drawn = tile if tsumogiri else 109
    hand = [36, 37, 38, 44, 45, 46, 52, 53, 54, 60, 61, 62, 108, 109]
    pool = [t for kind in tile_types(players) for t in range(kind * 4, kind * 4 + 4) if t not in [*hand, 48]]
    hands = [[t for t in hand if t != drawn]]
    for _ in range(players - 1):
        hands.append(pool[:13])
        del pool[:13]
    deal = [t for start in (0, 4, 8) for h in hands for t in h[start : start + 4]]
    tail = 8 if players == 3 else 4
    wall = deal + [h[12] for h in hands] + [drawn] + pool[:-tail] + [48] + pool[-tail:]
    assert sorted(wall) == [t for kind in tile_types(players) for t in range(kind * 4, kind * 4 + 4)]
    env = RiichiEnv(f"{players}p-red-single", seed=42, rule=getattr(GameRule, f"default_{preset}")())
    env.reset(wall=wall, oya=0)
    assert env.drawn_tile == drawn
    assert env.dora_indicators == [48]

    declare = next(a for a in env._get_legal_actions(0) if a.action_type == ActionType.RIICHI)
    assert declare.tile is None
    env.step({0: declare})
    assert env.get_observation(1).riichi_sutehais == [None] * players
    discard = next(a for a in env._get_legal_actions(0) if a.action_type == ActionType.DISCARD and a.tile == tile)
    env.step({0: discard})
    assert env.discard_is_riichi[0] == [True]

    def check_discard():
        for observer in range(players):
            obs = env.get_observation(observer)
            assert obs.riichi_sutehais == [tile] + [None] * (players - 1)
            assert obs.last_tedashis[0] == (None if tsumogiri and preset == "tenhou" else tile)
            if observer != 0:
                expected = [tile_types(players).index(tile // 4) / (len(tile_types(players)) - 1), 0.0, 1.0]
                assert list(memoryview(obs.encode_riichi_sutehais()).cast("f")[:3]) == pytest.approx(expected)
                width = len(tile_types(players))
                assert memoryview(obs.encode_extended()).cast("f")[208 * width] == 1.0

    check_discard()
    while env.phase == Phase.WaitResponse:
        env.step({p: Action(ActionType.PASS) for p in env.active_players})
    check_discard()
    env.reset()
    for observer in range(players):
        assert env.get_observation(observer).riichi_sutehais == [None] * players
