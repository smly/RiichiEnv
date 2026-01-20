import riichienv
from riichienv import AgariCalculator, Conditions, MeldType


def test_hand_parsing():
    # Test hand_from_text (13 tiles)
    text = "123m456p789s111z2z"  # 13 tiles
    hand = AgariCalculator.hand_from_text(text)
    tiles_list = list(hand.tiles_136)
    assert len(tiles_list) == 13

    # Test to_text reciprocity (canonical grouping)
    # 111z2z -> 1112z
    assert hand.to_text() == "123m456p789s1112z"

    # Test with Red 5
    # Need 13 tiles: 055m (3) + 456p (3) + 789s (3) + 1122z (4) = 13
    text_red = "055m456p789s1122z"
    hand_red = AgariCalculator.hand_from_text(text_red)
    tiles_list_red = list(hand_red.tiles_136)
    assert 16 in tiles_list_red
    assert hand_red.to_text() == "055m456p789s1122z"

    # Test calc_from_text (14 tiles)
    # 123m 456p 789s 111z 22z. Win on 2z.
    # Hand including win tile: 123m456p789s111z22z (14)
    res = AgariCalculator.calc_from_text("123m456p789s111z22z")
    assert res.agari
    assert res.han > 0

    # Test with Melds (13 tiles total)
    # 123m (3) + 456p (3) + 789s (3) + 2z (1) + Pon 1z (3) = 13
    melded_text = "123m456p789s2z(p1z0)"
    hand_melded = AgariCalculator.hand_from_text(melded_text)
    assert len(hand_melded.tiles_136) == 10  # 13 total - 3 melded
    assert len(hand_melded.melds) == 1
    m = hand_melded.melds[0]
    assert m.meld_type == MeldType.Peng

    # to_text: 123m456p789s2z(p1z0)
    assert hand_melded.to_text() == "123m456p789s2z(p1z0)"


def test_yaku_scenarios():
    def get_tile(s):
        tiles, _ = riichienv.parse_hand(s)
        tiles = list(tiles)
        return tiles[0]

    scenarios = [
        {
            "name": "Tanyao",
            "hand": "234m234p234s66m88s",
            "win_tile": "6m",
            "min_han": 1,
            "yaku_check": lambda y: 12 in y,  # Tanyao ID 12
        },
        {
            "name": "Pinfu",
            # 123m 456p 789s 23p 99m. Win 1p or 4p.
            "hand": "123m456p789s23p99m",
            "win_tile": "1p",
            "min_han": 1,
            "yaku_check": lambda y: 14 in y,  # Pinfu ID 14
        },
        {
            "name": "Yakuhai White",
            # 123m 456p 78s (Pon 5z) 88m. Win 9s.
            "hand": "123m456p78s88m(p5z0)",
            "win_tile": "9s",
            "min_han": 1,
            "yaku_check": lambda y: 7 in y,  # Yakuhai White (7)
        },
        {
            "name": "Honitsu",
            # 123m 567m 11m 33z 22z. Win 2z.
            "hand": "123m567m111m33z22z",
            "win_tile": "2z",
            "min_han": 3,
            "yaku_check": lambda y: 27 in y,  # Honitsu ID 27
        },
        {
            "name": "Red Dora Pinfu",
            "hand": "234m067p678s34m22z",
            "win_tile": "5m",
            "min_han": 2,
            "yaku_check": lambda y: 14 in y,  # Pinfu ID 14
        },
        {
            "name": "Regression Honroutou False Positive",
            "hand": "11s22z(p5z0)(456s0)(789m0)",
            "win_tile": "1s",
            "min_han": 1,
            "yaku_check": lambda y: (7 in y) and (24 not in y) and (31 not in y),
        },
    ]

    for s in scenarios:
        hand_str = s["hand"]
        win_tile_str = s["win_tile"]
        print(f"Testing {s['name']}...")

        calc = AgariCalculator.hand_from_text(hand_str)
        win_tile_val = get_tile(win_tile_str)

        res = calc.calc(win_tile_val, conditions=Conditions())

        if "yaku_check" in s:
            assert s["yaku_check"](res.yaku), f"{s['name']}: Yaku check failed. Got {res.yaku}"


def test_multiple_aka_dora():
    # Valid 14-tile hand: 1+2+3m, 4+0+6p, 7+0+9s, 1+2+3s, 2z+2z
    # Red 5m:16, Red 5p:52, Red 5s:88
    # Valid 14-tile hand with 3 sequences: 345m, 456p, 345s, 666s, 8p8p (All Simples -> Tanyao)
    # Red 5m:16, Red 5p:52, Red 5s:88
    # 6s: 92,93,94
    # 8p: 64,65

    tiles_136 = [
        8,
        12,
        16,  # 345m (with Red 5m)
        48,
        52,
        56,  # 456p (with Red 5p)
        80,
        84,
        88,  # 345s (with Red 5s)
        92,
        93,
        94,  # 6s triplet (Simple)
        64,  # 8p standing (Simple)
    ]
    tiles_136.sort()

    calc = AgariCalculator(tiles_136)
    res = calc.calc(65, [], Conditions(), [])  # Win on 8p(65) -> Tanyao

    assert res.agari
    assert not res.yakuman
    assert 32 in res.yaku
    assert 12 in res.yaku
    # Aka Dora ID (32) should be in yaku, but not duplicated (ID list is unique now)
    assert res.yaku.count(32) == 1
    assert res.han == 4  # 3 aka doras + 1 Tanyao


def test_only_aka_dora_fails():
    # Hand (No Yaku except Aka Dora):
    # 345m (Red 5m), 456p (Red 5p), 345s (Red 5s)
    # 9 s Triplet (Terminal)
    # 1 z Pair (East) - Assuming West round/South player -> No Yakuhai

    tiles_136 = [
        8,
        12,
        16,  # 345m (Red 5m)
        48,
        52,
        56,  # 456p (Red 5p)
        80,
        84,
        88,  # 345s (Red 5s)
        104,
        105,
        106,  # 9s Triplet (Terminal, breaks Tanyao)
        108,  # 1z Pair (East)
    ]
    tiles_136.sort()

    calc = AgariCalculator(tiles_136)

    # Conditions: Non-East player (South), Non-East Round (South) -> 1z is NOT Yakuhai
    cond = Conditions(player_wind=riichienv.Wind.South, round_wind=riichienv.Wind.South)

    # Win on 1z (109 West? No, 108 is East. 112 is South? No. 108,109,110,111 = East.)
    # 108 is Pair head. Win on 109 (East).
    res = calc.calc(109, [], cond, [])

    # Should have 3 Han (Aka Doras) but NO Yaku -> Agari=False
    # Note: If is_agari=False or Yaku Shibari fails, result must be agari=False.
    # We verify detection of 'Not Agari' primarily.
    assert not res.agari, "Should fail Yaku Shibari with only Aka Doras"
    # assert res.han == 3 # Omitted as implementation might return 0 if !agari


def test_kyoku4_regression():
    # 4p, 6p, 4m, 4p, 7m, 3m, 0m(5m), 1m, 7m, 5p, 7m, 0p(5p), 3p, 1m
    # Manzu: 1m(2), 3m(1), 4m(1), 5m(1), 7m(3)
    # Pinzu: 3p(1), 4p(2), 5p(2), 6p(1)

    standing = [48, 56, 12, 49, 24, 8, 16, 0, 25, 53, 26, 52, 44]
    win_tile = 0  # 1m

    # Case 1: All 14 tiles passed to constructor
    temp_tiles = sorted(standing + [win_tile])
    calc2 = AgariCalculator(temp_tiles, [])
    res = calc2.calc(
        win_tile, dora_indicators=[], ura_indicators=[], conditions=Conditions(tsumo=True, tsumo_first_turn=True)
    )

    print(f"DEBUG: Res agari={res.agari}, yaku={res.yaku}")
    assert res.agari
    assert 35 in res.yaku  # Tenhou
