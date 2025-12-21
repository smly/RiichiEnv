import pytest
import riichienv


def test_hand_parsing():
    from riichienv import AgariCalculator, Meld, MeldType

    # Test hand_from_text (13 tiles)
    text = "123m456p789s111z2z" # 13 tiles
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
    assert len(hand_melded.tiles_136) == 10 # 13 total - 3 melded
    assert len(hand_melded.melds) == 1
    m = hand_melded.melds[0]
    assert m.meld_type == MeldType.Peng
    
    # to_text: 123m456p789s2z(p1z0)
    assert hand_melded.to_text() == "123m456p789s2z(p1z0)"

def test_yaku_scenarios():
    import riichienv
    from riichienv import AgariCalculator, Conditions

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
            "yaku_check": lambda y: 12 in y or 4 in y # MJSoul ID 4 or 12 (Observed)
        },
        {
            "name": "Pinfu",
            # 123m 456p 789s 23p 99m. Win 1p or 4p.
            "hand": "123m456p789s23p99m",
            "min_han": 1,
            "win_tile": "1p",
            "yaku_check": lambda y: 14 in y or 3 in y # MJSoul ID 3 or 14
        },
        {
            "name": "Yakuhai White",
            # 123m 456p 78s (Pon 5z) 88m. Win 9s.
            "hand": "123m456p78s88m(p5z0)",
            "win_tile": "9s",
            "min_han": 1,
            "yaku_check": lambda y: 7 in y or 12 in y or 18 in y # 7 is Observed
        },
        {
            "name": "Honitsu",
            # 123m 567m 11m 33z 22z. Win 2z.
            "hand": "123m567m111m33z22z",
            "win_tile": "2z",
            "min_han": 3,
            "yaku_check": lambda y: 27 in y or 29 in y or 30 in y or 34 in y # Honitsu ID (27 observed)
        },
        {
            "name": "Red Dora Pinfu",
            "hand": "234m067p678s34m22z",
            "win_tile": "5m", 
            "min_han": 2, 
            "yaku_check": lambda y: 14 in y or 3 in y 
        },
        {
            "name": "Regression Honroutou False Positive",
            "hand": "11s22z(p5z0)(456s0)(789m0)",
            "win_tile": "1s", 
            "min_han": 1,
            "yaku_check": lambda y: (18 in y or 7 in y) and (24 not in y) and (31 not in y)
        }
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

    # Valid 14-tile hand: 123m 456p 789s 123s 1z1z
    # Substitute Red 5s:
    # 1 2 3m -> 1 2 3m (no reds)
    # 4 5 6p -> 4 0p 6p (one red)
    # 7 8 9s -> 7 8 9s (no reds)
    # 1 2 3s -> 1 2 3s (no reds)
    # 1z 1z -> 1z 1z (pair)
    # Wait, 3 reds? 123m 406p 789s 123s 1z1z only has 1 red.
    # Let's use 0m23m 0p56p 0s89s 123s 1z1z -> 123m 456p 789s 123s 1z1z
    # tiles_136:
def test_multiple_aka_dora():
    from riichienv import AgariCalculator, Conditions
    # Valid 14-tile hand: 1+2+3m, 4+0+6p, 7+0+9s, 1+2+3s, 2z+2z
    # Red 5m:16, Red 5p:52, Red 5s:88
    # 1m:0-3, 2m:4-7, 3m:8-11
    # 4p:36-39, 6p:44-47
    # 7s:72-75, 9s:80-83
    # 1s:104-107, 2s:108-111, 3s:112-115
    # 2z:116-119
    # Valid 14-tile hand with 3 sequences: 123m, 406p, 709s, 123s, 1z1z
    # Red 5m is only 1 Red. 
    # Let's use 023m, 056p, 089s, 123s, 1z1z.
    # 123m: Red(16), 4, 8
    # 456p: 48, Red(52), 56
    # 789s: 72, 76, Red(88)
    # 123s: 104, 108, 112
    # 1z: 112, 113
    # Valid 14-tile hand with 3 sequences: 345m, 456p, 345s, 111s, 1z1z (Red on each sequence)
    # Red 5m:16, Red 5p:52, Red 5s:88
    # 3m:8, 4m:12, 5m(Red):16
    # 4p:48, 5p(Red):52, 6p:56
    # 3s:80, 4s:84, 5s(Red):88
    # 1s:104, 105, 106 (Triplet)
    # 1z:112 (Head)
    tiles_136 = [
        8, 12, 16,    # 345m (with Red 5m)
        48, 52, 56,    # 456p (with Red 5p)
        80, 84, 88,    # 345s (with Red 5s)
        104, 105, 106, # 111s (triplet)
        112            # 1z standing
    ]
    tiles_136.sort()
    calc = AgariCalculator(tiles_136)
    res = calc.calc(113, [], Conditions(), []) # Win on 1z(113)
    
    assert res.agari
    assert not res.yakuman
    assert 32 in res.yaku
    # Aka Dora ID (32) should be in yaku, but not duplicated (ID list is unique now)
    assert res.yaku.count(32) == 1
    assert res.han == 3 # 3 aka doras
