import riichienv as rv
import riichienv.convert as cvt


def test_pinfu():
    # Hand from debug log:
    # DEBUG: Hand: [12, 17, 21, 68, 68, 80, 80, 83, 96, 104, 120, 120, 122]
    # DEBUG: Win Tile: 100 (8s)

    # 12: 4m
    # 17: 5m
    # 21: 6m
    # 68: 9p
    # 68: 9p (Duplicate!)
    # 80: 3s
    # 80: 3s (Duplicate!)
    # 83: 3s
    # 96: 7s
    # 104: 9s
    # 120: N
    # 120: N
    # 122: N

    hand = [12, 17, 21, 68, 68, 80, 80, 83, 96, 104, 120, 120, 122]
    win_tile = 100
    melds = []  # No melds

    # Conditions
    # Riichi = True
    cond = rv.Conditions(
        tsumo=False,
        riichi=True,
        player_wind=3,  # Doesn't strict matter for Pinfu unless Jikaze head
        round_wind=0,
    )
    res = rv.AgariCalculator(hand, melds).calc(
        win_tile=win_tile, dora_indicators=[], conditions=cond, ura_indicators=[]
    )

    # Check Pinfu (ID 14)
    assert 14 not in res.yaku


def test_agari_calc_from_text():
    hand = rv.AgariCalculator.hand_from_text("123m456p789s111z2z")
    win_tile = rv.parse_tile("2z")

    # Default: Oya (East), Ron Agari
    res = hand.calc(win_tile, conditions=rv.Conditions())
    assert res.agari
    assert res.han == 2
    assert res.fu == 40
    assert res.tsumo_agari_oya == 0
    assert res.tsumo_agari_ko == 0
    assert res.ron_agari == 3900

    # Ko (South)
    res = hand.calc(win_tile, conditions=rv.Conditions(tsumo=True, player_wind=rv.SOUTH))
    assert res.agari
    assert res.han == 2
    assert res.fu == 40
    assert res.tsumo_agari_oya == 1300
    assert res.tsumo_agari_ko == 700
    assert res.ron_agari == 0

    # Oya (East)
    res = hand.calc(win_tile, conditions=rv.Conditions(tsumo=True, player_wind=rv.EAST))
    assert res.agari
    assert res.han == 3
    assert res.fu == 40
    assert res.tsumo_agari_oya == 0
    assert res.tsumo_agari_ko == 2600
    assert res.ron_agari == 0

    # Ron (East)
    res = hand.calc(win_tile, conditions=rv.Conditions(tsumo=False, player_wind=rv.EAST))
    assert res.agari
    assert res.han == 2
    assert res.fu == 40
    assert res.tsumo_agari_oya == 0
    assert res.tsumo_agari_ko == 0
    assert res.ron_agari == 3900

    # Ko (West)
    res = hand.calc(win_tile, conditions=rv.Conditions(tsumo=True, player_wind=rv.WEST))
    assert res.agari
    assert res.han == 2
    assert res.fu == 40
    assert res.tsumo_agari_oya == 1300
    assert res.tsumo_agari_ko == 700
    assert res.ron_agari == 0

    # Ko (North)
    res = hand.calc(win_tile, conditions=rv.Conditions(tsumo=True, player_wind=rv.NORTH))
    assert res.agari
    assert res.han == 2
    assert res.fu == 40
    assert res.tsumo_agari_oya == 1300
    assert res.tsumo_agari_ko == 700
    assert res.ron_agari == 0
