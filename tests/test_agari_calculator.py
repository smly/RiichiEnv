import riichienv as rv


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
        player_wind=rv.Wind.North,  # Doesn't strict matter for Pinfu unless Jikaze head
        round_wind=rv.Wind.East,
    )
    res = rv.HandEvaluator(hand, melds).calc(
        win_tile=win_tile, dora_indicators=[], conditions=cond, ura_indicators=[]
    )

    # Check Pinfu (ID 14)
    assert 14 not in res.yaku


def test_agari_calc_from_text():
    hand = rv.HandEvaluator.hand_from_text("123m456p789s111z2z")
    win_tile = rv.parse_tile("2z")

    # Default: Oya (East), Ron Agari
    res = hand.calc(win_tile, conditions=rv.Conditions())
    assert res.is_win
    assert res.han == 2
    assert res.fu == 40
    assert res.tsumo_agari_oya == 0
    assert res.tsumo_agari_ko == 0
    assert res.ron_agari == 3900

    # Ko (South)
    res = hand.calc(win_tile, conditions=rv.Conditions(tsumo=True, player_wind=rv.Wind.South))
    assert res.is_win
    assert res.han == 2
    assert res.fu == 40
    assert res.tsumo_agari_oya == 1300
    assert res.tsumo_agari_ko == 700
    assert res.ron_agari == 0

    # Oya (East)
    res = hand.calc(win_tile, conditions=rv.Conditions(tsumo=True, player_wind=rv.Wind.East))
    assert res.is_win
    assert res.han == 3
    assert res.fu == 40
    assert res.tsumo_agari_oya == 0
    assert res.tsumo_agari_ko == 2600
    assert res.ron_agari == 0

    # Ron (East)
    res = hand.calc(win_tile, conditions=rv.Conditions(tsumo=False, player_wind=rv.Wind.East))
    assert res.is_win
    assert res.han == 2
    assert res.fu == 40
    assert res.tsumo_agari_oya == 0
    assert res.tsumo_agari_ko == 0
    assert res.ron_agari == 3900

    # Ko (West)
    res = hand.calc(win_tile, conditions=rv.Conditions(tsumo=True, player_wind=rv.Wind.West))
    assert res.is_win
    assert res.han == 2
    assert res.fu == 40
    assert res.tsumo_agari_oya == 1300
    assert res.tsumo_agari_ko == 700
    assert res.ron_agari == 0

    # Ko (North)
    res = hand.calc(win_tile, conditions=rv.Conditions(tsumo=True, player_wind=rv.Wind.North))
    assert res.is_win
    assert res.han == 2
    assert res.fu == 40
    assert res.tsumo_agari_oya == 1300
    assert res.tsumo_agari_ko == 700
    assert res.ron_agari == 0


def test_yaku_shibari():
    # Hand: 1, 2, 3m, 5p, 6p, 7p, 2s, 2s, 6s, 7s, 8s
    # Melds: Chi(5m, 6m, 7m) where 5m is Red (0m) or 5pr.
    # Actually, previous reproduction used:
    # Hand: 2m, 3m, 0p(5pr), 6p, 7p, 2s, 2s, 6s, 7s, 8s (10 tiles) + Chi(567m)
    # Win: 1m

    # Let's reconstruct cleanly:
    # Hand: 123m (1m win), 567p (with red 5p), 22s, 678s.
    # Melds: None (Closed) or Open? If Closed, Menzen Tsumo is Yaku.
    # Must be Open to fail Yaku Shibari if no other Yaku.

    # Hand: 2m, 3m, 5pr, 6p, 7p, 2s, 2s, 6s, 7s, 8s.
    # Meld: Chi 5,6,7m (Open).
    # Win: 1m (from Ron)

    # 2m=4, 3m=8
    # 5pr=52
    # 6p=56, 7p=60
    # 2s=76, 77 (pair)
    # 6s=92, 7s=96, 8s=100

    hand_tiles = [4, 8, 52, 56, 60, 76, 77, 92, 96, 100]

    # Meld 567m Open
    # 5m=16, 6m=20, 7m=24
    m = rv.Meld(rv.MeldType.Chi, [16, 20, 24], True)

    win_tile = 0  # 1m

    calc = rv.HandEvaluator(hand_tiles, [m])

    # Default conditions (Ron, no Riichi = No Yaku context except Dora)

    res = calc.calc(win_tile, dora_indicators=[], conditions=rv.Conditions())

    # Should not be agari (either shape invalid or Yaku Shibari)
    # Ideally should be Yaku Shibari if shape is valid.
    # But for now we just verify it doesn't allow a win.
    assert not res.is_win, "Yaku Shibari failed: Allowed agari with only Aka Dora"
