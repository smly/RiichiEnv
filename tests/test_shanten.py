import riichienv


def test_shanten_3p_no_manzu_sequence():
    """1111m111122233z: both 4P and 3P shanten = 1.

    Best decomposition: 111m(koutsu) + 111z(koutsu) + 222z(koutsu) + 33z(pair)
    = 3 mentsu + 1 pair. No manzu sequences needed.
    """
    tiles, _ = riichienv.parse_hand("1111m111122233z")
    hand = list(tiles)

    shanten_4p = riichienv.calculate_shanten(hand)
    shanten_3p = riichienv.calculate_shanten_3p(hand)

    assert shanten_4p == 1, f"4P shanten: expected 1, got {shanten_4p}"
    assert shanten_3p == 1, f"3P shanten: expected 1, got {shanten_3p}"


def test_shanten_3p_complete():
    """Complete hand: 4 koutsu + 1 pair."""
    tiles, _ = riichienv.parse_hand("111m111z222z333z44z")
    hand = list(tiles)

    assert riichienv.calculate_shanten(hand) == -1
    assert riichienv.calculate_shanten_3p(hand) == -1


def test_shanten_3p_pinzu_sequences():
    """Pinzu sequences are valid in 3P: complete hand with 3 shuntsu + pair."""
    tiles, _ = riichienv.parse_hand("123456789p11z")
    hand = list(tiles)

    assert riichienv.calculate_shanten(hand) == -1
    assert riichienv.calculate_shanten_3p(hand) == -1


def test_shanten_3p_tenpai():
    """Kokushi tenpai: valid in both 3P and 4P."""
    tiles, _ = riichienv.parse_hand("19m19p19s1234567z")
    hand = list(tiles)

    assert riichienv.calculate_shanten(hand) == 0
    assert riichienv.calculate_shanten_3p(hand) == 0


def test_shanten_3p_chiitoitsu():
    """Chiitoitsu tenpai."""
    tiles, _ = riichienv.parse_hand("1199m1199p1199s1z")
    hand = list(tiles)

    assert riichienv.calculate_shanten(hand) == 0
    assert riichienv.calculate_shanten_3p(hand) == 0


def test_shanten_3p_consistency_with_4p():
    """For valid 3P hands (no 2m-8m), 3P and 4P shanten should match."""
    test_hands = [
        "19m19p19s1234567z",
        "1199m1199p1199s1z",
        "111m999m111p11z",
        "1m9m1p9p1s9s12345z",
    ]
    for hand_str in test_hands:
        tiles, _ = riichienv.parse_hand(hand_str)
        hand = list(tiles)
        shanten_4p = riichienv.calculate_shanten(hand)
        shanten_3p = riichienv.calculate_shanten_3p(hand)
        assert shanten_4p == shanten_3p, (
            f"{hand_str}: 4P={shanten_4p} != 3P={shanten_3p}"
        )
