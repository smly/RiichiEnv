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
    tiles, _ = riichienv.parse_hand("123456789p11222z")
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


def test_shanten_3p_manzu_shuntsu_blocked():
    """3P blocks manzu shuntsu: 123m is a complete shuntsu in 4P but 3 isolated tiles in 3P."""
    # 123m456p789s11z (11 tiles): complete in 4P, 1-shanten in 3P
    tiles, _ = riichienv.parse_hand("123m456p789s11z")
    hand = list(tiles)
    assert riichienv.calculate_shanten(hand) == -1
    assert riichienv.calculate_shanten_3p(hand) == 1


def test_shanten_3p_manzu_shuntsu_blocked_13():
    """3P blocks manzu shuntsu: 123m makes tenpai in 4P but not in 3P (13 tiles)."""
    # 123m111z222z333z4z: tenpai in 4P (123m+3 koutsu+tanki), 2-shanten in 3P
    tiles, _ = riichienv.parse_hand("123m111z222z333z4z")
    hand = list(tiles)
    assert riichienv.calculate_shanten(hand) == 0
    assert riichienv.calculate_shanten_3p(hand) == 2


def test_shanten_3p_manzu_taatsu_blocked():
    """3P blocks manzu taatsu/kanchan partial sequences."""
    # 12m taatsu counts as partial in 4P but not in 3P
    tiles, _ = riichienv.parse_hand("12m111p222s333z44z")
    hand = list(tiles)
    assert riichienv.calculate_shanten(hand) == 0
    assert riichienv.calculate_shanten_3p(hand) == 1

    # 13m kanchan counts as partial in 4P but not in 3P
    tiles, _ = riichienv.parse_hand("13m111p222s333z44z")
    hand = list(tiles)
    assert riichienv.calculate_shanten(hand) == 0
    assert riichienv.calculate_shanten_3p(hand) == 1


def test_shanten_3p_consistency_with_4p():
    """For valid 3P hands (no 2m-8m), 3P and 4P shanten should match."""
    test_hands = [
        ("19m19p19s1234567z", 0, 0),  # Kokushi tenpai
        ("1199m1199p1199s1z", 0, 0),  # Chiitoitsu tenpai
        ("111m999m111p11z", -1, -1),  # Complete hand with 4 koutsu + pair
    ]
    for hand_str, expected_4p, expected_3p in test_hands:
        tiles, _ = riichienv.parse_hand(hand_str)
        hand = list(tiles)
        shanten_4p = riichienv.calculate_shanten(hand)
        shanten_3p = riichienv.calculate_shanten_3p(hand)
        assert shanten_4p == shanten_3p, f"{hand_str}: 4P={shanten_4p} != 3P={shanten_3p}"
        assert shanten_4p == expected_4p, f"{hand_str}: expected 4P={expected_4p}, got {shanten_4p}"
        assert shanten_3p == expected_3p, f"{hand_str}: expected 3P={expected_3p}, got {shanten_3p}"
