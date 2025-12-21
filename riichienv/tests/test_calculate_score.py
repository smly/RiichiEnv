from riichienv import calculate_score


def test_calculate_score() -> None:
    score = calculate_score(4, 30, False, True)
    assert score.pay_tsumo_oya == 3900
    assert score.pay_tsumo_ko == 2000
    assert score.total == 7900