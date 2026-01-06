import pytest

import riichienv.convert as cvt
from riichienv import RiichiEnv


@pytest.mark.skip(reason="Legacy python test - Rust implementation pending or parity missing")
def test_can_pon_aka():
    env = RiichiEnv()

    # Rank 4 is 5m. IDs: 16(aka), 17, 18, 19
    aka_5m = 16
    n1_5m = 17
    n2_5m = 18
    n3_5m = 19

    # Case 1: Two normals, no aka
    hand = [n1_5m, n2_5m, 0, 1, 2]
    options = env._can_pon(hand, n3_5m)  # Someone discards 5m
    assert len(options) == 1
    assert sorted(options[0]) == sorted([n1_5m, n2_5m])

    # Case 2: One normal, one aka
    hand = [aka_5m, n1_5m, 0, 1, 2]
    options = env._can_pon(hand, n2_5m)
    assert len(options) == 1
    assert sorted(options[0]) == sorted([aka_5m, n1_5m])

    # Case 3: Two normals, one aka
    hand = [aka_5m, n1_5m, n2_5m, 0, 1]
    options = env._can_pon(hand, n3_5m)
    assert len(options) == 2
    # Option 1: Two normals
    # Option 2: One normal, one aka
    mpsz_options = [sorted(cvt.tid_to_mpsz_list(opt)) for opt in options]
    assert ["5m", "5m"] in mpsz_options
    assert ["0m", "5m"] in mpsz_options


@pytest.mark.skip(reason="Legacy python test - Rust implementation pending or parity missing")
def test_can_chi_aka():
    env = RiichiEnv()

    # 3m, 4m, 5m (aka)
    # 3m: rank 2 (ids 8-11)
    # 4m: rank 3 (ids 12-15)
    # 5m: rank 4 (ids 16(aka), 17-19)
    # 6m: rank 5 (ids 20-23)

    t3m = 8
    t4m = 12
    aka_5m = 16
    n5m = 17
    t6m = 20

    # Discard 4m, Hand has 3m and 5m.
    # If hand has only normal 5m
    hand = [t3m, n5m, 100, 101]
    options = env._can_chi(hand, t4m)
    assert len(options) == 1
    assert sorted(options[0]) == sorted([t3m, n5m])

    # If hand has only aka 5m
    hand = [t3m, aka_5m, 100, 101]
    options = env._can_chi(hand, t4m)
    assert len(options) == 1
    assert sorted(options[0]) == sorted([t3m, aka_5m])

    # If hand has both normal 5m and aka 5m
    hand = [t3m, n5m, aka_5m, 100]
    options = env._can_chi(hand, t4m)
    assert len(options) == 2
    mpsz_options = [sorted(cvt.tid_to_mpsz_list(opt)) for opt in options]
    assert ["3m", "5m"] in mpsz_options
    assert ["0m", "3m"] in mpsz_options

    # Complex case: Discard 6m, Hand has 4m, 5m(normal), 5m(aka), 7m, 8m
    # 4-5-6 chi and 6-7-8 chi
    t7m = 24
    t8m = 28
    hand = [t4m, n5m, aka_5m, t7m, t8m, 100]
    options = env._can_chi(hand, t6m)
    # Pairs for 6m (rank 5): (3, 4) and (6, 7)
    # Candidates for rank 3 (4m): [t4m]
    # Candidates for rank 4 (5m): [aka_5m, n5m]
    # Candidates for rank 6 (7m): [t7m]
    # Candidates for rank 7 (8m): [t8m]
    # (3, 4) -> [t4m, aka_5m], [t4m, n5m]
    # (4, 6) -> [aka_5m, t7m], [n5m, t7m]
    # (6, 7) -> [t7m, t8m]
    assert len(options) == 5
    mpsz_options = [sorted(cvt.tid_to_mpsz_list(opt)) for opt in options]
    assert ["0m", "4m"] in mpsz_options
    assert ["4m", "5m"] in mpsz_options
    assert ["0m", "7m"] in mpsz_options
    assert ["5m", "7m"] in mpsz_options
    assert ["7m", "8m"] in mpsz_options
