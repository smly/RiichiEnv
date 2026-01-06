import pytest

from riichienv import RiichiEnv


class TestRiichiEnvRanksPoints:
    def test_ranks_distinct(self):
        env = RiichiEnv(seed=42)
        env.reset(scores=[30000, 20000, 40000, 10000])
        # P2: 40000 (1st)
        # P0: 30000 (2nd)
        # P1: 20000 (3rd)
        # P3: 10000 (4th)
        assert env.ranks() == [2, 3, 1, 4]

    def test_ranks_ties(self):
        env = RiichiEnv(seed=42)
        env.reset(oya=0, scores=[25000, 25000, 25000, 25000])
        assert env.ranks() == [1, 2, 3, 4]

        env.reset(oya=1, scores=[25000, 25000, 25000, 25000])
        assert env.ranks() == [1, 2, 3, 4]

        env.reset(oya=0, scores=[30000, 30000, 20000, 20000])
        # P0 and P1 tie for 1st/2nd. P0 (index 0) wins tie.
        # P2 and P3 tie for 3rd/4th. P2 (index 2) wins tie.
        assert env.ranks() == [1, 2, 3, 4]

    def test_points_basic(self):
        env = RiichiEnv(seed=42)
        # basic: soten_weight=1, soten_base=25000, jun_weight=[50, 10, -10, -50]
        # P0: 35000 -> (35000-25000)/1000 * 1 + 50 = 10 + 50 = 60
        # P1: 25000 -> (25000-25000)/1000 * 1 + 10 = 0 + 10 = 10
        # P2: 25000 -> (25000-25000)/1000 * 1 - 10 = 0 - 10 = -10
        # P3: 15000 -> (15000-25000)/1000 * 1 - 50 = -10 - 50 = -60
        env.reset(oya=0, scores=[35000, 25000, 25000, 15000])
        assert env.points("basic") == [60, 10, -10, -60]

    def test_points_ouza_tyoujyo(self):
        env = RiichiEnv(seed=42)
        # ouza-tyoujyo: soten_weight=0, soten_base=25000, jun_weight=[100, 40, -40, -100]
        env.reset(oya=0, scores=[40000, 30000, 20000, 10000])
        assert env.points("ouza-tyoujyo") == [100, 40, -40, -100]

    def test_points_ouza_normal(self):
        env = RiichiEnv(seed=42)
        # ouza-normal: soten_weight=0, soten_base=25000, jun_weight=[50, 20, -20, -50]
        env.reset(oya=0, scores=[40000, 30000, 20000, 10000])
        assert env.points("ouza-normal") == [50, 20, -20, -50]

    def test_points_invalid_rule(self):
        env = RiichiEnv(seed=42)
        env.reset()
        with pytest.raises(ValueError, match="Unknown preset rule: nonexistent"):
            env.points("nonexistent")
