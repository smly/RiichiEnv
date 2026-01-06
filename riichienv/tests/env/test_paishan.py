import riichienv.convert as cvt

from .helper import helper_setup_env


class TestPaishan:
    def test_paishan_to_wall(self):
        """Verify paishan parsing matches expectations."""
        s = "1m2m3m"
        wall = cvt.paishan_to_wall(s)
        assert len(wall) == 3
        tids = [cvt.mpsz_to_tid(t) for t in ["1m", "2m", "3m"]]
        assert wall[0] in [tids[0], tids[0] + 1, tids[0] + 2, tids[0] + 3]
        assert wall[1] in [tids[1], tids[1] + 1, tids[1] + 2, tids[1] + 3]

        # Test duplicates: "1m1m"
        s = "1m1m"
        wall = cvt.paishan_to_wall(s)
        assert len(wall) == 2
        base = cvt.mpsz_to_tid("1m")
        assert wall == [base, base + 1]

    def test_env_init_from_paishan(self):
        """Verify RiichiEnv extracts Dora from Paishan wall."""
        paishan_wall = list(range(136))
        env = helper_setup_env(wall=paishan_wall)

        # Check Dora
        assert len(env.dora_indicators) == 1
        # self.wall is reversed: [135, 134, ..., 0]
        # self.wall[4] should be 131.
        assert env.dora_indicators[0] == 131

        # Check Ura
        ura_markers = env._get_ura_markers()
        assert len(ura_markers) == 1
        # Ura is wall[5] = 130.
        # _get_ura_markers returns MJAI strings.
        assert ura_markers[0] == cvt.tid_to_mjai(130)

    def test_real_dora_reveal(self):
        paishan_wall_str = (
            "3s9s1m5s9s3m9s4p7z1z3m6p3m3p5z1z2s7m5z2m7p3z7p7z5m5m5s6m6p4p3p4s7m7s4m6s9m5p5m6m3s2m3s9m3z4z4z1s8p4s7z8s1p1m9p"
            "9m8s4z6z2z1s4s2m3m8s3p1m7s8m2s1p2m6s1z9p3z8p6z5z2p2z2z1m7p4p7s6z6z6s5p8m9m3p2p3s7s7p6p2s9p6m1p5p1z6p2p4m7m5z9s2s4p5s0s4m3z8m1s"
            "2z6m7m0m6s1p8s8m8p4z1s0p9p4s4m2p7z8p"
        )
        paishan_wall = cvt.paishan_to_wall(paishan_wall_str)
        env = helper_setup_env(wall=paishan_wall)

        assert len(env.dora_indicators) == 1
        assert cvt.tid_to_mpsz(env.dora_indicators[0]) == "4s"

        # Simulate Rinshan draw (Popping from Dead Wall side 0)
        wall = env.wall
        wall.pop(0)
        env.wall = wall
        env.rinshan_draw_count += 1
        env.pending_kan_dora_count = 1  # Manual usage requires setting pending count for correct index calc
        env._reveal_kan_dora()
        assert len(env.dora_indicators) == 2
        # Index 6 of original wall (now index 5) is 0p.
        assert cvt.tid_to_mpsz(env.dora_indicators[1]) == "0p"

    def test_kan_dora_reveal(self):
        """Verify Kan Dora reveal logic."""
        paishan_wall = list(range(136))
        env = helper_setup_env(wall=paishan_wall)

        # Initial: wall[4] (131).
        assert len(env.dora_indicators) == 1
        assert env.dora_indicators[0] == 131

        # Kan 1
        # Simulate Rinshan draw
        wall = env.wall
        wall.pop(0)
        env.wall = wall
        env.rinshan_draw_count += 1
        env.pending_kan_dora_count = 1  # Manual usage requires setting pending count for correct index calc
        env._reveal_kan_dora()
        assert len(env.dora_indicators) == 2
        # Next index: 6. self.wall[6] (new wall[5]) = 129.
        assert env.dora_indicators[1] == 129

        # Check Ura
        # Ura should be wall[5] (130) and wall[7] (128).
        ura_markers = env._get_ura_markers()
        assert len(ura_markers) == 2
        assert ura_markers[0] == cvt.tid_to_mjai(130)
        assert ura_markers[1] == cvt.tid_to_mjai(128)
