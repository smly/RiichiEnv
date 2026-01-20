from riichienv import Action, ActionType, Meld, MeldType, Phase

from ..helper import helper_setup_env


class TestPao:
    def test_daisangen_pao_tsumo(self) -> None:
        """
        Verify Daisangen Pao for Tsumo.
        P0 has 2 dragon melds, calls 3rd from P1, then Tsumos.
        """
        env = helper_setup_env(
            seed=42,
            oya=0,
            hands=[
                [132, 133, 0, 1, 2, 4, 5],
                [134] + [0] * 12,  # P1 has 134
                [0] * 13,
                [0] * 13,
            ],
            melds=[
                [
                    Meld(MeldType.Peng, [124, 125, 126], True),
                    Meld(MeldType.Peng, [128, 129, 130], True),
                ],
                [],
                [],
                [],
            ],
            current_player=1,
            phase=Phase.WaitAct,
            active_players=[1],
        )

        # P1 discards 3rd Red Dragon (134)
        env.step({1: Action(ActionType.Discard, tile=134)})

        # P0 calls Pon
        env.step({0: Action(ActionType.Pon, tile=134, consume_tiles=[132, 133])})

        # Verify Pao is established (Daisangen Yaku ID 37)
        assert env.pao[0].get(37) == 1
        env.step({0: Action(ActionType.Discard, tile=4)})

        # P0 draws completing tile for 2m pair (6 is 2m)
        env.current_player = 0
        env.phase = Phase.WaitAct
        env.active_players = [0]
        env.drawn_tile = 6
        h = env.hands
        h[0].append(6)
        h[0].sort()
        env.hands = h

        # Tsumo
        env.step({0: Action(ActionType.Tsumo)})

        # Verify scoring
        agari_res = env.agari_results[0]
        assert agari_res.agari
        assert agari_res.yaku == [37]  # Daisangen

        hora = next(m for m in reversed(env.mjai_log) if m["type"] == "hora")
        deltas = hora["deltas"]

        # Oya Yakuman Tsumo: 48000 total.
        # P1 (Pao) should pay 48000.
        assert deltas[0] == 48000
        assert deltas[1] == -48000
        assert deltas[2] == 0
        assert deltas[3] == 0

    def test_daisangen_pao_ron(self) -> None:
        """
        Verify Daisangen Pao for Ron from a 3rd party.
        """
        # Setup Wall to control draws
        # P1 needs 1m (ignored). P2 needs 101. P3 needs 102. P0 needs 103.
        # 53 tiles consumed by deal (14+13+13+13=53).
        # Index 53: P1 Turn 2.
        # Index 54: P2 Turn 1.
        # Index 55: P3 Turn 1.
        # Index 56: P0 Turn 2.
        wall = [0] * 200
        wall[54] = 101
        wall[55] = 102
        wall[56] = 103

        env = helper_setup_env(
            seed=42,
            oya=1,
            hands=[
                [132, 133, 0, 1, 2, 36, 120],  # P0: 7 tiles (Melds=6, Total 13)
                [134, 100] + list(range(10, 22)),  # P1 (Oya): 14 tiles
                [37] + list(range(40, 52)),  # P2: 13 tiles
                list(range(60, 73)),  # P3: 13 tiles
            ],
            melds=[
                [
                    Meld(MeldType.Peng, [124, 125, 126], True),
                    Meld(MeldType.Peng, [128, 129, 130], True),
                ],
                [],
                [],
                [],
            ],
            current_player=1,
            phase=Phase.WaitAct,
            active_players=[1],
            wall=wall,
        )

        # P1 discards 134
        env.step({1: Action(ActionType.Discard, tile=134)})

        # P0 Pons
        env.step({0: Action(ActionType.Pon, tile=134, consume_tiles=[132, 133])})

        assert env.pao[0].get(37) == 1

        # P0 Discards 120 (North). Hand: 0,1,2 (1m), 36(1p). Tanki wait on 1p.
        env.step({0: Action(ActionType.Discard, tile=120)})

        # To clear "Doujun Furiten", we must cycle a full turn so P0 draws again.

        # P1 Turn
        env.step({})
        env.step({1: Action(ActionType.Discard, tile=100)})

        # P2 Turn
        env.step({})
        env.step({2: Action(ActionType.Discard, tile=101)})

        # P3 Turn
        env.step({})
        env.step({3: Action(ActionType.Discard, tile=102)})

        # P0 Turn - Draws and Discards (clears Furiten)
        env.step({})
        env.step({0: Action(ActionType.Discard, tile=103)})

        # P1 Turn
        env.step({})
        env.step({1: Action(ActionType.Discard, tile=10)})

        # P2 Turn - Discards Winning Tile 37
        env.step({})
        env.step({2: Action(ActionType.Discard, tile=37)})

        # P0 Ron
        obs = env.get_obs_py([0])[0]
        ron_available = any(a.action_type == ActionType.Ron for a in obs.legal_actions())
        assert ron_available, f"Ron not available! Legal: {obs.legal_actions()}"

        env.step({0: Action(ActionType.Ron, tile=37)})
        assert env.is_done

        # Ko Yakuman Ron: 32000 points.
        hora = next(m for m in reversed(env.mjai_log) if m["type"] == "hora")
        deltas = hora["deltas"]

        assert deltas[0] == 32000
        assert deltas[1] == -16000  # Pao (Oya) - Responsible for Pao
        assert deltas[2] == -16000  # Discarder (Ko)
        assert deltas[3] == 0
