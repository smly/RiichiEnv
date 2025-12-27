import time
from pathlib import Path

from mahjong.hand_calculating.hand import HandCalculator

from riichienv import AgariCalculator, Conditions, ReplayGame

YAKUMAN_IDS = [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 49, 50]


def to_mahjong_args(ctx):
    from mahjong.constants import EAST, NORTH, SOUTH, WEST
    from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
    from mahjong.meld import Meld as MahjongMeld

    # Map winds
    wind_map = {0: EAST, 1: SOUTH, 2: WEST, 3: NORTH}

    config = HandConfig(
        is_tsumo=ctx.conditions.tsumo,
        is_riichi=ctx.conditions.riichi,
        is_ippatsu=ctx.conditions.ippatsu,
        is_rinshan=ctx.conditions.rinshan,
        is_chankan=ctx.conditions.chankan,
        is_haitei=ctx.conditions.haitei,
        is_houtei=ctx.conditions.houtei,
        is_daburu_riichi=ctx.conditions.double_riichi,
        player_wind=wind_map.get(ctx.conditions.player_wind, EAST),
        round_wind=wind_map.get(ctx.conditions.round_wind, EAST),
        options=OptionalRules(has_aka_dora=True, has_open_tanyao=True),
    )

    melds = []
    for m in ctx.melds:
        # riichienv.MeldType to mahjong.meld.Meld type
        # Peng=1, Gang=2, Angang=3, Addgang=4 in riichienv
        # mahjong uses CHI, PONG, KAN, etc.
        if m.meld_type == 0:
            m_type = MahjongMeld.CHI
        elif m.meld_type == 1:
            m_type = MahjongMeld.PON
        elif m.meld_type in [2, 3, 4]:
            m_type = MahjongMeld.KAN
        else:
            raise ValueError(f"Unexpected meld_type: {m.meld_type!r} in ctx.melds")

        # IMPORTANT: Sort tiles for Mahjong package!
        # Mahjong package's is_chi requires tiles to be sorted by 34-index.
        # Since riichienv tile IDs are basically 34_id * 4 (+ offset),
        # sorting by 136-id is equivalent to sorting by 34-id for this purpose.
        m_tiles = sorted(list(m.tiles))

        melds.append(MahjongMeld(m_type, m_tiles, opened=m.opened))

    all_tiles = list(ctx.tiles)
    for m in ctx.melds:
        all_tiles.extend(m.tiles)

    return {
        "tiles": all_tiles,
        "win_tile": ctx.agari_tile,
        "melds": melds,
        "dora_indicators": list(ctx.dora_indicators),
        "ura_indicators": list(ctx.ura_indicators),
        "config": config,
    }


def main() -> None:
    # Prefer data directory in sibling riichienv package; fall back to local data/ when running from project root.
    log_dir = Path("../riichienv/data/game_record_4p_thr_2025-12-14_out/")
    if not log_dir.exists():
        # Fallback to data directory relative to current working directory.
        log_dir = Path("data/game_record_4p_thr_2025-12-14_out/")

    total_agari = 0
    t_riichienv = 0.0
    t_riichienv_py = 0.0
    t_mahjong = 0.0

    for log_path in sorted(list(log_dir.glob("251214-*.json.gz"))):
        game = ReplayGame.from_json(str(log_path))

        for kyoku in game.take_kyokus():
            for ctx in kyoku.take_agari_contexts():
                total_agari += 1
                expected_fu = ctx.expected_fu
                expected_han = ctx.expected_han
                expected_yakuman = len(set(ctx.expected_yaku) & set(YAKUMAN_IDS)) > 0

                # Riichienv
                t0 = time.time()
                calc_r = ctx.create_calculator()
                res_r = ctx.calculate(calc_r)
                t_riichienv += time.time() - t0

                assert res_r.yakuman == expected_yakuman
                assert res_r.agari, f"Agari: {res_r.agari}"
                if expected_yakuman:
                    pass
                else:
                    assert res_r.han == expected_han, f"Han: {res_r.han} != {expected_han}"
                    assert res_r.fu == expected_fu, f"Fu: {res_r.fu} != {expected_fu}"

                # Riichienv (py)
                t0 = time.time()
                cond_py = Conditions()
                for attr in [
                    "tsumo",
                    "riichi",
                    "double_riichi",
                    "ippatsu",
                    "haitei",
                    "houtei",
                    "rinshan",
                    "chankan",
                    "tsumo_first_turn",
                    "player_wind",
                    "round_wind",
                    "kyoutaku",
                    "tsumi",
                ]:
                    setattr(cond_py, attr, getattr(ctx.conditions, attr))

                res_r_py = AgariCalculator(
                    tiles=ctx.tiles,
                    melds=ctx.melds,
                ).calc(ctx.agari_tile, ctx.dora_indicators, cond_py, ctx.ura_indicators)
                t_riichienv_py += time.time() - t0

                if not res_r_py.agari:
                    print(f"Error in {log_path} (riichienv-py): not agari")
                    print(f"  Closed Tiles (ctx.tiles): {ctx.tiles} (len={len(ctx.tiles)})")
                    print(f"  Win Tile (ctx.agari_tile): {ctx.agari_tile}")
                    print(f"  Melds (ctx.melds): {ctx.melds}")
                    # Try to see if win_tile is in tiles for Tsumo
                    if ctx.conditions.tsumo:
                        print(f"  Tsumo: win_tile in tiles? {ctx.agari_tile in ctx.tiles}")

                assert res_r_py.yakuman == expected_yakuman
                assert res_r_py.agari
                if expected_yakuman:
                    pass
                else:
                    assert res_r_py.han == expected_han, f"Han (py): {res_r_py.han} != {expected_han}"
                    assert res_r_py.fu == expected_fu, f"Fu (py): {res_r_py.fu} != {expected_fu}"

                # Mahjong package
                args = to_mahjong_args(ctx)

                mahjong_args = dict(args)
                ura_inds = mahjong_args.pop("ura_indicators")
                if ctx.conditions.riichi or ctx.conditions.double_riichi:
                    mahjong_args["dora_indicators"] = mahjong_args["dora_indicators"] + ura_inds

                t0 = time.time()
                mahjong_pkg_hand_calculator = HandCalculator()
                res_m = mahjong_pkg_hand_calculator.estimate_hand_value(**mahjong_args)
                t_mahjong += time.time() - t0

                if res_m.error:
                    print(f"Error in {log_path}: {res_m.error}")
                    print(f"  Seat: {ctx.seat}")
                    print(f"  Hand (closed): {ctx.tiles}")
                    print(f"  Melds: {ctx.melds}")
                    print(f"  Win tile: {ctx.agari_tile}")
                    print(
                        f"  Conditions: riichi={ctx.conditions.riichi}, tsumo={ctx.conditions.tsumo}, p_wind={ctx.conditions.player_wind}, r_wind={ctx.conditions.round_wind}"
                    )
                    print(f"  Riichienv results: han={res_r.han}, fu={res_r.fu}, yaku={res_r.yaku}")
                    print(f"  Expected: han={expected_han}, fu={expected_fu}")

                assert res_m.error is None, f"Agari: {res_m.error}"
                if expected_yakuman:
                    pass
                else:
                    if res_m.han != expected_han or res_m.fu != expected_fu:
                        print(f"Mismatch in {log_path}:")
                        print(f"  Riichienv: han={res_r.han}, fu={res_r.fu}")
                        print(f"  Mahjong:   han={res_m.han}, fu={res_m.fu}, yaku={res_m.yaku}")
                        print(f"  Expected:  han={expected_han}, fu={expected_fu}")

                    assert res_m.han == expected_han, f"Han: {res_m.han} != {expected_han}"
                    assert res_m.fu == expected_fu, f"Fu: {res_m.fu} != {expected_fu}"

    print(f"Total Agari situations: {total_agari}")
    print(f"riichienv: {t_riichienv:.4f}s ({total_agari / t_riichienv:.2f} agari/sec)")
    print(f"riichienv-py: {t_riichienv_py:.4f}s ({total_agari / t_riichienv_py:.2f} agari/sec)")
    print(f"mahjong:   {t_mahjong:.4f}s ({total_agari / t_mahjong:.2f} agari/sec)")


if __name__ == "__main__":
    main()
