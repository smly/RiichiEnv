import lzma
import time
import glob
import gzip
import json
import tempfile
from typing import Iterator

import tqdm

from riichienv import AgariCalculator, Conditions, ReplayGame
from mjsoul_parser import MjsoulPaifuParser, Paifu

YAKUMAN_IDS = [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 49, 50]
TARGET_FILE_PATTERN = "/data/mahjong_game_record_4p_*/*.bin.xz"

def iter_game_kyoku(paifu: Paifu) -> Iterator[Kyoku]:
    with tempfile.NamedTemporaryFile(delete=True) as f:
        with gzip.GzipFile(fileobj=f, mode="w") as g:
            g.write(json.dumps({"rounds": paifu.data}).encode("utf-8"))
        f.flush()
        f.seek(0)
        game = ReplayGame.from_json(f.name)

    for kyoku in game.take_kyokus():
        yield kyoku


def main():
    total_agari = 0
    t_riichienv_py = 0
    has_error = False

    target_files = list(glob.glob(TARGET_FILE_PATTERN))
    for path in tqdm.tqdm(target_files, desc="Processing files", ncols=100):
        with lzma.open(path, "rb") as f:
            data = f.read()
            paifu: Paifu = MjsoulPaifuParser.to_dict(data)

        for k, kyoku in enumerate(iter_game_kyoku(paifu)):
            try:
                for ctx in kyoku.take_agari_contexts():
                    total_agari += 1

                    expected_yakuman = len(set(ctx.expected_yaku) & set(YAKUMAN_IDS)) > 0
                    expected_han = ctx.expected_han
                    expected_fu = ctx.expected_fu

                    # Riichienv (py)
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

                    t0 = time.time()
                    res_r_py = AgariCalculator(
                        tiles=ctx.tiles,
                        melds=ctx.melds,
                    ).calc(ctx.agari_tile, ctx.dora_indicators, cond_py, ctx.ura_indicators)
                    t_riichienv_py += time.time() - t0

                    assert res_r_py.yakuman == expected_yakuman
                    assert res_r_py.agari
                    if not expected_yakuman:
                        assert res_r_py.han == expected_han, f"Han (py): {res_r_py.han} != {expected_han}"
                        assert res_r_py.fu == expected_fu, f"Fu (py): {res_r_py.fu} != {expected_fu}"
                        assert set(res_r_py.yaku) == set(ctx.expected_yaku), f"Yaku (py): {res_r_py.yaku} != {ctx.expected_yaku}"

            except AssertionError as e:
                print(e)
                print(res_r_py.yaku, ctx.expected_yaku)
                print(paifu.header["uuid"], k)
                has_error = True
                raise e
        

if __name__ == "__main__":
    main()
