from typing import Any
from enum import Enum
from pathlib import Path
import yaml
import json
import gzip

from riichienv import AgariCalculator, Conditions, Agari, Meld, MeldType as RiichiMeldType
from mahjong.tile import TilesConverter


with open("./data/yans.yml", "r") as f:
    fans = yaml.load(f, Loader=yaml.SafeLoader)["fans"]


def load_game_record(path: str) -> dict:
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    return data


Hand = list[str]

class MeldType(Enum):
    CHI = 0
    PENG = 1
    GANG = 2
    ANGANG = 3
    ADDGANG = 4


MeldTuple = tuple[MeldType, list[str]]


class Round:
    def __init__(self, new_round_data: dict) -> None:
        self.scores = new_round_data["scores"]
        # Order by seat index
        self.hands: list[Hand] = [
            new_round_data["tiles0"],
            new_round_data["tiles1"],
            new_round_data["tiles2"],
            new_round_data["tiles3"],
        ]
        self.melds: list[list[MeldTuple]] = [
            [],
            [],
            [],
            [],
        ]
        self.wliqi: list[bool] = [False] * 4
        self.liqi: list[bool] = [False] * 4
        self.ippatsu: list[bool] = [False] * 4
        self.rinshan: list[bool] = [False] * 4

        self.doras = new_round_data["doras"]
        self.left_tile_count = new_round_data["left_tile_count"]
        self.chang = new_round_data["chang"] # 場
        self.ju = new_round_data["ju"] # 局
        self.ben = new_round_data["ben"] # 本場
        self.liqibang = new_round_data["liqibang"] # 供託

    def __str__(self) -> str:
        return f"Round(chang={self.chang}, ju={self.ju}, ben={self.ben}, liqibang={self.liqibang})"

    def oya_seat(self) -> int:
        return self.ju % 4

    def _get_melds_for_mahjong_lib(self, seat: int) -> tuple[list[Meld], list[int]]:
        hola_tiles: list[int] = []
        melds_ = []
        for meld_tuple in self.melds[seat]:
            match meld_tuple[0]:
                case MeldType.CHI:
                    tiles = self._get_tiles(meld_tuple[1])
                    melds_.append(Meld(RiichiMeldType.Chi, tiles, True))
                    hola_tiles += tiles
                case MeldType.PENG:
                    tiles = self._get_tiles(meld_tuple[1])
                    melds_.append(Meld(RiichiMeldType.Peng, tiles, True))
                    hola_tiles += tiles
                case MeldType.GANG:
                    tiles = self._get_tiles(meld_tuple[1])
                    melds_.append(Meld(RiichiMeldType.Gang, tiles, True))
                    hola_tiles += tiles
                case MeldType.ANGANG:
                    tiles = self._get_tiles(meld_tuple[1])
                    melds_.append(Meld(RiichiMeldType.Angang, tiles, False))
                    hola_tiles += tiles
                case MeldType.ADDGANG:
                    # 対応するポンを明槓にする
                    meld_type, meld_tiles = meld_tuple
                    kan_tile_str: str = meld_tiles[0] # 1枚のみ
                    kan_tile_int: int = self._get_tile(kan_tile_str)
                    hola_tiles += [kan_tile_int]

                    for meld_idx, meld in enumerate(melds_):
                        if meld.meld_type == RiichiMeldType.Peng and (meld.tiles[0] // 4) == (kan_tile_int // 4):
                            new_tiles = sorted(meld.tiles + [kan_tile_int])
                            melds_[meld_idx] = Meld(RiichiMeldType.Addgang, new_tiles, True)
                            break
                    else:
                        assert False, "PENG not found"

        return melds_, hola_tiles

    def _get_tiles(self, tiles: list[str]) -> list[int]:
        """
        has_aka_dora=True のとき 0m, 0p, 0s は赤ドラとして 5m, 5p, 5s の1枚目の値としてエンコード

        ソートしないと一通、三色同順が正しく検証できない
        """
        tiles_ = TilesConverter.string_to_136_array(
            man="".join([tile[0] for tile in tiles if tile[1] == "m"]),
            pin="".join([tile[0] for tile in tiles if tile[1] == "p"]),
            sou="".join([tile[0] for tile in tiles if tile[1] == "s"]),
            honors="".join([tile[0] for tile in tiles if tile[1] == "z"]),
            has_aka_dora=True,
        )
        return list(sorted(tiles_))

    def _get_tile(self, tile: str) -> int:
        return TilesConverter.string_to_136_array(
            man=tile[0] if tile[1] == "m" else "",
            pin=tile[0] if tile[1] == "p" else "",
            sou=tile[0] if tile[1] == "s" else "",
            honors=tile[0] if tile[1] == "z" else "",
            has_aka_dora=True,
        )[0]

    def _apply_hule(self, action: dict[str, Any]) -> None:
        name, data = action["name"], action["data"]

        for hule in data["hules"]:
            seat = hule["seat"]
            tiles = list(sorted(self.hands[seat]))
            melds = self.melds[seat]
            is_zimo = hule["zimo"]

            assert self.liqi[seat] == hule["liqi"]
            hola_tiles = self._get_tiles(hule["hand"] + [hule["hu_tile"]])
            win_tile = self._get_tile(hule["hu_tile"])

            melds_ = None
            if len(self.melds[seat]) > 0:
                melds_, hola_tiles_in_melds = self._get_melds_for_mahjong_lib(seat)
                hola_tiles += hola_tiles_in_melds

            dora_indicators = self._get_tiles(self.doras)
            if self.liqi[seat]:
                dora_indicators += self._get_tiles(hule["li_doras"])

            # 和了判定の計算
            is_chankan = any(f["id"] == 3 for f in hule.get("fans", []))
            
            agari_calc = AgariCalculator(
                tiles=hola_tiles,
                melds=melds_,
            )
            r2 = agari_calc.calc(win_tile, dora_indicators, Conditions(
                tsumo=is_zimo,
                riichi=self.liqi[seat],
                double_riichi=self.wliqi[seat],
                ippatsu=self.ippatsu[seat],
                haitei=(self.left_tile_count == 0) and is_zimo,
                houtei=(self.left_tile_count == 0) and not is_zimo,
                rinshan=self.rinshan[seat],
                chankan=is_chankan,
                kyoutaku=self.liqibang,
                tsumi=self.ben,
                player_wind=(seat - self.ju + 4) % 4,
                round_wind=self.chang,
            ))

            # 和了判定の検証
            if not r2.agari:
                print(f"Agari Failed: seat={seat}")
                print(f"Hand: {hule['hand']} + {hule['hu_tile']}")
                print(f"Melds: {self.melds[seat]}")
                print(f"Liqi: {self.liqi[seat]}")
                print(f"Internal r2: {r2}")
            assert r2.agari
            if hule["yiman"] != r2.yakuman:
                print(f"Yakuman Mismatch: seat={seat}")
                print(f"Info: seat={seat}, ju={self.ju}, chang={self.chang}")
                print(f"Hand: {hule['hand']} + {hule['hu_tile']}")
                print(f"Expected Yakuman: {hule['yiman']}")
                print(f"Actual Yakuman: {r2.yakuman}")
                print(f"Actual Yaku: {r2.yaku}")
            assert hule["yiman"] == r2.yakuman
 
            if hule["yiman"]:
                assert r2.yakuman # 役満の役
            else:
                if r2.han != hule["count"]:
                    print(f"Han Mismatch: seat={seat}")
                    print(f"Info: seat={seat}, ju={self.ju}, chang={self.chang}")
                    print(f"Doras: {self.doras} + Li: {hule.get('li_doras', [])}")
                    print(f"Hand: {hule['hand']} + {hule['hu_tile']}")
                    print(f"Melds: {self.melds[seat]}")
                    print(f"Liqi: {self.liqi[seat]}")
                    print(f"Expected: Han={hule['count']}, Fans={hule['fans']}")
                    print(f"Actual: Han={r2.han}, Yaku IDs={r2.yaku}, Fu={r2.fu}")
                assert r2.han == hule["count"]
                assert r2.fu == hule["fu"]
            valid_fans = [f for f in hule["fans"] if f["id"] not in [31, 33]]
            r2_valid_yaku = [mjsoul_id for mjsoul_id in r2.yaku if mjsoul_id not in [31, 33]]
            if len(valid_fans) != len(r2_valid_yaku):
                 print(f"Mismatch Yaku Count: seat={seat}")
                 print(f"Expected Fans: {[f['id'] for f in valid_fans]}")
                 print(f"Actual IDs: {r2_valid_yaku}")
                 print(f"Hand: {hule['hand']} + {hule['hu_tile']}")
                 print(f"Agari: {r2}")
            assert len(valid_fans) == len(r2_valid_yaku)

            if not is_zimo:
                assert hule["point_rong"] == r2.ron_agari # 直接の得点
            else:
                # For self-draw, hule provides point_zimo_qin (from oya) and point_zimo_xian (from ko)
                # r2 provides tsumo_agari_oya (from oya) and tsumo_agari_ko (from ko)
                if hule["point_zimo_qin"] != r2.tsumo_agari_oya or hule["point_zimo_xian"] != r2.tsumo_agari_ko:
                    print(f"Point Mismatch (Tsumo): seat={seat}, ju={self.ju}, chang={self.chang}")
                    print(f"Expected: Qin={hule['point_zimo_qin']}, Xian={hule['point_zimo_xian']}")
                    print(f"Actual: Oya={r2.tsumo_agari_oya}, Ko={r2.tsumo_agari_ko}")
                    print(f"Han: {r2.han}, Fu: {r2.fu}")
                assert hule["point_zimo_qin"] == r2.tsumo_agari_oya # 親からの得点
                assert hule["point_zimo_xian"] == r2.tsumo_agari_ko # 子からの得点

    def apply_action(self, action : dict[str, Any]) -> None:
        name, data = action["name"], action["data"]

        if name != "Hule":
            self.rinshan = [False] * 4

        match name:
            case "DiscardTile":
                assert data["tile"] in self.hands[data["seat"]]
                if data["is_wliqi"]:
                    self.wliqi[data["seat"]] = True
                    self.ippatsu[data["seat"]] = True
                if data["is_liqi"]:
                    self.liqi[data["seat"]] = True
                    self.ippatsu[data["seat"]] = True

                if not data["is_liqi"]:
                    self.ippatsu[data["seat"]] = False

                self.hands[data["seat"]].remove(data["tile"])
                if len(data["doras"]) > 0:
                    self.doras = data["doras"]

            case "DealTile":
                self.hands[data["seat"]].append(data["tile"])
                self.left_tile_count -= 1
                if len(data["doras"]) > 0:
                    self.doras = data["doras"]
                    self.rinshan[data["seat"]] = True

            case "ChiPengGang":
                # 副露による一発消し
                self.ippatsu = [False] * 4

                meld_type = MeldType(data["type"])
                meld_tiles = data["tiles"]
                for tile_idx, tile_str in enumerate(data["tiles"]):
                    from_seat_idx = data["froms"][tile_idx]
                    if from_seat_idx == data["seat"]:
                        self.hands[from_seat_idx].remove(tile_str)

                self.melds[data["seat"]].append((meld_type, meld_tiles))

            case "AnGangAddGang":
                # 副露による一発消し
                self.ippatsu = [False] * 4

                if data["type"] == 3:
                    # 暗槓
                    tiles_ = [data["tiles"]] * 4
                    if data["tiles"][0] in ["0", "5"] and data["tiles"][1] != "z":
                        tiles_ = ["5" + data["tiles"][1]] * 3 + ["0" + data["tiles"][1]]

                    self.melds[data["seat"]].append((MeldType.ANGANG, tiles_))
                elif data["type"] == 2:
                    # 加槓
                    self.melds[data["seat"]].append((MeldType.ADDGANG, [data["tiles"]]))
                else:
                    assert False, "Invalid AnGangAddGang type"

            case "NoTile":
                pass

            case "LiuJu":
                pass

            case "Hule":
                self._apply_hule(action)

            case _:
                assert False, f"Unknown action: {name}"


def main() -> None:
    base_dir = "./data/game_record_4p_thr_2025-12-14_out/"
    for j, path in enumerate(list(sorted(Path(base_dir).glob("*.json.gz")))):
        if j % 100 == 0:
            print(j)
        data = load_game_record(path)
        for i, round_data in enumerate(data["rounds"]):
            mjai_records = []
            assert round_data[0]["name"] == "NewRound"
            assert round_data[-1]["name"] in ["LiuJu", "Hule", "NoTile"]

            r = Round(round_data[0]["data"])
            assert len(r.hands[r.oya_seat()]) == 14

            for action in round_data[1:]:
                r.apply_action(action)

    base_dir = "./data/game_record_4p_jad_2025-12-14_out/"
    for j, path in enumerate(list(sorted(Path(base_dir).glob("*.json.gz")))):
        if j % 100 == 0:
            print(j)
        data = load_game_record(path)
        for i, round_data in enumerate(data["rounds"]):
            mjai_records = []
            assert round_data[0]["name"] == "NewRound"
            assert round_data[-1]["name"] in ["LiuJu", "Hule", "NoTile"]

            r = Round(round_data[0]["data"])
            assert len(r.hands[r.oya_seat()]) == 14

            for action in round_data[1:]:
                r.apply_action(action)


if __name__ == "__main__":
    main()