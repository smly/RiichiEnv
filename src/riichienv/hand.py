from dataclasses import dataclass, field

from . import _riichienv as rust_core  # type: ignore
from ._riichienv import (  # type: ignore
    Meld,
    Wind,
)

WINDS = [Wind.East, Wind.South, Wind.West, Wind.North]


@dataclass
class Yaku:
    name: str
    name_en: str
    tenhou_id: int
    mjsoul_id: int


class YakuList:
    # 流し満貫は役定義なし
    yaku_list: list[Yaku] = [
        # 1 Han
        Yaku(name="門前清自摸和", name_en="Menzen Tsumo", mjsoul_id=1, tenhou_id=0),
        Yaku(name="立直", name_en="Riichi", mjsoul_id=2, tenhou_id=1),
        Yaku(name="一発", name_en="Ippatsu", mjsoul_id=30, tenhou_id=2),
        Yaku(name="槍槓", name_en="Chankan", mjsoul_id=3, tenhou_id=3),
        Yaku(name="嶺上開花", name_en="Rinshan Kaihou", mjsoul_id=4, tenhou_id=4),
        Yaku(name="海底摸月", name_en="Haitei Raoyue", mjsoul_id=5, tenhou_id=5),
        Yaku(name="河底撈魚", name_en="Houtei Raoyui", mjsoul_id=6, tenhou_id=6),
        Yaku(name="平和", name_en="Pinfu", mjsoul_id=14, tenhou_id=7),
        Yaku(name="断幺九", name_en="Tanyao", mjsoul_id=12, tenhou_id=8),
        Yaku(name="一盃口", name_en="Iipeiko", mjsoul_id=13, tenhou_id=9),
        Yaku(name="自風牌", name_en="Yakuhai (wind of place)", mjsoul_id=10, tenhou_id=10),
        Yaku(name="場風牌", name_en="Yakuhai (wind of round)", mjsoul_id=11, tenhou_id=11),
        Yaku(name="役牌 白", name_en="Yakuhai (haku)", mjsoul_id=7, tenhou_id=18),
        Yaku(name="役牌 發", name_en="Yakuhai (hatsu)", mjsoul_id=8, tenhou_id=19),
        Yaku(name="役牌 中", name_en="Yakuhai (chun)", mjsoul_id=9, tenhou_id=20),
        # 2 Han
        Yaku(name="ダブル立直", name_en="Double Riichi", mjsoul_id=18, tenhou_id=21),
        Yaku(name="七対子", name_en="Chiitoitsu", mjsoul_id=25, tenhou_id=22),
        Yaku(name="混全帯幺九", name_en="Chantai", mjsoul_id=15, tenhou_id=23),
        Yaku(name="一気通貫", name_en="Ittsu", mjsoul_id=16, tenhou_id=24),
        Yaku(name="三色同順", name_en="Sanshoku Doujun", mjsoul_id=17, tenhou_id=25),
        Yaku(name="三色同刻", name_en="Sanshoku Doukou", mjsoul_id=19, tenhou_id=26),
        Yaku(name="三槓子", name_en="San Kantsu", mjsoul_id=20, tenhou_id=27),
        Yaku(name="対々和", name_en="Toitoi", mjsoul_id=21, tenhou_id=28),
        Yaku(name="三暗刻", name_en="San Ankou", mjsoul_id=22, tenhou_id=29),
        Yaku(name="小三元", name_en="Shou Sangen", mjsoul_id=23, tenhou_id=30),
        Yaku(name="混老頭", name_en="Honroutou", mjsoul_id=24, tenhou_id=31),
        # 3 Han
        Yaku(name="二盃口", name_en="Ryanpeikou", mjsoul_id=28, tenhou_id=32),
        Yaku(name="純全帯幺九", name_en="Junchan", mjsoul_id=26, tenhou_id=33),
        Yaku(name="混一色", name_en="Honitsu", mjsoul_id=27, tenhou_id=34),
        # 6 Han
        Yaku(name="清一色", name_en="Chinitsu", mjsoul_id=29, tenhou_id=35),
        # Yakuman
        Yaku(name="天和", name_en="Tenhou", mjsoul_id=35, tenhou_id=37),
        Yaku(name="地和", name_en="Chiihou", mjsoul_id=36, tenhou_id=38),
        Yaku(name="大三元", name_en="Dai Sangen", mjsoul_id=37, tenhou_id=39),
        Yaku(name="四暗刻単騎", name_en="Su Ankou Tanki", mjsoul_id=48, tenhou_id=40),
        Yaku(name="四暗刻", name_en="Su Ankou", mjsoul_id=38, tenhou_id=41),
        Yaku(name="字一色", name_en="Tsuu iisou", mjsoul_id=39, tenhou_id=42),
        Yaku(name="緑一色", name_en="Ryuu iisou", mjsoul_id=40, tenhou_id=43),
        Yaku(name="清老頭", name_en="Chinroutou", mjsoul_id=41, tenhou_id=44),
        Yaku(name="九蓮宝燈", name_en="Chuuren Poutou", mjsoul_id=45, tenhou_id=45),
        Yaku(name="純正九蓮宝燈", name_en="Junsei Chuuren Poutou", mjsoul_id=47, tenhou_id=46),
        Yaku(name="国士無双", name_en="Kokushi Musou", mjsoul_id=42, tenhou_id=47),
        Yaku(name="国士無双十三面待ち", name_en="Kokushi Musou 13-men", mjsoul_id=49, tenhou_id=48),
        Yaku(name="大四喜", name_en="Dai Suusi", mjsoul_id=50, tenhou_id=49),
        Yaku(name="小四喜", name_en="Sho Suusi", mjsoul_id=43, tenhou_id=50),
        Yaku(name="四槓子", name_en="Su Kantsu", mjsoul_id=44, tenhou_id=51),
        # Extra
        Yaku(name="ドラ", name_en="Dora", mjsoul_id=31, tenhou_id=52),
        Yaku(name="赤ドラ", name_en="Aka Dora", mjsoul_id=32, tenhou_id=54),
        Yaku(name="裏ドラ", name_en="Ura Dora", mjsoul_id=33, tenhou_id=52),  # ドラと同じ
    ]

    @staticmethod
    def get_yaku_from_tenhou_id(tenhou_id: int) -> Yaku:
        for yaku in YakuList.yaku_list:
            if yaku.tenhou_id == tenhou_id:
                return yaku
        raise ValueError(f"Invalid tenhou_id: {tenhou_id}")

    @staticmethod
    def get_yaku_from_mjsoul_id(mjsoul_id: int) -> Yaku:
        for yaku in YakuList.yaku_list:
            if yaku.mjsoul_id == mjsoul_id:
                return yaku
        raise ValueError(f"Invalid mjsoul_id: {mjsoul_id}")


@dataclass
class WinResult:
    is_win: bool
    yakuman: bool = False
    ron_agari: int = 0
    tsumo_agari_oya: int = 0
    tsumo_agari_ko: int = 0
    yaku: list[int] = field(default_factory=list)
    han: int = 0
    fu: int = 0


@dataclass
class Conditions:
    tsumo: bool = False
    riichi: bool = False
    double_riichi: bool = False
    ippatsu: bool = False
    haitei: bool = False
    houtei: bool = False
    rinshan: bool = False
    chankan: bool = False
    tsumo_first_turn: bool = False

    player_wind: int | Wind = 0  # E,S,W,N = (0,1,2,3) or Wind enum values
    round_wind: int | Wind = 0  # E,S,W,N = (0,1,2,3) or Wind enum values

    riichi_sticks: int = 0
    honba: int = 0


class HandEvaluator:
    def __init__(self, tiles: list[int], melds: list[Meld] | None = None) -> None:
        self.tiles_136 = tiles
        self.melds = melds or []
        self._rust_melds = self.melds
        self.calc_rust = rust_core.HandEvaluator(self.tiles_136, self._rust_melds)

    @staticmethod
    def hand_from_text(hand_str_repr: str) -> "HandEvaluator":
        """
        # Hand representation parsing
        `Hand::from_text()` method accepts a string representation.
        It expects a 13-tile hand (plus extra tiles for Kans).
        """
        tiles, melds = rust_core.parse_hand(hand_str_repr)

        num_kans = 0
        current_tiles_count = len(tiles)
        for m in melds:
            current_tiles_count += len(m.tiles)
            if m.meld_type in [rust_core.MeldType.Daiminkan, rust_core.MeldType.Ankan, rust_core.MeldType.Kakan]:
                num_kans += 1

        expected_count = 13 + num_kans
        if current_tiles_count != expected_count:
            raise ValueError(f"Hand must have {expected_count} tiles (got {current_tiles_count})")

        tiles = list(tiles)
        tiles.sort()
        return HandEvaluator(tiles, melds)

    def to_text(self, win_tile: str | None = None) -> str:
        """
        Convert hand to text representation.
        """
        tiles = sorted(self.tiles_136)
        # TODO: Implement full meld string reconstruction if needed.
        # For now, focus on closed tiles.
        # User requested full implementation.
        # I'll implement basic tile string construction.

        result = self._tiles_to_string(tiles)

        for meld in self.melds:
            # Reconstruct meld string
            # Format: (XYZCI) etc.
            # This requires reconstructing the exact input format which includes 'call index'.
            # Rust Meld struct DOES NOT store 'call index'.
            # It only stores type, tiles, opened.
            # So we cannot perfectly reconstruct (p1z2) vs (p1z3).
            # We can only approximate or omit index?
            # User sample: (p1z1).
            # I'll assume index 0 if unknown.
            result += self._meld_to_string(meld)

        return result

    @staticmethod
    def calc_from_text(
        hand_str_repr_with_win_tile: str,
        dora_indicators: str | None = None,
        conditions: Conditions | None = None,
        ura_indicators: str | None = None,
    ) -> WinResult:
        """
        hand_str_repr_with_win_tile: str は 14 枚分の牌を想定する。最後の1枚を win_tile として扱う。
        """
        conditions = conditions or Conditions()
        tiles, melds = rust_core.parse_hand(hand_str_repr_with_win_tile)
        if not tiles and not melds:
            raise ValueError("Empty hand")

        if not tiles:
            raise ValueError("No standing tiles to check for win tile")

        win_tile = tiles[-1]
        tiles = list(tiles)
        tiles.sort()

        calc = HandEvaluator(tiles, melds)

        dora_inds = []
        if dora_indicators:
            dora_inds, _ = rust_core.parse_hand(dora_indicators)
            dora_inds = list(dora_inds)
            dora_inds.sort()

        ura_inds = []
        if ura_indicators:
            ura_inds, _ = rust_core.parse_hand(ura_indicators)
            ura_inds = list(ura_inds)
            ura_inds.sort()

        return calc.calc(win_tile, dora_inds, conditions, ura_inds)

    def _tiles_to_string(self, tiles: list[int]) -> str:
        # Group by suit: m, p, s, z
        man = []
        pin = []
        sou = []
        honors = []

        for t in tiles:
            is_red = t in [16, 52, 88]
            val = t // 4

            digit = 0
            if val < 9:
                digit = val + 1
                if is_red:
                    digit = 0
                man.append(digit)
            elif val < 18:
                digit = (val - 9) + 1
                if is_red:
                    digit = 0
                pin.append(digit)
            elif val < 27:
                digit = (val - 18) + 1
                if is_red:
                    digit = 0
                sou.append(digit)
            else:
                digit = (val - 27) + 1
                honors.append(digit)

        res = ""
        if man:
            res += "".join(map(str, man)) + "m"
        if pin:
            res += "".join(map(str, pin)) + "p"
        if sou:
            res += "".join(map(str, sou)) + "s"
        if honors:
            res += "".join(map(str, honors)) + "z"
        return res

    def _meld_to_string(self, meld: Meld) -> str:  # noqa: PLR0915
        # Reconstruct string from Meld
        # (XYZCI)
        # Meld has tiles[].
        # Determine suit from first tile.
        t0 = meld.tiles[0]
        val0 = t0 // 4

        suffix = "m"
        if val0 >= 9 and val0 < 18:
            suffix = "p"
        elif val0 >= 18 and val0 < 27:
            suffix = "s"
        elif val0 >= 27:
            suffix = "z"

        m_type = meld.meld_type

        # Digits
        digits = ""
        is_chi = m_type == rust_core.MeldType.Chi

        if is_chi:
            for t in meld.tiles:
                is_red = t in [16, 52, 88]
                v = t // 4

                d = 0
                if v < 9:
                    d = v + 1
                elif v < 18:
                    d = v - 9 + 1
                elif v < 27:
                    d = v - 18 + 1
                else:
                    d = v - 27 + 1

                if is_red:
                    d = 0
                digits += str(d)
        else:
            # Pon/Kan/Add: single digit
            has_red = any(t in [16, 52, 88] for t in meld.tiles)
            v = val0
            d = 0
            if v < 9:
                d = v + 1
            elif v < 18:
                d = v - 9 + 1
            elif v < 27:
                d = v - 18 + 1
            else:
                d = v - 27 + 1

            if has_red:
                d = 0
            digits = str(d)

        # Prefix
        prefix = ""

        if m_type == rust_core.MeldType.Pon:
            prefix = "p"
        elif m_type == rust_core.MeldType.Daiminkan:
            prefix = "k"
        elif m_type == rust_core.MeldType.Kakan:
            prefix = "s"
        elif m_type == rust_core.MeldType.Ankan:
            prefix = "k"
        # Closed Kan also 'k'? Usually closed kan not indicated in open string?
        # But user example (k2z) was "closed kan or daiminkan".
        # If Angang, maybe suffix missing?

        # Call index? Not stored. Use 0.

        return f"({prefix}{digits}{suffix}0)"  # Fake index 0

    def calc(
        self,
        win_tile: int,
        dora_indicators: list[int] | None = None,
        conditions: Conditions | None = None,
        ura_indicators: list[int] | None = None,
    ) -> WinResult:
        conditions = conditions or Conditions()
        if dora_indicators is None:
            dora_indicators = []
        if ura_indicators is None:
            ura_indicators = []
        # Convert winds to Rust Wind enum if they are integers
        p_wind = conditions.player_wind
        if isinstance(p_wind, int):
            p_wind = WINDS[p_wind % 4]

        r_wind = conditions.round_wind
        if isinstance(r_wind, int):
            r_wind = WINDS[r_wind % 4]

        rust_conditions = rust_core.Conditions(
            tsumo=conditions.tsumo,
            riichi=conditions.riichi,
            double_riichi=conditions.double_riichi,
            ippatsu=conditions.ippatsu,
            haitei=conditions.haitei,
            houtei=conditions.houtei,
            rinshan=conditions.rinshan,
            chankan=conditions.chankan,
            tsumo_first_turn=conditions.tsumo_first_turn,
            player_wind=p_wind,
            round_wind=r_wind,
            riichi_sticks=conditions.riichi_sticks,
            honba=conditions.honba,
        )

        dora_inds_136 = dora_indicators if dora_indicators else []
        ura_inds_136 = ura_indicators if ura_indicators else []

        rust_melds = self._rust_melds
        total_tiles = len(self.tiles_136) + len(rust_melds) * 3

        calc_obj = self.calc_rust
        if total_tiles % 3 == 1:
            temp_tiles = sorted(self.tiles_136 + [win_tile])
            calc_obj = rust_core.HandEvaluator(temp_tiles, rust_melds)

        res = calc_obj.calc(win_tile, dora_inds_136, ura_inds_136, rust_conditions)

        if not res.is_win:
            return WinResult(is_win=False)

        return WinResult(
            is_win=True,
            yakuman=res.yakuman,
            ron_agari=res.ron_agari,
            tsumo_agari_oya=res.tsumo_agari_oya,
            tsumo_agari_ko=res.tsumo_agari_ko,
            yaku=res.yaku,
            han=res.han,
            fu=res.fu,
        )

    def is_tenpai(self) -> bool:
        return self.calc_rust.is_tenpai()

    def get_waits(self) -> list[int]:
        return self.calc_rust.get_waits()
