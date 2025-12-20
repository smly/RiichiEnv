from ._riichienv import (
    Hand,
    Meld,
    MeldType,
    Score,
    is_agari,
    calculate_score,
    parse_hand,
    ReplayGame,
    Kyoku,
    AgariContext,
)
from .hand import AgariCalculator, Conditions, Agari

def _to_mahjong_args(self):
    from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
    from mahjong.meld import Meld as MahjongMeld
    from mahjong.constants import EAST, SOUTH, WEST, NORTH

    # Map winds
    wind_map = {0: EAST, 1: SOUTH, 2: WEST, 3: NORTH}

    config = HandConfig(
        is_tsumo=self.conditions.tsumo,
        is_riichi=self.conditions.riichi,
        is_ippatsu=self.conditions.ippatsu,
        is_rinshan=self.conditions.rinshan,
        is_chankan=self.conditions.chankan,
        is_haitei=self.conditions.haitei,
        is_houtei=self.conditions.houtei,
        is_daburu_riichi=self.conditions.double_riichi,
        player_wind=wind_map.get(self.conditions.player_wind, EAST),
        round_wind=wind_map.get(self.conditions.round_wind, EAST),
        options=OptionalRules(has_aka_dora=True, has_open_tanyao=True)
    )

    melds = []
    for m in self.melds:
        # riichienv.MeldType to mahjong.meld.Meld type
        # Peng=1, Gang=2, Angang=3, Addgang=4 in riichienv
        # mahjong uses CHI, PONG, KAN, etc.
        if m.meld_type == 0: m_type = MahjongMeld.CHI
        elif m.meld_type == 1: m_type = MahjongMeld.PON
        elif m.meld_type in [2, 3, 4]: m_type = MahjongMeld.KAN

        # IMPORTANT: Sort tiles for Mahjong package!
        # Mahjong package's is_chi requires tiles to be sorted by 34-index.
        # Since riichienv tile IDs are basically 34_id * 4 (+ offset),
        # sorting by 136-id is equivalent to sorting by 34-id for this purpose.
        m_tiles = sorted(list(m.tiles))

        melds.append(MahjongMeld(
            m_type,
            m_tiles,
            opened=m.opened
        ))

    all_tiles = list(self.tiles)
    for m in self.melds:
        all_tiles.extend(m.tiles)

    return {
        "tiles": all_tiles,
        "win_tile": self.agari_tile,
        "melds": melds,
        "dora_indicators": list(self.dora_indicators),
        "ura_indicators": list(self.ura_indicators),
        "config": config
    }

AgariContext.to_mahjong_args = _to_mahjong_args