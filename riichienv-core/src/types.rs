#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

pub const TILE_MAX: usize = 34;

/// A hand representation using a histogram of tile types (0-33).
#[derive(Debug, Clone)]
pub struct Hand {
    pub counts: [u8; TILE_MAX],
}

impl Hand {
    pub fn new(tiles: Option<Vec<u8>>) -> Self {
        let mut h = Hand {
            counts: [0; TILE_MAX],
        };
        if let Some(ts) = tiles {
            for t in ts {
                h.add(t);
            }
        }
        h
    }

    pub fn add(&mut self, t: u8) {
        if (t as usize) < TILE_MAX {
            self.counts[t as usize] += 1;
        }
    }

    pub fn remove(&mut self, t: u8) {
        if (t as usize) < TILE_MAX && self.counts[t as usize] > 0 {
            self.counts[t as usize] -= 1;
        }
    }

    fn __str__(&self) -> String {
        format!("Hand(counts={:?})", &self.counts[..])
    }
}

impl Default for Hand {
    fn default() -> Self {
        Hand {
            counts: [0; TILE_MAX],
        }
    }
}

#[cfg_attr(feature = "python", pyclass(eq, eq_int, from_py_object))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeldType {
    Chi = 0,
    Pon = 1,
    Daiminkan = 2,
    Ankan = 3,
    Kakan = 4,
}

/// Represents wind directions in mahjong, used for player seats and round wind.
#[cfg_attr(feature = "python", pyclass(eq, eq_int, from_py_object))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Wind {
    #[default]
    East = 0,
    South = 1,
    West = 2,
    North = 3,
}

impl From<u8> for Wind {
    fn from(val: u8) -> Self {
        match val % 4 {
            0 => Wind::East,
            1 => Wind::South,
            2 => Wind::West,
            3 => Wind::North,
            _ => unreachable!(),
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl Wind {
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

#[cfg_attr(feature = "python", pyclass(from_py_object))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Meld {
    pub meld_type: MeldType,
    pub tiles: Vec<u8>,
    pub opened: bool,
    pub from_who: i8,
    /// The tile claimed from another player's discard (for chi/pon/daiminkan).
    /// None for ankan/kakan or melds not involving a discard claim.
    pub called_tile: Option<u8>,
}

impl Meld {
    pub fn new(
        meld_type: MeldType,
        tiles: Vec<u8>,
        opened: bool,
        from_who: i8,
        called_tile: Option<u8>,
    ) -> Self {
        Self {
            meld_type,
            tiles,
            opened,
            from_who,
            called_tile,
        }
    }

    pub fn tiles_as_u32(&self) -> Vec<u32> {
        self.tiles.iter().map(|&t| t as u32).collect()
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl Meld {
    #[new]
    #[pyo3(signature = (meld_type, tiles, opened, from_who=-1, called_tile=None))]
    pub fn py_new(
        meld_type: MeldType,
        tiles: Vec<u8>,
        opened: bool,
        from_who: i8,
        called_tile: Option<u8>,
    ) -> Self {
        Self::new(meld_type, tiles, opened, from_who, called_tile)
    }

    #[getter]
    pub fn get_meld_type(&self) -> MeldType {
        self.meld_type
    }

    #[setter]
    pub fn set_meld_type(&mut self, meld_type: MeldType) {
        self.meld_type = meld_type;
    }

    #[getter]
    pub fn get_tiles(&self) -> Vec<u32> {
        self.tiles_as_u32()
    }

    #[setter]
    pub fn set_tiles(&mut self, tiles: Vec<u8>) {
        self.tiles = tiles;
    }

    #[getter]
    pub fn get_opened(&self) -> bool {
        self.opened
    }

    #[setter]
    pub fn set_opened(&mut self, opened: bool) {
        self.opened = opened;
    }

    #[getter]
    pub fn get_from_who(&self) -> i8 {
        self.from_who
    }

    #[setter]
    pub fn set_from_who(&mut self, from_who: i8) {
        self.from_who = from_who;
    }

    #[getter]
    pub fn get_called_tile(&self) -> Option<u8> {
        self.called_tile
    }
}

#[cfg_attr(feature = "python", pyclass(get_all, set_all, from_py_object))]
#[derive(Debug, Clone)]
pub struct Conditions {
    pub tsumo: bool,
    pub riichi: bool,
    pub double_riichi: bool,
    pub ippatsu: bool,
    pub haitei: bool,
    pub houtei: bool,
    pub rinshan: bool,
    pub player_wind: Wind,
    pub round_wind: Wind,
    pub chankan: bool,
    pub tsumo_first_turn: bool,
    pub riichi_sticks: u32,
    pub honba: u32,
    pub kita_count: u8,
    pub is_sanma: bool,
    pub num_players: u8,
}

impl Default for Conditions {
    fn default() -> Self {
        Self {
            tsumo: false,
            riichi: false,
            double_riichi: false,
            ippatsu: false,
            haitei: false,
            houtei: false,
            rinshan: false,
            player_wind: Wind::East,
            round_wind: Wind::East,
            chankan: false,
            tsumo_first_turn: false,
            riichi_sticks: 0,
            honba: 0,
            kita_count: 0,
            is_sanma: false,
            num_players: 4,
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl Conditions {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (tsumo=false, riichi=false, double_riichi=false, ippatsu=false, haitei=false, houtei=false, rinshan=false, chankan=false, tsumo_first_turn=false, player_wind=Wind::East, round_wind=Wind::East, riichi_sticks=0, honba=0, kita_count=0, is_sanma=false, num_players=4))]
    pub fn py_new(
        tsumo: bool,
        riichi: bool,
        double_riichi: bool,
        ippatsu: bool,
        haitei: bool,
        houtei: bool,
        rinshan: bool,
        chankan: bool,
        tsumo_first_turn: bool,
        player_wind: Wind,
        round_wind: Wind,
        riichi_sticks: u32,
        honba: u32,
        kita_count: u8,
        is_sanma: bool,
        num_players: u8,
    ) -> Self {
        Self {
            tsumo,
            riichi,
            double_riichi,
            ippatsu,
            haitei,
            houtei,
            rinshan,
            chankan,
            tsumo_first_turn,
            player_wind,
            round_wind,
            riichi_sticks,
            honba,
            kita_count,
            is_sanma,
            num_players,
        }
    }
}

#[cfg_attr(feature = "python", pyclass(get_all, set_all, from_py_object))]
#[derive(Debug, Clone)]
pub struct WinResult {
    pub is_win: bool,
    pub yakuman: bool,
    pub ron_agari: u32,
    pub tsumo_agari_oya: u32,
    pub tsumo_agari_ko: u32,
    pub yaku: Vec<u32>,
    pub han: u32,
    pub fu: u32,
    pub pao_payer: Option<u8>,
    pub has_win_shape: bool,
}

impl WinResult {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        is_win: bool,
        yakuman: bool,
        ron_agari: u32,
        tsumo_agari_oya: u32,
        tsumo_agari_ko: u32,
        yaku: Vec<u32>,
        han: u32,
        fu: u32,
        pao_payer: Option<u8>,
        has_win_shape: bool,
    ) -> Self {
        Self {
            is_win,
            yakuman,
            ron_agari,
            tsumo_agari_oya,
            tsumo_agari_ko,
            yaku,
            han,
            fu,
            pao_payer,
            has_win_shape,
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl WinResult {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (is_win, yakuman=false, ron_agari=0, tsumo_agari_oya=0, tsumo_agari_ko=0, yaku=vec![], han=0, fu=0, pao_payer=None, has_win_shape=false))]
    pub fn py_new(
        is_win: bool,
        yakuman: bool,
        ron_agari: u32,
        tsumo_agari_oya: u32,
        tsumo_agari_ko: u32,
        yaku: Vec<u32>,
        han: u32,
        fu: u32,
        pao_payer: Option<u8>,
        has_win_shape: bool,
    ) -> Self {
        Self::new(
            is_win,
            yakuman,
            ron_agari,
            tsumo_agari_oya,
            tsumo_agari_ko,
            yaku,
            han,
            fu,
            pao_payer,
            has_win_shape,
        )
    }

    pub fn yaku_list(&self) -> Vec<crate::yaku::Yaku> {
        self.yaku
            .iter()
            .filter_map(|&id| crate::yaku::get_yaku_by_id(id))
            .collect()
    }
}

pub fn is_terminal_tile(t: u8) -> bool {
    let t_type = t / 4;
    let rank = t_type % 9;
    let suit = t_type / 9;
    suit == 3 || rank == 0 || rank == 8
}

/// Check if a tile ID (0-135) is excluded in sanma (2m through 8m).
/// Manzu tiles: type 0-8 (IDs 0-35)
/// 1m = type 0 (IDs 0-3) - KEPT
/// 2m = type 1 (IDs 4-7) - EXCLUDED
/// ...
/// 8m = type 7 (IDs 28-31) - EXCLUDED
/// 9m = type 8 (IDs 32-35) - KEPT
pub fn is_sanma_excluded_tile(tile_id: u8) -> bool {
    let tile_type = tile_id / 4;
    // Manzu 2-8 (types 1-7)
    (1..=7).contains(&tile_type)
}

/// Standard dora wrapping for 4-player mahjong.
/// Input/output are tile types (0-33), not tile IDs.
pub fn standard_next_dora_tile(tile: u8) -> u8 {
    match tile {
        // Manzu (0-8): 1m->2m->...->9m->1m
        0..=8 => (tile + 1) % 9,
        // Pinzu (9-17): 1p->2p->...->9p->1p
        9..=17 => 9 + (tile - 9 + 1) % 9,
        // Souzu (18-26): 1s->2s->...->9s->1s
        18..=26 => 18 + (tile - 18 + 1) % 9,
        // Winds (27-30): E->S->W->N->E
        27..=30 => 27 + (tile - 27 + 1) % 4,
        // Dragons (31-33): Haku->Hatsu->Chun->Haku
        31..=33 => 31 + (tile - 31 + 1) % 3,
        _ => tile,
    }
}
