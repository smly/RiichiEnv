use pyo3::prelude::*;

pub const TILE_MAX: usize = 34;

/// A hand representation using a histogram of tile types (0-33).
#[derive(Debug, Clone)]
pub struct Hand {
    pub counts: [u8; TILE_MAX],
    // shuntsu_counts removed or ignored? Let's keep it private or ignore for now.
    pub shuntsu_counts: [u8; TILE_MAX],
}

impl Hand {
    pub fn new(tiles: Option<Vec<u8>>) -> Self {
        let mut h = Hand {
            counts: [0; TILE_MAX],
            shuntsu_counts: [0; TILE_MAX],
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
            shuntsu_counts: [0; TILE_MAX],
        }
    }
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeldType {
    Chi = 0,
    Peng = 1,
    Gang = 2,
    Angang = 3,
    Addgang = 4,
}

/// Represents wind directions in mahjong, used for player seats and round wind.
///
/// In mahjong, winds are assigned to players to indicate their seating position
/// and also used to denote the current round. The values map to integers as follows:
/// - East = 0: The dealer (oya) position in most rulesets
/// - South = 1: Player to dealer's right
/// - West = 2: Player across from dealer
/// - North = 3: Player to dealer's left
///
/// Wind values are used in scoring calculations and yaku determination,
/// particularly for yakuhai (wind honor tiles) and determining dealer bonuses.
#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wind {
    East = 0,
    South = 1,
    West = 2,
    North = 3,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Meld {
    #[pyo3(get, set)]
    pub meld_type: MeldType,
    #[pyo3(get, set)]
    pub tiles: Vec<u8>,
    #[pyo3(get, set)]
    pub opened: bool,
}

#[pymethods]
impl Meld {
    #[new]
    pub fn new(meld_type: MeldType, tiles: Vec<u8>, opened: bool) -> Self {
        Self {
            meld_type,
            tiles,
            opened,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Conditions {
    #[pyo3(get, set)]
    pub tsumo: bool,
    #[pyo3(get, set)]
    pub riichi: bool,
    #[pyo3(get, set)]
    pub double_riichi: bool,
    #[pyo3(get, set)]
    pub ippatsu: bool,
    #[pyo3(get, set)]
    pub haitei: bool,
    #[pyo3(get, set)]
    pub houtei: bool,
    #[pyo3(get, set)]
    pub rinshan: bool,
    #[pyo3(get, set)]
    pub player_wind: u8,
    #[pyo3(get, set)]
    pub round_wind: u8,
    #[pyo3(get, set)]
    pub chankan: bool,
    #[pyo3(get, set)]
    pub tsumo_first_turn: bool,
    #[pyo3(get, set)]
    pub kyoutaku: u32,
    #[pyo3(get, set)]
    pub tsumi: u32,
}

#[pymethods]
impl Conditions {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (tsumo=false, riichi=false, double_riichi=false, ippatsu=false, haitei=false, houtei=false, rinshan=false, chankan=false, tsumo_first_turn=false, player_wind=0, round_wind=0, kyoutaku=0, tsumi=0))]
    pub fn new(
        tsumo: bool,
        riichi: bool,
        double_riichi: bool,
        ippatsu: bool,
        haitei: bool,
        houtei: bool,
        rinshan: bool,
        chankan: bool,
        tsumo_first_turn: bool,
        player_wind: u8,
        round_wind: u8,
        kyoutaku: u32,
        tsumi: u32,
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
            kyoutaku,
            tsumi,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Agari {
    #[pyo3(get, set)]
    pub agari: bool,
    #[pyo3(get, set)]
    pub yakuman: bool,
    #[pyo3(get, set)]
    pub ron_agari: u32,
    #[pyo3(get, set)]
    pub tsumo_agari_oya: u32,
    #[pyo3(get, set)]
    pub tsumo_agari_ko: u32,
    #[pyo3(get, set)]
    pub yaku: Vec<u32>,
    #[pyo3(get, set)]
    pub han: u32,
    #[pyo3(get, set)]
    pub fu: u32,
}

#[pymethods]
impl Agari {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (agari, yakuman=false, ron_agari=0, tsumo_agari_oya=0, tsumo_agari_ko=0, yaku=vec![], han=0, fu=0))]
    pub fn new(
        agari: bool,
        yakuman: bool,
        ron_agari: u32,
        tsumo_agari_oya: u32,
        tsumo_agari_ko: u32,
        yaku: Vec<u32>,
        han: u32,
        fu: u32,
    ) -> Self {
        Self {
            agari,
            yakuman,
            ron_agari,
            tsumo_agari_oya,
            tsumo_agari_ko,
            yaku,
            han,
            fu,
        }
    }
}
