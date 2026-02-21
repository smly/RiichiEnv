use crate::types::{is_sanma_excluded_tile, standard_next_dora_tile};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GameSubMode3P {
    Single = 0,
    East = 1,
    Half = 2,
}

impl GameSubMode3P {
    pub fn from_game_mode(mode: u8) -> Self {
        match mode {
            3 => GameSubMode3P::Single,
            4 => GameSubMode3P::East,
            5 => GameSubMode3P::Half,
            _ => GameSubMode3P::East,
        }
    }

    pub fn game_mode_id(&self) -> u8 {
        3 + *self as u8
    }
}

/// 3P fixed configuration (no enum dispatch needed).
pub fn num_players() -> u8 {
    3
}

pub fn starting_score() -> i32 {
    35000
}

pub fn tile_set() -> Vec<u8> {
    (0..136u8).filter(|&t| !is_sanma_excluded_tile(t)).collect()
}

pub fn tenpai_pool() -> i32 {
    2000
}

/// Get the next dora tile for a given indicator tile (tile type 0-33).
/// In sanma, manzu wraps 1m(0)->9m(8) and 9m(8)->1m(0) directly.
pub fn get_next_dora_tile(tile: u8) -> u8 {
    if tile < 9 {
        // Manzu suit in sanma: only 0 (1m) and 8 (9m) exist
        if tile == 0 {
            8
        } else if tile == 8 {
            0
        } else {
            tile
        }
    } else {
        standard_next_dora_tile(tile)
    }
}
