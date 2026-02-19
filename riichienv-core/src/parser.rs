#![allow(clippy::useless_conversion)]
use crate::errors::{RiichiError, RiichiResult};
use crate::types::{Meld, MeldType};
#[cfg(feature = "python")]
use pyo3::prelude::*;
use std::iter::Peekable;
use std::str::Chars;

struct TileManager {
    used: [[bool; 4]; 34],
}

impl TileManager {
    fn new() -> Self {
        Self {
            used: [[false; 4]; 34],
        }
    }

    fn get_tile(&mut self, tile_34: usize, is_red: bool) -> Result<u8, String> {
        if tile_34 >= 34 {
            return Err(format!("Invalid tile ID: {}", tile_34));
        }

        let is_5 = tile_34 == 4 || tile_34 == 13 || tile_34 == 22;

        let search_indices: &[usize] = match (is_5, is_red) {
            (true, true) => &[0],
            (true, false) => &[1, 2, 3, 0],
            (false, _) => &[0, 1, 2, 3],
        };

        let target_idx = search_indices
            .iter()
            .find(|&&idx| !self.used[tile_34][idx])
            .copied()
            .ok_or_else(|| format!("No more copies of tile {}", tile_34))?;
        self.used[tile_34][target_idx] = true;
        Ok(((tile_34 * 4) + target_idx) as u8)
    }
}

pub fn parse_hand_internal(text: &str) -> RiichiResult<(Vec<u8>, Vec<Meld>)> {
    let mut tm = TileManager::new();
    let mut tiles_136 = Vec::new();
    let mut melds = Vec::new();

    let mut chars = text.chars().peekable();
    let mut pending_digits: Vec<char> = Vec::new();

    while let Some(&c) = chars.peek() {
        if c == '(' {
            chars.next();
            let meld = parse_meld(&mut chars, &mut tm)?;
            melds.push(meld);
        } else if c.is_ascii_digit() {
            chars.next();
            pending_digits.push(c);
        } else if is_suit(c) {
            chars.next();
            let suit_offset = match c {
                'm' => 0,
                'p' => 9,
                's' => 18,
                'z' => 27,
                _ => unreachable!(),
            };
            for d in &pending_digits {
                let val = d.to_digit(10).unwrap() as usize;
                let (tile_34, is_red) = if val == 0 {
                    (suit_offset + 4, true)
                } else {
                    (suit_offset + val - 1, false)
                };
                let tid = tm
                    .get_tile(tile_34, is_red)
                    .map_err(|e| RiichiError::Parse {
                        input: text.to_string(),
                        message: e,
                    })?;
                tiles_136.push(tid);
            }
            pending_digits.clear();
        } else {
            chars.next();
        }
    }

    if !pending_digits.is_empty() {
        return Err(RiichiError::Parse {
            input: text.to_string(),
            message: "Pending digits without suit".to_string(),
        });
    }

    Ok((tiles_136, melds))
}

pub fn parse_hand(text: &str) -> RiichiResult<(Vec<u32>, Vec<Meld>)> {
    let (tiles, melds) = parse_hand_internal(text)?;
    Ok((tiles.iter().map(|&x| x as u32).collect(), melds))
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "parse_hand")]
pub fn parse_hand_py(text: &str) -> PyResult<(Vec<u32>, Vec<Meld>)> {
    parse_hand(text).map_err(Into::into)
}

pub fn parse_tile(text: &str) -> RiichiResult<u8> {
    let (tiles, melds) = parse_hand_internal(text)?;
    if !melds.is_empty() {
        return Err(RiichiError::Parse {
            input: text.to_string(),
            message: "parse_tile expects a single tile, but found meld syntax in input".to_string(),
        });
    }
    if tiles.is_empty() {
        return Err(RiichiError::Parse {
            input: text.to_string(),
            message: "No tile found in string".to_string(),
        });
    }
    if tiles.len() != 1 {
        return Err(RiichiError::Parse {
            input: text.to_string(),
            message: format!(
                "Expected exactly one tile, but found {} tiles in string",
                tiles.len()
            ),
        });
    }
    Ok(tiles[0])
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "parse_tile")]
pub fn parse_tile_py(text: &str) -> PyResult<u8> {
    parse_tile(text).map_err(Into::into)
}

fn is_suit(c: char) -> bool {
    matches!(c, 'm' | 'p' | 's' | 'z')
}

fn parse_meld(chars: &mut Peekable<Chars>, tm: &mut TileManager) -> RiichiResult<Meld> {
    let mut content = String::new();
    while let Some(&c) = chars.peek() {
        if c == ')' {
            chars.next();
            break;
        }
        content.push(c);
        chars.next();
    }

    let (prefix, rest) = if let Some(stripped) = content.strip_prefix('p') {
        ('p', stripped)
    } else if let Some(stripped) = content.strip_prefix('k') {
        ('k', stripped)
    } else if let Some(stripped) = content.strip_prefix('s') {
        ('s', stripped)
    } else {
        (' ', content.as_str())
    };

    let mut digits = Vec::new();
    let remaining_str = rest;
    let mut suit_char = ' ';

    let mut idx = 0;
    let chars_vec: Vec<char> = remaining_str.chars().collect();
    while idx < chars_vec.len() && chars_vec[idx].is_ascii_digit() {
        digits.push(chars_vec[idx]);
        idx += 1;
    }

    if idx < chars_vec.len() {
        suit_char = chars_vec[idx];
        idx += 1;
    }

    let _call_idx = if idx < chars_vec.len() {
        let c = chars_vec[idx];
        if c.is_ascii_digit() {
            c.to_digit(10).unwrap()
        } else {
            0
        }
    } else {
        0
    };

    let suit_offset = match suit_char {
        'm' => 0,
        'p' => 9,
        's' => 18,
        'z' => 27,
        _ => {
            return Err(RiichiError::Parse {
                input: content.clone(),
                message: format!("Invalid suit in meld: {}", suit_char),
            })
        }
    };

    let mut tiles_136 = Vec::new();

    if prefix == ' ' {
        // Chi
        if digits.len() != 3 {
            return Err(RiichiError::Parse {
                input: content.clone(),
                message: "Chi meld requires 3 digits".to_string(),
            });
        }
        for d in digits {
            let val = d.to_digit(10).unwrap() as usize;
            let (tile_34, is_red) = if val == 0 {
                (suit_offset + 4, true)
            } else {
                (suit_offset + val - 1, false)
            };
            tiles_136.push(
                tm.get_tile(tile_34, is_red)
                    .map_err(|e| RiichiError::Parse {
                        input: content.clone(),
                        message: e,
                    })?,
            );
        }
        tiles_136.sort();
        Ok(Meld::new(MeldType::Chi, tiles_136, true, -1, None))
    } else {
        let val_d = digits[0].to_digit(10).unwrap() as usize;
        let (base_34, is_red_indicated) = if val_d == 0 {
            (suit_offset + 4, true)
        } else {
            (suit_offset + val_d - 1, false)
        };

        let count = match prefix {
            'p' => 3,
            'k' | 's' => 4,
            _ => 3,
        };

        let mut got_red = false;
        if is_red_indicated {
            tiles_136.push(tm.get_tile(base_34, true).map_err(|e| RiichiError::Parse {
                input: content.clone(),
                message: e,
            })?);
            got_red = true;
        }

        while tiles_136.len() < count {
            if let Ok(t) = tm.get_tile(base_34, false) {
                tiles_136.push(t);
            } else if !got_red {
                if let Ok(t) = tm.get_tile(base_34, true) {
                    tiles_136.push(t);
                    got_red = true;
                } else {
                    return Err(RiichiError::Parse {
                        input: content.clone(),
                        message: format!("Not enough tiles for meld of {}", base_34),
                    });
                }
            } else {
                return Err(RiichiError::Parse {
                    input: content.clone(),
                    message: format!("Not enough tiles for meld of {}", base_34),
                });
            }
        }

        tiles_136.sort();

        let mtype = match prefix {
            'p' => MeldType::Pon,
            'k' => {
                if _call_idx == 0 {
                    MeldType::Ankan
                } else {
                    MeldType::Daiminkan
                }
            }
            's' => MeldType::Kakan,
            _ => unreachable!(),
        };

        let opened = mtype != MeldType::Ankan;

        Ok(Meld::new(mtype, tiles_136, opened, -1, None))
    }
}

pub fn tid_to_mjai(tid: u8) -> String {
    // Check Red 5s
    if tid == 16 {
        return "5mr".to_string();
    }
    if tid == 52 {
        return "5pr".to_string();
    }
    if tid == 88 {
        return "5sr".to_string();
    }

    let kind = tid / 36;
    if kind < 3 {
        let suit_char = match kind {
            0 => "m",
            1 => "p",
            2 => "s",
            _ => unreachable!(),
        };
        let offset = tid % 36;
        let num = offset / 4 + 1;
        format!("{}{}", num, suit_char)
    } else {
        let offset = tid - 108;
        let num = offset / 4 + 1;
        let honors = ["E", "S", "W", "N", "P", "F", "C"];
        if (1..=7).contains(&num) {
            honors[num as usize - 1].to_string()
        } else {
            format!("{}z", num)
        }
    }
}

pub fn mjai_to_tid(mjai: &str) -> Option<u8> {
    // Honors
    let honors = ["E", "S", "W", "N", "P", "F", "C"];
    if let Some(pos) = honors.iter().position(|&h| h == mjai) {
        return Some(108 + (pos as u8) * 4);
    }

    // Red 5s
    if mjai == "5mr" {
        return Some(16);
    }
    if mjai == "5pr" {
        return Some(52);
    }
    if mjai == "5sr" {
        return Some(88);
    }

    // MPS
    if mjai.len() < 2 {
        return None;
    }
    let num_char = mjai.chars().next()?;
    let suit_char = mjai.chars().nth(1)?;
    let num = num_char.to_digit(10)? as u8;
    if num == 0 {
        let suit_idx = match suit_char {
            'm' => 0,
            'p' => 1,
            's' => 2,
            _ => return None,
        };
        return Some(suit_idx * 36 + 16);
    }
    if !(1..=9).contains(&num) {
        return None;
    }
    let suit_idx = match suit_char {
        'm' => 0,
        'p' => 1,
        's' => 2,
        'z' => {
            return Some(108 + (num - 1) * 4);
        }
        _ => return None,
    };

    let base = suit_idx * 36 + (num - 1) * 4;
    if num == 5 {
        Some(base + 1)
    } else {
        Some(base)
    }
}
