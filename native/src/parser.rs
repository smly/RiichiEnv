#![allow(clippy::useless_conversion)]
use crate::types::{Meld, MeldType};
use pyo3::prelude::*;
use std::iter::Peekable;
use std::str::Chars;

struct TileManager {
    // For each tile 0..34, keep usage count or stack of available.
    // However, 136 format implies specific IDs.
    // 0..33 * 4 + 0..3 = 136 total.
    // Normal: 0, 1, 2, 3 offset.
    // Red 5s logic:
    // 5m (4): 16, 17, 18, 19. 16 is red (0m). 17-19 black.
    // 5p (13): 52, 53, 54, 55. 52 is red (0p). 53-55 black.
    // 5s (22): 88, 89, 90, 91. 88 is red (0s). 89-91 black.

    // We track which indices (0..4) are used for each tile (0..33).
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

// Parses string like "123m456p..." or "(...)"
pub fn parse_hand_internal(text: &str) -> PyResult<(Vec<u8>, Vec<Meld>)> {
    let mut tm = TileManager::new();
    let mut tiles_136 = Vec::new();
    let mut melds = Vec::new();

    let mut chars = text.chars().peekable();

    // Accumulate digits for a suit block
    let mut pending_digits: Vec<char> = Vec::new();

    while let Some(&c) = chars.peek() {
        if c == '(' {
            chars.next(); // consume '('
            let meld = parse_meld(&mut chars, &mut tm)?;
            melds.push(meld);
        } else if c.is_ascii_digit() {
            chars.next();
            pending_digits.push(c);
        } else if is_suit(c) {
            chars.next();
            // Flush pending
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
                    (suit_offset + 4, true) // 0 means red 5
                } else {
                    (suit_offset + val - 1, false)
                };
                let tid = tm
                    .get_tile(tile_34, is_red)
                    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
                tiles_136.push(tid);
            }
            pending_digits.clear();
        } else {
            // Ignore whitespace or error?
            chars.next();
        }
    }

    if !pending_digits.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Pending digits without suit",
        ));
    }

    Ok((tiles_136, melds))
}

// Parses string like "123m456p..." or "(...)"
#[pyfunction]
pub fn parse_hand(text: &str) -> PyResult<(Vec<u32>, Vec<Meld>)> {
    let (tiles, melds) = parse_hand_internal(text)?;
    Ok((tiles.iter().map(|&x| x as u32).collect(), melds))
}

/// Parse a single tile string into a 136-format tile ID.
///
/// The function accepts tile notation in the format: `{rank}{suit}` where:
/// - `rank` is a digit 0-9 (0 represents red five)
/// - `suit` is one of: m (man/characters), p (pin/circles), s (sou/bamboo), z (honors)
///
/// # Arguments
///
/// * `text` - A string containing exactly one tile (e.g., "2z", "5m", "0p")
///
/// # Returns
///
/// Returns a u8 tile ID in the 136-format where:
/// - Each of the 34 unique tiles has 4 copies (0-135 total)
/// - Tile ID = (tile_type * 4) + copy_index
/// - For red fives (0m, 0p, 0s), the ID corresponds to the first copy (index 0)
///
/// # Examples
///
/// ```python
/// import riichienv as rv
///
/// # Regular tiles
/// tile_id = rv.parse_tile("1m")  # Returns 0 (1-man, first copy)
/// tile_id = rv.parse_tile("5m")  # Returns 17 (5-man, black, first available)
/// tile_id = rv.parse_tile("1z")  # Returns 108 (East wind, first copy)
///
/// # Red fives (special notation using 0)
/// tile_id = rv.parse_tile("0m")  # Returns 16 (red 5-man)
/// tile_id = rv.parse_tile("0p")  # Returns 52 (red 5-pin)
/// tile_id = rv.parse_tile("0s")  # Returns 88 (red 5-sou)
///
/// # Use with AgariCalculator
/// hand = rv.AgariCalculator.hand_from_text("123m456p789s111z2z")
/// win_tile = rv.parse_tile("2z")  # Parse the winning tile
/// result = hand.calc(win_tile, conditions=rv.Conditions())
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - The input contains meld syntax (parentheses)
/// - The input contains no tiles or multiple tiles
/// - The input has invalid tile notation
#[pyfunction]
pub fn parse_tile(text: &str) -> PyResult<u8> {
    let (tiles, melds) = parse_hand_internal(text)?;
    if !melds.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "parse_tile expects a single tile, but found meld syntax in input",
        ));
    }
    if tiles.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No tile found in string",
        ));
    }
    if tiles.len() != 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Expected exactly one tile, but found {} tiles in string",
            tiles.len()
        )));
    }
    Ok(tiles[0])
}

fn is_suit(c: char) -> bool {
    matches!(c, 'm' | 'p' | 's' | 'z')
}

fn parse_meld(chars: &mut Peekable<Chars>, tm: &mut TileManager) -> PyResult<Meld> {
    // Formats:
    // (XYZCI) -> Chi
    // (pXCIr) -> Pon
    // (kXCIr) -> Kan (Closed/Daimin)
    // (sXCIr) -> AddedKan

    // Read unitl ')'
    let mut content = String::new();
    while let Some(&c) = chars.peek() {
        if c == ')' {
            chars.next();
            break;
        }
        content.push(c);
        chars.next();
    }

    // Analyze prefix
    let (prefix, rest) = if let Some(stripped) = content.strip_prefix('p') {
        ('p', stripped)
    } else if let Some(stripped) = content.strip_prefix('k') {
        ('k', stripped)
    } else if let Some(stripped) = content.strip_prefix('s') {
        ('s', stripped)
    } else {
        (' ', content.as_str()) // Chi or error?
    };

    // Chi format: "123m1" -> digits, suit, index
    // Other format: "1z1" -> digit (or digits?), suit, index, optional 'r'?

    // Parse digits
    let mut digits = Vec::new();
    let remaining_str = rest;
    let mut suit_char = ' ';

    // Extract digits loop
    let mut idx = 0;
    let chars_vec: Vec<char> = remaining_str.chars().collect();
    while idx < chars_vec.len() && chars_vec[idx].is_ascii_digit() {
        digits.push(chars_vec[idx]);
        idx += 1;
    }

    if idx < chars_vec.len() {
        suit_char = chars_vec[idx];
        idx += 1; // consume suit
    }

    // Check call index
    let _call_idx = if idx < chars_vec.len() {
        let c = chars_vec[idx];
        if c.is_ascii_digit() {
            c.to_digit(10).unwrap()
        } else {
            0
        } // Default or error?
    } else {
        0
    }; // Default if not present (Closed Kan?)

    // Remaining (e.g. 'r'?)

    let suit_offset = match suit_char {
        'm' => 0,
        'p' => 9,
        's' => 18,
        'z' => 27,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid suit in meld: {}",
                suit_char
            )))
        }
    };

    let mut tiles_136 = Vec::new();

    // Construct tiles
    // If Chi: 3 digits.
    if prefix == ' ' {
        // Chi
        if digits.len() != 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Chi meld requires 3 digits",
            ));
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
                    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?,
            );
        }
        tiles_136.sort(); // standardize
        Ok(Meld::new(MeldType::Chi, tiles_136, true, -1))
    } else {
        // Pon/Kan/AddedKan
        // Usually 1 digit for type (e.g. '1' in 'p1z1'). Or '0' for red pon.
        // Pon: 3 tiles. Kan: 4 tiles.
        // If 1 digit given, it implies triplets/quads of that tile.
        // If >1 digit? e.g. "4445" for kan? No, standard format is simplified?
        // Examples: (p0m2), (k2z), (s3z3). Single digit + suit.
        // Note: For red pon (p0m2), it means a pon of 5m involving red 5.
        // How to select which are used?
        // Pon uses 3 copies. Kan uses 4.

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

        // Populate tiles
        // If red indicated, make sure to include red.
        // If not red indicated, prefer blacks, but if ran out?

        let mut got_red = false;
        if is_red_indicated {
            // Need red
            tiles_136.push(
                tm.get_tile(base_34, true)
                    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?,
            );
            got_red = true;
        }

        // Fill remaining
        while tiles_136.len() < count {
            // Try black first if red already got, or if red not indicated (but might pick red if black out? No, try black)
            // Actually tm.get_tile will error if I specifically ask for one.
            // I should ask for "standard" (false) and fallback?
            // My get_tile logic handles fallback slightly? No. 'is_red' arg is stricter.
            // Let's modify usage.
            // If I need any tile, prefer black.
            if let Ok(t) = tm.get_tile(base_34, false) {
                tiles_136.push(t);
            } else if !got_red {
                // Try red if I haven't got it and ran out of black?
                // Usually Pon of 5 without red means [5, 5, 5] black.
                // But strings might be distinct.
                if let Ok(t) = tm.get_tile(base_34, true) {
                    tiles_136.push(t);
                    got_red = true;
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Not enough tiles for meld of {}",
                        base_34
                    )));
                }
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Not enough tiles for meld of {}",
                    base_34
                )));
            }
        }

        tiles_136.sort();

        let mtype = match prefix {
            'p' => MeldType::Peng,
            'k' => {
                // (k2z) -> Call index optional. "if not available, can is considered closed".
                // In my parse logic, I defaulted call_idx to 0 if digits missing.
                // But digits were handled by the loop.
                // call_idx comes AFTER suit.
                // Original check `if idx < chars_vec.len()..`
                // So if call index is present (non-zero?), it's open (Daiminkan).
                // If 0 or missing, usually Ankan.
                // Wait, (k2z) -> 2z count 4. No index. Closed Kan (Ankan).
                // (k2z3) -> Index 3. Open Kan (Daiminkan/Gang).
                // So I need to know if index was present?
                // My logic defaults to 0 if missing.
                // Assuming index 0 implies Ankan (since index is 1,2,3 for players).
                // But Self is 0. You don't call from self.
                // So 0 means Ankan.
                // 1,2,3 means Gang.
                // Wait, what variable holds call index? `_call_idx`.
                // I need to use it.
                // Let's assume idx 1, 2, 3 -> Open.
                // (k2z) -> 0 -> Ankan.
                if _call_idx == 0 {
                    MeldType::Angang
                } else {
                    MeldType::Gang
                }
            }
            's' => MeldType::Addgang,
            _ => unreachable!(),
        };

        // For Angang/Addgang, 'opened' flag logic?
        // Angang: opened=false.
        // Addgang: opened=true.
        // Gang: opened=true.
        // Peng: opened=true.
        // Chi: opened=true.

        let opened = mtype != MeldType::Angang;

        Ok(Meld::new(mtype, tiles_136, opened, -1))
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

#[allow(dead_code)]
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
        // Support 0m/0p/0s just in case
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
    // If it's a 5, and not red, it should be base+1 (17, 53, 89)
    if num == 5 {
        Some(base + 1)
    } else {
        Some(base)
    }
}
