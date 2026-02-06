/// Yaku possibility checker based on observable information
///
/// This module provides functions to determine whether certain yaku (winning hands)
/// are possible or impossible based on visible information (melds, discards, dora indicators).
use crate::types::Meld;

/// Result of yaku possibility check
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum YakuPossibility {
    /// Yaku is definitely possible (e.g., pon of dragons visible)
    Possible,
    /// Yaku is definitely impossible (e.g., terminals in melds for tanyao)
    Impossible,
    /// Yaku possibility is unknown (conservative: assume possible)
    Unknown,
}

impl YakuPossibility {
    pub fn to_f32(self) -> f32 {
        match self {
            YakuPossibility::Possible | YakuPossibility::Unknown => 1.0,
            YakuPossibility::Impossible => 0.0,
        }
    }
}

/// Check if tanyao (all simples) is possible
pub fn check_tanyao(melds: &[Meld]) -> YakuPossibility {
    for meld in melds {
        for &tile in &meld.tiles {
            let tile_type = (tile / 4) as usize;
            // Check for terminals (0,8,9,17,18,26) or honors (27-33)
            if tile_type % 9 == 0 || tile_type % 9 == 8 || tile_type >= 27 {
                return YakuPossibility::Impossible;
            }
        }
    }
    YakuPossibility::Unknown
}

/// Count visible tiles of a specific type
fn count_visible_tiles(tile_type: usize, discards: &[u32], dora_indicators: &[u32]) -> usize {
    let mut count = 0;

    for &tile in discards {
        if (tile / 4) as usize == tile_type {
            count += 1;
        }
    }

    for &tile in dora_indicators {
        if (tile / 4) as usize == tile_type {
            count += 1;
        }
    }

    count
}

/// Check if a specific yakuhai (honor tile) yaku is possible
pub fn check_yakuhai(
    tile_type: usize,
    melds: &[Meld],
    discards: &[u32],
    dora_indicators: &[u32],
) -> YakuPossibility {
    // Check if there's a pon/kan of this tile in melds
    for meld in melds {
        if !meld.tiles.is_empty() {
            let meld_tile_type = (meld.tiles[0] / 4) as usize;
            if meld_tile_type == tile_type && meld.tiles.len() >= 3 {
                return YakuPossibility::Possible;
            }
        }
    }

    // Check if 3+ tiles are visible (impossible to make pon)
    let visible = count_visible_tiles(tile_type, discards, dora_indicators);
    if visible >= 3 {
        return YakuPossibility::Impossible;
    }

    YakuPossibility::Unknown
}

/// Check if honitsu (half flush) or chinitsu (full flush) is possible
pub fn check_flush(melds: &[Meld]) -> (YakuPossibility, YakuPossibility) {
    if melds.is_empty() {
        return (YakuPossibility::Unknown, YakuPossibility::Unknown);
    }

    let mut has_man = false;
    let mut has_pin = false;
    let mut has_sou = false;
    let mut has_honor = false;

    for meld in melds {
        for &tile in &meld.tiles {
            let tile_type = (tile / 4) as usize;
            if tile_type < 9 {
                has_man = true;
            } else if tile_type < 18 {
                has_pin = true;
            } else if tile_type < 27 {
                has_sou = true;
            } else {
                has_honor = true;
            }
        }
    }

    let num_suits = [has_man, has_pin, has_sou].iter().filter(|&&x| x).count();

    let honitsu = if num_suits == 0 {
        // Only honors (unlikely but possible)
        YakuPossibility::Unknown
    } else if num_suits == 1 {
        // One suit + maybe honors = possible honitsu
        YakuPossibility::Possible
    } else {
        // 2+ suits = impossible
        YakuPossibility::Impossible
    };

    let chinitsu = if num_suits == 0 {
        YakuPossibility::Unknown
    } else if num_suits == 1 && !has_honor {
        // One suit only = possible chinitsu
        YakuPossibility::Possible
    } else if num_suits >= 2 || has_honor {
        // 2+ suits or honors = impossible
        YakuPossibility::Impossible
    } else {
        YakuPossibility::Unknown
    };

    (honitsu, chinitsu)
}

/// Check if toitoi (all triplets) is possible
pub fn check_toitoi(melds: &[Meld]) -> YakuPossibility {
    for meld in melds {
        // If there's any chi (sequence), toitoi is impossible
        if meld.tiles.len() == 3 {
            let t0 = (meld.tiles[0] / 4) as usize;
            let t1 = (meld.tiles[1] / 4) as usize;
            let t2 = (meld.tiles[2] / 4) as usize;

            // Check if it's a sequence (consecutive numbers)
            if t0 + 1 == t1 && t1 + 1 == t2 && t0 < 27 {
                return YakuPossibility::Impossible;
            }
        }
    }

    // If all melds are pon/kan, toitoi is possible
    if !melds.is_empty()
        && melds
            .iter()
            .all(|m| m.tiles.len() >= 3 && m.tiles[0] / 4 == m.tiles[1] / 4)
    {
        return YakuPossibility::Possible;
    }

    YakuPossibility::Unknown
}

/// Check if chiitoitsu (seven pairs) is possible
pub fn check_chiitoitsu(melds: &[Meld]) -> YakuPossibility {
    // Chiitoitsu cannot have any melds (kan is not allowed)
    if !melds.is_empty() {
        return YakuPossibility::Impossible;
    }
    YakuPossibility::Unknown
}

/// Check if shousangen (small three dragons) is possible
pub fn check_shousangen(
    _melds: &[Meld],
    discards: &[u32],
    dora_indicators: &[u32],
) -> YakuPossibility {
    // Check visibility of each dragon tile (31=white, 32=green, 33=red)
    let dragons = [31, 32, 33];
    let mut impossible_count = 0;

    for &dragon_type in &dragons {
        // If 4 tiles are visible, this dragon is impossible
        let visible = count_visible_tiles(dragon_type, discards, dora_indicators);
        if visible >= 4 {
            impossible_count += 1;
        }
    }

    // If any one dragon type has all 4 visible, shousangen is impossible
    if impossible_count >= 1 {
        return YakuPossibility::Impossible;
    }

    YakuPossibility::Unknown
}

/// Check if daisangen (big three dragons) is possible
pub fn check_daisangen(
    melds: &[Meld],
    discards: &[u32],
    dora_indicators: &[u32],
) -> YakuPossibility {
    let dragons = [31, 32, 33];
    let mut pon_count = 0;

    // Count how many dragon pons are visible
    for &dragon_type in &dragons {
        let has_pon = melds.iter().any(|m| {
            m.tiles.len() >= 3 && !m.tiles.is_empty() && (m.tiles[0] / 4) as usize == dragon_type
        });

        if has_pon {
            pon_count += 1;
        } else {
            // Check if this dragon is impossible
            let visible = count_visible_tiles(dragon_type, discards, dora_indicators);
            if visible >= 2 {
                // 2+ in river/dora + not in melds = impossible to complete
                return YakuPossibility::Impossible;
            }
        }
    }

    if pon_count == 3 {
        return YakuPossibility::Possible;
    }

    YakuPossibility::Unknown
}

/// Check if tsuuiisou (all honors) is possible
pub fn check_tsuuiisou(melds: &[Meld]) -> YakuPossibility {
    for meld in melds {
        for &tile in &meld.tiles {
            let tile_type = (tile / 4) as usize;
            if tile_type < 27 {
                // Number tile found = impossible
                return YakuPossibility::Impossible;
            }
        }
    }
    YakuPossibility::Unknown
}

/// Check if chinroutou (all terminals) is possible
pub fn check_chinroutou(melds: &[Meld]) -> YakuPossibility {
    for meld in melds {
        for &tile in &meld.tiles {
            let tile_type = (tile / 4) as usize;
            // Must be terminals only (0, 8, 9, 17, 18, 26)
            if tile_type >= 27 || (!tile_type % 9 == 0 && tile_type % 9 != 8) {
                return YakuPossibility::Impossible;
            }
        }
    }
    YakuPossibility::Unknown
}

/// Check if honroutou (all terminals and honors) is possible
pub fn check_honroutou(melds: &[Meld]) -> YakuPossibility {
    for meld in melds {
        for &tile in &meld.tiles {
            let tile_type = (tile / 4) as usize;
            // Must be terminals (0,8,9,17,18,26) or honors (27+)
            if tile_type < 27 && !tile_type % 9 == 0 && tile_type % 9 != 8 {
                // Simples (2-8) found = impossible
                return YakuPossibility::Impossible;
            }
        }
    }
    YakuPossibility::Unknown
}

/// Check if kokushi musou (thirteen orphans) is possible
pub fn check_kokushi(melds: &[Meld], discards: &[u32], dora_indicators: &[u32]) -> YakuPossibility {
    // Kokushi cannot have any melds (must be closed hand)
    if !melds.is_empty() {
        return YakuPossibility::Impossible;
    }

    // Kokushi requires all 13 terminal/honor types: 0,8,9,17,18,26,27,28,29,30,31,32,33
    let required_types = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];

    // Check if any of the 13 required types have 4 tiles visible (impossible to collect)
    for &tile_type in &required_types {
        let visible = count_visible_tiles(tile_type, discards, dora_indicators);
        if visible >= 4 {
            // All 4 tiles of this type are visible = impossible to make kokushi
            return YakuPossibility::Impossible;
        }
    }

    YakuPossibility::Unknown
}

/// Check if chanta (outside hand) is possible
pub fn check_chanta(melds: &[Meld]) -> YakuPossibility {
    // Chanta requires terminals or honors in every group
    // If there are simples-only melds (pon of 2-8, chi with no terminals), it's impossible
    for meld in melds {
        if meld.tiles.is_empty() {
            continue;
        }

        let mut has_terminal_or_honor = false;
        for &tile in &meld.tiles {
            let tile_type = (tile / 4) as usize;
            // Check if terminal (0,8,9,17,18,26) or honor (27+)
            if tile_type % 9 == 0 || tile_type % 9 == 8 || tile_type >= 27 {
                has_terminal_or_honor = true;
                break;
            }
        }

        if !has_terminal_or_honor {
            // This meld contains only simples (2-8) = impossible chanta
            return YakuPossibility::Impossible;
        }
    }

    YakuPossibility::Unknown
}

/// Check if junchan (terminals in all groups) is possible
pub fn check_junchan(melds: &[Meld]) -> YakuPossibility {
    // Junchan requires terminals (no honors) in every group
    for meld in melds {
        if meld.tiles.is_empty() {
            continue;
        }

        let mut has_terminal = false;
        for &tile in &meld.tiles {
            let tile_type = (tile / 4) as usize;
            // Check if honor
            if tile_type >= 27 {
                // Honor tile found = impossible junchan
                return YakuPossibility::Impossible;
            }
            // Check if terminal (0,8,9,17,18,26)
            if tile_type % 9 == 0 || tile_type % 9 == 8 {
                has_terminal = true;
            }
        }

        if !has_terminal {
            // This meld has no terminal = impossible junchan
            return YakuPossibility::Impossible;
        }
    }

    YakuPossibility::Unknown
}

/// Check if sanshoku doujun (three colored straight) is possible
/// This checks if it's definitely impossible based on melds
pub fn check_sanshoku_doujun(melds: &[Meld]) -> YakuPossibility {
    // For sanshoku to be impossible, we need to see that certain sequences can't be completed
    // This is complex to determine from partial information, so we're conservative
    // We can only mark it impossible if there are honor melds or if the melds show incompatible patterns

    for meld in melds {
        for &tile in &meld.tiles {
            let tile_type = (tile / 4) as usize;
            // If there are honor tiles in melds, sequences are not possible
            if tile_type >= 27 {
                // However, this doesn't make sanshoku impossible if other sequences exist
                // We're being very conservative here
                continue;
            }
        }
    }

    // Conservative: assume possible unless we have strong evidence
    YakuPossibility::Unknown
}

/// Check if iipeikou (pure double sequence) is possible
pub fn check_iipeikou(melds: &[Meld]) -> YakuPossibility {
    // Iipeikou requires closed hand (no melds)
    if !melds.is_empty() {
        return YakuPossibility::Impossible;
    }

    YakuPossibility::Unknown
}

/// Check if ittsu (straight) is possible
pub fn check_ittsu(melds: &[Meld]) -> YakuPossibility {
    // Ittsu requires 123-456-789 of one suit
    // We check if this is definitely impossible based on what we see

    // Count if we have honors in melds (doesn't make ittsu impossible, but limits possibilities)
    for meld in melds {
        for &tile in &meld.tiles {
            let tile_type = (tile / 4) as usize;
            if tile_type >= 27 {
                // Honor melds don't prevent ittsu
                continue;
            }
        }
    }

    // Conservative: assume possible
    YakuPossibility::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MeldType;

    #[test]
    fn test_tanyao() {
        // Tanyao possible with no melds
        assert_eq!(check_tanyao(&[]), YakuPossibility::Unknown);

        // Tanyao impossible with terminal tile
        let meld = Meld {
            meld_type: MeldType::Peng,
            tiles: vec![0, 1, 2], // 1m (terminal)
            opened: true,
            from_who: -1,
        };
        assert_eq!(check_tanyao(&[meld]), YakuPossibility::Impossible);

        // Tanyao possible with all simples
        let meld = Meld {
            meld_type: MeldType::Peng,
            tiles: vec![16, 17, 18], // 5m (simple)
            opened: true,
            from_who: -1,
        };
        assert_eq!(check_tanyao(&[meld]), YakuPossibility::Unknown);
    }

    #[test]
    fn test_toitoi() {
        // Toitoi impossible with chi
        let meld = Meld {
            meld_type: MeldType::Chi,
            tiles: vec![0, 4, 8], // 1m-2m-3m
            opened: true,
            from_who: -1,
        };
        assert_eq!(check_toitoi(&[meld]), YakuPossibility::Impossible);

        // Toitoi possible with pon
        let meld = Meld {
            meld_type: MeldType::Peng,
            tiles: vec![16, 17, 18], // 5m pon
            opened: true,
            from_who: -1,
        };
        assert_eq!(check_toitoi(&[meld]), YakuPossibility::Possible);
    }
}
