/// Calculate shanten number for a given hand using simple algorithm
/// This is a basic implementation without external dependencies
/// Returns shanten number (-1 for tenpai, 0+ for not tenpai)
pub fn calculate_shanten(hand_tiles: &[u32]) -> i32 {
    // Convert tile IDs to tile counts (0-33)
    let mut tile_counts = [0u8; 34];
    for &tile in hand_tiles {
        let tile_type = (tile / 4) as usize;
        if tile_type < 34 {
            tile_counts[tile_type] += 1;
        }
    }

    // Calculate different shanten patterns and take minimum
    let standard_shanten = calculate_standard_shanten(&tile_counts);
    let seven_pairs_shanten = calculate_seven_pairs_shanten(&tile_counts);
    let thirteen_orphans_shanten = calculate_thirteen_orphans_shanten(&tile_counts);

    standard_shanten
        .min(seven_pairs_shanten)
        .min(thirteen_orphans_shanten)
}

/// Calculate standard form shanten (4 melds + 1 pair)
fn calculate_standard_shanten(tiles: &[u8; 34]) -> i32 {
    // Simplified shanten calculation
    let mut melds = 0;
    let mut taatsu = 0;
    let mut pairs = 0;

    for &count in tiles.iter().take(34) {
        if count >= 3 {
            melds += 1;
        } else if count == 2 {
            pairs += 1;
        }
    }

    // Check for sequences (for number tiles only)
    for suit_start in [0, 9, 18] {
        for i in 0..7 {
            let idx = suit_start + i;
            if idx + 2 < 34 && tiles[idx] > 0 && tiles[idx + 1] > 0 && tiles[idx + 2] > 0 {
                taatsu += 1;
            }
        }
    }

    // Estimate shanten
    let _total_groups = melds + taatsu;
    let has_pair = pairs > 0 || melds + taatsu >= 5;

    if melds == 4 && has_pair {
        -1 // Tenpai
    } else if melds == 4 {
        0 // 1-shanten
    } else {
        (4 - melds - (taatsu.min(4 - melds))) + if has_pair { 0 } else { 1 }
    }
}

/// Calculate seven pairs shanten
fn calculate_seven_pairs_shanten(tiles: &[u8; 34]) -> i32 {
    let mut pairs = 0;

    for &count in tiles {
        if count >= 2 {
            pairs += 1;
        }
    }

    6 - pairs
}

/// Calculate thirteen orphans shanten
fn calculate_thirteen_orphans_shanten(tiles: &[u8; 34]) -> i32 {
    let terminals_and_honors = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
    let mut unique_count = 0;
    let mut has_pair = false;

    for &idx in &terminals_and_honors {
        if tiles[idx] > 0 {
            unique_count += 1;
            if tiles[idx] >= 2 {
                has_pair = true;
            }
        }
    }

    13 - unique_count - if has_pair { 1 } else { 0 }
}

/// Calculate effective tiles (tiles that reduce shanten when drawn)
pub fn calculate_effective_tiles(hand_tiles: &[u32]) -> u32 {
    let current_shanten = calculate_shanten(hand_tiles);
    let mut effective_count = 0;

    for tile_type in 0..34 {
        let count_in_hand = hand_tiles.iter().filter(|&&t| (t / 4) == tile_type).count();
        if count_in_hand >= 4 {
            continue;
        }

        let mut new_hand = hand_tiles.to_vec();
        new_hand.push(tile_type * 4);
        let new_shanten = calculate_shanten(&new_hand);

        if new_shanten < current_shanten {
            effective_count += 1;
        }
    }

    effective_count
}

/// Calculate best ukeire (number of tiles that improve hand)
pub fn calculate_best_ukeire(hand_tiles: &[u32], visible_tiles: &[u32]) -> u32 {
    let mut max_ukeire = 0;
    let mut visible_counts = [0u32; 34];

    for &tile in visible_tiles {
        let tile_type = (tile / 4) as usize;
        if tile_type < 34 {
            visible_counts[tile_type] += 1;
        }
    }

    let current_shanten = calculate_shanten(hand_tiles);

    for (idx, _) in hand_tiles.iter().enumerate() {
        let new_hand: Vec<u32> = hand_tiles
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != idx)
            .map(|(_, &t)| t)
            .collect();

        let new_shanten = calculate_shanten(&new_hand);
        if new_shanten > current_shanten {
            continue;
        }

        let mut ukeire = 0;
        for tile_type in 0..34 {
            let mut test_hand = new_hand.clone();
            test_hand.push(tile_type * 4);
            let test_shanten = calculate_shanten(&test_hand);

            if test_shanten < new_shanten {
                ukeire += 4 - visible_counts[tile_type as usize];
            }
        }

        max_ukeire = max_ukeire.max(ukeire);
    }

    max_ukeire
}
