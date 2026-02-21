use crate::types::TILE_MAX;

// Shanten tables (Nyanten / Cryolite algorithm)
// Hash tables: [tile_index][cumulative_count][tile_count]
#[rustfmt::skip]
const SHUPAI_TABLE: [[[u32; 5]; 15]; 9] = [
    [ // i = 0
        [0,0,0,0,0], [0,139150,0,0,0], [0,105150,244300,0,0],
        [0,75750,180900,320050,0], [0,51810,127560,232710,371860],
        [0,33490,85300,161050,266200], [0,0,33490,85300,161050],
        [0,0,0,33490,85300], [0,0,0,0,33490],
        [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0],
        [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0],
    ],
    [ // i = 1
        [0,0,0,0,0], [0,43130,0,0,0], [0,34995,78125,0,0],
        [0,27120,62115,105245,0], [0,19980,47100,82095,125225],
        [0,13925,33905,61025,96020], [0,9130,23055,43035,70155],
        [0,5595,14725,28650,48630], [0,3180,8775,17905,31830],
        [0,1660,4840,10435,19565], [0,0,1660,4840,10435],
        [0,0,0,1660,4840], [0,0,0,0,1660],
        [0,0,0,0,0], [0,0,0,0,0],
    ],
    [ // i = 2
        [0,0,0,0,0], [0,11880,0,0,0], [0,10374,22254,0,0],
        [0,8688,19062,30942,0], [0,6937,15625,25999,37879],
        [0,5251,12188,20876,31250], [0,3745,8996,15933,24621],
        [0,2499,6244,11495,18432], [0,1548,4047,7792,13043],
        [0,882,2430,4929,8674], [0,456,1338,2886,5385],
        [0,210,666,1548,3096], [0,84,294,750,1632],
        [0,28,112,322,778], [0,0,28,112,322],
    ],
    [ // i = 3
        [0,0,0,0,0], [0,2878,0,0,0], [0,2693,5571,0,0],
        [0,2438,5131,8009,0], [0,2118,4556,7249,10127],
        [0,1753,3871,6309,9002], [0,1372,3125,5243,7681],
        [0,1007,2379,4132,6250], [0,687,1694,3066,4819],
        [0,432,1119,2126,3498], [0,247,679,1366,2373],
        [0,126,373,805,1492], [0,56,182,429,861],
        [0,21,77,203,450], [0,6,27,83,209],
    ],
    [ // i = 4
        [0,0,0,0,0], [0,620,0,0,0], [0,610,1230,0,0],
        [0,590,1200,1820,0], [0,555,1145,1755,2375],
        [0,503,1058,1648,2258], [0,435,938,1493,2083],
        [0,355,790,1293,1848], [0,270,625,1060,1563],
        [0,190,460,815,1250], [0,122,312,582,937],
        [0,70,192,382,652], [0,35,105,227,417],
        [0,15,50,120,242], [0,5,20,55,125],
    ],
    [ // i = 5
        [0,0,0,0,0], [0,125,0,0,0], [0,125,250,0,0],
        [0,125,250,375,0], [0,124,249,374,499],
        [0,121,245,370,495], [0,115,236,360,485],
        [0,105,220,341,465], [0,90,195,310,431],
        [0,72,162,267,382], [0,53,125,215,320],
        [0,35,88,160,250], [0,20,55,108,180],
        [0,10,30,65,118], [0,4,14,34,69],
    ],
    [ // i = 6
        [0,0,0,0,0], [0,25,0,0,0], [0,25,50,0,0],
        [0,25,50,75,0], [0,25,50,75,100],
        [0,25,50,75,100], [0,25,50,75,100],
        [0,25,50,75,100], [0,24,49,74,99],
        [0,22,46,71,96], [0,19,41,65,90],
        [0,15,34,56,80], [0,10,25,44,66],
        [0,6,16,31,50], [0,3,9,19,34],
    ],
    [ // i = 7
        [0,0,0,0,0], [0,5,0,0,0], [0,5,10,0,0],
        [0,5,10,15,0], [0,5,10,15,20],
        [0,5,10,15,20], [0,5,10,15,20],
        [0,5,10,15,20], [0,5,10,15,20],
        [0,5,10,15,20], [0,5,10,15,20],
        [0,5,10,15,20], [0,4,9,14,19],
        [0,3,7,12,17], [0,2,5,9,14],
    ],
    [ // i = 8
        [0,0,0,0,0], [0,1,0,0,0], [0,1,2,0,0],
        [0,1,2,3,0], [0,1,2,3,4],
        [0,1,2,3,4], [0,1,2,3,4],
        [0,1,2,3,4], [0,1,2,3,4],
        [0,1,2,3,4], [0,1,2,3,4],
        [0,1,2,3,4], [0,1,2,3,4],
        [0,1,2,3,4], [0,1,2,3,4],
    ],
];

#[rustfmt::skip]
const ZIPAI_TABLE: [[[u32; 5]; 15]; 7] = [
    [ // i = 0
        [0,0,0,0,0], [0,11880,0,0,0], [0,10374,22254,0,0],
        [0,8688,19062,30942,0], [0,6937,15625,25999,37879],
        [0,5251,12188,20876,31250], [0,0,5251,12188,20876],
        [0,0,0,5251,12188], [0,0,0,0,5251],
        [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0],
        [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0],
    ],
    [ // i = 1
        [0,0,0,0,0], [0,2878,0,0,0], [0,2693,5571,0,0],
        [0,2438,5131,8009,0], [0,2118,4556,7249,10127],
        [0,1753,3871,6309,9002], [0,1372,3125,5243,7681],
        [0,1007,2379,4132,6250], [0,687,1694,3066,4819],
        [0,432,1119,2126,3498], [0,0,432,1119,2126],
        [0,0,0,432,1119], [0,0,0,0,432],
        [0,0,0,0,0], [0,0,0,0,0],
    ],
    [ // i = 2
        [0,0,0,0,0], [0,620,0,0,0], [0,610,1230,0,0],
        [0,590,1200,1820,0], [0,555,1145,1755,2375],
        [0,503,1058,1648,2258], [0,435,938,1493,2083],
        [0,355,790,1293,1848], [0,270,625,1060,1563],
        [0,190,460,815,1250], [0,122,312,582,937],
        [0,70,192,382,652], [0,35,105,227,417],
        [0,15,50,120,242], [0,0,15,50,120],
    ],
    [ // i = 3
        [0,0,0,0,0], [0,125,0,0,0], [0,125,250,0,0],
        [0,125,250,375,0], [0,124,249,374,499],
        [0,121,245,370,495], [0,115,236,360,485],
        [0,105,220,341,465], [0,90,195,310,431],
        [0,72,162,267,382], [0,53,125,215,320],
        [0,35,88,160,250], [0,20,55,108,180],
        [0,10,30,65,118], [0,4,14,34,69],
    ],
    [ // i = 4
        [0,0,0,0,0], [0,25,0,0,0], [0,25,50,0,0],
        [0,25,50,75,0], [0,25,50,75,100],
        [0,25,50,75,100], [0,25,50,75,100],
        [0,25,50,75,100], [0,24,49,74,99],
        [0,22,46,71,96], [0,19,41,65,90],
        [0,15,34,56,80], [0,10,25,44,66],
        [0,6,16,31,50], [0,3,9,19,34],
    ],
    [ // i = 5
        [0,0,0,0,0], [0,5,0,0,0], [0,5,10,0,0],
        [0,5,10,15,0], [0,5,10,15,20],
        [0,5,10,15,20], [0,5,10,15,20],
        [0,5,10,15,20], [0,5,10,15,20],
        [0,5,10,15,20], [0,5,10,15,20],
        [0,5,10,15,20], [0,4,9,14,19],
        [0,3,7,12,17], [0,2,5,9,14],
    ],
    [ // i = 6
        [0,0,0,0,0], [0,1,0,0,0], [0,1,2,0,0],
        [0,1,2,3,0], [0,1,2,3,4],
        [0,1,2,3,4], [0,1,2,3,4],
        [0,1,2,3,4], [0,1,2,3,4],
        [0,1,2,3,4], [0,1,2,3,4],
        [0,1,2,3,4], [0,1,2,3,4],
        [0,1,2,3,4], [0,1,2,3,4],
    ],
];

// Hierarchical key lookup tables (generated by scripts/convert_nyanten_tables.py)
static SHUPAI_KEYS: &[u8; 405_350] = include_bytes!("data/nyanten_shupai_keys.bin");
static ZIPAI_KEYS: &[u8; 43_130] = include_bytes!("data/nyanten_zipai_keys.bin");
static KEYS1: &[u8; 15_876] = include_bytes!("data/nyanten_keys1.bin");
static KEYS2: &[u8; 22_680] = include_bytes!("data/nyanten_keys2.bin");
static KEYS3: &[u8; 49_500] = include_bytes!("data/nyanten_keys3.bin");

#[inline]
fn hash_shupai(tiles: &[u8]) -> usize {
    let mut n: usize = 0;
    let mut h: usize = 0;
    for (i, &c) in tiles.iter().enumerate() {
        let c = c as usize;
        n += c;
        h += SHUPAI_TABLE[i][n][c] as usize;
    }
    h
}

#[inline]
fn hash_zipai(tiles: &[u8]) -> usize {
    let mut n: usize = 0;
    let mut h: usize = 0;
    for (i, &c) in tiles.iter().enumerate() {
        let c = c as usize;
        n += c;
        h += ZIPAI_TABLE[i][n][c] as usize;
    }
    h
}

fn calc_normal(tiles: &[u8; TILE_MAX], len_div3: u8) -> i8 {
    let m = len_div3 as usize;
    let k0_m = SHUPAI_KEYS[hash_shupai(&tiles[0..9])] as usize;
    let k0_p = SHUPAI_KEYS[hash_shupai(&tiles[9..18])] as usize;
    let k1 = KEYS1[k0_m * 126 + k0_p] as usize;
    let k0_s = SHUPAI_KEYS[hash_shupai(&tiles[18..27])] as usize;
    let k2 = KEYS2[k1 * 126 + k0_s] as usize;
    let k0_z = ZIPAI_KEYS[hash_zipai(&tiles[27..34])] as usize;
    let replacement = KEYS3[(k2 * 55 + k0_z) * 5 + m];
    (replacement as i8) - 1
}

fn calc_chitoi(tiles: &[u8; TILE_MAX]) -> i8 {
    let mut pairs = 0u8;
    let mut kinds = 0u8;
    for &c in tiles.iter() {
        if c > 0 {
            kinds += 1;
            if c >= 2 {
                pairs += 1;
            }
        }
    }
    let redunct = 7u8.saturating_sub(kinds) as i8;
    7 - pairs as i8 + redunct - 1
}

fn calc_kokushi(tiles: &[u8; TILE_MAX]) -> i8 {
    const TERMINALS: [usize; 13] = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
    let mut kinds = 0i8;
    let mut has_pair = false;
    for &idx in &TERMINALS {
        if tiles[idx] > 0 {
            kinds += 1;
            if tiles[idx] >= 2 {
                has_pair = true;
            }
        }
    }
    14 - kinds - has_pair as i8 - 1
}

pub fn calc_shanten_from_counts(tehai: &[u8; TILE_MAX], tehai_len_div3: u8) -> i8 {
    let mut shanten = calc_normal(tehai, tehai_len_div3);
    if shanten <= 0 || tehai_len_div3 < 4 {
        return shanten;
    }
    shanten = shanten.min(calc_chitoi(tehai));
    if shanten > 0 {
        shanten.min(calc_kokushi(tehai))
    } else {
        shanten
    }
}

/// Calculate shanten from 136-tile ID hand.
#[cfg(feature = "python")]
pub fn calculate_shanten(hand_tiles: &[u32]) -> i32 {
    let mut tile_counts = [0u8; TILE_MAX];
    for &tile in hand_tiles {
        let tile_type = (tile / 4) as usize;
        if tile_type < TILE_MAX {
            tile_counts[tile_type] += 1;
        }
    }
    let num_tiles: u8 = tile_counts.iter().sum();
    let len_div3 = num_tiles / 3;
    calc_shanten_from_counts(&tile_counts, len_div3) as i32
}

/// Calculate effective tiles (tiles that reduce shanten when drawn)
#[cfg(feature = "python")]
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
#[cfg(feature = "python")]
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
