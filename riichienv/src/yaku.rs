use crate::types::{Hand, TILE_MAX, Meld};
use crate::agari::{self, Mentsu, Division};

#[derive(Debug, Clone, Default)]
pub struct YakuResult {
    pub han: u8,
    pub fu: u8,
    pub yaku_ids: Vec<u32>,
    pub yaku_names: Vec<String>,
}

pub struct YakuContext {
    pub is_menzen: bool,
    pub is_reach: bool,
    pub is_ippatsu: bool,
    pub is_tsumo: bool,
    pub is_haitei: bool,
    pub is_houtei: bool,
    pub is_rinshan: bool,
    pub is_chankan: bool,
    pub is_daburu_reach: bool,
    pub dora_count: u8,
    pub aka_dora: u8,
    pub bakaze: u8, // 27=East, 28=South, etc.
    pub jikaze: u8,
}

impl Default for YakuContext {
    fn default() -> Self {
        Self {
            is_menzen: true,
            is_reach: false,
            is_ippatsu: false,
            is_tsumo: false,
            is_haitei: false,
            is_houtei: false,
            is_rinshan: false,
            is_chankan: false,
            is_daburu_reach: false,
            dora_count: 0,
            aka_dora: 0,
            bakaze: 27,
            jikaze: 27,
        }
    }
}

pub fn calculate_yaku(hand: &Hand, melds: &[Meld], ctx: &YakuContext, win_tile: u8) -> YakuResult {
    let divisions = agari::find_divisions(hand);
    let mut best_res = YakuResult::default();

    if divisions.is_empty() {
        if agari::is_kokushi(hand) {
            best_res.han = 13;
            best_res.yaku_ids.push(42);
            best_res.yaku_names.push("Kokushi Musou".to_string());
            return best_res;
        }
        if agari::is_chiitoitsu(hand) {
            best_res.han = 2;
            best_res.fu = 25;
            best_res.yaku_ids.push(25);
            best_res.yaku_names.push("Chiitoitsu".to_string());
            
            if is_tanyao(hand, melds) { best_res.han += 1; best_res.yaku_ids.push(12); best_res.yaku_names.push("Tanyao".to_string()); }
            if is_chinitsu(hand, melds) { best_res.han += 6; best_res.yaku_ids.push(29); best_res.yaku_names.push("Chinitsu".to_string()); }
            else if is_honitsu(hand, melds) { best_res.han += 3; best_res.yaku_ids.push(27); best_res.yaku_names.push("Honitsu".to_string()); }
            if is_honroutou(hand, melds) { best_res.han += 2; best_res.yaku_ids.push(24); best_res.yaku_names.push("Honroutou".to_string()); }

            apply_static_yaku(&mut best_res, ctx);
            return best_res;
        }
        return best_res;
    }

    for div in divisions {
        let mut win_group_indices = Vec::new();
        if div.head == win_tile {
            win_group_indices.push(None);
        }
        for (idx, m) in div.body.iter().enumerate() {
            match m {
                Mentsu::Koutsu(t) => if *t == win_tile { win_group_indices.push(Some(idx)); }
                Mentsu::Shuntsu(t) => if win_tile >= *t && win_tile <= *t + 2 { win_group_indices.push(Some(idx)); }
            }
        }

        if win_group_indices.is_empty() { continue; }

        for wg_idx in win_group_indices {
            let mut res = YakuResult::default();
            
            // Basic Yaku
            if ctx.is_reach && !ctx.is_daburu_reach { res.han += 1; res.yaku_ids.push(2); res.yaku_names.push("Riichi".to_string()); }
            if ctx.is_daburu_reach { res.han += 2; res.yaku_ids.push(18); res.yaku_names.push("Double Riichi".to_string()); }
            if ctx.is_ippatsu { res.han += 1; res.yaku_ids.push(30); res.yaku_names.push("Ippatsu".to_string()); }
            if ctx.is_menzen && ctx.is_tsumo { res.han += 1; res.yaku_ids.push(1); res.yaku_names.push("Menzen Tsumo".to_string()); }
            if is_tanyao(hand, melds) { res.han += 1; res.yaku_ids.push(12); res.yaku_names.push("Tanyao".to_string()); }

            // Pinfu check
            if check_pinfu(&div, melds, ctx, wg_idx, win_tile) {
                res.han += 1;
                res.yaku_ids.push(14);
                res.yaku_names.push("Pinfu".to_string());
                res.fu = if ctx.is_tsumo { 20 } else { 30 };
            } else {
                res.fu = calculate_fu_with_waiting(&div, melds, ctx, wg_idx, win_tile);
            }

            // Yakuhai
            let yakuhai_tiles = [ctx.bakaze, ctx.jikaze, 31, 32, 33];
            for &t in &yakuhai_tiles {
                let count = div.body.iter().filter(|m| match m { Mentsu::Koutsu(tile) => *tile == t, _ => false }).count()
                          + melds.iter().filter(|m| m.tiles[0] == t && m.meld_type != crate::types::MeldType::Chi).count();
                if count > 0 {
                    res.han += count as u8;
                    let id = match t {
                        31 => 7, 32 => 8, 33 => 9,
                        _ => if t == ctx.bakaze { 11 } else { 10 },
                    };
                    for _ in 0..count { res.yaku_ids.push(id); res.yaku_names.push("Yakuhai".to_string()); }
                }
            }

            // Dragons check (Daisangen / Shousangen)
            let haku_koutsu = div.body.iter().any(|m| match m { Mentsu::Koutsu(t) => *t == 31, _ => false }) || melds.iter().any(|m| m.tiles[0] == 31 && m.meld_type != crate::types::MeldType::Chi);
            let hatsu_koutsu = div.body.iter().any(|m| match m { Mentsu::Koutsu(t) => *t == 32, _ => false }) || melds.iter().any(|m| m.tiles[0] == 32 && m.meld_type != crate::types::MeldType::Chi);
            let chun_koutsu = div.body.iter().any(|m| match m { Mentsu::Koutsu(t) => *t == 33, _ => false }) || melds.iter().any(|m| m.tiles[0] == 33 && m.meld_type != crate::types::MeldType::Chi);
            
            if haku_koutsu && hatsu_koutsu && chun_koutsu {
                res.han = 13;
                res.yaku_ids.clear();
                res.yaku_ids.push(37);
                res.yaku_names.clear();
                res.yaku_names.push("Daisangen".to_string());
                if res.han > best_res.han { best_res = res; }
                continue;
            } else {
                let dragon_koutsu_count = (if haku_koutsu {1} else {0}) + (if hatsu_koutsu {1} else {0}) + (if chun_koutsu {1} else {0});
                let dragon_pair_count = (if div.head == 31 {1} else {0}) + (if div.head == 32 {1} else {0}) + (if div.head == 33 {1} else {0});
                if dragon_koutsu_count == 2 && dragon_pair_count == 1 {
                    res.han += 2;
                    res.yaku_ids.push(23); // Shousangen
                    res.yaku_names.push("Shousangen".to_string());
                }
            }

            // Winds (Daisushii / Shousushii)
            let mut wind_koutsu_count = 0;
            let mut wind_pair_count = 0;
            for w in 27..=30 {
                let has_koutsu = div.body.iter().any(|m| match m { Mentsu::Koutsu(t) => *t == w, _ => false }) || melds.iter().any(|m| m.tiles[0] == w && m.meld_type != crate::types::MeldType::Chi);
                if has_koutsu {
                    wind_koutsu_count += 1;
                } else if div.head == w {
                    wind_pair_count += 1;
                }
            }
            if wind_koutsu_count == 4 {
                res.han = 26; // Double Yakuman? Or 13? Usually 26 in MJSoul for Daisushii
                res.yaku_ids.clear();
                res.yaku_ids.push(50);
                res.yaku_names.clear();
                res.yaku_names.push("Daisushii".to_string());
                if res.han > best_res.han { best_res = res; }
                continue;
            } else if wind_koutsu_count == 3 && wind_pair_count == 1 {
                res.han = 13;
                res.yaku_ids.clear();
                res.yaku_ids.push(43);
                res.yaku_names.clear();
                res.yaku_names.push("Shousushii".to_string());
                if res.han > best_res.han { best_res = res; }
                continue;
            }

            // Toitoi
            let koutsu_total = div.body.iter().filter(|m| match m { Mentsu::Koutsu(_) => true, _ => false }).count()
                             + melds.iter().filter(|m| m.meld_type != crate::types::MeldType::Chi).count();
            if koutsu_total == 4 { res.han += 2; res.yaku_ids.push(21); res.yaku_names.push("Toitoi".to_string()); }

            // San Ankou / Su Ankou
            let mut closed_koutsu_count = 0;
            for (idx, m) in div.body.iter().enumerate() {
                if let Mentsu::Koutsu(_) = m {
                    // Ron on closed triplet makes it open
                    if !ctx.is_tsumo && Some(idx) == wg_idx { continue; }
                    closed_koutsu_count += 1;
                }
            }
            for m in melds {
                if m.meld_type == crate::types::MeldType::Angang {
                    closed_koutsu_count += 1;
                }
            }
            if closed_koutsu_count == 4 {
                res.han = 13;
                res.yaku_ids.clear();
                res.yaku_ids.push(38);
                res.yaku_names.clear();
                res.yaku_names.push("Su Ankou".to_string());
                if res.han > best_res.han { best_res = res; }
                continue;
            } else if closed_koutsu_count == 3 {
                res.han += 2;
                res.yaku_ids.push(22);
                res.yaku_names.push("San Ankou".to_string());
            }

            // Iipeiko / Ryanpeikou (Closed only)
            if ctx.is_menzen {
                let mut shuntsu_tiles = Vec::new();
                for m in &div.body {
                    if let Mentsu::Shuntsu(t) = m { shuntsu_tiles.push(*t); }
                }
                shuntsu_tiles.sort();
                let mut identical_pairs = 0;
                let mut i = 0;
                while i + 1 < shuntsu_tiles.len() {
                    if shuntsu_tiles[i] == shuntsu_tiles[i+1] {
                        identical_pairs += 1;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                if identical_pairs == 2 {
                    res.han += 3;
                    res.yaku_ids.push(28);
                    res.yaku_names.push("Ryanpeikou".to_string());
                } else if identical_pairs == 1 {
                    res.han += 1;
                    res.yaku_ids.push(13);
                    res.yaku_names.push("Iipeiko".to_string());
                }
            }

            // Ittsu / Sanshoku Doujun
            if check_ittsu(&div, melds) {
                res.han += if ctx.is_menzen { 2 } else { 1 };
                res.yaku_ids.push(16);
                res.yaku_names.push("Iittsu".to_string());
            }
            if is_sanshoku_doujun(&div, melds) {
                res.han += if ctx.is_menzen { 2 } else { 1 };
                res.yaku_ids.push(17);
                res.yaku_names.push("Sanshoku Doujun".to_string());
            }
            if is_sanshoku_doukou(&div, melds) {
                res.han += 2;
                res.yaku_ids.push(19);
                res.yaku_names.push("Sanshoku Doukou".to_string());
            }

            // Honitsu / Chinitsu
            if is_chinitsu(hand, melds) {
                res.han += if ctx.is_menzen { 6 } else { 5 };
                res.yaku_ids.push(29);
                res.yaku_names.push("Chinitsu".to_string());
            } else if is_honitsu(hand, melds) {
                res.han += if ctx.is_menzen { 3 } else { 2 };
                res.yaku_ids.push(27);
                res.yaku_names.push("Honitsu".to_string());
            }

            // Chantai / Junchan / Honroutou
            if is_honroutou(hand, melds) {
                res.han += 2;
                res.yaku_ids.push(24);
                res.yaku_names.push("Honroutou".to_string());
            } else if is_junchan(&div, melds) {
                res.han += if ctx.is_menzen { 3 } else { 2 };
                res.yaku_ids.push(26);
                res.yaku_names.push("Junchan".to_string());
            } else if is_chantai(&div, melds) {
                res.han += if ctx.is_menzen { 2 } else { 1 };
                res.yaku_ids.push(15);
                res.yaku_names.push("Chantai".to_string());
            }

            // Extra
            if ctx.is_haitei { res.han += 1; res.yaku_ids.push(5); res.yaku_names.push("Haitei".to_string()); }
            if ctx.is_houtei { res.han += 1; res.yaku_ids.push(6); res.yaku_names.push("Houtei".to_string()); }
            if ctx.is_rinshan { res.han += 1; res.yaku_ids.push(4); res.yaku_names.push("Rinshan".to_string()); }
            if ctx.is_chankan { res.han += 1; res.yaku_ids.push(3); res.yaku_names.push("Chankan".to_string()); }

            // Dora
            if ctx.dora_count > 0 {
                res.han += ctx.dora_count;
                res.yaku_ids.push(31);
                res.yaku_names.push(format!("Dora"));
            }
            if ctx.aka_dora > 0 {
                res.han += ctx.aka_dora;
                res.yaku_ids.push(32);
                res.yaku_names.push(format!("Aka Dora"));
            }

            if res.han > best_res.han || (res.han == best_res.han && res.fu > best_res.fu) {
                best_res = res;
            }
        }
    }

    best_res
}

fn calculate_fu_with_waiting(div: &Division, melds: &[Meld], ctx: &YakuContext, wg_idx: Option<usize>, win_tile: u8) -> u8 {
    let mut fu = 20;
    if ctx.is_tsumo { fu += 2; } else if ctx.is_menzen { fu += 10; }

    if div.head == ctx.bakaze { fu += 2; }
    if div.head == ctx.jikaze { fu += 2; }
    if div.head >= 31 { fu += 2; }

    // Waiting fu
    match wg_idx {
        None => fu += 2, // Tanki
        Some(idx) => {
            match div.body[idx] {
                Mentsu::Koutsu(_) => {
                    // Ron on closed koutsu is 2 fu, but handled in triplet processing below
                }
                Mentsu::Shuntsu(t) => {
                    // Kanchan
                    if win_tile == t + 1 { fu += 2; }
                    // Penchan
                    else if (win_tile == t + 2 && (t % 9 == 0)) || (win_tile == t && (t % 9 == 6)) {
                        fu += 2;
                    }
                }
            }
        }
    }

    for (idx, m) in div.body.iter().enumerate() {
        match m {
            Mentsu::Koutsu(t) => {
                let mut f = 4; // Closed 
                if !ctx.is_tsumo && Some(idx) == wg_idx { f = 2; } // Ron on closed triplet is open
                if is_terminal(*t) { f *= 2; }
                fu += f;
            }
            _ => {}
        }
    }
    for m in melds {
        if m.tiles.len() >= 3 && m.tiles[0] == m.tiles[1] {
            let mut f = 2; // Open
            if !m.opened { f = 4; } // Angang
            if is_terminal(m.tiles[0]) { f *= 2; }
            if m.meld_type == crate::types::MeldType::Gang || m.meld_type == crate::types::MeldType::Angang || m.meld_type == crate::types::MeldType::Addgang {
                f *= 4;
            }
            fu += f;
        }
    }

    if fu == 20 && !ctx.is_tsumo { fu = 30; } // Open Ron minimum 30 fu

    ((fu + 9) / 10) * 10
}

fn check_pinfu(div: &Division, melds: &[Meld], ctx: &YakuContext, wg_idx: Option<usize>, win_tile: u8) -> bool {
    // Pinfu requires Menzen, all sequences, non-yakuhai head
    if !ctx.is_menzen { return false; }
    if melds.len() > 0 { return false; } // No melds (including closed Kan)
    for m in &div.body {
        if let Mentsu::Koutsu(_) = m { return false; }
    }
    if is_yakuhai_tile(div.head, ctx) { return false; }

    // Waiting must be ryanmen
    match wg_idx {
        Some(idx) => {
            if let Mentsu::Shuntsu(t) = div.body[idx] {
                // Ryanmen: win_tile is t or t+2, and it's not a penchan wait
                if win_tile == t {
                    if t % 9 == 6 { return false; } // 78(9) is penchan
                    return true;
                }
                if win_tile == t + 2 {
                    if t % 9 == 0 { return false; } // (1)23 is penchan
                    return true;
                }
            }
        }
        _ => {}
    }
    false
}

fn is_yakuhai_tile(tile: u8, ctx: &YakuContext) -> bool {
    tile >= 31 || tile == ctx.bakaze || tile == ctx.jikaze
}

fn is_honroutou(hand: &Hand, melds: &[Meld]) -> bool {
    for (i, &count) in hand.counts.iter().enumerate() {
        if count > 0 && !is_terminal(i as u8) { return false; }
    }
    if melds.iter().any(|m| m.tiles.iter().any(|&t| !is_terminal(t))) { return false; }
    true
}

fn is_junchan(div: &Division, melds: &[Meld]) -> bool {
    if !is_number_terminal(div.head) { return false; }
    for m in &div.body {
        match m {
            Mentsu::Koutsu(t) => if !is_number_terminal(*t) { return false; },
            Mentsu::Shuntsu(t) => if !is_number_terminal(*t) && !is_number_terminal(t+2) { return false; },
        }
    }
    for m in melds {
        if m.tiles.iter().all(|&t| !is_number_terminal(t)) { return false; }
    }
    true
}

fn is_chantai(div: &Division, melds: &[Meld]) -> bool {
    if !is_terminal(div.head) { return false; }
    let mut has_honor = is_honor(div.head);
    for m in &div.body {
        match m {
            Mentsu::Koutsu(t) => { if !is_terminal(*t) { return false; } if is_honor(*t) { has_honor = true; } },
            Mentsu::Shuntsu(t) => if !is_terminal(*t) && !is_terminal(t+2) { return false; },
        }
    }
    for m in melds {
        if m.tiles.iter().all(|&t| !is_terminal(t)) { return false; }
        if m.tiles.iter().any(|&t| is_honor(t)) { has_honor = true; }
    }
    has_honor
}

fn is_terminal(tile: u8) -> bool { tile < 27 && (tile % 9 == 0 || tile % 9 == 8) || tile >= 27 }
fn is_number_terminal(tile: u8) -> bool { tile < 27 && (tile % 9 == 0 || tile % 9 == 8) }
fn is_honor(tile: u8) -> bool { tile >= 27 }

fn is_honitsu(hand: &Hand, melds: &[Meld]) -> bool {
    let mut suits = [false; 3];
    let mut has_honor = false;
    for (i, &count) in hand.counts.iter().enumerate() {
        if count > 0 {
            if i < 9 { suits[0] = true; }
            else if i < 18 { suits[1] = true; }
            else if i < 27 { suits[2] = true; }
            else { has_honor = true; }
        }
    }
    for meld in melds {
        for &t in &meld.tiles {
            let idx = t as usize;
            if idx < 9 { suits[0] = true; }
            else if idx < 18 { suits[1] = true; }
            else if idx < 27 { suits[2] = true; }
            else { has_honor = true; }
        }
    }
    suits.iter().filter(|&&b| b).count() == 1 && has_honor
}

fn is_chinitsu(hand: &Hand, melds: &[Meld]) -> bool {
    let mut suits = [false; 3];
    for (i, &count) in hand.counts.iter().enumerate() {
        if count > 0 {
            if i >= 27 { return false; }
            if i < 9 { suits[0] = true; }
            else if i < 18 { suits[1] = true; }
            else if i < 27 { suits[2] = true; }
        }
    }
    for meld in melds {
        for &t in &meld.tiles {
            let idx = t as usize;
            if idx >= 27 { return false; }
            if idx < 9 { suits[0] = true; }
            else if idx < 18 { suits[1] = true; }
            else if idx < 27 { suits[2] = true; }
        }
    }
    suits.iter().filter(|&&b| b).count() == 1
}

fn apply_static_yaku(res: &mut YakuResult, ctx: &YakuContext) {
    if ctx.is_reach && !ctx.is_daburu_reach { res.han += 1; res.yaku_ids.push(2); }
    if ctx.is_daburu_reach { res.han += 2; res.yaku_ids.push(18); }
    if ctx.is_ippatsu { res.han += 1; res.yaku_ids.push(30); }
    if ctx.is_menzen && ctx.is_tsumo { res.han += 1; res.yaku_ids.push(1); }
    if ctx.is_haitei { res.han += 1; res.yaku_ids.push(5); }
    if ctx.is_houtei { res.han += 1; res.yaku_ids.push(6); }
    if ctx.is_rinshan { res.han += 1; res.yaku_ids.push(4); }
    if ctx.is_chankan { res.han += 1; res.yaku_ids.push(3); }
    
    if ctx.dora_count > 0 {
        res.han += ctx.dora_count;
        res.yaku_ids.push(31);
    }
    if ctx.aka_dora > 0 {
        res.han += ctx.aka_dora;
        res.yaku_ids.push(32);
    }
}

fn check_ittsu(div: &Division, melds: &[Meld]) -> bool {
    for suit_offset in [0, 9, 18] {
        let mut has_123 = false;
        let mut has_456 = false;
        let mut has_789 = false;
        
        // Check Body
        for m in &div.body {
            if let Mentsu::Shuntsu(t) = m {
                if *t == suit_offset { has_123 = true; }
                else if *t == suit_offset + 3 { has_456 = true; }
                else if *t == suit_offset + 6 { has_789 = true; }
            }
        }
        
        // Check Melds
        for m in melds {
            if m.meld_type == crate::types::MeldType::Chi {
                let t = m.tiles[0];
                if t == suit_offset { has_123 = true; }
                else if t == suit_offset + 3 { has_456 = true; }
                else if t == suit_offset + 6 { has_789 = true; }
            }
        }
        
        if has_123 && has_456 && has_789 { return true; }
    }
    false
}

fn is_sanshoku_doujun(div: &Division, melds: &[Meld]) -> bool {
    for i in 0..7 {
        let mut has_man = false;
        let mut has_pin = false;
        let mut has_sou = false;
        for m in &div.body {
            if let Mentsu::Shuntsu(t) = m {
                if *t == i { has_man = true; }
                if *t == i + 9 { has_pin = true; }
                if *t == i + 18 { has_sou = true; }
            }
        }
        for m in melds {
            if m.meld_type == crate::types::MeldType::Chi {
                let t = m.tiles[0];
                if t == i { has_man = true; }
                if t == i + 9 { has_pin = true; }
                if t == i + 18 { has_sou = true; }
            }
        }
        if has_man && has_pin && has_sou { return true; }
    }
    false
}

fn is_tanyao(hand: &Hand, melds: &[Meld]) -> bool {
    let terminals = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
    for &t in &terminals {
        if hand.counts[t] > 0 { return false; }
    }
    for meld in melds {
        for &t in &meld.tiles {
             if terminals.contains(&(t as usize)) { return false; }
        }
    }
    true
}

fn is_sanshoku_doukou(div: &Division, melds: &[Meld]) -> bool {
    for i in 0..9 {
        let mut has_man = false;
        let mut has_pin = false;
        let mut has_sou = false;
        for m in &div.body {
            if let Mentsu::Koutsu(t) = m {
                if *t == i { has_man = true; }
                if *t == i + 9 { has_pin = true; }
                if *t == i + 18 { has_sou = true; }
            }
        }
        for m in melds {
            if m.meld_type != crate::types::MeldType::Chi {
                let t = m.tiles[0];
                if t == i { has_man = true; }
                if t == i + 9 { has_pin = true; }
                if t == i + 18 { has_sou = true; }
            }
        }
        if has_man && has_pin && has_sou { return true; }
    }
    false
}
