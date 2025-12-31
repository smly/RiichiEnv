use crate::agari::{self, Division, Mentsu};
use crate::types::{Hand, Meld};

// Yaku IDs based on yans.yml
pub const ID_TSUMO: u32 = 1;
pub const ID_RIICHI: u32 = 2;
pub const ID_CHANKAN: u32 = 3;
pub const ID_RINSHAN: u32 = 4;
pub const ID_HAITEI: u32 = 5;
pub const ID_HOUTEI: u32 = 6;
pub const ID_HAKU: u32 = 7;
pub const ID_HATSU: u32 = 8;
pub const ID_CHUN: u32 = 9;
pub const ID_JIKAZE: u32 = 10;
pub const ID_BAKAZE: u32 = 11;
pub const ID_TANYAO: u32 = 12;
pub const ID_IPEIKO: u32 = 13;
pub const ID_PINFU: u32 = 14;
pub const ID_CHANTA: u32 = 15;
pub const ID_ITTSU: u32 = 16;
pub const ID_SANSHOKU: u32 = 17;
pub const ID_DOUBLE_RIICHI: u32 = 18;
pub const ID_SANSHOKU_DOKO: u32 = 19;
pub const ID_SANKANTSU: u32 = 20;
pub const ID_TOITOI: u32 = 21;
pub const ID_SANANKOU: u32 = 22;
pub const ID_SHOSANGEN: u32 = 23;
pub const ID_HONROUTO: u32 = 24;
pub const ID_CHITOITSU: u32 = 25;
pub const ID_JUNCHAN: u32 = 26;
pub const ID_HONITSU: u32 = 27;
pub const ID_RYANPEIKO: u32 = 28;
pub const ID_CHINITSU: u32 = 29;
pub const ID_IPPATSU: u32 = 30;
pub const ID_DORA: u32 = 31;
pub const ID_AKADORA: u32 = 32;
pub const ID_URADORA: u32 = 33;
#[allow(dead_code)]
pub const ID_NUKIDORA: u32 = 34;
pub const ID_TENHO: u32 = 35;
pub const ID_CHIHO: u32 = 36;
pub const ID_DAISANGEN: u32 = 37;
pub const ID_SUANKO: u32 = 38;
pub const ID_TSUISO: u32 = 39;
pub const ID_RYUISOU: u32 = 40;
pub const ID_CHINROUTO: u32 = 41;
pub const ID_KOKUSHI: u32 = 42;
pub const ID_SHOUSUUSHI: u32 = 43;
pub const ID_SUKANTSU: u32 = 44;
pub const ID_CHUUREN: u32 = 45;
pub const ID_JUNSEI_CHUUREN: u32 = 47;
pub const ID_SUANKO_TANKI: u32 = 48;
pub const ID_KOKUSHI_13: u32 = 49;
pub const ID_DAISUUSHI: u32 = 50;

#[derive(Debug, Clone, Default)]
pub struct YakuResult {
    pub han: u8,
    pub fu: u8,
    pub yaku_ids: Vec<u32>,
    pub yaku_names: Vec<String>,
    pub yakuman_count: u8,
}

#[derive(Debug)]
pub struct YakuContext {
    pub is_menzen: bool,
    pub is_reach: bool,
    pub is_ippatsu: bool,
    pub is_tsumo: bool,
    pub is_haitei: bool,
    pub is_houtei: bool,
    pub is_rinshan: bool,
    pub is_chankan: bool,
    pub is_tsumo_first_turn: bool,
    pub is_daburu_reach: bool,
    pub dora_count: u8,
    pub aka_dora: u8,
    pub ura_dora_count: u8,
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
            is_tsumo_first_turn: false,
            is_daburu_reach: false,
            dora_count: 0,
            aka_dora: 0,
            ura_dora_count: 0,
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
            let is_13_wait = hand.counts[win_tile as usize] == 2;
            if is_13_wait {
                best_res.han = 26;
                best_res.yakuman_count = 2;
                best_res.yaku_ids.push(ID_KOKUSHI_13);
                best_res
                    .yaku_names
                    .push("Kokushi Musou 13-wait".to_string());
            } else {
                best_res.han = 13;
                best_res.yakuman_count = 1;
                best_res.yaku_ids.push(ID_KOKUSHI);
                best_res.yaku_names.push("Kokushi Musou".to_string());
            }
            return best_res;
        }
        if agari::is_chiitoitsu(hand) {
            best_res.han = 2;
            best_res.fu = 25;
            best_res.yaku_ids.push(ID_CHITOITSU);
            best_res.yaku_names.push("Chiitoitsu".to_string());

            if is_tanyao(hand, melds) {
                best_res.han += 1;
                best_res.yaku_ids.push(12);
                best_res.yaku_names.push("Tanyao".to_string());
            }
            if is_chinitsu(hand, melds) {
                best_res.han += 6;
                best_res.yaku_ids.push(29);
                best_res.yaku_names.push("Chinitsu".to_string());
            } else if is_honitsu(hand, melds) {
                best_res.han += 3;
                best_res.yaku_ids.push(27);
                best_res.yaku_names.push("Honitsu".to_string());
            }
            if is_honroutou(hand, melds) {
                best_res.han += 2;
                best_res.yaku_ids.push(24);
                best_res.yaku_names.push("Honroutou".to_string());
            }

            apply_yakuman(
                &mut best_res,
                hand,
                melds,
                ctx,
                &Division {
                    head: 0,
                    body: Vec::new(),
                },
                None,
                win_tile,
            ); // Simplified call
            apply_static_yaku(&mut best_res, ctx);
            return best_res;
        }
        return best_res;
    }

    for div in &divisions {
        let mut win_group_indices = Vec::new();
        if div.head == win_tile {
            win_group_indices.push(None);
        }
        for (idx, m) in div.body.iter().enumerate() {
            match m {
                Mentsu::Koutsu(t) => {
                    if *t == win_tile {
                        win_group_indices.push(Some(idx));
                    }
                }
                Mentsu::Shuntsu(t) => {
                    if win_tile >= *t && win_tile <= *t + 2 {
                        win_group_indices.push(Some(idx));
                    }
                }
            }
        }

        if win_group_indices.is_empty() {
            continue;
        }

        for wg_idx in win_group_indices {
            let mut res = YakuResult::default();

            apply_yakuman(&mut res, hand, melds, ctx, div, wg_idx, win_tile);
            if res.han >= 13 {
                if res.han > best_res.han {
                    best_res = res;
                }
                continue;
            }

            // Static Yaku and Dora are handled by apply_static_yaku
            apply_static_yaku(&mut res, ctx);

            // Tanyao
            // Tanyao
            if is_tanyao(hand, melds) {
                res.han += 1;
                res.yaku_ids.push(ID_TANYAO); // Corrected to 12
                res.yaku_names.push("Tanyao".to_string());
            }

            // Pinfu check
            if check_pinfu(div, melds, ctx, wg_idx, win_tile) {
                res.han += 1;
                res.yaku_ids.push(ID_PINFU);
                res.yaku_names.push("Pinfu".to_string());
                res.fu = if ctx.is_tsumo { 20 } else { 30 };
            } else {
                res.fu = calculate_fu_with_waiting(div, melds, ctx, wg_idx, win_tile);
            }

            // Yakuhai
            // IDs: 7: Haku, 8: Hatsu, 9: Chun, 10: Jikaze, 11: Bakaze
            let yakuhai_tiles = [31, 32, 33, ctx.bakaze, ctx.jikaze];
            for (i, &t) in yakuhai_tiles.iter().enumerate() {
                let count = div
                    .body
                    .iter()
                    .filter(|m| matches!(m, Mentsu::Koutsu(tile) if *tile == t))
                    .count()
                    + melds
                        .iter()
                        .filter(|m| m.tiles[0] == t && m.meld_type != crate::types::MeldType::Chi)
                        .count();
                if count > 0 {
                    res.han += count as u8;
                    let id = match t {
                        31 => ID_HAKU,
                        32 => ID_HATSU,
                        33 => ID_CHUN,
                        _ => {
                            if i == 3 {
                                ID_BAKAZE
                            }
                            // Bakaze iteration
                            else {
                                ID_JIKAZE
                            } // Jikaze iteration
                        }
                    };
                    res.yaku_ids.push(id);
                    res.yaku_names.push("Yakuhai".to_string());
                }
            }

            // Dragons check (Daisangen / Shousangen)
            let haku_koutsu = div.body.iter().any(|m| match m {
                Mentsu::Koutsu(t) => *t == 31,
                _ => false,
            }) || melds
                .iter()
                .any(|m| m.tiles[0] == 31 && m.meld_type != crate::types::MeldType::Chi);
            let hatsu_koutsu = div.body.iter().any(|m| match m {
                Mentsu::Koutsu(t) => *t == 32,
                _ => false,
            }) || melds
                .iter()
                .any(|m| m.tiles[0] == 32 && m.meld_type != crate::types::MeldType::Chi);
            let chun_koutsu = div.body.iter().any(|m| match m {
                Mentsu::Koutsu(t) => *t == 33,
                _ => false,
            }) || melds
                .iter()
                .any(|m| m.tiles[0] == 33 && m.meld_type != crate::types::MeldType::Chi);

            if haku_koutsu && hatsu_koutsu && chun_koutsu {
                // Already handled in apply_yakuman (Daisangen)
            } else {
                let dragon_koutsu_count = (if haku_koutsu { 1 } else { 0 })
                    + (if hatsu_koutsu { 1 } else { 0 })
                    + (if chun_koutsu { 1 } else { 0 });
                let dragon_pair_count = (if div.head == 31 { 1 } else { 0 })
                    + (if div.head == 32 { 1 } else { 0 })
                    + (if div.head == 33 { 1 } else { 0 });
                if dragon_koutsu_count == 2 && dragon_pair_count == 1 {
                    res.han += 2;
                    res.yaku_ids.push(ID_SHOSANGEN);
                    res.yaku_names.push("Shousangen".to_string());
                }
            }

            // Toitoi
            let koutsu_total = div
                .body
                .iter()
                .filter(|m| matches!(m, Mentsu::Koutsu(_)))
                .count()
                + melds
                    .iter()
                    .filter(|m| m.meld_type != crate::types::MeldType::Chi)
                    .count();
            if koutsu_total == 4 {
                res.han += 2;
                res.yaku_ids.push(ID_TOITOI);
                res.yaku_names.push("Toitoi".to_string());
            }

            // San Ankou
            let mut closed_koutsu_count = 0;
            for (idx, m) in div.body.iter().enumerate() {
                if let Mentsu::Koutsu(_) = m {
                    // Ron on closed triplet makes it open (if it was part of a Shanpon wait)
                    if !ctx.is_tsumo && Some(idx) == wg_idx {
                        continue;
                    }
                    closed_koutsu_count += 1;
                }
            }
            for m in melds {
                if m.meld_type == crate::types::MeldType::Angang {
                    closed_koutsu_count += 1;
                }
            }
            if closed_koutsu_count == 3 {
                res.han += 2;
                res.yaku_ids.push(ID_SANANKOU);
                res.yaku_names.push("San Ankou".to_string());
            }

            // San Kantsu
            let kantsu_count = melds
                .iter()
                .filter(|m| {
                    m.meld_type == crate::types::MeldType::Gang
                        || m.meld_type == crate::types::MeldType::Angang
                        || m.meld_type == crate::types::MeldType::Addgang
                })
                .count();
            if kantsu_count == 3 {
                res.han += 2;
                res.yaku_ids.push(ID_SANKANTSU);
                res.yaku_names.push("San Kantsu".to_string());
            }

            // Iipeiko / Ryanpeikou (Closed only)
            if ctx.is_menzen {
                let mut shuntsu_tiles = Vec::new();
                for m in &div.body {
                    if let Mentsu::Shuntsu(t) = m {
                        shuntsu_tiles.push(*t);
                    }
                }
                shuntsu_tiles.sort();
                let mut identical_pairs = 0;
                let mut i = 0;
                while i + 1 < shuntsu_tiles.len() {
                    if shuntsu_tiles[i] == shuntsu_tiles[i + 1] {
                        identical_pairs += 1;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                if identical_pairs == 2 {
                    res.han += 3;
                    res.yaku_ids.push(ID_RYANPEIKO);
                    res.yaku_names.push("Ryanpeikou".to_string());
                } else if identical_pairs == 1 {
                    res.han += 1;
                    res.yaku_ids.push(ID_IPEIKO);
                    res.yaku_names.push("Iipeiko".to_string());
                }
            }

            // Ittsu / Sanshoku Doujun
            if check_ittsu(div, melds) {
                res.han += if ctx.is_menzen { 2 } else { 1 };
                res.yaku_ids.push(ID_ITTSU);
                res.yaku_names.push("Ittsu".to_string());
            }
            if is_sanshoku_doujun(div, melds) {
                res.han += if ctx.is_menzen { 2 } else { 1 };
                res.yaku_ids.push(ID_SANSHOKU);
                res.yaku_names.push("Sanshoku Doujun".to_string());
            }
            if is_sanshoku_doukou(div, melds) {
                res.han += 2;
                res.yaku_ids.push(ID_SANSHOKU_DOKO);
                res.yaku_names.push("Sanshoku Doukou".to_string());
            }

            // Honitsu / Chinitsu
            if is_chinitsu(hand, melds) {
                res.han += if ctx.is_menzen { 6 } else { 5 };
                res.yaku_ids.push(ID_CHINITSU);
                res.yaku_names.push("Chinitsu".to_string());
            } else if is_honitsu(hand, melds) {
                res.han += if ctx.is_menzen { 3 } else { 2 };
                res.yaku_ids.push(ID_HONITSU);
                res.yaku_names.push("Honitsu".to_string());
            }

            // Chantai / Junchan / Honroutou
            if is_honroutou(hand, melds) {
                res.han += 2;
                res.yaku_ids.push(ID_HONROUTO);
                res.yaku_names.push("Honroutou".to_string());
            } else if is_junchan(div, melds) {
                res.han += if ctx.is_menzen { 3 } else { 2 };
                res.yaku_ids.push(ID_JUNCHAN);
                res.yaku_names.push("Junchan".to_string());
            } else if is_chantai(div, melds) {
                res.han += if ctx.is_menzen { 2 } else { 1 };
                res.yaku_ids.push(ID_CHANTA);
                res.yaku_names.push("Chantai".to_string());
            }

            // Static Yaku and Dora are handled by apply_static_yaku (already called at start of loop)

            if res.han > best_res.han || (res.han == best_res.han && res.fu > best_res.fu) {
                best_res = res;
            }
        }
    }

    best_res
}

fn calculate_fu_with_waiting(
    div: &Division,
    melds: &[Meld],
    ctx: &YakuContext,
    wg_idx: Option<usize>,
    win_tile: u8,
) -> u8 {
    let mut fu: u8 = 20;
    if ctx.is_tsumo {
        fu += 2;
    } else if ctx.is_menzen {
        fu += 10;
    }

    if div.head == ctx.bakaze {
        fu += 2;
    }
    if div.head == ctx.jikaze {
        fu += 2;
    }
    if div.head >= 31 {
        fu += 2;
    }

    // Waiting fu
    match wg_idx {
        None => fu += 2, // Tanki
        Some(idx) => {
            match div.body[idx] {
                Mentsu::Koutsu(_) => {
                    // Ron on closed koutsu is 2 fu, but handled in triplet processing below
                }
                Mentsu::Shuntsu(t) => {
                    // Kanchan or Penchan
                    if win_tile == t + 1
                        || (win_tile == t + 2 && (t % 9 == 0))
                        || (win_tile == t && (t % 9 == 6))
                    {
                        fu += 2;
                    }
                }
            }
        }
    }

    for (idx, m) in div.body.iter().enumerate() {
        if let Mentsu::Koutsu(t) = m {
            let mut f = 4; // Closed
            if !ctx.is_tsumo && Some(idx) == wg_idx {
                f = 2;
            } // Ron on closed triplet is open
            if is_terminal(*t) {
                f *= 2;
            }
            fu += f;
        }
    }
    for m in melds {
        if m.tiles.len() >= 3 && m.tiles[0] == m.tiles[1] {
            let mut f = 2; // Open
            if !m.opened {
                f = 4;
            } // Angang
            if is_terminal(m.tiles[0]) {
                f *= 2;
            }
            if m.meld_type == crate::types::MeldType::Gang
                || m.meld_type == crate::types::MeldType::Angang
                || m.meld_type == crate::types::MeldType::Addgang
            {
                f *= 4;
            }
            fu += f;
        }
    }

    if fu == 20 && !ctx.is_tsumo {
        fu = 30;
    } // Open Ron minimum 30 fu

    fu.div_ceil(10) * 10
}

fn check_pinfu(
    div: &Division,
    melds: &[Meld],
    ctx: &YakuContext,
    wg_idx: Option<usize>,
    win_tile: u8,
) -> bool {
    // Pinfu requires Menzen, all sequences, non-yakuhai head
    if !ctx.is_menzen {
        return false;
    }
    if !melds.is_empty() {
        return false;
    } // No melds (including closed Kan)
    for m in &div.body {
        if let Mentsu::Koutsu(_) = m {
            return false;
        }
    }
    if is_yakuhai_tile(div.head, ctx) {
        return false;
    }

    // Waiting must be ryanmen
    if let Some(idx) = wg_idx {
        if let Mentsu::Shuntsu(t) = div.body[idx] {
            // Ryanmen: win_tile is t or t+2, and it's not a penchan wait
            if win_tile == t {
                if t % 9 == 6 {
                    return false;
                } // 78(9) is penchan
                return true;
            }
            if win_tile == t + 2 {
                if t % 9 == 0 {
                    return false;
                } // (1)23 is penchan
                return true;
            }
        }
    }
    false
}

fn is_yakuhai_tile(tile: u8, ctx: &YakuContext) -> bool {
    tile >= 31 || tile == ctx.bakaze || tile == ctx.jikaze
}

fn is_honroutou(hand: &Hand, melds: &[Meld]) -> bool {
    for (i, &count) in hand.counts.iter().enumerate() {
        if count > 0 && !is_terminal(i as u8) {
            return false;
        }
    }
    if melds
        .iter()
        .any(|m| m.tiles.iter().any(|&t| !is_terminal(t)))
    {
        return false;
    }
    true
}

fn is_junchan(div: &Division, melds: &[Meld]) -> bool {
    if !is_number_terminal(div.head) {
        return false;
    }
    for m in &div.body {
        match m {
            Mentsu::Koutsu(t) => {
                if !is_number_terminal(*t) {
                    return false;
                }
            }
            Mentsu::Shuntsu(t) => {
                if !is_number_terminal(*t) && !is_number_terminal(t + 2) {
                    return false;
                }
            }
        }
    }
    for m in melds {
        if m.tiles.iter().all(|&t| !is_number_terminal(t)) {
            return false;
        }
    }
    true
}

fn is_chantai(div: &Division, melds: &[Meld]) -> bool {
    if !is_terminal(div.head) {
        return false;
    }
    let mut has_honor = is_honor(div.head);
    for m in &div.body {
        match m {
            Mentsu::Koutsu(t) => {
                if !is_terminal(*t) {
                    return false;
                }
                if is_honor(*t) {
                    has_honor = true;
                }
            }
            Mentsu::Shuntsu(t) => {
                if !is_terminal(*t) && !is_terminal(t + 2) {
                    return false;
                }
            }
        }
    }
    for m in melds {
        if m.tiles.iter().all(|&t| !is_terminal(t)) {
            return false;
        }
        if m.tiles.iter().any(|&t| is_honor(t)) {
            has_honor = true;
        }
    }
    has_honor
}

fn is_terminal(tile: u8) -> bool {
    tile >= 27 || tile.is_multiple_of(9) || (tile % 9 == 8)
}
fn is_number_terminal(tile: u8) -> bool {
    tile < 27 && (tile.is_multiple_of(9) || tile % 9 == 8)
}
fn is_honor(tile: u8) -> bool {
    tile >= 27
}

fn is_honitsu(hand: &Hand, melds: &[Meld]) -> bool {
    let mut suits = [false; 3];
    let mut has_honor = false;
    for (i, &count) in hand.counts.iter().enumerate() {
        if count > 0 {
            if i < 9 {
                suits[0] = true;
            } else if i < 18 {
                suits[1] = true;
            } else if i < 27 {
                suits[2] = true;
            } else {
                has_honor = true;
            }
        }
    }
    for meld in melds {
        for &t in &meld.tiles {
            let idx = t as usize;
            if idx < 9 {
                suits[0] = true;
            } else if idx < 18 {
                suits[1] = true;
            } else if idx < 27 {
                suits[2] = true;
            } else {
                has_honor = true;
            }
        }
    }
    suits.iter().filter(|&&b| b).count() == 1 && has_honor
}

fn is_chinitsu(hand: &Hand, melds: &[Meld]) -> bool {
    let mut suits = [false; 3];
    for (i, &count) in hand.counts.iter().enumerate() {
        if count > 0 {
            if i >= 27 {
                return false;
            }
            if i < 9 {
                suits[0] = true;
            } else if i < 18 {
                suits[1] = true;
            } else if i < 27 {
                suits[2] = true;
            }
        }
    }
    for meld in melds {
        for &t in &meld.tiles {
            let idx = t as usize;
            if idx >= 27 {
                return false;
            }
            if idx < 9 {
                suits[0] = true;
            } else if idx < 18 {
                suits[1] = true;
            } else if idx < 27 {
                suits[2] = true;
            }
        }
    }
    suits.iter().filter(|&&b| b).count() == 1
}

fn apply_static_yaku(res: &mut YakuResult, ctx: &YakuContext) {
    // Riichi
    if ctx.is_reach && !ctx.is_daburu_reach {
        res.han += 1;
        res.yaku_ids.push(ID_RIICHI);
    }
    if ctx.is_daburu_reach {
        res.han += 2;
        res.yaku_ids.push(ID_DOUBLE_RIICHI);
    }
    if ctx.is_ippatsu {
        res.han += 1;
        res.yaku_ids.push(ID_IPPATSU);
    }
    if ctx.is_menzen && ctx.is_tsumo {
        res.han += 1;
        res.yaku_ids.push(ID_TSUMO);
    }
    if ctx.is_haitei {
        res.han += 1;
        res.yaku_ids.push(ID_HAITEI);
    }
    if ctx.is_houtei {
        res.han += 1;
        res.yaku_ids.push(ID_HOUTEI);
    }
    if ctx.is_rinshan {
        res.han += 1;
        res.yaku_ids.push(ID_RINSHAN);
    }
    if ctx.is_chankan {
        res.han += 1;
        res.yaku_ids.push(ID_CHANKAN);
    }

    if ctx.dora_count > 0 {
        res.han += ctx.dora_count;
        res.yaku_ids.push(ID_DORA);
    }
    if ctx.aka_dora > 0 {
        res.han += ctx.aka_dora;
        res.yaku_ids.push(ID_AKADORA);
    }
    if ctx.ura_dora_count > 0 {
        res.han += ctx.ura_dora_count;
        res.yaku_ids.push(ID_URADORA);
    }
}

fn apply_yakuman(
    res: &mut YakuResult,
    hand: &Hand,
    melds: &[Meld],
    ctx: &YakuContext,
    div: &Division,
    wg_idx: Option<usize>,
    win_tile: u8,
) {
    let mut yakuman_count = 0;

    // Kokushi / Chiitoitsu are handled at the start of calculate_yaku
    if div.head == 0 && div.body.is_empty() {
        // Special Case: Kokushi / Chiitoitsu call from start of calculate_yaku
        // We only care about Tenhou/Chiihou/Tsuuiisou etc. that don't need divisions
    }

    // Tsuu iisou (All Honors)
    if is_tsuu_iisou(hand, melds) {
        yakuman_count += 1;
        res.yaku_ids.push(ID_TSUISO);
        res.yaku_names.push("Tsuu iisou".to_string());
    }

    // Chinroutou (All Terminals)
    if is_chinroutou(hand, melds) {
        yakuman_count += 1;
        res.yaku_ids.push(ID_CHINROUTO);
        res.yaku_names.push("Chinroutou".to_string());
    }

    // Ryuu iisou (All Green)
    if is_ryuu_iisou(hand, melds) {
        yakuman_count += 1;
        res.yaku_ids.push(ID_RYUISOU);
        res.yaku_names.push("Ryuu iisou".to_string());
    }

    // Su Kantsu (Four Kans)
    if melds
        .iter()
        .filter(|m| {
            m.meld_type == crate::types::MeldType::Gang
                || m.meld_type == crate::types::MeldType::Angang
                || m.meld_type == crate::types::MeldType::Addgang
        })
        .count()
        == 4
    {
        yakuman_count += 1;
        res.yaku_ids.push(ID_SUKANTSU);
        res.yaku_names.push("Su Kantsu".to_string());
    }

    // Chuuren Poutou
    if ctx.is_menzen && (div.body.len() + melds.len()) == 4 {
        // Only check if we have a valid standard division
        if is_chuuren_poutou(hand) {
            let is_9_wait = is_chuuren_9_wait(hand, win_tile);
            if is_9_wait {
                yakuman_count += 2;
                res.yaku_ids.push(ID_JUNSEI_CHUUREN);
                res.yaku_names.push("Chuuren Poutou 9-wait".to_string());
            } else {
                yakuman_count += 1;
                res.yaku_ids.push(ID_CHUUREN);
                res.yaku_names.push("Chuuren Poutou".to_string());
            }
        }
    }

    // Tenhou / Chiihou
    if ctx.is_tsumo_first_turn && ctx.is_menzen && ctx.is_tsumo {
        if ctx.jikaze == 27 {
            // Oya (East)
            yakuman_count += 1;
            res.yaku_ids.push(ID_TENHO);
            res.yaku_names.push("Tenhou".to_string());
        } else {
            yakuman_count += 1;
            res.yaku_ids.push(ID_CHIHO);
            res.yaku_names.push("Chiihou".to_string());
        }
    }

    // Divisions-based Yakuman
    // Su Ankou
    let mut closed_koutsu_count = 0;
    for (idx, m) in div.body.iter().enumerate() {
        if let Mentsu::Koutsu(_) = m {
            if !ctx.is_tsumo && Some(idx) == wg_idx {
                continue;
            }
            closed_koutsu_count += 1;
        }
    }
    for m in melds {
        if m.meld_type == crate::types::MeldType::Angang {
            closed_koutsu_count += 1;
        }
    }
    if closed_koutsu_count == 4 {
        if wg_idx.is_none() {
            yakuman_count += 2;
            res.yaku_ids.push(ID_SUANKO_TANKI); // Su Ankou Tanki
            res.yaku_names.push("Su Ankou Tanki".to_string());
        } else {
            yakuman_count += 1;
            res.yaku_ids.push(ID_SUANKO);
            res.yaku_names.push("Su Ankou".to_string());
        }
    }

    // Daisangen
    let haku_koutsu = div.body.iter().any(|m| match m {
        Mentsu::Koutsu(t) => *t == 31,
        _ => false,
    }) || melds.iter().any(|m| m.tiles.contains(&31));
    let hatsu_koutsu = div.body.iter().any(|m| match m {
        Mentsu::Koutsu(t) => *t == 32,
        _ => false,
    }) || melds.iter().any(|m| m.tiles.contains(&32));
    let chun_koutsu = div.body.iter().any(|m| match m {
        Mentsu::Koutsu(t) => *t == 33,
        _ => false,
    }) || melds.iter().any(|m| m.tiles.contains(&33));

    if haku_koutsu && hatsu_koutsu && chun_koutsu {
        yakuman_count += 1;
        res.yaku_ids.push(ID_DAISANGEN);
        res.yaku_names.push("Daisangen".to_string());
    }

    // Winds
    let mut wind_koutsu_count = 0;
    let mut wind_pair_count = 0;
    for w in 27..=30 {
        let has_koutsu = div.body.iter().any(|m| match m {
            Mentsu::Koutsu(t) => *t == w,
            _ => false,
        }) || melds
            .iter()
            .any(|m| m.tiles[0] == w && m.meld_type != crate::types::MeldType::Chi);
        if has_koutsu {
            wind_koutsu_count += 1;
        } else if div.head == w {
            wind_pair_count += 1;
        }
    }
    if wind_koutsu_count == 4 {
        yakuman_count += 2; // Double Yakuman
        res.yaku_ids.push(ID_DAISUUSHI);
        res.yaku_names.push("Daisushii".to_string());
    } else if wind_koutsu_count == 3 && wind_pair_count == 1 {
        yakuman_count += 1;
        res.yaku_ids.push(ID_SHOUSUUSHI);
        res.yaku_names.push("Shousushii".to_string());
    }

    if yakuman_count > 0 {
        res.han = 13 * yakuman_count;
        res.yakuman_count = yakuman_count;
    }
}

fn is_tsuu_iisou(hand: &Hand, melds: &[Meld]) -> bool {
    for (i, &count) in hand.counts.iter().enumerate() {
        if count > 0 && i < 27 {
            return false;
        }
    }
    for m in melds {
        if m.tiles.iter().any(|&t| t < 27) {
            return false;
        }
    }
    true
}

fn is_chinroutou(hand: &Hand, melds: &[Meld]) -> bool {
    for (i, &count) in hand.counts.iter().enumerate() {
        if count > 0 && !is_number_terminal(i as u8) {
            return false;
        }
    }
    for m in melds {
        if m.tiles.iter().any(|&t| !is_number_terminal(t)) {
            return false;
        }
    }
    true
}

fn is_ryuu_iisou(hand: &Hand, melds: &[Meld]) -> bool {
    // 2, 3, 4, 6, 8 of sou + Green Dragon (32)
    let green_tiles = [19, 20, 21, 23, 25, 32];
    for (i, &count) in hand.counts.iter().enumerate() {
        if count > 0 && !green_tiles.contains(&(i as u8)) {
            return false;
        }
    }
    for m in melds {
        if m.tiles.iter().any(|&t| !green_tiles.contains(&t)) {
            return false;
        }
    }
    true
}

fn is_chuuren_poutou(hand: &Hand) -> bool {
    // Must be chinitsu (already checked or implicit if this is called)
    // Check if hand contains 1112345678999 + any one of 1-9
    let mut counts = [0u8; 9];
    let mut suit = None;

    for (i, &count) in hand.counts.iter().enumerate() {
        if count > 0 {
            if i >= 27 {
                return false;
            }
            let s = i / 9;
            if let Some(prev_s) = suit {
                if prev_s != s {
                    return false;
                }
            } else {
                suit = Some(s);
            }
            counts[i % 9] = count;
        }
    }

    if counts[0] < 3 || counts[8] < 3 {
        return false;
    }
    if counts[1..8].contains(&0) {
        return false;
    }
    true
}

fn is_chuuren_9_wait(hand: &Hand, win_tile: u8) -> bool {
    if win_tile >= 27 {
        return false;
    }
    let val = (win_tile % 9) as usize;
    let counts = &hand.counts[(win_tile / 9 * 9) as usize..(win_tile / 9 * 9 + 9) as usize];

    if val == 0 || val == 8 {
        counts[val] == 4
    } else {
        counts[val] == 2
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
                if *t == suit_offset {
                    has_123 = true;
                } else if *t == suit_offset + 3 {
                    has_456 = true;
                } else if *t == suit_offset + 6 {
                    has_789 = true;
                }
            }
        }

        // Check Melds
        for m in melds {
            if m.meld_type == crate::types::MeldType::Chi {
                let t = m.tiles[0];
                if t == suit_offset {
                    has_123 = true;
                } else if t == suit_offset + 3 {
                    has_456 = true;
                } else if t == suit_offset + 6 {
                    has_789 = true;
                }
            }
        }

        if has_123 && has_456 && has_789 {
            return true;
        }
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
                if *t == i {
                    has_man = true;
                }
                if *t == i + 9 {
                    has_pin = true;
                }
                if *t == i + 18 {
                    has_sou = true;
                }
            }
        }
        for m in melds {
            if m.meld_type == crate::types::MeldType::Chi {
                let t = m.tiles[0];
                if t == i {
                    has_man = true;
                }
                if t == i + 9 {
                    has_pin = true;
                }
                if t == i + 18 {
                    has_sou = true;
                }
            }
        }
        if has_man && has_pin && has_sou {
            return true;
        }
    }
    false
}

fn is_tanyao(hand: &Hand, melds: &[Meld]) -> bool {
    let terminals = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
    for &t in &terminals {
        if hand.counts[t] > 0 {
            return false;
        }
    }
    for meld in melds {
        for &t in &meld.tiles {
            if terminals.contains(&(t as usize)) {
                return false;
            }
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
                if *t == i {
                    has_man = true;
                }
                if *t == i + 9 {
                    has_pin = true;
                }
                if *t == i + 18 {
                    has_sou = true;
                }
            }
        }
        for m in melds {
            if m.meld_type != crate::types::MeldType::Chi {
                let t = m.tiles[0];
                if t == i {
                    has_man = true;
                }
                if t == i + 9 {
                    has_pin = true;
                }
                if t == i + 18 {
                    has_sou = true;
                }
            }
        }
        if has_man && has_pin && has_sou {
            return true;
        }
    }
    false
}
