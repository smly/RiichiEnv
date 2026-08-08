//! SP-specific lean yaku/fu calculator.
//!
//! Used by `sp::base_score_tsumo` to bypass `HandEvaluator` + `yaku::calculate_yaku`
//! for standard tsumo wins encountered during SP DP. The full path remains as a
//! fallback (returned `None` from this module).
//!
//! Scope:
//!   - Standard 14-tile (one pair + four mentsu) tsumo wins.
//!   - All common Mahjong yaku (riichi/menzen tsumo/yakuhai/tanyao/pinfu/
//!     iipeikou/honitsu/chinitsu/toitoi/sanankou/sanshoku doujun/sanshoku doukou/
//!     ittsu/junchan/chanta/honroutou/shousangen/ryanpeikou).
//!   - Aka dora + regular dora.
//!
//! Out of scope (returns `None`, falls back):
//!   - Yakuman shapes (caller's HandEvaluator path handles these).
//!   - Chitoitsu / Kokushi (not in `agari_table`).
//!   - Open hands (the compact agari table is keyed by 14 concealed tiles).
//!   - Hands containing kans (kan fu / suukantsu).
//!   - Ura dora distribution (caller computes it on top of our base score).
//!
use crate::agari_table::{self, MAX_DIVS_PER_KEY};
use crate::sp::SpInput;
use crate::types::{Meld, MeldType, TILE_MAX};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeanScore {
    pub han: u32,
    pub fu: u32,
    pub yakuman: bool,
}

const ID_BASE_HAN_OFFSET: u32 = 0;

/// Compute (han, fu, yakuman_flag) for a SP-style tsumo win, or `None` if the
/// caller should fall back to the full HandEvaluator path.
///
/// `counts_13` is the 13-tile in-hand count vector (excluding the winning tile);
/// `win_tile` is the 0..34 tile id that was drawn to complete the hand.
/// `akas_in_hand` flags which red 5xrs are present in the in-hand portion.
pub fn compute_for_sp_tsumo(
    input: &SpInput,
    counts_13: &[u8; TILE_MAX],
    win_tile: u8,
    akas_in_hand: [bool; 3],
) -> Option<LeanScore> {
    if win_tile as usize >= TILE_MAX || counts_13[win_tile as usize] >= 4 {
        return None;
    }
    // Reject hands with kans — fu and suukantsu would diverge.
    for m in &input.melds {
        match m.meld_type {
            MeldType::Daiminkan | MeldType::Ankan | MeldType::Kakan => return None,
            _ => {}
        }
    }

    let mut counts_14 = *counts_13;
    counts_14[win_tile as usize] += 1;

    // Aggregate counts including melds (for yaku detection like tanyao).
    let mut full_counts = counts_14;
    for m in &input.melds {
        for &t136 in &m.tiles {
            let tt = (t136 / 4) as usize;
            if tt < TILE_MAX {
                full_counts[tt] += 1;
            }
        }
    }

    let assume_riichi = input.is_menzen && input.can_riichi;

    // Chitoitsu shape detection (7 distinct pairs, menzen, no melds).
    // We don't early-return here: a 14-tile hand with 7 distinct pairs can
    // ALSO admit a ryanpeikou-style standard decomposition (e.g. 2 sets of
    // identical shuntsu + 1 pair). 高点法 (highest-score rule) requires us
    // to evaluate both interpretations and return the higher-scoring one.
    let mut chitoi_han: Option<u32> = None;
    if input.is_menzen && input.melds.is_empty() {
        let mut all_two = true;
        let mut pair_count = 0u8;
        for &c in &counts_14 {
            match c {
                0 => {}
                2 => pair_count += 1,
                _ => {
                    all_two = false;
                    break;
                }
            }
        }
        if all_two && pair_count == 7 {
            let (suit_usage, flags) = scan_hand(&counts_14);
            let mut han: u32 = 2; // chiitoitsu
            han += 1; // menzen tsumo
            if assume_riichi {
                han += 1;
            }
            // tanyao
            if !flags.has_terminal && !suit_usage.has_z {
                han += 1;
            }
            // honitsu / chinitsu
            if flags.single_numbered_suit {
                if suit_usage.has_z {
                    han += 3;
                } else {
                    han += 6;
                }
            }
            // honroutou (all yaocchi pairs)
            if flags.all_yaocchi {
                han += 2;
            }
            // dora — counts_14 includes the win tile, so the aka check
            // correctly attributes the red 5x even when it's the win itself.
            let regular_dora = count_regular_dora(input, &counts_14);
            let red_present = |t34: usize| counts_14[t34] >= 1;
            let mut aka_dora = 0u32;
            if akas_in_hand[0] && red_present(4) {
                aka_dora += 1;
            }
            if akas_in_hand[1] && red_present(13) {
                aka_dora += 1;
            }
            if akas_in_hand[2] && red_present(22) {
                aka_dora += 1;
            }
            han += regular_dora + aka_dora;
            chitoi_han = Some(han);
        }
    }

    // Look up decompositions from the topology-indexed compact table.
    // Returns a slice of CompactDiv (each = u32, packed pair_idx +
    // kotsu/shuntsu tile14 indices). Working set per SP sample is small
    // enough to fit in L1 (~250 unique keys × 4-byte div + index value).
    //
    // For pure-chitoi shapes (no 4-set+pair decomposition possible), the
    // lookup miss is expected — we'll return the chitoi score we already
    // computed. For shapes that admit BOTH chitoi AND a standard division
    // (e.g. ryanpeikou-via-double-shuntsu), we fall through to compute the
    // standard score and pick the higher-scoring interpretation.
    let standard_lookup = agari_table::lookup_compact(&counts_14);
    let (tile14, list) = match standard_lookup {
        Some((t, l)) if !l.is_empty() => (t, l),
        _ => {
            return chitoi_han.map(|han| LeanScore {
                han,
                fu: 25,
                yakuman: false,
            });
        }
    };

    // Quick yakuman shape detection: defer to the legacy path so we never
    // produce a wrong score by missing yakuman.
    if might_be_yakuman(&full_counts, &input.melds) {
        return None;
    }

    // Single-pass hand scan: SuitUsage + HandFlags in one walk over the counts.
    let (suit_usage, flags) = scan_hand(&full_counts);

    // Aggregate dora count (regular indicators -> next tile + aka).
    let regular_dora = count_regular_dora(input, &full_counts);
    // Aka attribution: an in-hand red 5x must be present somewhere in the
    // 14-tile post-win hand (counts_14 = counts_13 + win_tile). Previously
    // this checked counts_13 only, which incorrectly dropped the aka bit
    // when the win tile WAS the player's only copy of that 5x (e.g. drew
    // aka 5s into a hand without prior 5s). Mortal counts this aka via its
    // post-deal `state.akas_in_hand` flag — we need the same semantics here.
    let red_present = |t34: usize| counts_14[t34] >= 1;
    let mut aka_dora = count_meld_aka(&input.melds);
    if akas_in_hand[0] && red_present(4) {
        aka_dora += 1;
    }
    if akas_in_hand[1] && red_present(13) {
        aka_dora += 1;
    }
    if akas_in_hand[2] && red_present(22) {
        aka_dora += 1;
    }

    let bakaze = norm_wind(input.bakaze);
    let jikaze = norm_wind(input.jikaze);
    let is_oya = jikaze == 27;

    // Hoist meld decomposition out of the per-div loop (constant for one call).
    let mut melds_kotsu_buf = [0u8; 4];
    let mut melds_shuntsu_buf = [0u8; 4];
    let (n_mk, n_ms) = collect_melds(&input.melds, &mut melds_kotsu_buf, &mut melds_shuntsu_buf);
    let melds_kotsu = &melds_kotsu_buf[..n_mk];
    let melds_shuntsu = &melds_shuntsu_buf[..n_ms];


    // For each table-decomp + the open-meld portion, compute (han, fu).
    let mut best: Option<(u32, u32)> = None; // (han, fu) — pick by total score
    for &template in list {
        let div = absolute_div_compact(template, &tile14);

        let Some((han, fu)) = score_one_div(
            input,
            &div,
            &full_counts,
            counts_13,
            win_tile,
            akas_in_hand,
            assume_riichi,
            bakaze,
            jikaze,
            is_oya,
            &suit_usage,
            &flags,
            regular_dora,
            aka_dora,
            melds_kotsu,
            melds_shuntsu,
        ) else {
            return None;
        };
        match best {
            None => best = Some((han, fu)),
            Some((bh, bf)) => {
                if score_lt(bh, bf, han, fu, is_oya) {
                    best = Some((han, fu));
                }
            }
        }
    }

    // 高点法: compare standard best with chitoi (if applicable) and return
    // the higher-scoring interpretation.
    let standard = best;
    match (standard, chitoi_han) {
        (Some((s_han, s_fu)), Some(c_han)) => {
            if score_lt(c_han, 25, s_han, s_fu, is_oya) {
                Some(LeanScore { han: s_han, fu: s_fu, yakuman: false })
            } else {
                Some(LeanScore { han: c_han, fu: 25, yakuman: false })
            }
        }
        (Some((s_han, s_fu)), None) => Some(LeanScore {
            han: s_han,
            fu: s_fu,
            yakuman: false,
        }),
        (None, Some(c_han)) => Some(LeanScore {
            han: c_han,
            fu: 25,
            yakuman: false,
        }),
        (None, None) => None,
    }
}

/// Yaku-presence-only fast path. Returns `Some(true)` if the (counts_13, win_tile)
/// has any yaku, `Some(false)` if definitely none, and `None` if the lean
/// path is out of scope (caller should fall back to score_tsumo).
///
/// Skips fu computation, dora counting, aka counting, and (han, fu) tally —
/// stops at the first division with any yaku flag set. ~5× faster than
/// `compute_for_sp_tsumo` for "no yaku" hands (which is the worst case for
/// SP's `has_yaku_tenpai_after_best_discard` Pass 2).
pub fn has_any_yaku_for_sp_tsumo(
    input: &SpInput,
    counts_13: &[u8; TILE_MAX],
    win_tile: u8,
) -> Option<bool> {
    if win_tile as usize >= TILE_MAX || counts_13[win_tile as usize] >= 4 {
        return None;
    }
    // Reject hands with kans — fu/yakuman semantics diverge.
    for m in &input.melds {
        match m.meld_type {
            MeldType::Daiminkan | MeldType::Ankan | MeldType::Kakan => return None,
            _ => {}
        }
    }

    let mut counts_14 = *counts_13;
    counts_14[win_tile as usize] += 1;
    let mut full_counts = counts_14;
    for m in &input.melds {
        for &t136 in &m.tiles {
            let tt = (t136 / 4) as usize;
            if tt < TILE_MAX {
                full_counts[tt] += 1;
            }
        }
    }

    let assume_riichi = input.is_menzen && input.can_riichi;

    // Menzen tsumo / riichi guarantee yaku.
    if input.is_menzen || assume_riichi {
        return Some(true);
    }

    // Chitoitsu fast path: 7 distinct pairs, menzen-only — already covered above.
    // (We're here only if !is_menzen, so chitoitsu doesn't apply.)

    let bakaze = norm_wind(input.bakaze);
    let jikaze = norm_wind(input.jikaze);

    // Yakuhai-in-meld: any pon/kan of dragon/seat/round wind tile.
    for m in &input.melds {
        for &t136 in &m.tiles {
            let tt = t136 / 4;
            if matches!(tt, 31 | 32 | 33) || tt == bakaze || tt == jikaze {
                return Some(true);
            }
        }
    }

    // Structural shape-only yaku (computable without decomposition):
    //   - tanyao: full has no yaocchi
    //   - honitsu / chinitsu: full uses only 1 numbered suit
    //   - honroutou: full all yaocchi
    //   - sanshoku doukou (kotsu shape): same n with full[n], full[n+9], full[n+18] ≥ 3
    //   - in-hand yakuhai kotsu: yakuhai count ≥ 3 in counts_14
    let (suit_usage, flags) = scan_hand(&full_counts);
    if !flags.has_terminal && !suit_usage.has_z {
        return Some(true); // tanyao
    }
    if flags.single_numbered_suit {
        return Some(true); // honitsu/chinitsu (open)
    }
    if flags.all_yaocchi {
        return Some(true); // honroutou (kotsu-only decomp guaranteed)
    }
    for n in 0..9usize {
        if full_counts[n] >= 3 && full_counts[n + 9] >= 3 && full_counts[n + 18] >= 3 {
            return Some(true); // sanshoku doukou
        }
    }
    let mut yh_tiles = [31u8, 32, 33, 0, 0];
    let mut n_yh = 3usize;
    if (27..=30).contains(&bakaze) { yh_tiles[n_yh] = bakaze; n_yh += 1; }
    if (27..=30).contains(&jikaze) && jikaze != bakaze { yh_tiles[n_yh] = jikaze; n_yh += 1; }
    for i in 0..n_yh {
        if counts_14[yh_tiles[i] as usize] >= 3 {
            return Some(true); // in-hand yakuhai kotsu
        }
    }

    // Fall back to per-division check: requires shape decomposition for yaku
    // like sanshoku doujun, ittsu, sanankou, junchan, chanta. Look up
    // agari_table and check yaku flags per division — stop at first yaku.
    let (tile14, list) = agari_table::lookup_compact(&counts_14)?;
    if list.is_empty() {
        return Some(false);
    }

    // Quick yakuman shape detection: defer to the legacy path so we never
    // miss yakuman.
    if might_be_yakuman(&full_counts, &input.melds) {
        return None;
    }

    // Per-division yaku-flag check. We don't need han/fu — just yaku presence.
    let mut melds_kotsu_buf = [0u8; 4];
    let mut melds_shuntsu_buf = [0u8; 4];
    let (n_mk, n_ms) = collect_melds(&input.melds, &mut melds_kotsu_buf, &mut melds_shuntsu_buf);
    let melds_kotsu = &melds_kotsu_buf[..n_mk];
    let melds_shuntsu = &melds_shuntsu_buf[..n_ms];

    for &template in list {
        let div = absolute_div_compact(template, &tile14);
        if has_any_yaku_for_div(
            input,
            &div,
            win_tile,
            bakaze,
            jikaze,
            &suit_usage,
            &flags,
            melds_kotsu,
            melds_shuntsu,
        ) {
            return Some(true);
        }
    }
    Some(false)
}

/// Yaku-flag-only check for a single decomposition. Skips fu compute and
/// han accumulation; returns `true` at the first yaku found.
#[allow(clippy::too_many_arguments)]
fn has_any_yaku_for_div(
    input: &SpInput,
    div: &AbsoluteDiv,
    win_tile: u8,
    bakaze: u8,
    jikaze: u8,
    suit_usage: &SuitUsage,
    flags: &HandFlags,
    melds_kotsu: &[u8],
    melds_shuntsu: &[u8],
) -> bool {
    // Verify win_tile is in this division (else caller never reaches scoring).
    let mut wait_buf = [WaitKind::Tanki; 5];
    let n_waits = classify_waits(div, win_tile, &mut wait_buf);
    if n_waits == 0 {
        return false;
    }

    let melds_empty = melds_kotsu.is_empty() && melds_shuntsu.is_empty();
    let mut all_kotsu_buf = [0u8; 8];
    let mut all_shuntsu_buf = [0u8; 8];
    let (all_kotsu, all_shuntsu): (&[u8], &[u8]) = if melds_empty {
        (div.kotsu(), div.shuntsu())
    } else {
        let mut n_k = 0usize;
        for &k in div.kotsu() { all_kotsu_buf[n_k] = k; n_k += 1; }
        for &k in melds_kotsu { all_kotsu_buf[n_k] = k; n_k += 1; }
        let mut n_s = 0usize;
        for &s in div.shuntsu() { all_shuntsu_buf[n_s] = s; n_s += 1; }
        for &s in melds_shuntsu { all_shuntsu_buf[n_s] = s; n_s += 1; }
        (&all_kotsu_buf[..n_k], &all_shuntsu_buf[..n_s])
    };

    // Yakuhai (kotsu of dragon/seat/round wind) — terminate early.
    for &k in all_kotsu {
        if matches!(k, 31 | 32 | 33) || k == bakaze || k == jikaze {
            return true;
        }
    }

    // Tanyao / honitsu / chinitsu / honroutou: already pre-checked in caller,
    // but they remain TRUE for this div too (shape-only). Re-check for safety.
    if !flags.has_terminal && !suit_usage.has_z { return true; }
    if flags.single_numbered_suit { return true; }
    if flags.all_yaocchi { return true; }

    // Toitoi: 4 kotsu (no shuntsu in division).
    if all_shuntsu.is_empty() && !all_kotsu.is_empty() {
        return true;
    }

    // Bitmasks for sanshoku doujun / ittsu / sanshoku doukou.
    let mut shuntsu_mask: u32 = 0;
    for &s in all_shuntsu { shuntsu_mask |= 1u32 << s; }
    let smask_m = shuntsu_mask & 0x1FF;
    let smask_p = (shuntsu_mask >> 9) & 0x1FF;
    let smask_s = (shuntsu_mask >> 18) & 0x1FF;
    const ITTSU_PATTERN: u32 = (1 << 0) | (1 << 3) | (1 << 6);
    if (smask_m & ITTSU_PATTERN) == ITTSU_PATTERN
        || (smask_p & ITTSU_PATTERN) == ITTSU_PATTERN
        || (smask_s & ITTSU_PATTERN) == ITTSU_PATTERN
    {
        return true;
    }
    if (smask_m & smask_p & smask_s) != 0 {
        return true;
    }
    let mut kotsu_mask: u64 = 0;
    for &k in all_kotsu { kotsu_mask |= 1u64 << k; }
    let kmask_m = (kotsu_mask & 0x1FF) as u32;
    let kmask_p = ((kotsu_mask >> 9) & 0x1FF) as u32;
    let kmask_s = ((kotsu_mask >> 18) & 0x1FF) as u32;
    if (kmask_m & kmask_p & kmask_s) != 0 {
        return true;
    }

    // Sanankou (≥ 3 closed kotsu in tsumo).
    if div.n_kotsu >= 3 {
        return true;
    }

    // Shousangen: 2 dragon kotsu + 1 dragon pair.
    let dragon_kotsu_bits = (kotsu_mask >> 31) & 0b111;
    let dragon_kotsu_count = dragon_kotsu_bits.count_ones() as usize;
    let dragon_pair = (31..=33).contains(&div.pair_tile);
    if dragon_kotsu_count == 2 && dragon_pair {
        return true;
    }

    // Junchan / Chanta.
    if flags.has_terminal {
        let mentsu_all_have_terminal = all_kotsu.iter().all(|&t| is_terminal(t))
            && all_shuntsu.iter().all(|&t| t % 9 == 0 || t % 9 == 6)
            && is_terminal(div.pair_tile);
        let mentsu_all_have_yaocchi = all_kotsu.iter().all(|&t| is_yaocchi(t))
            && all_shuntsu.iter().all(|&t| t % 9 == 0 || t % 9 == 6)
            && is_yaocchi(div.pair_tile);
        let honroutou_local = flags.all_yaocchi
            && all_shuntsu.is_empty()
            && all_kotsu.iter().all(|&t| is_yaocchi(t))
            && is_yaocchi(div.pair_tile);
        let junchan = !honroutou_local
            && !all_shuntsu.is_empty()
            && !suit_usage.has_z
            && mentsu_all_have_terminal;
        let chanta = !honroutou_local && !junchan && !all_shuntsu.is_empty()
            && mentsu_all_have_yaocchi;
        if junchan || chanta {
            return true;
        }
    }

    // Iipeikou / ryanpeikou (menzen only — caller already returned true for
    // is_menzen, so this is dead code under our caller).
    if input.is_menzen {
        let (iipeikou, ryanpeikou) = detect_peikou(div.shuntsu());
        if iipeikou || ryanpeikou {
            return true;
        }
    }

    // Pinfu (menzen + 4 shuntsu + non-yakuhai pair + ryanmen wait).
    if input.is_menzen
        && div.n_shuntsu == 4
        && div.n_kotsu == 0
        && melds_kotsu.is_empty()
        && melds_shuntsu.is_empty()
        && !is_yakuhai_pair(div.pair_tile, bakaze, jikaze)
    {
        for i in 0..n_waits {
            if wait_buf[i] == WaitKind::Ryanmen {
                return true;
            }
        }
    }

    false
}

/// Pick the higher-scoring (han, fu). If han caps to mangan/etc., higher han
/// always wins; below mangan we compare base points.
fn score_lt(a_han: u32, a_fu: u32, b_han: u32, b_fu: u32, _is_oya: bool) -> bool {
    if a_han != b_han {
        return a_han < b_han;
    }
    a_fu < b_fu
}

fn norm_wind(w: u8) -> u8 {
    if (27..=30).contains(&w) {
        w
    } else {
        27 + (w & 0b11)
    }
}

fn is_yaocchi(t: u8) -> bool {
    t >= 27 || matches!(t % 9, 0 | 8)
}

/// True only for the six terminals (1m/9m/1p/9p/1s/9s). Excludes honors.
fn is_terminal(t: u8) -> bool {
    if t >= 27 {
        return false;
    }
    matches!(t % 9, 0 | 8)
}

fn count_regular_dora(input: &SpInput, full_counts: &[u8; TILE_MAX]) -> u32 {
    let mut n = 0u32;
    for &ind136 in &input.dora_indicators {
        let ind34 = (ind136 / 4) as usize;
        let dora34 = next_tile(ind34);
        n += full_counts[dora34] as u32;
    }
    n
}

fn next_tile(t34: usize) -> usize {
    if t34 < 27 {
        // numbered 0..26 (3 suits × 9)
        let suit = t34 / 9;
        let num = t34 % 9;
        let next_num = if num == 8 { 0 } else { num + 1 };
        suit * 9 + next_num
    } else if (27..=30).contains(&t34) {
        // winds: 27→28→29→30→27
        if t34 == 30 {
            27
        } else {
            t34 + 1
        }
    } else {
        // dragons: 31→32→33→31
        if t34 == 33 {
            31
        } else {
            t34 + 1
        }
    }
}

fn count_meld_aka(melds: &[Meld]) -> u32 {
    let mut n = 0u32;
    for m in melds {
        for &t in &m.tiles {
            if t == 16 || t == 52 || t == 88 {
                n += 1;
            }
        }
    }
    n
}

fn might_be_yakuman(full_counts: &[u8; TILE_MAX], melds: &[Meld]) -> bool {
    // Daisangen (3 dragon kotsu — counts of 31/32/33 each ≥ 3 in the FULL hand).
    if full_counts[31] >= 3 && full_counts[32] >= 3 && full_counts[33] >= 3 {
        return true;
    }
    // Tsuuiisou (字一色: all honors).
    let any_numbered = (0..27).any(|t| full_counts[t] > 0);
    if !any_numbered {
        return true;
    }
    // Daisuushii / Shousuushii: 3+ wind kotsu OR 3 wind kotsu + wind pair.
    let wind_kotsu_count = (27..=30).filter(|&t| full_counts[t] >= 3).count();
    if wind_kotsu_count >= 3 {
        return true;
    }
    // Suuankou: 4 concealed kotsu. Conservatively flag if there are 4 closed
    // kotsu in counts (no melds besides ankan, which we already rejected).
    if melds.is_empty() {
        let n_kotsu_in_hand = (0..TILE_MAX).filter(|&t| full_counts[t] >= 3).count();
        if n_kotsu_in_hand >= 4 {
            return true;
        }
    }
    // Chinroutou: all terminals. (1m/9m/1p/9p/1s/9s only.)
    let any_non_terminal_in_numbered = (0..27).any(|t| {
        let n = t % 9;
        full_counts[t] > 0 && (1..=7).contains(&n)
    });
    let any_honor = (27..34).any(|t| full_counts[t] > 0);
    if !any_non_terminal_in_numbered && !any_honor {
        return true;
    }
    // Ryuiisou: only green tiles (2s, 3s, 4s, 6s, 8s, 6z = 32).
    let green_set: [usize; 6] = [19, 20, 21, 23, 25, 32]; // 2-3-4-6-8 in s + 發
    let any_non_green = (0..TILE_MAX).any(|t| full_counts[t] > 0 && !green_set.contains(&t));
    if !any_non_green {
        return true;
    }
    // Chuuren (only menzen single-suit 1112345678999 + extra).
    if melds.is_empty() {
        for suit in 0..3 {
            let base = suit * 9;
            // Tiles outside this suit must be 0
            let other_suit_present = (0..TILE_MAX)
                .any(|t| (t < base || t >= base + 9) && full_counts[t] > 0);
            if other_suit_present {
                continue;
            }
            // Pattern: counts[base] >= 3, counts[base+8] >= 3, others ≥ 1
            let p = &full_counts[base..base + 9];
            if p[0] >= 3 && p[8] >= 3 && (1..8).all(|i| p[i] >= 1) {
                return true;
            }
        }
    }
    false
}

#[derive(Debug, Clone, Copy)]
struct SuitUsage {
    has_m: bool,
    has_p: bool,
    has_s: bool,
    has_z: bool,
}

impl SuitUsage {
    #[allow(dead_code)]
    fn n_numbered_suits(&self) -> usize {
        self.has_m as usize + self.has_p as usize + self.has_s as usize
    }
}

/// Precomputed flags about the full 14-tile hand (in-hand + melds). Used to
/// short-circuit yaku checks that can't possibly apply.
#[derive(Debug, Clone, Copy)]
struct HandFlags {
    /// At least one tile is a terminal (1m/9m/1p/9p/1s/9s).
    has_terminal: bool,
    /// Every tile is yaocchi (terminal or honor).
    all_yaocchi: bool,
    /// Hand uses only one numbered suit (chinitsu/honitsu eligible).
    single_numbered_suit: bool,
    /// At least one dragon (haku/hatsu/chun) is present.
    /// Used at the kotsu loop's outer guard for shousangen / future yakuman gating.
    #[allow(dead_code)]
    has_dragon: bool,
}

/// Single-pass scan: compute `SuitUsage` and `HandFlags` together. Each
/// previously walked the 34-tile counts independently — fusing them halves
/// the iteration cost.
fn scan_hand(counts: &[u8; TILE_MAX]) -> (SuitUsage, HandFlags) {
    let mut su = SuitUsage {
        has_m: false,
        has_p: false,
        has_s: false,
        has_z: false,
    };
    let mut has_terminal = false;
    let mut has_simple = false;
    // Numbered suits 0..27.
    for suit in 0..3 {
        let base = suit * 9;
        let mut suit_present = false;
        for n in 0..9 {
            let c = counts[base + n];
            if c > 0 {
                suit_present = true;
                if n == 0 || n == 8 {
                    has_terminal = true;
                } else {
                    has_simple = true;
                }
            }
        }
        match suit {
            0 => su.has_m = suit_present,
            1 => su.has_p = suit_present,
            _ => su.has_s = suit_present,
        }
    }
    // Honors 27..34.
    for t in 27..34 {
        if counts[t] > 0 {
            su.has_z = true;
            break;
        }
    }
    let has_dragon = counts[31] > 0 || counts[32] > 0 || counts[33] > 0;
    let n_numbered = su.has_m as usize + su.has_p as usize + su.has_s as usize;
    let flags = HandFlags {
        has_terminal,
        all_yaocchi: !has_simple,
        single_numbered_suit: n_numbered == 1,
        has_dragon,
    };
    (su, flags)
}

/// Translate a packed `CompactDiv` (u32 with tile14-index slots) into an
/// `AbsoluteDiv` with real tile ids. The CompactDiv is a single u32 word —
/// fits in 1 register — and tile14 is already sorted ascending so the
/// kotsu/shuntsu arrays come out sorted naturally.
#[inline]
fn absolute_div_compact(
    template: agari_table::CompactDiv,
    tile14: &agari_table::Tile14,
) -> AbsoluteDiv {
    let tiles = &tile14.tiles;
    let n_k = template.n_kotsu() as usize;
    let n_s = template.n_shuntsu() as usize;
    let mut out = AbsoluteDiv {
        pair_tile: tiles[template.pair_idx() as usize],
        n_kotsu: n_k,
        kotsu: [0; 4],
        n_shuntsu: n_s,
        shuntsu: [0; 4],
    };
    for i in 0..n_k {
        out.kotsu[i] = tiles[template.kotsu_idx(i) as usize];
    }
    for i in 0..n_s {
        out.shuntsu[i] = tiles[template.shuntsu_idx(i) as usize];
    }
    out
}

#[derive(Debug, Clone, Copy)]
struct AbsoluteDiv {
    pair_tile: u8,
    n_kotsu: usize,
    kotsu: [u8; 4],
    n_shuntsu: usize,
    shuntsu: [u8; 4],
}

impl AbsoluteDiv {
    fn kotsu(&self) -> &[u8] {
        &self.kotsu[..self.n_kotsu]
    }
    fn shuntsu(&self) -> &[u8] {
        &self.shuntsu[..self.n_shuntsu]
    }
}

/// Score a single division. Returns (han, fu) or None if we don't fully cover
/// the yaku set (caller falls back).
#[allow(clippy::too_many_arguments)]
fn score_one_div(
    input: &SpInput,
    div: &AbsoluteDiv,
    _full_counts: &[u8; TILE_MAX],
    counts_13: &[u8; TILE_MAX],
    win_tile: u8,
    akas_in_hand: [bool; 3],
    assume_riichi: bool,
    bakaze: u8,
    jikaze: u8,
    _is_oya: bool,
    suit_usage: &SuitUsage,
    flags: &HandFlags,
    regular_dora: u32,
    aka_dora: u32,
    melds_kotsu: &[u8],
    melds_shuntsu: &[u8],
) -> Option<(u32, u32)> {
    let _ = (counts_13, akas_in_hand);

    // For the common menzen-no-melds case, the divs' mentsu are already the
    // full set; skip the copy into all_*_buf entirely.
    let melds_empty = melds_kotsu.is_empty() && melds_shuntsu.is_empty();
    let mut all_kotsu_buf = [0u8; 8];
    let mut all_shuntsu_buf = [0u8; 8];
    let (all_kotsu, all_shuntsu): (&[u8], &[u8]) = if melds_empty {
        (div.kotsu(), div.shuntsu())
    } else {
        let mut n_k = 0usize;
        for &k in div.kotsu() {
            all_kotsu_buf[n_k] = k;
            n_k += 1;
        }
        for &k in melds_kotsu {
            all_kotsu_buf[n_k] = k;
            n_k += 1;
        }
        let mut n_s = 0usize;
        for &s in div.shuntsu() {
            all_shuntsu_buf[n_s] = s;
            n_s += 1;
        }
        for &s in melds_shuntsu {
            all_shuntsu_buf[n_s] = s;
            n_s += 1;
        }
        (&all_kotsu_buf[..n_k], &all_shuntsu_buf[..n_s])
    };

    // Bitmasks for O(1) yaku detection. kotsu_mask: 34 bits, one per tile type.
    // shuntsu_mask: 27 bits, one per shuntsu starting tile (0..=24, so bits 25,26 always 0).
    let mut kotsu_mask: u64 = 0;
    for &k in all_kotsu {
        kotsu_mask |= 1u64 << k;
    }
    let mut shuntsu_mask: u32 = 0;
    for &s in all_shuntsu {
        shuntsu_mask |= 1u32 << s;
    }

    let pair_tile = div.pair_tile;

    // ---------- Yaku: tanyao ---------- (impossible if any yaocchi present)
    let tanyao = !flags.has_terminal && !suit_usage.has_z;

    // ---------- Yaku: honitsu / chinitsu ---------- (need single numbered suit)
    let chinitsu = flags.single_numbered_suit && !suit_usage.has_z;
    let honitsu = flags.single_numbered_suit && suit_usage.has_z;

    // ---------- Yaku: honroutou ---------- (need all yaocchi)
    let honroutou = flags.all_yaocchi
        && all_shuntsu.is_empty()
        && all_kotsu.iter().all(|&t| is_yaocchi(t))
        && is_yaocchi(pair_tile);

    // ---------- Yaku: junchan / chanta ---------- (need every mentsu yaocchi)
    let (junchan, chanta) = if !flags.has_terminal {
        (false, false)
    } else {
        let mentsu_all_have_terminal = all_kotsu.iter().all(|&t| is_terminal(t))
            && all_shuntsu.iter().all(|&t| t % 9 == 0 || t % 9 == 6)
            && is_terminal(pair_tile);
        let mentsu_all_have_yaocchi = all_kotsu.iter().all(|&t| is_yaocchi(t))
            && all_shuntsu.iter().all(|&t| t % 9 == 0 || t % 9 == 6)
            && is_yaocchi(pair_tile);
        let j = !honroutou
            && !all_shuntsu.is_empty()
            && !suit_usage.has_z  // junchan = no honors
            && mentsu_all_have_terminal;
        let c = !honroutou && !j && !all_shuntsu.is_empty() && mentsu_all_have_yaocchi;
        (j, c)
    };

    // ---------- Yaku: ittsu / sanshoku doujun ---------- (bitmask-based O(1))
    let smask_m = shuntsu_mask & 0x1FF; // bits 0..8 (m suit shuntsu starts 0..=8)
    let smask_p = (shuntsu_mask >> 9) & 0x1FF;
    let smask_s = (shuntsu_mask >> 18) & 0x1FF;
    // Ittsu: starts 0, 3, 6 (1-3, 4-6, 7-9) all present in same suit.
    const ITTSU_PATTERN: u32 = (1 << 0) | (1 << 3) | (1 << 6);
    let ittsu = (smask_m & ITTSU_PATTERN) == ITTSU_PATTERN
        || (smask_p & ITTSU_PATTERN) == ITTSU_PATTERN
        || (smask_s & ITTSU_PATTERN) == ITTSU_PATTERN;
    // Sanshoku doujun: same shuntsu start in all 3 numbered suits.
    let sanshoku_doujun = (smask_m & smask_p & smask_s) != 0;

    // ---------- Yaku: sanshoku doukou ---------- (bitmask-based O(1))
    let kmask_m = (kotsu_mask & 0x1FF) as u32;
    let kmask_p = ((kotsu_mask >> 9) & 0x1FF) as u32;
    let kmask_s = ((kotsu_mask >> 18) & 0x1FF) as u32;
    let sanshoku_doukou = (kmask_m & kmask_p & kmask_s) != 0;

    // ---------- Yaku: toitoi ----------
    let toitoi = all_shuntsu.is_empty();

    // ---------- Yaku: sanankou (≥ 3 closed kotsu in tsumo)
    let n_ankou = div.n_kotsu;
    let sanankou = n_ankou >= 3;

    // ---------- Yaku: shousangen ---------- (bitmask-based O(1))
    let dragon_kotsu_bits = (kotsu_mask >> 31) & 0b111; // bits for 31, 32, 33
    let dragon_kotsu_count = dragon_kotsu_bits.count_ones() as usize;
    let dragon_pair = (31..=33).contains(&pair_tile);
    let shousangen = dragon_kotsu_count == 2 && dragon_pair;

    // ---------- Yaku: yakuhai ----------
    let mut yakuhai_han = 0u32;
    for &k in all_kotsu {
        // White, Green, Red dragons
        if matches!(k, 31 | 32 | 33) {
            yakuhai_han += 1;
        } else if k == bakaze {
            yakuhai_han += 1;
        }
        if k == jikaze && k != bakaze {
            // Avoid double-counting when bakaze == jikaze (連風牌 is +2 han via two yakuhai)
            // Wait — yakuhai counts each kotsu separately. If bakaze==jikaze and the player
            // has a kotsu of that wind, it's 2 han (one for bakaze, one for jikaze).
            // The check `k == bakaze` above already added 1; this adds the 2nd.
            yakuhai_han += 1;
        } else if k == jikaze && k == bakaze {
            yakuhai_han += 1; // 2nd han for 連風牌
        }
    }
    let _ = ID_BASE_HAN_OFFSET;

    // ---------- Yaku: iipeikou / ryanpeikou (menzen only) ----------
    let (iipeikou, ryanpeikou) = if input.is_menzen {
        detect_peikou(div.shuntsu())
    } else {
        (false, false)
    };

    // ---------- Wait placement enumeration (高点法) ----------
    let pair_is_yakuhai = is_yakuhai_pair(pair_tile, bakaze, jikaze);
    let mut wait_buf = [WaitKind::Tanki; 5];
    let n_waits = classify_waits(div, win_tile, &mut wait_buf);
    if n_waits == 0 {
        // win_tile not in this division — caller never reaches here for valid agari.
        return None;
    }

    // ---------- Yaku tally common across placements ----------
    let mut han: u32 = 0;
    // Han contributions independent of wait placement (pinfu is added per-placement).
    if input.is_menzen {
        han += 1; // menzen tsumo
    }
    if assume_riichi {
        han += 1;
    }
    han += yakuhai_han;
    if tanyao {
        han += 1;
    }
    if iipeikou && !ryanpeikou {
        han += 1;
    }
    if ryanpeikou {
        han += 3;
    }
    if sanshoku_doujun {
        han += if input.is_menzen { 2 } else { 1 };
    }
    if sanshoku_doukou {
        han += 2;
    }
    if ittsu {
        han += if input.is_menzen { 2 } else { 1 };
    }
    if toitoi {
        han += 2;
    }
    if sanankou {
        han += 2;
    }
    if shousangen {
        han += 2;
    }
    if honroutou {
        han += 2;
    }
    if junchan {
        han += if input.is_menzen { 3 } else { 2 };
    } else if chanta {
        han += if input.is_menzen { 2 } else { 1 };
    }
    if chinitsu {
        han += if input.is_menzen { 6 } else { 5 };
    } else if honitsu {
        han += if input.is_menzen { 3 } else { 2 };
    }
    // Dora
    han += regular_dora + aka_dora;

    let pinfu_eligible = input.is_menzen
        && div.n_shuntsu == 4
        && div.n_kotsu == 0
        && melds_kotsu.is_empty()
        && melds_shuntsu.is_empty()
        && !pair_is_yakuhai;

    // Yaku presence check (without pinfu, which depends on wait kind).
    let non_pinfu_yaku = (input.is_menzen)
        || assume_riichi
        || yakuhai_han > 0
        || tanyao
        || iipeikou
        || ryanpeikou
        || sanshoku_doujun
        || sanshoku_doukou
        || ittsu
        || toitoi
        || sanankou
        || shousangen
        || honroutou
        || junchan
        || chanta
        || chinitsu
        || honitsu;

    // Iterate every wait placement and pick the best-scoring (han, fu).
    let mut best: Option<(u32, u32)> = None;
    for i in 0..n_waits {
        let wk = wait_buf[i];
        let pinfu = pinfu_eligible && wk == WaitKind::Ryanmen;
        let placement_han = han + if pinfu { 1 } else { 0 };
        if !non_pinfu_yaku && !pinfu {
            // No yaku at all in this placement — not a valid win.
            continue;
        }
        let fu = compute_fu(
            div,
            &melds_kotsu,
            &melds_shuntsu,
            bakaze,
            jikaze,
            wk,
            pinfu,
            input.is_menzen,
        );
        match best {
            None => best = Some((placement_han, fu)),
            Some((bh, bf)) => {
                if score_lt(bh, bf, placement_han, fu, false) {
                    best = Some((placement_han, fu));
                }
            }
        }
    }
    best.or(Some((0, 0)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WaitKind {
    Tanki,
    Shanpon,
    Ryanmen,
    Kanchan,
    Penchan,
}

/// Enumerate every valid placement of `win_tile` within `div`. The same
/// (counts_14, win_tile) shape can support multiple wait kinds — the highest-
/// scoring one wins (高点法). Returns at most 4 placements (the win can land
/// in at most one mentsu per slot).
fn classify_waits(div: &AbsoluteDiv, win_tile: u8, out: &mut [WaitKind; 5]) -> usize {
    let mut n = 0usize;
    if win_tile == div.pair_tile {
        out[n] = WaitKind::Tanki;
        n += 1;
    }
    for &k in div.kotsu() {
        if k == win_tile {
            out[n] = WaitKind::Shanpon;
            n += 1;
        }
    }
    for &start in div.shuntsu() {
        if win_tile >= start && win_tile <= start + 2 {
            let pos = win_tile - start;
            let suit = start / 9;
            let in_suit_start = start - suit * 9;
            let kind = if pos == 1 {
                WaitKind::Kanchan
            } else if (in_suit_start == 0 && pos == 2) || (in_suit_start == 6 && pos == 0) {
                WaitKind::Penchan
            } else {
                WaitKind::Ryanmen
            };
            out[n] = kind;
            n += 1;
        }
    }
    n
}

fn is_yakuhai_pair(t: u8, bakaze: u8, jikaze: u8) -> bool {
    matches!(t, 31 | 32 | 33) || t == bakaze || t == jikaze
}

fn detect_peikou(in_hand_shuntsu: &[u8]) -> (bool, bool) {
    // Count duplicate shuntsu within the in-hand mentsu (open chi don't count for iipeikou).
    let mut sorted = [0u8; 4];
    let n = in_hand_shuntsu.len();
    sorted[..n].copy_from_slice(in_hand_shuntsu);
    sorted[..n].sort_unstable();
    let mut pairs = 0;
    let mut i = 0;
    while i + 1 < n {
        if sorted[i] == sorted[i + 1] {
            pairs += 1;
            i += 2;
        } else {
            i += 1;
        }
    }
    (pairs >= 1 && pairs < 2, pairs >= 2)
}

/// Decompose `melds` into stack buffers for kotsu (pons) and shuntsu (chis).
/// Returns `(n_kotsu, n_shuntsu)`. Each buffer holds at most 4 entries (mahjong
/// allows ≤ 4 melds total). Kans and special meld types are ignored — caller
/// must reject these before invoking lean.
fn collect_melds(
    melds: &[Meld],
    kotsu: &mut [u8; 4],
    shuntsu: &mut [u8; 4],
) -> (usize, usize) {
    let mut nk = 0usize;
    let mut ns = 0usize;
    for m in melds {
        match m.meld_type {
            MeldType::Pon => {
                kotsu[nk] = m.tiles[0] / 4;
                nk += 1;
            }
            MeldType::Chi => {
                let lowest = m.tiles.iter().min().copied().unwrap_or(0) / 4;
                shuntsu[ns] = lowest;
                ns += 1;
            }
            _ => {}
        }
    }
    (nk, ns)
}

#[allow(clippy::too_many_arguments)]
fn compute_fu(
    div: &AbsoluteDiv,
    melds_kotsu: &[u8],
    melds_shuntsu: &[u8],
    bakaze: u8,
    jikaze: u8,
    wait_kind: WaitKind,
    pinfu: bool,
    is_menzen: bool,
) -> u32 {
    if pinfu {
        // Tsumo pinfu fu = 20 (no extra base, no tsumo bonus, no wait fu).
        return 20;
    }

    let mut fu: u32 = 20;
    // Tsumo bonus
    fu += 2;

    // Closed kotsu (ankou): 4 (simple) / 8 (yaocchi)
    for &t in div.kotsu() {
        fu += if is_yaocchi(t) { 8 } else { 4 };
    }
    // Open kotsu (pon): 2 / 4
    for &t in melds_kotsu {
        fu += if is_yaocchi(t) { 4 } else { 2 };
    }
    // Pair
    if matches!(div.pair_tile, 31 | 32 | 33) {
        fu += 2;
    } else if div.pair_tile == bakaze {
        fu += 2;
        if div.pair_tile == jikaze {
            fu += 2; // 連風牌 (we use Tenhou's "連風牌は4符" rule)
        }
    } else if div.pair_tile == jikaze {
        fu += 2;
    }
    // Wait fu
    fu += match wait_kind {
        WaitKind::Tanki | WaitKind::Kanchan | WaitKind::Penchan => 2,
        WaitKind::Ryanmen | WaitKind::Shanpon => 0,
    };

    let _ = melds_shuntsu;

    // Open hand: no kuipinfu floor needed for tsumo (we already added the tsumo
    // +2 bonus). However an open hand with NO fu sources (only shuntsu chi +
    // ryanmen wait + non-yakuhai pair + tsumo) has fu = 22. After rounding up
    // to next 10 → 30. That's the kuipinfu rule baked into the rounding.
    if is_menzen && wait_kind != WaitKind::Tanki && wait_kind != WaitKind::Kanchan
        && wait_kind != WaitKind::Penchan
    {
        // menzen ron 10 fu — N/A for tsumo. Skip.
    }

    // Round up to next 10
    fu = fu.div_ceil(10) * 10;
    fu
}

const _: usize = MAX_DIVS_PER_KEY; // keep import alive
