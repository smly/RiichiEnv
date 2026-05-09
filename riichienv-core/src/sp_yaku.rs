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
//!   - Open hands without kans.
//!   - Aka dora + regular dora.
//!
//! Out of scope (returns `None`, falls back):
//!   - Yakuman shapes (caller's HandEvaluator path handles these).
//!   - Chitoitsu / Kokushi (not in `agari_table`).
//!   - Hands containing kans (kan fu / suukantsu).
//!   - Ura dora distribution (caller computes it on top of our base score).
//!
//! Numerical equivalence is verified against the legacy path on every real-replay
//! SP input via `debug_only_mortal_sp::harness::lean_vs_legacy_match`.

use crate::agari_table::{self, Division, MAX_DIVS_PER_KEY};
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

    // Look up decompositions from the precomputed table.
    let (list, offsets, sp_perm, hp) = agari_table::lookup_canonical(&counts_14)?;
    if list.n == 0 {
        return None;
    }

    // Quick yakuman shape detection: defer to the legacy path so we never
    // produce a wrong score by missing yakuman.
    if might_be_yakuman(&full_counts, &input.melds) {
        return None;
    }

    let assume_riichi = input.is_menzen && input.can_riichi;
    // Total yaocchi count flags
    let only_terminals_in_hand = is_chinroutou(&full_counts);
    let _ = only_terminals_in_hand;
    // For honitsu/chinitsu detection
    let suit_usage = SuitUsage::from(&full_counts);

    // Aggregate dora count (regular indicators -> next tile + aka).
    let regular_dora = count_regular_dora(input, &full_counts);
    // Mirror the legacy path's aka semantics: only count an in-hand red if the
    // corresponding deakaized 5x is actually present in counts_13. (When the
    // win tile *is* the red 5x but counts_13 has 0 of that tile, the legacy
    // encoder drops the red bit because it can't be attached to any in-hand
    // tile.)
    let red_present = |t34: usize| counts_13[t34] >= 1;
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

    // Hand-level structural flags — precomputed once so the per-div yaku
    // checks can early-exit on patterns that cannot apply.
    let flags = HandFlags::compute(&full_counts, &suit_usage);

    // For each table-decomp + the open-meld portion, compute (han, fu).
    let mut best: Option<(u32, u32)> = None; // (han, fu) — pick by total score
    for k in 0..list.n as usize {
        let template = &list.divs[k];
        let div = absolute_div(template, &offsets, &sp_perm, &hp);

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

    let (han, fu) = best?;
    Some(LeanScore {
        han,
        fu,
        yakuman: false,
    })
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

fn is_chinroutou(full_counts: &[u8; TILE_MAX]) -> bool {
    for t in 0..27 {
        let n = t % 9;
        if full_counts[t] > 0 && (1..=7).contains(&n) {
            return false;
        }
    }
    for t in 27..34 {
        if full_counts[t] > 0 {
            return false;
        }
    }
    true
}

#[derive(Debug, Clone, Copy)]
struct SuitUsage {
    has_m: bool,
    has_p: bool,
    has_s: bool,
    has_z: bool,
}

impl SuitUsage {
    fn from(counts: &[u8; TILE_MAX]) -> Self {
        Self {
            has_m: (0..9).any(|t| counts[t] > 0),
            has_p: (9..18).any(|t| counts[t] > 0),
            has_s: (18..27).any(|t| counts[t] > 0),
            has_z: (27..34).any(|t| counts[t] > 0),
        }
    }
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
    /// Every tile is a terminal (no simples, no honors). (Unused for yaku
    /// decisions today but useful as a documented predicate for future
    /// chinroutou handling.)
    #[allow(dead_code)]
    all_terminal: bool,
    /// Hand uses only one numbered suit (chinitsu/honitsu eligible).
    single_numbered_suit: bool,
    /// At least one dragon (haku/hatsu/chun) is present.
    has_dragon: bool,
}

impl HandFlags {
    fn compute(counts: &[u8; TILE_MAX], suit_usage: &SuitUsage) -> Self {
        let mut has_terminal = false;
        let mut has_simple = false;
        for t in 0..27 {
            if counts[t] > 0 {
                let n = t % 9;
                if n == 0 || n == 8 {
                    has_terminal = true;
                } else {
                    has_simple = true;
                }
            }
        }
        let has_dragon = counts[31] > 0 || counts[32] > 0 || counts[33] > 0;
        let all_yaocchi = !has_simple;
        let all_terminal = !has_simple && !suit_usage.has_z;
        Self {
            has_terminal,
            all_yaocchi,
            all_terminal,
            single_numbered_suit: suit_usage.n_numbered_suits() == 1,
            has_dragon,
        }
    }
}

/// Translate a canonical-form `Division` from `agari_table` into one with
/// absolute tile ids (using the per-suit shifts and permutations).
fn absolute_div(
    canonical: &Division,
    offsets: &[u8; 4],
    sp_perm: &[u8; 3],
    hp: &[u8; 7],
) -> AbsoluteDiv {
    let mut out = AbsoluteDiv {
        pair_tile: agari_table::apply_offset_perm(canonical.pair_tile, offsets, sp_perm, hp),
        n_kotsu: canonical.n_kotsu as usize,
        kotsu: [0; 4],
        n_shuntsu: canonical.n_shuntsu as usize,
        shuntsu: [0; 4],
    };
    for i in 0..out.n_kotsu {
        out.kotsu[i] =
            agari_table::apply_offset_perm(canonical.kotsu_tiles[i], offsets, sp_perm, hp);
    }
    for i in 0..out.n_shuntsu {
        out.shuntsu[i] =
            agari_table::apply_offset_perm(canonical.shuntsu_starts[i], offsets, sp_perm, hp);
    }
    // Sort for canonical comparison
    let n_k = out.n_kotsu;
    out.kotsu[..n_k].sort_unstable();
    let n_s = out.n_shuntsu;
    out.shuntsu[..n_s].sort_unstable();
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

    // All kotsu (closed in-hand + open from melds)
    let mut all_kotsu_buf = [0u8; 8];
    let mut n_all_kotsu = 0usize;
    for &k in div.kotsu() {
        all_kotsu_buf[n_all_kotsu] = k;
        n_all_kotsu += 1;
    }
    for &k in melds_kotsu {
        all_kotsu_buf[n_all_kotsu] = k;
        n_all_kotsu += 1;
    }
    // All shuntsu (closed in-hand + open from chi melds)
    let mut all_shuntsu_buf = [0u8; 8];
    let mut n_all_shuntsu = 0usize;
    for &s in div.shuntsu() {
        all_shuntsu_buf[n_all_shuntsu] = s;
        n_all_shuntsu += 1;
    }
    for &s in melds_shuntsu {
        all_shuntsu_buf[n_all_shuntsu] = s;
        n_all_shuntsu += 1;
    }
    let all_kotsu = &all_kotsu_buf[..n_all_kotsu];
    let all_shuntsu = &all_shuntsu_buf[..n_all_shuntsu];

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

    // ---------- Yaku: ittsu ---------- (need at least 3 shuntsu)
    let ittsu = all_shuntsu.len() >= 3 && detect_ittsu(all_shuntsu);

    // ---------- Yaku: sanshoku doujun ---------- (need ≥3 shuntsu)
    let sanshoku_doujun = all_shuntsu.len() >= 3 && detect_sanshoku_doujun(all_shuntsu);

    // ---------- Yaku: sanshoku doukou ---------- (need ≥3 kotsu)
    let sanshoku_doukou = all_kotsu.len() >= 3 && detect_sanshoku_doukou(all_kotsu);

    // ---------- Yaku: toitoi ----------
    let toitoi = all_shuntsu.is_empty();

    // ---------- Yaku: sanankou (≥ 3 closed kotsu in tsumo)
    let n_ankou = div.n_kotsu;
    let sanankou = n_ankou >= 3;

    // ---------- Yaku: shousangen ---------- (needs dragons)
    let (dragon_kotsu_count, dragon_pair) = if flags.has_dragon {
        (
            all_kotsu.iter().filter(|&&t| (31..=33).contains(&t)).count(),
            (31..=33).contains(&pair_tile),
        )
    } else {
        (0, false)
    };
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

fn detect_ittsu(all_shuntsu: &[u8]) -> bool {
    for suit in 0..3u8 {
        let base = suit * 9;
        if all_shuntsu.contains(&base)
            && all_shuntsu.contains(&(base + 3))
            && all_shuntsu.contains(&(base + 6))
        {
            return true;
        }
    }
    false
}

fn detect_sanshoku_doujun(all_shuntsu: &[u8]) -> bool {
    for start in 0..7u8 {
        if all_shuntsu.contains(&start)
            && all_shuntsu.contains(&(start + 9))
            && all_shuntsu.contains(&(start + 18))
        {
            return true;
        }
    }
    false
}

fn detect_sanshoku_doukou(all_kotsu: &[u8]) -> bool {
    for n in 0..9u8 {
        if all_kotsu.contains(&n)
            && all_kotsu.contains(&(n + 9))
            && all_kotsu.contains(&(n + 18))
        {
            return true;
        }
    }
    false
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
