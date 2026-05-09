//! Generates `riichienv-core/src/data/agari_table.bin`.
//!
//! Standard 14-tile (one pair + four mentsu) winning shapes only.
//! For each unique 34-tile counts vector, store the list of valid pair+mentsu
//! decompositions. Run with:
//!
//!   cargo run --release --bin build_agari_table
//!
//! The enumeration strategy: build per-suit "shape catalogues" indexed by
//! `(total_tiles, contains_pair, decompositions)`, then combine across the
//! three numbered suits + honors with pair-position bookkeeping. This avoids
//! the 5^34 brute-force tile-count enumeration.
//!
//! Output is checked into the repo. Re-run only when the encoding changes.
//!
//! NOTE: The per-suit enumerator is independently re-derived from the
//! algorithm description (山岡忠夫's mahjong score algorithm) and does not copy
//! code from any other implementation. The output is purely combinatorial
//! data (decompositions of mahjong winning shapes), itself uncopyrightable.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use riichienv_core::agari_table::{
    DIVISION_BYTES, Division, canonicalize_full, encode_division, key_from_counts,
};

const MAX_MENTSU_PER_SUIT: usize = 4;
const PAIR_TILE_OFFSETS: [u8; 4] = [0, 9, 18, 27]; // m, p, s, z

#[derive(Clone, Debug)]
struct SuitShape {
    /// Per-tile counts within this suit (length 9 for numbered, 7 for honors).
    counts: Vec<u8>,
    /// Optional pair-tile index within this suit. Length 1 if has pair, 0 otherwise.
    pair_idx: Option<u8>,
    /// Kotsu (triplet) tile indices within this suit.
    kotsu: Vec<u8>,
    /// Shuntsu starting tile indices within this suit (only for numbered suits).
    shuntsu: Vec<u8>,
}

impl SuitShape {
    fn empty(width: usize) -> Self {
        Self {
            counts: vec![0; width],
            pair_idx: None,
            kotsu: Vec::new(),
            shuntsu: Vec::new(),
        }
    }

    fn n_mentsu(&self) -> usize {
        self.kotsu.len() + self.shuntsu.len()
    }
}

fn main() {
    // Step 1: enumerate per-suit decompositions for numbered suits and honors.
    // For each suit, build a Vec<SuitShape>. We allow shapes with 0..=4 mentsu
    // and 0..=1 pair. Later we filter by total = 14 and exactly 1 pair across
    // all four suits combined.
    let numbered_shapes = enumerate_numbered_suit_shapes();
    let honor_shapes = enumerate_honor_shapes();
    println!(
        "numbered suit raw shapes: {} | honor shapes: {}",
        numbered_shapes.len(),
        honor_shapes.len()
    );

    // Step 2: bucket per-suit shapes by (total_tiles, has_pair) so we can
    // cross-combine only compatible buckets. Across all four suits the totals
    // must sum to 14 and exactly one suit holds the pair (which means one
    // suit has total ≡ 2 mod 3 and the others ≡ 0 mod 3).
    let mut numbered_buckets: HashMap<(usize, bool), Vec<&SuitShape>> = HashMap::new();
    for sh in &numbered_shapes {
        let total = sh.counts.iter().map(|&c| c as usize).sum();
        numbered_buckets
            .entry((total, sh.pair_idx.is_some()))
            .or_default()
            .push(sh);
    }
    let mut honor_buckets: HashMap<(usize, bool), Vec<&SuitShape>> = HashMap::new();
    for sh in &honor_shapes {
        let total = sh.counts.iter().map(|&c| c as usize).sum();
        honor_buckets
            .entry((total, sh.pair_idx.is_some()))
            .or_default()
            .push(sh);
    }

    let mut hand_table: HashMap<u128, Vec<Division>> = HashMap::new();

    // Enumerate suit total assignments: each suit gets a total in 0..=14, sum=14,
    // exactly one suit has the pair. Numbered suits: pair-bearing total ∈ {2,5,8,11,14},
    // non-pair total ∈ {0,3,6,9,12}. Honor suit can also bear the pair.
    let pair_totals = [2usize, 5, 8, 11, 14];
    let no_pair_totals = [0usize, 3, 6, 9, 12];
    // For honors, with no shuntsu, valid totals: 0,2,3,5,6,8,9,11,12 (combinations
    // of pair (2) and kotsu (3) up to 4 honors). The buckets will simply be empty
    // for invalid totals, so we can iterate all 0..=14 and let the bucket lookup
    // filter.
    let z_pair_totals: Vec<usize> = (0..=14).filter(|t| t % 3 == 2).collect();
    let z_no_pair_totals: Vec<usize> = (0..=14).filter(|t| t % 3 == 0).collect();

    // Iterate over which suit bears the pair: m, p, s, or z (4 choices).
    for pair_suit in 0..4 {
        // For each (m_total, p_total, s_total, z_total) summing to 14 with
        // appropriate pair/no-pair bucket constraints.
        let m_totals: &[usize] = if pair_suit == 0 { &pair_totals } else { &no_pair_totals };
        let p_totals: &[usize] = if pair_suit == 1 { &pair_totals } else { &no_pair_totals };
        let s_totals: &[usize] = if pair_suit == 2 { &pair_totals } else { &no_pair_totals };
        let z_totals: &[usize] = if pair_suit == 3 { &z_pair_totals } else { &z_no_pair_totals };

        for &mt in m_totals {
            if mt > 14 {
                continue;
            }
            for &pt in p_totals {
                if mt + pt > 14 {
                    continue;
                }
                for &st in s_totals {
                    if mt + pt + st > 14 {
                        continue;
                    }
                    let zt_needed = 14 - mt - pt - st;
                    if !z_totals.contains(&zt_needed) {
                        continue;
                    }

                    let m_bucket = numbered_buckets.get(&(mt, pair_suit == 0));
                    let p_bucket = numbered_buckets.get(&(pt, pair_suit == 1));
                    let s_bucket = numbered_buckets.get(&(st, pair_suit == 2));
                    let z_bucket = honor_buckets.get(&(zt_needed, pair_suit == 3));
                    let (Some(mb), Some(pb), Some(sb), Some(zb)) = (m_bucket, p_bucket, s_bucket, z_bucket) else {
                        continue;
                    };

                    for m in mb {
                        for p in pb {
                            for s in sb {
                                for z in zb {
                                    let total_mentsu =
                                        m.n_mentsu() + p.n_mentsu() + s.n_mentsu() + z.n_mentsu();
                                    if total_mentsu != 4 {
                                        continue;
                                    }
                                    add_combo(m, p, s, z, &mut hand_table);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    println!("unique 14-tile shapes with valid decomp: {}", hand_table.len());
    let max_divs = hand_table.values().map(|v| v.len()).max().unwrap_or(0);
    let total_divs: usize = hand_table.values().map(|v| v.len()).sum();
    println!("max divs per shape: {}, total divs: {}", max_divs, total_divs);
    if max_divs > 4 {
        // Print one example for diagnosis.
        for (_, v) in hand_table.iter() {
            if v.len() == max_divs {
                println!("  example with {} divs: {:?}", max_divs, v);
                break;
            }
        }
        panic!("max_divs {} exceeds MAX_DIVS_PER_KEY=4", max_divs);
    }

    // Sort entries by key for determinism.
    let mut entries: Vec<(u128, Vec<Division>)> = hand_table.into_iter().collect();
    entries.sort_by_key(|(k, _)| *k);
    for (_, divs) in entries.iter_mut() {
        divs.sort_unstable();
    }

    // Encode: [u128 LE key] [u8 n_div] [n_div * DIVISION_BYTES]
    let mut raw: Vec<u8> = Vec::with_capacity(entries.len() * 32);
    for (key, divs) in &entries {
        raw.extend_from_slice(&key.to_le_bytes());
        raw.push(divs.len() as u8);
        for d in divs {
            encode_division(d, &mut raw);
        }
    }
    println!("raw encoded size: {} bytes", raw.len());

    let out_path: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("data")
        .join("agari_table.bin");
    let mut f = File::create(&out_path).expect("create output");
    f.write_all(&raw).expect("write");
    drop(f);
    let final_size = std::fs::metadata(&out_path).expect("stat").len();
    println!("wrote {} ({} bytes)", out_path.display(), final_size);

    let expected_raw: usize =
        entries.iter().map(|(_, d)| 16 + 1 + d.len() * DIVISION_BYTES).sum();
    assert_eq!(expected_raw, raw.len());
}

fn add_combo(
    m: &SuitShape,
    p: &SuitShape,
    s: &SuitShape,
    z: &SuitShape,
    hand_table: &mut HashMap<u128, Vec<Division>>,
) {
    let mut counts = [0u8; 34];
    counts[0..9].copy_from_slice(&m.counts);
    counts[9..18].copy_from_slice(&p.counts);
    counts[18..27].copy_from_slice(&s.counts);
    counts[27..34].copy_from_slice(&z.counts);

    let pair_tile = pair_tile_global(m, p, s, z);
    let mut div = Division::new(pair_tile);
    let mut all_kotsu: Vec<u8> = Vec::new();
    let mut all_shuntsu: Vec<u8> = Vec::new();
    for (offset, suit) in [(0u8, m), (9, p), (18, s)] {
        for &k in &suit.kotsu {
            all_kotsu.push(k + offset);
        }
        for &sh in &suit.shuntsu {
            all_shuntsu.push(sh + offset);
        }
    }
    for &k in &z.kotsu {
        all_kotsu.push(k + 27);
    }
    all_kotsu.sort_unstable();
    all_shuntsu.sort_unstable();
    for &k in &all_kotsu {
        div.push_kotsu(k);
    }
    for &sh in &all_shuntsu {
        div.push_shuntsu(sh);
    }

    // Full canonicalize: shift each numbered suit, permute numbered suits, AND
    // permute honors. Translate the division's tile ids to the same canonical
    // form (sorted-suit, sorted-honor coordinates).
    let (canon, offsets, suit_perm, honor_perm) = canonicalize_full(&counts);
    let mut inv_suit = [0u8; 3];
    for (sorted_idx, &orig) in suit_perm.iter().enumerate() {
        inv_suit[orig as usize] = sorted_idx as u8;
    }
    let mut inv_honor = [0u8; 7];
    for (sorted_idx, &orig) in honor_perm.iter().enumerate() {
        inv_honor[orig as usize] = sorted_idx as u8;
    }
    let xlate = |t: u8| -> u8 {
        if t >= 27 {
            let orig_pos = (t - 27) as usize;
            return 27 + inv_honor[orig_pos];
        }
        let orig_suit = (t / 9) as usize;
        let pos = t % 9;
        let shifted_pos = pos - offsets[orig_suit];
        let sorted_suit = inv_suit[orig_suit] as usize;
        (sorted_suit as u8) * 9 + shifted_pos
    };
    div.pair_tile = xlate(div.pair_tile);
    for i in 0..div.n_kotsu as usize {
        div.kotsu_tiles[i] = xlate(div.kotsu_tiles[i]);
    }
    for i in 0..div.n_shuntsu as usize {
        div.shuntsu_starts[i] = xlate(div.shuntsu_starts[i]);
    }
    // Re-sort kotsu/shuntsu after translation.
    let mut k = div.kotsu_tiles;
    k[..div.n_kotsu as usize].sort_unstable();
    div.kotsu_tiles = k;
    let mut s = div.shuntsu_starts;
    s[..div.n_shuntsu as usize].sort_unstable();
    div.shuntsu_starts = s;

    let key = key_from_counts(&canon);
    let entry = hand_table.entry(key).or_insert_with(Vec::new);
    if !entry.contains(&div) {
        entry.push(div);
    }
}

fn pair_tile_global(m: &SuitShape, p: &SuitShape, s: &SuitShape, z: &SuitShape) -> u8 {
    if let Some(idx) = m.pair_idx {
        return idx + PAIR_TILE_OFFSETS[0];
    }
    if let Some(idx) = p.pair_idx {
        return idx + PAIR_TILE_OFFSETS[1];
    }
    if let Some(idx) = s.pair_idx {
        return idx + PAIR_TILE_OFFSETS[2];
    }
    if let Some(idx) = z.pair_idx {
        return idx + PAIR_TILE_OFFSETS[3];
    }
    panic!("no pair?");
}

/// Enumerate every (decomposable) numbered-suit shape: 0..=4 mentsu and 0..=1
/// pair. Output is one entry per (count_vec, decomposition) — the same
/// count_vec may appear multiple times with different decompositions, which
/// is what we want when later combining across suits.
fn enumerate_numbered_suit_shapes() -> Vec<SuitShape> {
    let mut out = Vec::new();
    // Mentsu choices for one suit: kotsu at i (0..9) and shuntsu starting at i (0..7).
    // Up to 4 mentsu from a multiset of these 16 choices, with replacement, sorted.
    // We iterate sorted (a <= b <= c <= d) by index in the choice list.
    let mut choices: Vec<(bool, u8)> = Vec::new();
    for i in 0..9u8 {
        choices.push((true, i)); // kotsu
    }
    for i in 0..7u8 {
        choices.push((false, i)); // shuntsu
    }
    let n_choices = choices.len();
    // n_mentsu = 0..=4
    for n_mentsu in 0..=MAX_MENTSU_PER_SUIT {
        // Sorted multi-index combinations.
        if n_mentsu == 0 {
            // No mentsu shape (just maybe a pair or empty).
            push_with_optional_pair(&SuitShape::empty(9), &mut out);
            continue;
        }
        let mut idx = vec![0usize; n_mentsu];
        loop {
            // Build counts
            let mut counts = [0u8; 9];
            let mut kotsu = Vec::new();
            let mut shuntsu = Vec::new();
            let mut overflow = false;
            for &k in &idx {
                let (is_kotsu, t) = choices[k];
                if is_kotsu {
                    counts[t as usize] += 3;
                    if counts[t as usize] > 4 {
                        overflow = true;
                        break;
                    }
                    kotsu.push(t);
                } else {
                    counts[t as usize] += 1;
                    counts[(t + 1) as usize] += 1;
                    counts[(t + 2) as usize] += 1;
                    if counts[t as usize] > 4
                        || counts[(t + 1) as usize] > 4
                        || counts[(t + 2) as usize] > 4
                    {
                        overflow = true;
                        break;
                    }
                    shuntsu.push(t);
                }
            }
            if !overflow {
                let shape = SuitShape {
                    counts: counts.to_vec(),
                    pair_idx: None,
                    kotsu,
                    shuntsu,
                };
                push_with_optional_pair(&shape, &mut out);
            }
            // Increment sorted multi-index
            if !next_sorted_multi_index(&mut idx, n_choices) {
                break;
            }
        }
    }
    out
}

fn push_with_optional_pair(base: &SuitShape, out: &mut Vec<SuitShape>) {
    out.push(base.clone()); // no pair
    // try each tile as pair
    for p in 0..9u8 {
        let mut shape = base.clone();
        shape.counts[p as usize] += 2;
        if shape.counts[p as usize] > 4 {
            continue;
        }
        shape.pair_idx = Some(p);
        out.push(shape);
    }
}

/// Honor enumeration: kotsu of any honor tile (0..7), pair of any honor tile.
/// No shuntsu.
fn enumerate_honor_shapes() -> Vec<SuitShape> {
    let mut out = Vec::new();
    // Choose up to 4 distinct honors as kotsu (kotsu of same honor exhausts the 4 copies anyway,
    // and one of them as pair would need 5 copies — impossible).
    // For each subset of 0..=4 honors, that subset becomes kotsu.
    let honors: Vec<u8> = (0..7u8).collect();
    for n_kotsu in 0..=MAX_MENTSU_PER_SUIT {
        // Choose n_kotsu honors out of 7 (since same honor can't appear twice as kotsu, and pair-of-honor + kotsu-of-same is also impossible: 3+2=5 > 4).
        let mut sel = vec![0usize; n_kotsu];
        if n_kotsu == 0 {
            push_honor_shape_with_optional_pair(&honors, &[], &mut out);
        } else {
            // Initialize sel to [0,1,2,...,n_kotsu-1]
            for (k, v) in sel.iter_mut().enumerate() {
                *v = k;
            }
            loop {
                push_honor_shape_with_optional_pair(&honors, &sel, &mut out);
                if !next_combination(&mut sel, honors.len()) {
                    break;
                }
            }
        }
    }
    out
}

fn push_honor_shape_with_optional_pair(
    honors: &[u8],
    kotsu_indices: &[usize],
    out: &mut Vec<SuitShape>,
) {
    let mut counts = [0u8; 7];
    let mut kotsu = Vec::new();
    for &i in kotsu_indices {
        counts[i] += 3;
        kotsu.push(honors[i]);
    }
    out.push(SuitShape {
        counts: counts.to_vec(),
        pair_idx: None,
        kotsu: kotsu.clone(),
        shuntsu: Vec::new(),
    });
    // pair of an honor that is NOT already a kotsu
    for p in 0..7u8 {
        if kotsu_indices.contains(&(p as usize)) {
            continue;
        }
        let mut shape_counts = counts;
        shape_counts[p as usize] += 2;
        out.push(SuitShape {
            counts: shape_counts.to_vec(),
            pair_idx: Some(p),
            kotsu: kotsu.clone(),
            shuntsu: Vec::new(),
        });
    }
}

/// Next sorted-with-replacement multi-index: idx is an n-tuple over [0..max),
/// non-decreasing. Returns false when the last permutation has been emitted.
fn next_sorted_multi_index(idx: &mut [usize], max: usize) -> bool {
    if idx.is_empty() {
        return false;
    }
    let n = idx.len();
    // Find rightmost position we can bump.
    let mut i = n;
    while i > 0 {
        i -= 1;
        if idx[i] + 1 < max {
            idx[i] += 1;
            // Reset all positions to the right to idx[i] (preserve sorted)
            let v = idx[i];
            for j in i + 1..n {
                idx[j] = v;
            }
            return true;
        }
    }
    false
}

/// Next k-combination of [0..n) (strictly increasing indices).
fn next_combination(sel: &mut [usize], n: usize) -> bool {
    let k = sel.len();
    if k == 0 {
        return false;
    }
    let mut i = k;
    while i > 0 {
        i -= 1;
        // sel[i] can go up to n - (k - i)
        if sel[i] + 1 + (k - 1 - i) < n {
            sel[i] += 1;
            for j in i + 1..k {
                sel[j] = sel[j - 1] + 1;
            }
            return true;
        }
    }
    false
}
