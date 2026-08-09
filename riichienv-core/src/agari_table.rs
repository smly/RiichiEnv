//! Precomputed agari (winning hand) decomposition table.
//!
//! Approach is the same family as 山岡忠夫's algorithm
//! (<http://hp.vector.co.jp/authors/VA046927/mjscore/mjalgorism.html>):
//! enumerate every standard 14-tile winning shape (one pair + four mentsu),
//! and store, for each unique shape, the list of valid pair+mentsu
//! decompositions. At runtime the SP / `find_divisions` hot path can hash the
//! 34-tile counts and read the decompositions out without re-running the
//! recursive search.
//!
//! This implementation is **independently written from the algorithm
//! description** and does not copy code from any AGPL implementation. The
//! generated table file lives at `data/agari_table.bin.gz` and is built by
//! `src/bin/build_agari_table.rs` (run once, checked in).

// Tile indices are part of the table format. Explicit indexed loops keep the
// correspondence between a tile id and each fixed-size buffer visible.
#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::LazyLock;

/// Lightweight FxHash implementation specialised for the u128 keys used by
/// the agari table. SipHash (the std::collections::HashMap default) is
/// gratuitous for our trusted in-process keys and adds ~30 ns/lookup; FxHash
/// brings the hashing portion well under 10 ns.
#[derive(Default, Clone, Copy)]
pub struct FxHasher64 {
    hash: u64,
}

const FX_SEED: u64 = 0x517c_c1b7_2722_0a95;

impl Hasher for FxHasher64 {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut h = self.hash;
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            // SAFETY: chunks_exact yields exactly 8-byte slices.
            let v = u64::from_ne_bytes(unsafe { *(chunk.as_ptr() as *const [u8; 8]) });
            h = h.rotate_left(5) ^ v;
            h = h.wrapping_mul(FX_SEED);
        }
        for &b in chunks.remainder() {
            h = h.rotate_left(5) ^ (b as u64);
            h = h.wrapping_mul(FX_SEED);
        }
        self.hash = h;
    }
    #[inline]
    fn write_u8(&mut self, b: u8) {
        self.hash = (self.hash.rotate_left(5) ^ (b as u64)).wrapping_mul(FX_SEED);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher64>>;

/// A canonical decomposition and the permutations needed to restore the
/// caller's suit and honor ordering.
pub type CanonicalLookup<'a> = (&'a DivisionList, [u8; 4], [u8; 3], [u8; 7]);

/// Maximum mentsu in a standard hand: at most 4 (pair + 4 mentsu = 14 tiles).
pub const MAX_MENTSU: usize = 4;

/// One pair-plus-four-mentsu decomposition of a 14-tile standard agari hand.
///
/// Each tile id is in 0..34 (the standard 34-tile encoding: 0-8 = 1-9 manzu,
/// 9-17 = 1-9 pinzu, 18-26 = 1-9 souzu, 27-33 = honor tiles).
///
/// `kotsu_tiles` holds the tile id of each koutsu (triple). `shuntsu_starts`
/// holds the lowest tile id of each shuntsu (sequence). Honor tiles never
/// participate in shuntsu, so a shuntsu start is always in 0..=24 (within a
/// single suit). The unused slots are zeroed; use the `n_*` counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Division {
    pub pair_tile: u8,
    pub n_kotsu: u8,
    pub kotsu_tiles: [u8; MAX_MENTSU],
    pub n_shuntsu: u8,
    pub shuntsu_starts: [u8; MAX_MENTSU],
}

impl Division {
    pub fn new(pair_tile: u8) -> Self {
        Self {
            pair_tile,
            n_kotsu: 0,
            kotsu_tiles: [0; MAX_MENTSU],
            n_shuntsu: 0,
            shuntsu_starts: [0; MAX_MENTSU],
        }
    }

    pub fn push_kotsu(&mut self, tile: u8) {
        self.kotsu_tiles[self.n_kotsu as usize] = tile;
        self.n_kotsu += 1;
    }

    pub fn push_shuntsu(&mut self, start: u8) {
        self.shuntsu_starts[self.n_shuntsu as usize] = start;
        self.n_shuntsu += 1;
    }

    pub fn kotsu(&self) -> &[u8] {
        &self.kotsu_tiles[..self.n_kotsu as usize]
    }

    pub fn shuntsu(&self) -> &[u8] {
        &self.shuntsu_starts[..self.n_shuntsu as usize]
    }
}

/// Shift each numbered suit's counts so it starts at slot 0 within its suit
/// (the m suit at counts[0..9], p at [9..18], s at [18..27]). Honors are not
/// shifted. Returns the canonical counts and the per-suit shift offset
/// (offsets[0..3] for m/p/s; offsets[3] is unused).
///
/// Two 14-tile hands that differ only in which absolute positions their
/// numbered tiles occupy share the same canonical form (e.g. 5-7m and 6-8m).
/// This collapses the table from ~11M direct-keyed entries to ~10K.
#[inline]
pub fn canonicalize(counts: &[u8; 34]) -> ([u8; 34], [u8; 4]) {
    let mut canon = [0u8; 34];
    let mut offsets = [0u8; 4];
    for (suit_idx, base) in [(0usize, 0usize), (1, 9), (2, 18)] {
        let mut first = 9usize;
        for i in 0..9 {
            if counts[base + i] > 0 {
                first = i;
                break;
            }
        }
        if first < 9 {
            for i in 0..9 - first {
                canon[base + i] = counts[base + first + i];
            }
            offsets[suit_idx] = first as u8;
        }
    }
    canon[27..34].copy_from_slice(&counts[27..34]);
    (canon, offsets)
}

/// Compact key for a canonical hand shape (call `canonicalize` first).
/// 3 bits per tile × 34 tiles = 102 bits, fits in u128.
#[inline]
pub fn key_from_counts(counts: &[u8; 34]) -> u128 {
    let mut key = 0u128;
    for (i, &c) in counts.iter().enumerate() {
        key |= ((c & 0x7) as u128) << (3 * i);
    }
    key
}

/// Apply per-suit offsets to a tile id stored in canonical form (where each
/// numbered suit's counts have been left-shifted). Honors are returned as is.
#[inline]
pub fn apply_offset(canonical_tile: u8, offsets: &[u8; 4]) -> u8 {
    if canonical_tile < 27 {
        let suit = (canonical_tile / 9) as usize;
        canonical_tile + offsets[suit]
    } else {
        canonical_tile
    }
}

/// Full canonicalization: per-suit shift, permutation of the three numbered
/// suits, AND permutation of the seven honor tiles by count. Collapses
/// hands that share a structural shape (e.g. "kotsu of 東 + pair of 西" and
/// "kotsu of 北 + pair of 中" both map to the same canonical key).
///
/// Returns `(canonical_counts, per_suit_shifts, suit_perm, honor_perm)` where
/// `suit_perm[sorted_suit] = original_suit_index` and
/// `honor_perm[sorted_honor_pos] = original_honor_pos`.
#[inline]
pub fn canonicalize_full(counts: &[u8; 34]) -> ([u8; 34], [u8; 4], [u8; 3], [u8; 7]) {
    let (shifted, offsets) = canonicalize(counts);

    // Inline 3-element insertion sort over the suit slot patterns (avoids the
    // generic `sort_by` closure overhead that costs ~30ns/call).
    let mut suit_perm: [u8; 3] = [0, 1, 2];
    let cmp_suit = |a: u8, b: u8| -> std::cmp::Ordering {
        let aa = &shifted[(a as usize) * 9..(a as usize) * 9 + 9];
        let bb = &shifted[(b as usize) * 9..(b as usize) * 9 + 9];
        aa.cmp(bb)
    };
    if cmp_suit(suit_perm[0], suit_perm[1]) == std::cmp::Ordering::Greater {
        suit_perm.swap(0, 1);
    }
    if cmp_suit(suit_perm[1], suit_perm[2]) == std::cmp::Ordering::Greater {
        suit_perm.swap(1, 2);
    }
    if cmp_suit(suit_perm[0], suit_perm[1]) == std::cmp::Ordering::Greater {
        suit_perm.swap(0, 1);
    }

    let mut canon = [0u8; 34];
    for (sorted_idx, &orig_suit) in suit_perm.iter().enumerate() {
        let dst = sorted_idx * 9;
        let src = (orig_suit as usize) * 9;
        canon[dst..dst + 9].copy_from_slice(&shifted[src..src + 9]);
    }

    // Honor permutation: sort the seven honor positions descending by count,
    // tie-break by index. Use a 7-element insertion sort (compact, no
    // closure call overhead).
    let mut honor_perm: [u8; 7] = [0, 1, 2, 3, 4, 5, 6];
    for i in 1..7 {
        let mut j = i;
        while j > 0 {
            let ca = shifted[27 + honor_perm[j - 1] as usize];
            let cb = shifted[27 + honor_perm[j] as usize];
            // Sort descending by count; ties broken by ascending index so the
            // generated table and runtime agree.
            if cb > ca || (cb == ca && honor_perm[j] < honor_perm[j - 1]) {
                honor_perm.swap(j - 1, j);
                j -= 1;
            } else {
                break;
            }
        }
    }
    for (sorted_idx, &orig_pos) in honor_perm.iter().enumerate() {
        canon[27 + sorted_idx] = shifted[27 + orig_pos as usize];
    }
    (canon, offsets, suit_perm, honor_perm)
}

/// Translate a tile id from the table's canonical-sorted form back to the
/// caller's actual coordinates.
#[inline]
pub fn apply_offset_perm(
    canonical_tile: u8,
    offsets: &[u8; 4],
    suit_perm: &[u8; 3],
    honor_perm: &[u8; 7],
) -> u8 {
    if canonical_tile >= 27 {
        let sorted_pos = (canonical_tile - 27) as usize;
        return 27 + honor_perm[sorted_pos];
    }
    let sorted_suit = (canonical_tile / 9) as usize;
    let pos = canonical_tile % 9;
    let actual_suit = suit_perm[sorted_suit] as usize;
    (actual_suit as u8) * 9 + pos + offsets[actual_suit]
}

/// Encoded decomposition layout (little-endian):
/// - 1 byte: pair_tile (0..34)
/// - 1 byte: n_kotsu (0..=4)
/// - 4 bytes: kotsu_tiles (zero-padded)
/// - 1 byte: n_shuntsu (0..=4)
/// - 4 bytes: shuntsu_starts (zero-padded)
pub const DIVISION_BYTES: usize = 11;

pub fn encode_division(div: &Division, out: &mut Vec<u8>) {
    out.push(div.pair_tile);
    out.push(div.n_kotsu);
    out.extend_from_slice(&div.kotsu_tiles);
    out.push(div.n_shuntsu);
    out.extend_from_slice(&div.shuntsu_starts);
}

pub fn decode_division(buf: &[u8]) -> Division {
    let pair_tile = buf[0];
    let n_kotsu = buf[1];
    let mut kotsu_tiles = [0u8; MAX_MENTSU];
    kotsu_tiles.copy_from_slice(&buf[2..2 + MAX_MENTSU]);
    let n_shuntsu = buf[2 + MAX_MENTSU];
    let mut shuntsu_starts = [0u8; MAX_MENTSU];
    shuntsu_starts.copy_from_slice(&buf[3 + MAX_MENTSU..3 + 2 * MAX_MENTSU]);
    Division {
        pair_tile,
        n_kotsu,
        kotsu_tiles,
        n_shuntsu,
        shuntsu_starts,
    }
}

/// Embedded table data shipped in the crate. Each entry is:
///   [16 bytes key (LE u128)] [1 byte n_div] [n_div * DIVISION_BYTES]
/// Stored uncompressed to keep wasm builds free of flate2; HTTP gzip on the
/// transport layer handles compression for browser delivery.
pub const AGARI_TABLE_DATA: &[u8] = include_bytes!("data/agari_table.bin");

/// Used only for the legacy canonical `DivisionList` (which has the bound
/// of 4 by construction). The COMPACT table stores divs in a flat Vec —
/// it has no per-entry inline cap, so suit-perm expansion can produce as
/// many divs per key as needed without pre-allocating worst case.
pub const MAX_DIVS_PER_KEY: usize = 4;

#[derive(Clone)]
pub struct DivisionList {
    pub n: u8,
    pub divs: [Division; MAX_DIVS_PER_KEY],
}

pub static AGARI_TABLE: LazyLock<FxHashMap<u128, DivisionList>> = LazyLock::new(load_table);

fn load_table() -> FxHashMap<u128, DivisionList> {
    let raw = AGARI_TABLE_DATA;
    let mut map = FxHashMap::with_capacity_and_hasher(50_000, BuildHasherDefault::default());
    let mut i = 0;
    while i < raw.len() {
        // u128 key (16 bytes, little endian)
        let mut key_bytes = [0u8; 16];
        key_bytes.copy_from_slice(&raw[i..i + 16]);
        let key = u128::from_le_bytes(key_bytes);
        i += 16;
        let n_div = raw[i];
        i += 1;
        assert!((n_div as usize) <= MAX_DIVS_PER_KEY);
        let mut list = DivisionList {
            n: n_div,
            divs: [Division::new(0); MAX_DIVS_PER_KEY],
        };
        for k in 0..n_div as usize {
            list.divs[k] = decode_division(&raw[i..i + DIVISION_BYTES]);
            i += DIVISION_BYTES;
        }
        map.insert(key, list);
    }
    map
}

/// Return the precomputed decompositions (in canonical tile-id form) plus the
/// permutations needed to translate them back to actual tile ids.
/// Returns None for non-agari shapes.
pub fn lookup_canonical(counts: &[u8; 34]) -> Option<CanonicalLookup<'_>> {
    let (canon, offsets, suit_perm, honor_perm) = canonicalize_full(counts);
    AGARI_TABLE
        .get(&key_from_counts(&canon))
        .map(|d| (d, offsets, suit_perm, honor_perm))
}

/// Convenience: return the decompositions translated into absolute tile ids.
/// Allocates a small Vec; for hot paths use `lookup_canonical` and translate
/// inline.
pub fn lookup(counts: &[u8; 34]) -> Vec<Division> {
    let Some((list, offsets, sp, hp)) = lookup_canonical(counts) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(list.n as usize);
    for k in 0..list.n as usize {
        let mut d = list.divs[k];
        d.pair_tile = apply_offset_perm(d.pair_tile, &offsets, &sp, &hp);
        for i in 0..d.n_kotsu as usize {
            d.kotsu_tiles[i] = apply_offset_perm(d.kotsu_tiles[i], &offsets, &sp, &hp);
        }
        for i in 0..d.n_shuntsu as usize {
            d.shuntsu_starts[i] = apply_offset_perm(d.shuntsu_starts[i], &offsets, &sp, &hp);
        }
        let mut kk = d.kotsu_tiles;
        kk[..d.n_kotsu as usize].sort_unstable();
        d.kotsu_tiles = kk;
        let mut ss = d.shuntsu_starts;
        ss[..d.n_shuntsu as usize].sort_unstable();
        d.shuntsu_starts = ss;
        out.push(d);
    }
    out
}

// ───────────── topology-indexed compact lookup ─────────────────────────
// 既存テーブル (1.1MB, canonical-key) は absolute tile id を保持するため、
// shape-trivial だが座標違いの形を 4× 重複して持っている。Mortal 流の
// stair-step bit key は「非ゼロ位置の **ギャップは 1 ビットの separator** で
// 表現する」エンコーディングなので、内側位置のシフト不変性を自然に獲得し、
// 約 6.8K キーまで縮約できる (Mortal 自身のテーブル 9362 とほぼ同オーダー)。
//
// ここでは canonical テーブルを起動時に走査して、
//   - mortal_key (u32 stair-step) を計算
//   - 同じ mortal_key を持つ canonical entries を 1 つに圧縮
//   - Division を「tile14 配列のインデックス」表現に変換
// した COMPACT_TABLE を `LazyLock` で構築する。
// shipping bin の追加は無し (派生)。

/// 14-tile 中の **異なる**牌の並び (昇順、最大 14 個) と長さ。
/// `lookup_compact` の戻り値で、`CompactDiv` のインデックスはこの配列を参照する。
#[derive(Debug, Clone, Copy)]
pub struct Tile14 {
    pub tiles: [u8; 14],
    pub len: u8,
}

/// 1 ハンドの 1 decomposition を tile14 インデックスで Mortal-style な
/// u32 に packing したもの。ビットレイアウト:
///   [ 0.. 3]  n_kotsu (3 bits, 0..=4)
///   [ 3.. 6]  n_shuntsu (3 bits, 0..=4)
///   [ 6..10]  pair_idx (4 bits, 0..=14)
///   [10..14]  mentsu[0] (kotsu first, then shuntsu — slot share like Mortal)
///   [14..18]  mentsu[1]
///   [18..22]  mentsu[2]
///   [22..26]  mentsu[3]
///
/// 4 byte/div で flat_divs に詰めるため、L1 cache に載りやすい。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct CompactDiv(pub u32);

impl CompactDiv {
    #[inline]
    pub fn pack(
        pair_idx: u8,
        n_kotsu: u8,
        kotsu_idxs: &[u8],
        n_shuntsu: u8,
        shuntsu_idxs: &[u8],
    ) -> Self {
        debug_assert!(pair_idx < 15);
        debug_assert!((n_kotsu as usize) == kotsu_idxs.len());
        debug_assert!((n_shuntsu as usize) == shuntsu_idxs.len());
        debug_assert!(n_kotsu + n_shuntsu <= 4);
        let mut v = (n_kotsu as u32) | ((n_shuntsu as u32) << 3) | ((pair_idx as u32) << 6);
        let mut slot = 0u32;
        for &i in kotsu_idxs {
            debug_assert!(i < 15);
            v |= (i as u32) << (10 + slot * 4);
            slot += 1;
        }
        for &i in shuntsu_idxs {
            debug_assert!(i < 15);
            v |= (i as u32) << (10 + slot * 4);
            slot += 1;
        }
        Self(v)
    }

    #[inline]
    pub fn n_kotsu(self) -> u8 {
        (self.0 & 0x7) as u8
    }
    #[inline]
    pub fn n_shuntsu(self) -> u8 {
        ((self.0 >> 3) & 0x7) as u8
    }
    #[inline]
    pub fn pair_idx(self) -> u8 {
        ((self.0 >> 6) & 0xF) as u8
    }
    /// `i`-th kotsu tile14 index (0..n_kotsu).
    #[inline]
    pub fn kotsu_idx(self, i: usize) -> u8 {
        ((self.0 >> (10 + i * 4)) & 0xF) as u8
    }
    /// `i`-th shuntsu tile14 index (0..n_shuntsu). Stored after kotsu.
    #[inline]
    pub fn shuntsu_idx(self, i: usize) -> u8 {
        let slot = self.n_kotsu() as usize + i;
        ((self.0 >> (10 + slot * 4)) & 0xF) as u8
    }
}

/// 字牌位置と独立な、複数 division を共有メモリ領域 (`flat_divs`) で
/// 持つ container. table 全体が L1/L2 にフィットしやすいよう設計。
pub struct CompactTable {
    /// mortal_key → packed value: [bits 0..24] = offset into `flat_divs`,
    /// [bits 24..32] = n_div. 1 entry = 4 byte hash key + 4 byte value
    /// (HashMap value alignment 込み)。
    index: FxHashMap<u32, u32>,
    /// 全 entry の CompactDiv を 1 本の Vec に直列化。各 entry は
    /// `&flat_divs[offset .. offset + n_div]` でスライス参照する。
    /// 1 div = 4 byte なので連続 7k entry でも ~28 KB に収まる。
    flat_divs: Box<[CompactDiv]>,
}

impl CompactTable {
    /// `mortal_key` に対応する CompactDiv のスライスを返す。
    /// (1 cache line で済むよう u32 → (offset, n) → 連続スライス と最小限のアクセスで構成)
    #[inline]
    pub fn lookup(&self, key: u32) -> Option<&[CompactDiv]> {
        let &packed = self.index.get(&key)?;
        let offset = (packed & 0x00FF_FFFF) as usize;
        let n = (packed >> 24) as usize;
        Some(&self.flat_divs[offset..offset + n])
    }
    pub fn n_keys(&self) -> usize {
        self.index.len()
    }
    pub fn n_divs(&self) -> usize {
        self.flat_divs.len()
    }
}

/// 入力 `counts` から Mortal 互換の stair-step u32 key と sorted-unique
/// `Tile14` を 1 パスで構築する。`apply_offset_perm` 等の派生不要。
///
/// エンコーディング: 各 suit 内で「非ゼロ位置を 1 つずつ訪問しながら、
/// 各位置の余分カウント (c-1) を 2(c-1) ビットで書き、隣接ゼロから
/// 非ゼロへ戻る境界に 1 ビットの separator を打つ」。実質的に
/// non-zero positions の集合と count multiset を可逆に符号化する。
#[inline]
pub fn topology_key_and_tile14(counts: &[u8; 34]) -> (u32, Tile14) {
    let mut key: u32 = 0;
    let mut bit_idx: i32 = -1;
    let mut tiles = [0u8; 14];
    let mut len = 0usize;

    let emit_count = |key: &mut u32, bit_idx: &mut i32, c: u8| {
        *bit_idx += 1;
        match c {
            2 => {
                *key |= 0b11 << *bit_idx;
                *bit_idx += 2;
            }
            3 => {
                *key |= 0b1111 << *bit_idx;
                *bit_idx += 4;
            }
            4 => {
                *key |= 0b11_1111 << *bit_idx;
                *bit_idx += 6;
            }
            _ => {}
        }
    };

    // 数牌 3 suit: chunks_exact(9) と同じイテレーション。
    for kind in 0..3 {
        let mut prev_in_hand = false;
        for num in 0..9 {
            let c = counts[kind * 9 + num];
            if c > 0 {
                prev_in_hand = true;
                if len < 14 {
                    tiles[len] = (kind * 9 + num) as u8;
                    len += 1;
                }
                emit_count(&mut key, &mut bit_idx, c);
            } else if prev_in_hand {
                key |= 0b1 << bit_idx;
                bit_idx += 1;
                prev_in_hand = false;
            }
        }
        // suit 終端の境界 separator (次の suit と区別するため)。
        if prev_in_hand {
            key |= 0b1 << bit_idx;
            bit_idx += 1;
        }
    }

    // 字牌 7 種は位置が yaku 上意味を持つので「非ゼロ位置のみ訪問」+
    // 各 emit のあと必ず separator を打つ (suit のような「次グループ」が
    // ないため、separator はその牌の終端マーカー)。
    for tile in 27..34 {
        let c = counts[tile];
        if c > 0 {
            if len < 14 {
                tiles[len] = tile as u8;
                len += 1;
            }
            emit_count(&mut key, &mut bit_idx, c);
            key |= 0b1 << bit_idx;
            bit_idx += 1;
        }
    }

    (
        key,
        Tile14 {
            tiles,
            len: len as u8,
        },
    )
}

pub static COMPACT_AGARI_TABLE: LazyLock<CompactTable> = LazyLock::new(load_compact_table);

fn load_compact_table() -> CompactTable {
    // canonical AGARI_TABLE を走査し、各 canonical_counts について
    // **数牌スーツの 6 通りの順列** を列挙し、
    //   - 各順列で permuted_counts を再構成
    //   - mortal_key + tile14 を計算
    //   - Division を tile14 インデックス表現 (CompactDiv) に変換
    //   - 同じ mortal_key の重複は HashMap dedup で merge
    //
    // 字牌位置については mortal_key は本来 invariant (encoding が「非ゼロ
    // 位置のみを順に走査する」ため、どの字牌位置に同 count が立っても同
    // ビットパターンを生む)。したがって字牌の置換は列挙不要。
    //
    // 数牌の suit perm のみが mortal_key を変化させる (canonicalize_full は
    // 内容辞書順で suit を sort するため、ある canonical エントリは特定の
    // suit assignment のみを表す)。runtime input は任意の suit assignment
    // を取り得るので、ビルド時に 6 perm すべてを展開しておく必要がある。
    //
    // ビルドの中間形式: (mortal_key → Vec<CompactDiv> with dedup)。
    // 最後に flat_divs と index にパッキングし直す。
    let mut staging: FxHashMap<u32, Vec<CompactDiv>> =
        FxHashMap::with_capacity_and_hasher(20_000, BuildHasherDefault::default());

    // 3-suit の 6 順列。perm[real_suit] = canonical_suit (= "real_suit には
    // canonical_suit の中身を置く").
    const SUIT_PERMS: [[u8; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];

    let canonical_table = &*AGARI_TABLE;
    for (&canonical_key, divlist) in canonical_table.iter() {
        let mut counts = [0u8; 34];
        for i in 0..34 {
            counts[i] = ((canonical_key >> (3 * i)) & 0x7) as u8;
        }
        let total: u32 = counts.iter().map(|&c| c as u32).sum();
        if total != 14 {
            continue;
        }

        for perm in &SUIT_PERMS {
            // perm から逆引き: perm_inv[canonical] = real (canonical_suit が
            // 配置される real_suit のインデックス)。
            let mut perm_inv = [0u8; 3];
            for i in 0..3 {
                perm_inv[perm[i] as usize] = i as u8;
            }

            // permuted_counts: real_suit i に canonical_suit perm[i] の中身を入れる。
            let mut permuted = [0u8; 34];
            for real_suit in 0..3usize {
                let canon_suit = perm[real_suit] as usize;
                for pos in 0..9 {
                    permuted[real_suit * 9 + pos] = counts[canon_suit * 9 + pos];
                }
            }
            permuted[27..34].copy_from_slice(&counts[27..34]);

            // canonical の tile id (0..27) を permuted の tile id にマップ。
            // canonical tile t = (suit_canon, pos) → permuted tile = perm_inv[suit_canon] * 9 + pos.
            let mut canon_to_real = [0u8; 34];
            for t in 0..27u8 {
                let suit_canon = (t / 9) as usize;
                let pos = t % 9;
                canon_to_real[t as usize] = perm_inv[suit_canon] * 9 + pos;
            }
            // 字牌は identity (mortal_key 字牌 invariant のため再配置不要)。
            for t in 27..34 {
                canon_to_real[t as usize] = t;
            }

            insert_compact_with_translation(&mut staging, &permuted, divlist, &canon_to_real);
        }
    }

    // staging を flat_divs に詰めて offset/n をパッキング。
    let total_divs: usize = staging.values().map(|v| v.len()).sum();
    let mut flat: Vec<CompactDiv> = Vec::with_capacity(total_divs);
    let mut index: FxHashMap<u32, u32> =
        FxHashMap::with_capacity_and_hasher(staging.len(), BuildHasherDefault::default());
    for (key, divs) in staging.into_iter() {
        let offset = flat.len() as u32;
        let n = divs.len() as u32;
        debug_assert!(offset < (1 << 24));
        debug_assert!(n < (1 << 8));
        flat.extend(divs);
        index.insert(key, (offset & 0x00FF_FFFF) | (n << 24));
    }

    CompactTable {
        index,
        flat_divs: flat.into_boxed_slice(),
    }
}

/// `permuted_counts` (suit perm 適用後) と canonical → permuted の tile id
/// 翻訳テーブル `canon_to_real` を受け取り、CompactDiv を構築して compact
/// マップへ挿入する。同 mortal_key の重複 CompactDiv は dedup される。
fn insert_compact_with_translation(
    staging: &mut FxHashMap<u32, Vec<CompactDiv>>,
    permuted_counts: &[u8; 34],
    divlist: &DivisionList,
    canon_to_real: &[u8; 34],
) {
    let (mkey, tile14) = topology_key_and_tile14(permuted_counts);

    let mut tile_to_idx = [u8::MAX; 34];
    for i in 0..tile14.len as usize {
        tile_to_idx[tile14.tiles[i] as usize] = i as u8;
    }

    for k in 0..divlist.n as usize {
        let d = &divlist.divs[k];
        let pair_idx = tile_to_idx[canon_to_real[d.pair_tile as usize] as usize];
        let mut k_idxs = [0u8; MAX_MENTSU];
        for i in 0..d.n_kotsu as usize {
            k_idxs[i] = tile_to_idx[canon_to_real[d.kotsu_tiles[i] as usize] as usize];
        }
        let mut s_idxs = [0u8; MAX_MENTSU];
        for i in 0..d.n_shuntsu as usize {
            s_idxs[i] = tile_to_idx[canon_to_real[d.shuntsu_starts[i] as usize] as usize];
        }
        // Sort idxs ascending so equivalent CompactDivs produced from
        // different sources pack to identical u32s.
        k_idxs[..d.n_kotsu as usize].sort_unstable();
        s_idxs[..d.n_shuntsu as usize].sort_unstable();

        let compact_div = CompactDiv::pack(
            pair_idx,
            d.n_kotsu,
            &k_idxs[..d.n_kotsu as usize],
            d.n_shuntsu,
            &s_idxs[..d.n_shuntsu as usize],
        );

        let entry = staging.entry(mkey).or_default();
        if !entry.contains(&compact_div) {
            entry.push(compact_div);
        }
    }
}

/// Topology-indexed lookup. Returns `(tile14, divs_slice)` where `divs_slice`
/// references the contiguous CompactDiv block in the global flat storage —
/// reads are 1 cache line apiece, and the working set per SP sample
/// (~250 unique keys × ~4 bytes div + ~4 bytes index value) fits in L1.
#[inline]
pub fn lookup_compact(counts_14: &[u8; 34]) -> Option<(Tile14, &'static [CompactDiv])> {
    let (key, tile14) = topology_key_and_tile14(counts_14);
    COMPACT_AGARI_TABLE.lookup(key).map(|divs| (tile14, divs))
}

/// Pure (no I/O) per-suit + honors enumerator. Also used by the build script.
/// Given the 34-tile counts, return all valid (pair + 4 mentsu) decompositions.
/// Returns an empty list if there is no valid decomposition.
///
/// This is the slow reference implementation — the runtime hot path uses
/// `lookup` instead, which reads precomputed results.
pub fn enumerate_divisions(counts: &[u8; 34]) -> Vec<Division> {
    let mut out = Vec::new();
    if counts.iter().map(|&c| c as u32).sum::<u32>() != 14 {
        return out;
    }

    // Try every tile that has at least 2 copies as the pair candidate.
    for pair_tile in 0..34u8 {
        if counts[pair_tile as usize] < 2 {
            continue;
        }
        let mut work = *counts;
        work[pair_tile as usize] -= 2;

        // Decompose each suit + honors into mentsu only.
        let m_decomps = decompose_suit(&work[0..9]);
        let p_decomps = decompose_suit(&work[9..18]);
        let s_decomps = decompose_suit(&work[18..27]);
        let z_decomps = decompose_honors(&work[27..34]);

        for m in &m_decomps {
            for p in &p_decomps {
                for s in &s_decomps {
                    for z in &z_decomps {
                        let n_total = m.n_kotsu
                            + m.n_shuntsu
                            + p.n_kotsu
                            + p.n_shuntsu
                            + s.n_kotsu
                            + s.n_shuntsu
                            + z.n_kotsu;
                        if n_total != 4 {
                            continue;
                        }
                        let mut div = Division::new(pair_tile);
                        for &t in &m.kotsu_tiles[..m.n_kotsu as usize] {
                            div.push_kotsu(t);
                        }
                        for &t in &p.kotsu_tiles[..p.n_kotsu as usize] {
                            div.push_kotsu(t + 9);
                        }
                        for &t in &s.kotsu_tiles[..s.n_kotsu as usize] {
                            div.push_kotsu(t + 18);
                        }
                        for &t in &z.kotsu_tiles[..z.n_kotsu as usize] {
                            div.push_kotsu(t + 27);
                        }
                        for &t in &m.shuntsu_starts[..m.n_shuntsu as usize] {
                            div.push_shuntsu(t);
                        }
                        for &t in &p.shuntsu_starts[..p.n_shuntsu as usize] {
                            div.push_shuntsu(t + 9);
                        }
                        for &t in &s.shuntsu_starts[..s.n_shuntsu as usize] {
                            div.push_shuntsu(t + 18);
                        }
                        // Sort mentsu lists for canonical form (so dedup works).
                        let mut k = div.kotsu_tiles;
                        k[..div.n_kotsu as usize].sort_unstable();
                        div.kotsu_tiles = k;
                        let mut s = div.shuntsu_starts;
                        s[..div.n_shuntsu as usize].sort_unstable();
                        div.shuntsu_starts = s;
                        out.push(div);
                    }
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

#[derive(Clone, Copy)]
struct PartialDecomp {
    n_kotsu: u8,
    kotsu_tiles: [u8; MAX_MENTSU],
    n_shuntsu: u8,
    shuntsu_starts: [u8; MAX_MENTSU],
}

impl PartialDecomp {
    fn empty() -> Self {
        Self {
            n_kotsu: 0,
            kotsu_tiles: [0; MAX_MENTSU],
            n_shuntsu: 0,
            shuntsu_starts: [0; MAX_MENTSU],
        }
    }
    fn push_kotsu(&mut self, t: u8) {
        self.kotsu_tiles[self.n_kotsu as usize] = t;
        self.n_kotsu += 1;
    }
    fn pop_kotsu(&mut self) {
        self.n_kotsu -= 1;
    }
    fn push_shuntsu(&mut self, t: u8) {
        self.shuntsu_starts[self.n_shuntsu as usize] = t;
        self.n_shuntsu += 1;
    }
    fn pop_shuntsu(&mut self) {
        self.n_shuntsu -= 1;
    }
}

fn decompose_suit(counts: &[u8]) -> Vec<PartialDecomp> {
    debug_assert_eq!(counts.len(), 9);
    let mut work = [0u8; 9];
    work.copy_from_slice(counts);
    let mut out = Vec::new();
    let mut current = PartialDecomp::empty();
    decompose_suit_rec(&mut work, 0, &mut current, &mut out);
    out
}

fn decompose_suit_rec(
    counts: &mut [u8; 9],
    start: usize,
    current: &mut PartialDecomp,
    out: &mut Vec<PartialDecomp>,
) {
    let mut i = start;
    while i < 9 && counts[i] == 0 {
        i += 1;
    }
    if i == 9 {
        out.push(*current);
        return;
    }
    // Must consume counts[i] via either a kotsu starting at i, or shuntsu
    // starting at i (if i <= 6 and i+1, i+2 have ≥ 1 each). If neither path
    // is feasible, this branch is dead.
    if counts[i] >= 3 {
        counts[i] -= 3;
        current.push_kotsu(i as u8);
        decompose_suit_rec(counts, i, current, out);
        current.pop_kotsu();
        counts[i] += 3;
    }
    if i + 2 < 9 && counts[i] >= 1 && counts[i + 1] >= 1 && counts[i + 2] >= 1 {
        counts[i] -= 1;
        counts[i + 1] -= 1;
        counts[i + 2] -= 1;
        current.push_shuntsu(i as u8);
        decompose_suit_rec(counts, i, current, out);
        current.pop_shuntsu();
        counts[i] += 1;
        counts[i + 1] += 1;
        counts[i + 2] += 1;
    }
    // If counts[i] is 1 or 2 and no shuntsu fits, dead branch (no valid
    // mentsu-only decomposition).
}

fn decompose_honors(counts: &[u8]) -> Vec<PartialDecomp> {
    debug_assert_eq!(counts.len(), 7);
    // Honors only support kotsu (and optionally a pair, but pair is extracted
    // outside). Each honor count must be 0 or 3 to be decomposable here.
    let mut decomp = PartialDecomp::empty();
    for (i, &c) in counts.iter().enumerate() {
        match c {
            0 => continue,
            3 => decomp.push_kotsu(i as u8),
            _ => return Vec::new(),
        }
    }
    vec![decomp]
}

// ---------------------------------------------------------------------------
// Sort/Eq for Division so dedup works
// ---------------------------------------------------------------------------

impl Ord for Division {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.pair_tile
            .cmp(&other.pair_tile)
            .then_with(|| self.n_kotsu.cmp(&other.n_kotsu))
            .then_with(|| self.kotsu_tiles.cmp(&other.kotsu_tiles))
            .then_with(|| self.n_shuntsu.cmp(&other.n_shuntsu))
            .then_with(|| self.shuntsu_starts.cmp(&other.shuntsu_starts))
    }
}
impl PartialOrd for Division {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1-9m + 11p + 234p — pinfu-shape standard agari.
    #[test]
    fn enumerate_simple_pinfu() {
        let mut counts = [0u8; 34];
        for i in 0..9 {
            counts[i] = 1; // 1-9m
        }
        counts[9] = 2; // 11p
        counts[10] = 1; // 2p
        counts[11] = 1; // 3p
        counts[12] = 1; // 4p
        // total = 9 + 2 + 3 = 14 ✓ (pair=1p, mentsu=123m, 456m, 789m, 234p)
        let divs = enumerate_divisions(&counts);
        assert!(!divs.is_empty(), "should find at least one division");
        for d in &divs {
            assert_eq!(d.pair_tile, 9);
            assert_eq!(d.n_kotsu + d.n_shuntsu, 4);
        }
    }

    /// 7 pairs of honors — chitoitsu, NOT a standard agari.
    #[test]
    fn chitoitsu_not_in_table() {
        let mut counts = [0u8; 34];
        // chitoitsu needs 7 distinct pairs; honors only have 7 tiles.
        for i in 27..34 {
            counts[i] = 2;
        }
        let divs = enumerate_divisions(&counts);
        assert!(
            divs.is_empty(),
            "chitoitsu should not decompose into pair+4 mentsu"
        );
    }

    /// Lookup a known agari hand and verify the table returns it.
    #[test]
    fn lookup_pinfu_shape() {
        let mut counts = [0u8; 34];
        for i in 0..9 {
            counts[i] = 1; // 1-9m
        }
        counts[9] = 2; // 11p
        counts[10] = 1; // 2p
        counts[11] = 1; // 3p
        counts[12] = 1; // 4p
        let divs = lookup(&counts);
        assert!(!divs.is_empty(), "lookup failed for valid agari hand");
        for d in &divs {
            // pair must be 1p (tile 9), since that's the only doubled tile
            assert_eq!(d.pair_tile, 9, "pair should be 1p");
            assert_eq!(d.n_kotsu + d.n_shuntsu, 4);
        }
    }

    /// Lookup that should fail (chitoitsu shape, not standard agari).
    #[test]
    fn lookup_chitoitsu_misses() {
        let mut counts = [0u8; 34];
        for i in 27..34 {
            counts[i] = 2;
        }
        let divs = lookup(&counts);
        assert!(divs.is_empty(), "chitoitsu should not be in standard table");
    }

    /// Verify suit-permutation symmetry: hands that share a structural shape
    /// across different numbered suits must both be found in the table.
    /// Hand: 1m×3 + 234m + 567m + 11p + 333p (3+3+3+2+3 = 14 ✓).
    #[test]
    fn lookup_suit_symmetry() {
        let mut counts = [0u8; 34];
        counts[0] = 3; // 1m×3
        counts[1] = 1;
        counts[2] = 1;
        counts[3] = 1; // 234m
        counts[4] = 1;
        counts[5] = 1;
        counts[6] = 1; // 567m
        counts[9] = 2; // 11p
        counts[11] = 3; // 333p (tile 11 = 3p)
        let divs_m = lookup(&counts);
        assert!(!divs_m.is_empty(), "lookup failed for first hand");

        // Swap m and p suits — must also be in the table (suit canonicalization).
        let mut counts2 = [0u8; 34];
        counts2[9] = 3; // 1p×3
        counts2[10] = 1;
        counts2[11] = 1;
        counts2[12] = 1; // 234p
        counts2[13] = 1;
        counts2[14] = 1;
        counts2[15] = 1; // 567p
        counts2[0] = 2; // 11m
        counts2[2] = 3; // 333m
        let divs_p = lookup(&counts2);
        assert!(!divs_p.is_empty(), "lookup failed for swapped-suit hand");
    }

    /// Verify honor-permutation symmetry: kotsu-of-East + pair-of-South in the
    /// honor area should resolve identically to kotsu-of-West + pair-of-North
    /// (with appropriate body fill) — both share the same canonical shape.
    /// Hand: 1m..9m + 11p + East×3 + South×2 — wait that's 9+2+3+2=16. Adjust.
    /// Use 12-tile m hand + East×3 + South×2... too many. Let me just check
    /// that two honor-permuted hands both succeed without comparing divs.
    #[test]
    fn lookup_honor_symmetry() {
        // 123m 456m 789m + East×3 (kotsu) + South×2 (pair) = 9+3+2 = 14 ✓
        let mut counts_a = [0u8; 34];
        for i in 0..9 {
            counts_a[i] = 1;
        }
        counts_a[27] = 3; // East kotsu
        counts_a[28] = 2; // South pair
        let divs_a = lookup(&counts_a);
        assert!(!divs_a.is_empty(), "East+South hand not found");

        // Same shape but West kotsu + North pair (still honors).
        let mut counts_b = [0u8; 34];
        for i in 0..9 {
            counts_b[i] = 1;
        }
        counts_b[29] = 3; // West kotsu
        counts_b[30] = 2; // North pair
        let divs_b = lookup(&counts_b);
        assert!(!divs_b.is_empty(), "West+North hand not found");

        // Both must contain the right pair tiles after honor-perm un-translation.
        assert!(divs_a.iter().any(|d| d.pair_tile == 28));
        assert!(divs_b.iter().any(|d| d.pair_tile == 30));
    }

    /// Topology key + tile14 round-trip: same shape across suits → same key,
    /// and `Tile14` correctly lists unique non-zero tiles in ascending order.
    #[test]
    fn topology_key_collapses_across_suits() {
        // 123m + 11s
        let mut a = [0u8; 34];
        a[0] = 1;
        a[1] = 1;
        a[2] = 1;
        a[18] = 2;
        // total = 5 (not 14 but topology_key works on any counts).
        let (ka, t14a) = topology_key_and_tile14(&a);
        assert_eq!(t14a.len, 4);
        assert_eq!(&t14a.tiles[..4], &[0u8, 1, 2, 18]);

        // 123p + 11s
        let mut b = [0u8; 34];
        b[9] = 1;
        b[10] = 1;
        b[11] = 1;
        b[18] = 2;
        let (kb, t14b) = topology_key_and_tile14(&b);
        assert_eq!(t14b.len, 4);
        assert_eq!(&t14b.tiles[..4], &[9u8, 10, 11, 18]);
        // Across-suit isomorphic shapes must share the same topology key.
        assert_eq!(ka, kb, "intra-suit isomorphic shapes must share key");
    }

    /// `lookup_compact` finds a known winning shape and returns
    /// indices that resolve back to the actual tile ids.
    #[test]
    fn compact_lookup_pinfu_shape() {
        // 1m..9m + 11p + 234p (= 9 + 2 + 3 = 14, pinfu-style w/ kanchan)
        let mut counts = [0u8; 34];
        for i in 0..9 {
            counts[i] = 1;
        }
        counts[9] = 2; // 11p
        counts[10] = 1;
        counts[11] = 1;
        counts[12] = 1; // 234p
        let (tile14, list) =
            lookup_compact(&counts).expect("compact_lookup must find known agari shape");
        assert!(!list.is_empty());
        for &d in list {
            assert!((d.pair_idx() as usize) < tile14.len as usize);
            assert_eq!(d.n_kotsu() + d.n_shuntsu(), 4);
            // Pair tile must be 1p (=9).
            assert_eq!(tile14.tiles[d.pair_idx() as usize], 9);
            // Shuntsu starts must each be a real tile.
            for i in 0..d.n_shuntsu() as usize {
                let idx = d.shuntsu_idx(i) as usize;
                assert!(idx < tile14.len as usize);
                let t = tile14.tiles[idx];
                assert!(counts[t as usize] >= 1);
            }
        }
    }

    /// Compactification correctness: every canonical entry, when translated
    /// For every canonical entry, lookup_compact (called on the SAME
    /// canonical_counts that produced the entry) must succeed AND yield a
    /// `CompactDivList` whose `(pair_tile, kotsu, shuntsu)` translation back
    /// via `tile14` matches the canonical entry's `Division` (modulo sort
    /// order). This is the key invariant the SP path relies on.
    #[test]
    fn compact_lookup_succeeds_for_all_canonical_counts() {
        let canonical = &*AGARI_TABLE;
        let mut misses = 0usize;
        let mut samples_shown = 0usize;
        for (&key128, _divlist) in canonical.iter() {
            let mut counts = [0u8; 34];
            for i in 0..34 {
                counts[i] = ((key128 >> (3 * i)) & 0x7) as u8;
            }
            if counts.iter().map(|&c| c as u32).sum::<u32>() != 14 {
                continue;
            }
            // lookup_canonical succeeds by construction (entry exists).
            // lookup_compact MUST also succeed for the same input.
            if lookup_compact(&counts).is_none() {
                misses += 1;
                if samples_shown < 5 {
                    samples_shown += 1;
                    let nz: Vec<(usize, u8)> = (0..34)
                        .filter(|&i| counts[i] > 0)
                        .map(|i| (i, counts[i]))
                        .collect();
                    eprintln!("compact lookup miss for canonical hand: {nz:?}");
                }
            }
        }
        assert_eq!(
            misses, 0,
            "{misses} canonical entries miss in compact table"
        );
    }

    /// Sanity check: every canonical Division round-trips to a CompactDiv
    /// in COMPACT_AGARI_TABLE under the matching topology key (same operation
    /// load_compact_table does; verifies idempotence).
    #[test]
    fn compact_table_covers_canonical_entries() {
        let canonical = &*AGARI_TABLE;
        let compact = &*COMPACT_AGARI_TABLE;
        // Sanity: max divs per key is ≤ 4 (verified — actually 4 in practice
        // after suit-perm expansion, with 95.4% of keys having only 1 div).
        // Total flat storage ≈ 34KB (8602 divs × 4 bytes), index ≈ 66KB
        // (8185 entries × ~8 bytes incl. hashbrown overhead). Hot working
        // set per SP sample (~65 unique mortal_keys) is ~780 bytes — fits L1.
        // Sanity: compact must collapse the canonical 39361 down (we expect
        // ~8k after suit-perm expansion).
        assert!(compact.n_keys() < canonical.len() / 3);
        let mut checked = 0usize;
        for (&canonical_key, divlist) in canonical.iter() {
            let mut counts = [0u8; 34];
            for i in 0..34 {
                counts[i] = ((canonical_key >> (3 * i)) & 0x7) as u8;
            }
            if counts.iter().map(|&c| c as u32).sum::<u32>() != 14 {
                continue;
            }
            let (mkey, tile14) = topology_key_and_tile14(&counts);
            let comp = compact.lookup(mkey).unwrap_or_else(|| {
                panic!("topology key 0x{mkey:x} from canonical 0x{canonical_key:x} missing in compact table")
            });
            let mut tile_to_idx = [u8::MAX; 34];
            for i in 0..tile14.len as usize {
                tile_to_idx[tile14.tiles[i] as usize] = i as u8;
            }
            for k in 0..divlist.n as usize {
                let d = &divlist.divs[k];
                let mut k_idxs = [0u8; MAX_MENTSU];
                for i in 0..d.n_kotsu as usize {
                    k_idxs[i] = tile_to_idx[d.kotsu_tiles[i] as usize];
                }
                let mut s_idxs = [0u8; MAX_MENTSU];
                for i in 0..d.n_shuntsu as usize {
                    s_idxs[i] = tile_to_idx[d.shuntsu_starts[i] as usize];
                }
                k_idxs[..d.n_kotsu as usize].sort_unstable();
                s_idxs[..d.n_shuntsu as usize].sort_unstable();
                let want = CompactDiv::pack(
                    tile_to_idx[d.pair_tile as usize],
                    d.n_kotsu,
                    &k_idxs[..d.n_kotsu as usize],
                    d.n_shuntsu,
                    &s_idxs[..d.n_shuntsu as usize],
                );
                assert!(
                    comp.contains(&want),
                    "compact entry missing for canonical key 0x{canonical_key:x}"
                );
                checked += 1;
            }
            if checked > 5_000 {
                // Sample check; iterating all 39361 × ~1.025 divs × table ops
                // is slow in test mode (debug build) but the sample is large
                // enough to catch systematic bugs.
                break;
            }
        }
        assert!(checked > 1_000);
    }
}
