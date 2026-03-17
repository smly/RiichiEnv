use rand::prelude::*;
use rand::rngs::StdRng;
use sha2::{Digest, Sha256};

use crate::types::{TILES_4P, is_sanma_excluded_tile};

/// # 3P dead wall layout (9 stacks = 18 tiles)
///
/// In 3P mahjong the dead wall has 9 stacks (18 tiles):
/// 4 stacks for rinshan + 5 stacks for dora indicators.
///
/// ```text
///             Dead wall (9 stacks = 18 tiles)                 Live wall
///   |<─────────────────────────────────────────────────>|<────────── ... ──>
///
///   Stack:  R1   R2   R3   R4    D1   D2   D3   D4   D5
///   Upper: [r0] [r2] [r4] [r6]  [d1] [d3] [d5] [d7] [d9]   ...
///   Lower: [r1] [r3] [r5] [r7]  [u1] [u3] [u5] [u7] [u9]   ...
///          ├── rinshan (8) ────┤├──── dora indicators (10) ────┤
/// ```
///
/// - **R1-R4** (4 stacks = 8 tiles): rinshan draw area for kan and kita.
///   Kita draws a rinshan tile but does NOT reveal a new dora indicator —
///   only kan triggers dora revelation.
/// - **D1-D5** (5 stacks = 10 tiles): dora indicator stacks.
///   Each stack holds one omote (dora indicator) and one ura (ura-dora).
///
/// For comparison, 4P has 7 stacks (14 tiles): 2 rinshan + 5 dora.
/// 3P doubles the rinshan area because kita also draws from it.
///
/// # Vec layout after reversal
///
/// ```text
///   tiles[0..8]    = rinshan R1-R4 (draw_rinshan_tile draws from here)
///   tiles[8..18]   = dora stacks D1-D5 (pre-extracted, not drawn directly)
///   tiles[18..108] = live wall (pop draws from here)
/// ```
/// Wall state for 3-player mahjong (108 tiles, sanma hardcoded).
#[derive(Debug, Clone)]
pub struct WallState3P {
    pub tiles: Vec<u8>,
    pub dora_indicators: Vec<u8>,
    /// Pre-extracted dora indicator tiles (omote) in order D1..D5.
    pub dora_indicator_tiles: [u8; 5],
    /// Pre-extracted ura dora indicator tiles in order U1..U5.
    pub ura_indicator_tiles: [u8; 5],
    pub rinshan_draw_count: u8,
    pub pending_kan_dora_count: u8,
    pub drawable_count: u8,
    pub wall_digest: String,
    pub salt: String,
    pub seed: Option<u64>,
    pub hand_index: u64,
}

impl WallState3P {
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            tiles: Vec::new(),
            dora_indicators: Vec::new(),
            dora_indicator_tiles: [0; 5],
            ura_indicator_tiles: [0; 5],
            rinshan_draw_count: 0,
            pending_kan_dora_count: 0,
            drawable_count: 0,
            wall_digest: String::new(),
            salt: String::new(),
            seed,
            hand_index: 0,
        }
    }

    pub fn shuffle(&mut self) {
        // 3P: 108 tiles (no 2m-8m)
        let mut w: Vec<u8> = (0..TILES_4P as u8)
            .filter(|&t| !is_sanma_excluded_tile(t))
            .collect();

        let mut rng = if let Some(episode_seed) = self.seed {
            let hand_seed = splitmix64(episode_seed.wrapping_add(self.hand_index));
            self.hand_index = self.hand_index.wrapping_add(1);
            StdRng::seed_from_u64(hand_seed)
        } else {
            self.hand_index = self.hand_index.wrapping_add(1);
            StdRng::from_rng(&mut rand::rng())
        };

        w.shuffle(&mut rng);
        self.salt = format!("{:016x}", rng.next_u64());

        // Calculate digest
        let mut hasher = Sha256::new();
        hasher.update(self.salt.as_bytes());
        for &t in &w {
            hasher.update([t]);
        }
        self.wall_digest = format!("{:x}", hasher.finalize());

        w.reverse();
        self.tiles = w;

        // Pre-extract dora/ura indicators from the dead wall layout.
        // After reversal: D_i omote at tiles[8+2i], ura at tiles[9+2i].
        for i in 0..5 {
            self.dora_indicator_tiles[i] = self.tiles[8 + 2 * i];
            self.ura_indicator_tiles[i] = self.tiles[9 + 2 * i];
        }

        self.dora_indicators.clear();
        self.dora_indicators.push(self.dora_indicator_tiles[0]);
        self.rinshan_draw_count = 0;
        self.pending_kan_dora_count = 0;
        self.drawable_count = 0;
    }

    /// Draw a tile for rinshan (dead wall draw after kan or kita).
    ///
    /// Takes from the dedicated rinshan area (8 tiles) at the front of the
    /// wall via `remove(0)`.  The maximum number of rinshan draws is 8
    /// (4 kan + 4 kita), which exactly matches the 8 dedicated slots.
    pub fn draw_rinshan_tile(&mut self) -> Option<u8> {
        if self.tiles.is_empty() {
            return None;
        }
        let tile = self.tiles.remove(0);
        self.drawable_count = self.drawable_count.saturating_sub(1);
        Some(tile)
    }

    pub fn load_wall(&mut self, tiles: Vec<u8>) {
        let mut t = tiles;

        // MjSoul 3P dead wall layout (positions 94-107):
        //   Positions 94-99: dora stacks 1-3 (each pair [X,X+1] = ura,omote)
        //     Stack1=[98,99]  Stack2=[96,97]  Stack3=[94,95]
        //   Positions 100-107: rinshan draw area (8 tiles for up to 8 draws: kans+kitas)
        // Dora stacks 4-5 extend into the live wall area (positions 90-93):
        //     Stack4=[92,93]  Stack5=[90,91]
        // These are pre-extracted before any draws, so it's safe even if
        // those live wall positions are later drawn during normal play.
        if t.len() == 108 {
            // D1..D5 omote indicators
            self.dora_indicator_tiles = [t[99], t[97], t[95], t[93], t[91]];
            // U1..U5 ura indicators
            self.ura_indicator_tiles = [t[98], t[96], t[94], t[92], t[90]];
        }

        t.reverse();
        self.tiles = t;
        self.dora_indicators.clear();
        self.dora_indicators.push(self.dora_indicator_tiles[0]);
        self.rinshan_draw_count = 0;
        self.pending_kan_dora_count = 0;
        self.drawable_count = 0;
    }
}

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for issue #183: `draw_rinshan_tile` must never
    /// consume dora/ura indicator tiles.
    #[test]
    fn test_rinshan_draw_does_not_consume_dora_indicators() {
        for seed in 0..1000 {
            let mut wall = WallState3P::new(Some(seed));
            wall.shuffle();

            // Save the dora/ura indicator tile IDs (these must never be drawn).
            let dora_tiles: Vec<u8> = wall.dora_indicator_tiles.to_vec();
            let ura_tiles: Vec<u8> = wall.ura_indicator_tiles.to_vec();
            let indicator_set: std::collections::HashSet<u8> =
                dora_tiles.iter().chain(ura_tiles.iter()).copied().collect();

            // Simulate dealing: pop ~40 tiles from the back (live wall).
            for _ in 0..40 {
                wall.tiles.pop();
            }
            wall.drawable_count = (wall.tiles.len() as u8).saturating_sub(14);

            // Simulate up to 8 rinshan draws (4 kans + 4 kitas max).
            for draw_num in 0..8 {
                if wall.drawable_count == 0 {
                    break;
                }
                let t = wall
                    .draw_rinshan_tile()
                    .expect("draw_rinshan_tile should succeed");
                wall.rinshan_draw_count += 1;

                assert!(
                    !indicator_set.contains(&t),
                    "seed={seed}, draw #{draw_num}: rinshan drew tile {t} \
                     which is a dora/ura indicator! dora={dora_tiles:?}, ura={ura_tiles:?}"
                );
            }
        }
    }

    /// Verify the dead wall layout: 8 rinshan slots at [0..8], dora
    /// indicators at [8..18].
    #[test]
    fn test_dead_wall_layout_has_8_rinshan_slots() {
        let mut wall = WallState3P::new(Some(42));
        wall.shuffle();

        assert_eq!(wall.tiles.len(), 108);
        for i in 0..5 {
            assert_eq!(wall.dora_indicator_tiles[i], wall.tiles[8 + 2 * i]);
            assert_eq!(wall.ura_indicator_tiles[i], wall.tiles[9 + 2 * i]);
        }
    }

    /// Verify that the old behavior (always remove(0) with only 4 rinshan
    /// slots) WOULD have consumed dora indicators on the 5th+ draw.
    #[test]
    fn test_old_layout_remove0_would_consume_dora_indicators() {
        // Simulate the OLD layout: dora at tiles[4..14], only 4 rinshan slots.
        let mut found_collision = false;
        for seed in 0..100 {
            let mut wall = WallState3P::new(Some(seed));
            wall.shuffle();

            // Reconstruct old-style indicator positions (tiles[4..14])
            let mut old_indicators = std::collections::HashSet::new();
            for i in 0..5 {
                old_indicators.insert(wall.tiles[4 + 2 * i]);
                old_indicators.insert(wall.tiles[5 + 2 * i]);
            }

            // Simulate dealing
            for _ in 0..40 {
                wall.tiles.pop();
            }

            // Old behavior: always remove(0) with 4 rinshan slots
            for draw_num in 0..8 {
                if wall.tiles.is_empty() {
                    break;
                }
                let t = wall.tiles.remove(0);
                if draw_num >= 4 && old_indicators.contains(&t) {
                    found_collision = true;
                    break;
                }
            }
            if found_collision {
                break;
            }
        }
        assert!(
            found_collision,
            "Expected the old remove(0) approach with 4 rinshan slots to \
             consume dora indicators on the 5th+ draw"
        );
    }
}
