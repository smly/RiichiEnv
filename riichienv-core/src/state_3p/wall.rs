use rand::prelude::*;
use rand::rngs::StdRng;
use sha2::{Digest, Sha256};

use crate::types::is_sanma_excluded_tile;

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
            wall_digest: String::new(),
            salt: String::new(),
            seed,
            hand_index: 0,
        }
    }

    pub fn shuffle(&mut self) {
        // 3P: 108 tiles (no 2m-8m)
        let mut w: Vec<u8> = (0..136u8).filter(|&t| !is_sanma_excluded_tile(t)).collect();

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

        // Pre-extract dora/ura indicators from standard layout.
        // After reversal: D_i omote at tiles[4+2i], ura at tiles[5+2i].
        for i in 0..5 {
            self.dora_indicator_tiles[i] = self.tiles[4 + 2 * i];
            self.ura_indicator_tiles[i] = self.tiles[5 + 2 * i];
        }

        self.dora_indicators.clear();
        self.dora_indicators.push(self.dora_indicator_tiles[0]);
        self.rinshan_draw_count = 0;
        self.pending_kan_dora_count = 0;
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
    }
}

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}
