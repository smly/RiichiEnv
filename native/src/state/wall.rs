use rand::prelude::*;
use rand::rngs::StdRng;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone)]
pub struct WallState {
    pub tiles: Vec<u8>,
    pub dora_indicators: Vec<u8>,
    pub rinshan_draw_count: u8,
    pub pending_kan_dora_count: u8,
    pub wall_digest: String,
    pub salt: String,
    pub seed: Option<u64>,
    pub hand_index: u64,
}

impl WallState {
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            tiles: Vec::new(),
            dora_indicators: Vec::new(),
            rinshan_draw_count: 0,
            pending_kan_dora_count: 0,
            wall_digest: String::new(),
            salt: String::new(),
            seed,
            hand_index: 0,
        }
    }

    pub fn shuffle(&mut self) {
        let mut w: Vec<u8> = (0..136).collect();
        let mut rng = if let Some(episode_seed) = self.seed {
            let hand_seed = splitmix64(episode_seed.wrapping_add(self.hand_index));
            self.hand_index = self.hand_index.wrapping_add(1);
            StdRng::seed_from_u64(hand_seed)
        } else {
            self.hand_index = self.hand_index.wrapping_add(1);
            StdRng::from_entropy()
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

        w.reverse(); // Match original logic (pop from end)
        self.tiles = w;

        // Initial dora indicator (index 4 in the dead wall, which is at the start of the vector)
        // 0-3: Rinshan
        // 4: Dora Indicator 1
        // 5: Ura Dora 1
        // ...
        self.dora_indicators.clear();
        if self.tiles.len() > 5 {
            self.dora_indicators.push(self.tiles[4]);
        }
        self.rinshan_draw_count = 0;
        self.pending_kan_dora_count = 0;
    }

    // Load wall tiles from a provided vector (used when manually setting the wall)
    pub fn load_wall(&mut self, tiles: Vec<u8>) {
        // When loading manually (e.g. from event), we don't shuffle or use seed
        // But we expect tiles to be set.
        // If passed wall is normal order, we reverse it?
        // Usually reset() takes `wall: Option<Vec<u8>>`.
        // If provided, we assume it needs reversing if it mimics shuffling?
        // Original reset() -> if let Some(mut w) = wall { w.reverse(); self.wall = w; }
        let mut t = tiles;
        t.reverse();
        self.tiles = t;
        self.dora_indicators.clear();
        if self.tiles.len() > 5 {
            self.dora_indicators.push(self.tiles[4]);
        }
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
