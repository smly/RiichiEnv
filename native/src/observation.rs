use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};
use serde::{Deserialize, Serialize};

use crate::action::{Action, ActionType};
use crate::shanten;
use crate::types::{Meld, MeldType};
use crate::yaku_checker;
use ndarray::prelude::*;

/// Helper: write a scalar value broadcast across 34 tile positions into a flat buffer.
/// buf layout: channel-major, i.e. buf[(ch_offset + ch) * 34 + tile] = val
#[inline]
fn broadcast_scalar(buf: &mut [f32], ch_offset: usize, ch: usize, val: f32) {
    let start = (ch_offset + ch) * 34;
    for j in 0..34 {
        buf[start + j] = val;
    }
}

/// Helper: set a single value in the flat buffer.
#[inline]
fn set_val(buf: &mut [f32], ch_offset: usize, ch: usize, tile: usize, val: f32) {
    buf[(ch_offset + ch) * 34 + tile] = val;
}

/// Helper: add a value in the flat buffer.
#[inline]
fn add_val(buf: &mut [f32], ch_offset: usize, ch: usize, tile: usize, val: f32) {
    buf[(ch_offset + ch) * 34 + tile] += val;
}

/// Get the next tile in sequence for dora calculation
/// Tile type: 0-33 (m1-m9: 0-8, p1-p9: 9-17, s1-s9: 18-26, winds: 27-30, dragons: 31-33)
fn get_next_tile(tile: u32) -> u8 {
    let tile_type = (tile / 4) / 9; // 0=man, 1=pin, 2=sou, 3+=honors
    let tile_num = (tile / 4) % 9;

    if tile_type < 3 {
        // Number tiles (1-9)
        let next_num = if tile_num == 8 { 0 } else { tile_num + 1 };
        ((tile_type * 9 + next_num) * 4) as u8
    } else {
        // Honor tiles (winds 27-30, dragons 31-33)
        let base = tile / 4;
        if (27..31).contains(&base) {
            // Winds: E->S->W->N->E
            let wind_idx = base - 27;
            let next_wind = (wind_idx + 1) % 4;
            ((27 + next_wind) * 4) as u8
        } else if (31..34).contains(&base) {
            // Dragons: White->Green->Red->White
            let dragon_idx = base - 31;
            let next_dragon = (dragon_idx + 1) % 3;
            ((31 + next_dragon) * 4) as u8
        } else {
            tile as u8 // Fallback
        }
    }
}

#[pyclass(module = "riichienv._riichienv")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    #[pyo3(get)]
    pub player_id: u8,
    #[pyo3(get)]
    pub hands: Vec<Vec<u32>>,
    #[pyo3(get)]
    pub melds: Vec<Vec<Meld>>,
    #[pyo3(get)]
    pub discards: Vec<Vec<u32>>,
    #[pyo3(get)]
    pub dora_indicators: Vec<u32>,
    #[pyo3(get)]
    pub scores: Vec<i32>,
    #[pyo3(get)]
    pub riichi_declared: Vec<bool>,

    #[serde(skip)]
    pub _legal_actions: Vec<Action>,

    pub events: Vec<String>,

    #[pyo3(get)]
    pub honba: u8,
    #[pyo3(get)]
    pub riichi_sticks: u32,
    #[pyo3(get)]
    pub round_wind: u8,
    #[pyo3(get)]
    pub oya: u8,
    #[pyo3(get)]
    pub kyoku_index: u8,
    #[pyo3(get)]
    pub waits: Vec<u8>,
    #[pyo3(get)]
    pub is_tenpai: bool,
    #[pyo3(get)]
    pub tsumogiri_flags: Vec<Vec<bool>>,
    #[pyo3(get)]
    pub riichi_sutehais: Vec<Option<u8>>,
    #[pyo3(get)]
    pub last_tedashis: Vec<Option<u8>>,
    #[pyo3(get)]
    pub last_discard: Option<u32>,
}

/// Internal (non-PyO3) methods that write features directly into a flat f32 buffer.
/// Buffer layout: channel-major, buf[(ch_offset + ch) * 34 + tile] = value.
impl Observation {
    /// Write 74 base encode channels into buf starting at ch_offset.
    fn encode_base_into(&self, buf: &mut [f32], ch_offset: usize) {
        // Hand (ch 0-3) + Red (ch 4)
        if (self.player_id as usize) < self.hands.len() {
            let mut counts = [0u8; 34];
            for &t in &self.hands[self.player_id as usize] {
                let idx = (t as usize) / 4;
                if idx < 34 {
                    counts[idx] += 1;
                    if t == 16 || t == 52 || t == 88 {
                        set_val(buf, ch_offset, 4, idx, 1.0);
                    }
                }
            }
            for (i, &c) in counts.iter().enumerate() {
                if c >= 1 {
                    set_val(buf, ch_offset, 0, i, 1.0);
                }
                if c >= 2 {
                    set_val(buf, ch_offset, 1, i, 1.0);
                }
                if c >= 3 {
                    set_val(buf, ch_offset, 2, i, 1.0);
                }
                if c >= 4 {
                    set_val(buf, ch_offset, 3, i, 1.0);
                }
            }
        }

        // Melds (Self) (ch 5-8)
        if (self.player_id as usize) < self.melds.len() {
            for (m_idx, meld) in self.melds[self.player_id as usize].iter().enumerate() {
                if m_idx >= 4 {
                    break;
                }
                for &t in &meld.tiles {
                    let idx = (t as usize) / 4;
                    if idx < 34 {
                        set_val(buf, ch_offset, 5 + m_idx, idx, 1.0);
                    }
                }
            }
        }

        // Dora Indicators (ch 9)
        for &t in &self.dora_indicators {
            let idx = (t as usize) / 4;
            if idx < 34 {
                set_val(buf, ch_offset, 9, idx, 1.0);
            }
        }

        // Self discards last 4 (ch 10-13)
        if (self.player_id as usize) < self.discards.len() {
            let discs = &self.discards[self.player_id as usize];
            for (i, &t) in discs.iter().rev().take(4).enumerate() {
                let idx = (t as usize) / 4;
                if idx < 34 {
                    set_val(buf, ch_offset, 10 + i, idx, 1.0);
                }
            }
        }

        // Opponents discards last 4 (ch 14-25)
        for i in 1..4u8 {
            let opp_id = (self.player_id + i) % 4;
            if (opp_id as usize) < self.discards.len() {
                let discs = &self.discards[opp_id as usize];
                for (j, &t) in discs.iter().rev().take(4).enumerate() {
                    let idx = (t as usize) / 4;
                    if idx < 34 {
                        let ch = 14 + (i as usize - 1) * 4 + j;
                        set_val(buf, ch_offset, ch, idx, 1.0);
                    }
                }
            }
        }

        // Discard counts (ch 26-29)
        for (player_idx, discs) in self.discards.iter().enumerate() {
            let count_norm = (discs.len() as f32) / 24.0;
            broadcast_scalar(buf, ch_offset, 26 + player_idx, count_norm);
        }

        // Tiles left in wall (ch 30)
        let mut tiles_used = 0;
        for discs in &self.discards {
            tiles_used += discs.len();
        }
        for melds_list in &self.melds {
            for meld in melds_list {
                tiles_used += meld.tiles.len();
                // Subtract 1 for claimed tile (already counted in discards)
                if meld.called_tile.is_some() {
                    tiles_used -= 1;
                }
            }
        }
        if (self.player_id as usize) < self.hands.len() {
            tiles_used += self.hands[self.player_id as usize].len();
        }
        tiles_used += self.dora_indicators.len();
        let tiles_left = (136_i32 - tiles_used as i32).max(0) as f32;
        broadcast_scalar(buf, ch_offset, 30, tiles_left / 70.0);

        // Riichi (ch 31-34)
        if (self.player_id as usize) < self.riichi_declared.len()
            && self.riichi_declared[self.player_id as usize]
        {
            broadcast_scalar(buf, ch_offset, 31, 1.0);
        }
        for i in 1..4u8 {
            let opp_id = (self.player_id + i) % 4;
            if (opp_id as usize) < self.riichi_declared.len()
                && self.riichi_declared[opp_id as usize]
            {
                broadcast_scalar(buf, ch_offset, 32 + (i as usize - 1), 1.0);
            }
        }

        // Winds (ch 35-36)
        let rw = self.round_wind as usize;
        if 27 + rw < 34 {
            set_val(buf, ch_offset, 35, 27 + rw, 1.0);
        }
        let seat = (self.player_id + 4 - self.oya) % 4;
        if 27 + (seat as usize) < 34 {
            set_val(buf, ch_offset, 36, 27 + (seat as usize), 1.0);
        }

        // Honba/Sticks (ch 37-38)
        broadcast_scalar(buf, ch_offset, 37, (self.honba as f32) / 10.0);
        broadcast_scalar(buf, ch_offset, 38, (self.riichi_sticks as f32) / 5.0);

        // Scores (ch 39-46)
        for i in 0..4 {
            if i < self.scores.len() {
                broadcast_scalar(
                    buf,
                    ch_offset,
                    39 + i,
                    (self.scores[i].clamp(0, 100000) as f32) / 100000.0,
                );
                broadcast_scalar(
                    buf,
                    ch_offset,
                    43 + i,
                    (self.scores[i].clamp(0, 30000) as f32) / 30000.0,
                );
            }
        }

        // Waits (ch 47)
        for &t in &self.waits {
            if (t as usize) < 34 {
                set_val(buf, ch_offset, 47, t as usize, 1.0);
            }
        }

        // Is Tenpai (ch 48)
        broadcast_scalar(buf, ch_offset, 48, if self.is_tenpai { 1.0 } else { 0.0 });

        // Rank (ch 49-52)
        let my_score = self
            .scores
            .get(self.player_id as usize)
            .copied()
            .unwrap_or(0);
        let mut rank = 0;
        for &s in &self.scores {
            if s > my_score {
                rank += 1;
            }
        }
        if rank < 4 {
            broadcast_scalar(buf, ch_offset, 49 + rank, 1.0);
        }

        // Kyoku (ch 53)
        broadcast_scalar(buf, ch_offset, 53, (self.kyoku_index as f32) / 8.0);

        // Round Progress (ch 54)
        let round_progress = (self.round_wind as f32) * 4.0 + (self.kyoku_index as f32);
        broadcast_scalar(buf, ch_offset, 54, round_progress / 7.0);

        // Dora Count (ch 55-58)
        let mut dora_counts = [0u8; 4];
        for (player_idx, dora_count) in dora_counts.iter_mut().enumerate() {
            if player_idx < self.melds.len() {
                for meld in &self.melds[player_idx] {
                    for &tile in &meld.tiles {
                        for &dora_ind in &self.dora_indicators {
                            let dora_tile = get_next_tile(dora_ind);
                            if (tile / 4) == (dora_tile / 4) {
                                *dora_count += 1;
                            }
                        }
                    }
                }
            }
            if player_idx < self.discards.len() {
                for &tile in &self.discards[player_idx] {
                    for &dora_ind in &self.dora_indicators {
                        let dora_tile = get_next_tile(dora_ind);
                        if ((tile / 4) as u8) == (dora_tile / 4) {
                            *dora_count += 1;
                        }
                    }
                }
            }
        }
        if (self.player_id as usize) < self.hands.len() {
            for &tile in &self.hands[self.player_id as usize] {
                for &dora_ind in &self.dora_indicators {
                    let dora_tile = get_next_tile(dora_ind);
                    if ((tile / 4) as u8) == (dora_tile / 4) {
                        dora_counts[self.player_id as usize] += 1;
                    }
                }
            }
        }
        for (i, &dc) in dora_counts.iter().enumerate() {
            broadcast_scalar(buf, ch_offset, 55 + i, (dc as f32) / 12.0);
        }

        // Melds Count (ch 59-62)
        for (player_idx, melds_list) in self.melds.iter().enumerate() {
            broadcast_scalar(
                buf,
                ch_offset,
                59 + player_idx,
                (melds_list.len() as f32) / 4.0,
            );
        }

        // Tiles Seen (ch 63)
        let mut seen = [0u8; 34];
        if (self.player_id as usize) < self.hands.len() {
            for &t in &self.hands[self.player_id as usize] {
                seen[(t as usize) / 4] += 1;
            }
        }
        for mlist in &self.melds {
            for m in mlist {
                for &t in &m.tiles {
                    seen[(t as usize) / 4] += 1;
                }
            }
        }
        for dlist in &self.discards {
            for &t in dlist {
                seen[(t as usize) / 4] += 1;
            }
        }
        for &t in &self.dora_indicators {
            seen[(t as usize) / 4] += 1;
        }
        for (i, &s) in seen.iter().enumerate() {
            set_val(buf, ch_offset, 63, i, (s as f32) / 4.0);
        }

        // Extended discards self (ch 64-67)
        if (self.player_id as usize) < self.discards.len() {
            let discs = &self.discards[self.player_id as usize];
            for (i, &t) in discs.iter().rev().skip(4).take(4).enumerate() {
                let idx = (t as usize) / 4;
                if idx < 34 {
                    set_val(buf, ch_offset, 64 + i, idx, 1.0);
                }
            }
        }

        // Extended discards opponent 1 (ch 68-69)
        let opp1_id = ((self.player_id + 1) % 4) as usize;
        if opp1_id < self.discards.len() {
            let discs = &self.discards[opp1_id];
            for (i, &t) in discs.iter().rev().skip(4).take(2).enumerate() {
                let idx = (t as usize) / 4;
                if idx < 34 {
                    set_val(buf, ch_offset, 68 + i, idx, 1.0);
                }
            }
        }

        // Tsumogiri flags (ch 70-73)
        for player_idx in 0..4 {
            if player_idx < self.tsumogiri_flags.len()
                && !self.tsumogiri_flags[player_idx].is_empty()
            {
                let last_tsumogiri = *self.tsumogiri_flags[player_idx].last().unwrap_or(&false);
                broadcast_scalar(
                    buf,
                    ch_offset,
                    70 + player_idx,
                    if last_tsumogiri { 1.0 } else { 0.0 },
                );
            }
        }
    }

    /// Write 4 discard history decay channels into buf starting at ch_offset.
    fn encode_discard_decay_into(&self, buf: &mut [f32], ch_offset: usize) {
        let decay_rate = 0.2f32;
        for player_idx in 0..4 {
            if player_idx >= self.discards.len() {
                continue;
            }
            let discs = &self.discards[player_idx];
            let max_len = discs.len();
            if max_len == 0 {
                continue;
            }
            for (turn, &tile) in discs.iter().enumerate() {
                let tile_idx = (tile as usize) / 4;
                if tile_idx < 34 {
                    let age = (max_len - 1 - turn) as f32;
                    let weight = (-decay_rate * age).exp();
                    add_val(buf, ch_offset, player_idx, tile_idx, weight);
                }
            }
        }
    }

    /// Write 16 shanten efficiency channels (broadcast) into buf starting at ch_offset.
    /// 4 players × 4 features = 16 channels, each broadcast to 34 tiles.
    fn encode_shanten_into(&self, buf: &mut [f32], ch_offset: usize) {
        let mut all_visible: Vec<u32> = Vec::new();
        for discs in &self.discards {
            all_visible.extend(discs.iter().copied());
        }
        for melds_list in &self.melds {
            for meld in melds_list {
                all_visible.extend(meld.tiles.iter().map(|&x| x as u32));
            }
        }
        all_visible.extend(self.dora_indicators.iter().copied());

        for player_idx in 0..4 {
            if player_idx >= self.hands.len() {
                continue;
            }
            let base_ch = player_idx * 4;

            if player_idx == self.player_id as usize {
                let hand = &self.hands[player_idx];
                let shanten_val = shanten::calculate_shanten(hand);
                let effective = shanten::calculate_effective_tiles(hand);
                let best_ukeire = shanten::calculate_best_ukeire(hand, &all_visible);

                broadcast_scalar(buf, ch_offset, base_ch, (shanten_val as f32).max(0.0) / 8.0);
                broadcast_scalar(buf, ch_offset, base_ch + 1, (effective as f32) / 34.0);
                broadcast_scalar(buf, ch_offset, base_ch + 2, (best_ukeire as f32) / 80.0);
            } else {
                broadcast_scalar(buf, ch_offset, base_ch, 0.5);
                broadcast_scalar(buf, ch_offset, base_ch + 1, 0.5);
                broadcast_scalar(buf, ch_offset, base_ch + 2, 0.5);
            }

            let turn_count = if player_idx < self.discards.len() {
                self.discards[player_idx].len() as f32
            } else {
                0.0
            };
            broadcast_scalar(buf, ch_offset, base_ch + 3, (turn_count / 18.0).min(1.0));
        }
    }

    /// Write 4 ankan overview channels into buf starting at ch_offset.
    fn encode_ankan_into(&self, buf: &mut [f32], ch_offset: usize) {
        for (player_idx, melds) in self.melds.iter().enumerate() {
            if player_idx >= 4 {
                break;
            }
            for meld in melds {
                if matches!(meld.meld_type, MeldType::Ankan) {
                    if let Some(&tile) = meld.tiles.first() {
                        let tile_type = (tile / 4) as usize;
                        if tile_type < 34 {
                            set_val(buf, ch_offset, player_idx, tile_type, 1.0);
                        }
                    }
                }
            }
        }
    }

    /// Write 80 fuuro overview channels into buf starting at ch_offset.
    /// Layout: player(4) × meld(4) × tile_slot(5) flattened = 80 channels, each spatial (34).
    fn encode_fuuro_into(&self, buf: &mut [f32], ch_offset: usize) {
        for (player_idx, melds) in self.melds.iter().enumerate() {
            if player_idx >= 4 {
                break;
            }
            for (meld_idx, meld) in melds.iter().enumerate() {
                if meld_idx >= 4 {
                    break;
                }
                for (tile_slot_idx, &tile) in meld.tiles.iter().enumerate() {
                    if tile_slot_idx >= 4 {
                        break;
                    }
                    let tile_type = (tile / 4) as usize;
                    if tile_type < 34 {
                        // channel = player*20 + meld*5 + slot
                        let ch = player_idx * 20 + meld_idx * 5 + tile_slot_idx;
                        set_val(buf, ch_offset, ch, tile_type, 1.0);
                    }
                    // Check for aka (red five: 5m=16, 5p=52, 5s=88)
                    if matches!(tile, 16 | 52 | 88) {
                        let ch = player_idx * 20 + meld_idx * 5 + 4;
                        set_val(buf, ch_offset, ch, (tile / 4) as usize, 1.0);
                    }
                }
            }
        }
    }

    /// Write 11 action availability channels (broadcast) into buf starting at ch_offset.
    fn encode_action_avail_into(&self, buf: &mut [f32], ch_offset: usize) {
        for action in &self._legal_actions {
            match action.action_type {
                ActionType::Riichi => broadcast_scalar(buf, ch_offset, 0, 1.0),
                ActionType::Chi => {
                    let tiles = &action.consume_tiles;
                    if tiles.len() == 2 {
                        let t0 = tiles[0] / 4;
                        let t1 = tiles[1] / 4;
                        let diff = (t1 as i32 - t0 as i32).abs();
                        if diff == 1 {
                            if t0 < t1 {
                                broadcast_scalar(buf, ch_offset, 1, 1.0);
                            } else {
                                broadcast_scalar(buf, ch_offset, 3, 1.0);
                            }
                        } else if diff == 2 {
                            broadcast_scalar(buf, ch_offset, 2, 1.0);
                        }
                    }
                }
                ActionType::Pon => broadcast_scalar(buf, ch_offset, 4, 1.0),
                ActionType::Daiminkan => broadcast_scalar(buf, ch_offset, 5, 1.0),
                ActionType::Ankan => broadcast_scalar(buf, ch_offset, 6, 1.0),
                ActionType::Kakan => broadcast_scalar(buf, ch_offset, 7, 1.0),
                ActionType::Tsumo | ActionType::Ron => broadcast_scalar(buf, ch_offset, 8, 1.0),
                ActionType::KyushuKyuhai => broadcast_scalar(buf, ch_offset, 9, 1.0),
                ActionType::Pass => broadcast_scalar(buf, ch_offset, 10, 1.0),
                _ => {}
            }
        }
    }

    /// Write 5 discard candidates channels (broadcast) into buf starting at ch_offset.
    fn encode_discard_cand_into(&self, buf: &mut [f32], ch_offset: usize) {
        let player_idx = self.player_id as usize;
        if player_idx >= self.hands.len() {
            return;
        }

        let hand = &self.hands[player_idx];
        let current_shanten = shanten::calculate_shanten(hand);

        broadcast_scalar(buf, ch_offset, 0, hand.len() as f32 / 34.0);

        let mut keep_count = 0;
        let mut increase_count = 0;
        for (idx, _) in hand.iter().enumerate() {
            let new_hand: Vec<u32> = hand
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(_, &t)| t)
                .collect();
            let new_shanten = shanten::calculate_shanten(&new_hand);
            if new_shanten == current_shanten {
                keep_count += 1;
            } else if new_shanten > current_shanten {
                increase_count += 1;
            }
        }
        if !hand.is_empty() {
            broadcast_scalar(buf, ch_offset, 1, keep_count as f32 / hand.len() as f32);
            broadcast_scalar(buf, ch_offset, 2, increase_count as f32 / hand.len() as f32);
        }
        broadcast_scalar(
            buf,
            ch_offset,
            3,
            if current_shanten == -1 { 1.0 } else { 0.0 },
        );
        broadcast_scalar(
            buf,
            ch_offset,
            4,
            if player_idx < self.riichi_declared.len() && self.riichi_declared[player_idx] {
                1.0
            } else {
                0.0
            },
        );
    }

    /// Write 3 pass context channels (broadcast) into buf starting at ch_offset.
    fn encode_pass_ctx_into(&self, buf: &mut [f32], ch_offset: usize) {
        if let Some(tile) = self.last_discard {
            let tile_type = (tile / 4) as usize;
            broadcast_scalar(buf, ch_offset, 0, tile_type as f32 / 33.0);
            broadcast_scalar(
                buf,
                ch_offset,
                1,
                if matches!(tile, 16 | 52 | 88) {
                    1.0
                } else {
                    0.0
                },
            );

            let dora_tiles: Vec<u8> = self
                .dora_indicators
                .iter()
                .map(|&ind| get_next_tile(ind))
                .collect();
            broadcast_scalar(
                buf,
                ch_offset,
                2,
                if dora_tiles.contains(&(tile as u8)) {
                    1.0
                } else {
                    0.0
                },
            );
        }
    }

    /// Write 9 last tedashis channels (broadcast) into buf starting at ch_offset.
    fn encode_last_ted_into(&self, buf: &mut [f32], ch_offset: usize) {
        let dora_tiles: Vec<u8> = self
            .dora_indicators
            .iter()
            .map(|&ind| get_next_tile(ind))
            .collect();

        let mut opp_idx = 0;
        for player_id in 0..4 {
            if player_id == self.player_id as usize {
                continue;
            }
            if let Some(tile) = self.last_tedashis[player_id] {
                let tile_type = (tile / 4) as usize;
                broadcast_scalar(buf, ch_offset, opp_idx * 3, tile_type as f32 / 33.0);
                broadcast_scalar(
                    buf,
                    ch_offset,
                    opp_idx * 3 + 1,
                    if matches!(tile, 16 | 52 | 88) {
                        1.0
                    } else {
                        0.0
                    },
                );
                broadcast_scalar(
                    buf,
                    ch_offset,
                    opp_idx * 3 + 2,
                    if dora_tiles.contains(&tile) { 1.0 } else { 0.0 },
                );
            }
            opp_idx += 1;
        }
    }

    /// Write 9 riichi sutehais channels (broadcast) into buf starting at ch_offset.
    fn encode_riichi_sute_into(&self, buf: &mut [f32], ch_offset: usize) {
        let dora_tiles: Vec<u8> = self
            .dora_indicators
            .iter()
            .map(|&ind| get_next_tile(ind))
            .collect();

        let mut opp_idx = 0;
        for player_id in 0..4 {
            if player_id == self.player_id as usize {
                continue;
            }
            if let Some(tile) = self.riichi_sutehais[player_id] {
                let tile_type = (tile / 4) as usize;
                broadcast_scalar(buf, ch_offset, opp_idx * 3, tile_type as f32 / 33.0);
                broadcast_scalar(
                    buf,
                    ch_offset,
                    opp_idx * 3 + 1,
                    if matches!(tile, 16 | 52 | 88) {
                        1.0
                    } else {
                        0.0
                    },
                );
                broadcast_scalar(
                    buf,
                    ch_offset,
                    opp_idx * 3 + 2,
                    if dora_tiles.contains(&tile) { 1.0 } else { 0.0 },
                );
            }
            opp_idx += 1;
        }
    }

    /// Write 89 win projection channels into buf starting at ch_offset.
    fn encode_win_projection_into(&self, buf: &mut [f32], ch_offset: usize) {
        use crate::types::TILE_MAX;
        use crate::win_projection;

        let player_idx = self.player_id as usize;
        if player_idx >= self.hands.len() {
            return;
        }

        let hand_136 = &self.hands[player_idx];

        let mut tehai = [0u8; TILE_MAX];
        let mut akas_in_hand = [false; 3];
        for &t in hand_136 {
            let tile_type = (t / 4) as usize;
            if tile_type < TILE_MAX {
                tehai[tile_type] += 1;
            }
            match t {
                16 => akas_in_hand[0] = true,
                52 => akas_in_hand[1] = true,
                88 => akas_in_hand[2] = true,
                _ => {}
            }
        }

        let mut tiles_seen = [0u8; TILE_MAX];
        let mut akas_seen = [false; 3];
        for &t in hand_136 {
            let tt = (t / 4) as usize;
            if tt < TILE_MAX {
                tiles_seen[tt] += 1;
            }
            match t {
                16 => akas_seen[0] = true,
                52 => akas_seen[1] = true,
                88 => akas_seen[2] = true,
                _ => {}
            }
        }
        for melds_list in &self.melds {
            for meld in melds_list {
                for &tile in &meld.tiles {
                    let tt = (tile / 4) as usize;
                    if tt < TILE_MAX {
                        tiles_seen[tt] += 1;
                    }
                    match tile {
                        16 => akas_seen[0] = true,
                        52 => akas_seen[1] = true,
                        88 => akas_seen[2] = true,
                        _ => {}
                    }
                }
            }
        }
        for discs in &self.discards {
            for &tile in discs {
                let tt = (tile / 4) as usize;
                if tt < TILE_MAX {
                    tiles_seen[tt] += 1;
                }
                match tile {
                    16 => akas_seen[0] = true,
                    52 => akas_seen[1] = true,
                    88 => akas_seen[2] = true,
                    _ => {}
                }
            }
        }
        for &t in &self.dora_indicators {
            let tt = (t / 4) as usize;
            if tt < TILE_MAX {
                tiles_seen[tt] += 1;
            }
            match t {
                16 => akas_seen[0] = true,
                52 => akas_seen[1] = true,
                88 => akas_seen[2] = true,
                _ => {}
            }
        }
        // Subtract double-counted tiles: claimed tiles appear in both discards and melds
        for melds_list in &self.melds {
            for meld in melds_list {
                if let Some(ct) = meld.called_tile {
                    let tt = (ct as u32 / 4) as usize;
                    if tt < TILE_MAX && tiles_seen[tt] > 0 {
                        tiles_seen[tt] -= 1;
                    }
                }
            }
        }
        for ts in tiles_seen.iter_mut().take(TILE_MAX) {
            *ts = (*ts).min(4);
        }

        let num_melds = if player_idx < self.melds.len() {
            self.melds[player_idx].len()
        } else {
            0
        };
        let tehai_len_div3 = (4 - num_melds) as u8;
        let is_menzen = if player_idx < self.melds.len() {
            self.melds[player_idx]
                .iter()
                .all(|m| matches!(m.meld_type, MeldType::Ankan))
        } else {
            true
        };

        let seat = (self.player_id + 4 - self.oya) % 4;
        let is_oya = seat == 0;
        let round_wind_tile = 27 + self.round_wind;
        let seat_wind_tile = 27 + seat;

        let dora_indicators_34: Vec<u8> = self
            .dora_indicators
            .iter()
            .map(|&t| (t / 4) as u8)
            .collect();

        let mut num_doras_in_fuuro = 0u8;
        let mut num_aka_in_fuuro = 0u8;
        if player_idx < self.melds.len() {
            for meld in &self.melds[player_idx] {
                for &tile in &meld.tiles {
                    for &ind in &dora_indicators_34 {
                        if tile / 4 == win_projection::next_dora_tile(ind) {
                            num_doras_in_fuuro += 1;
                        }
                    }
                    match tile {
                        16 | 52 | 88 => num_aka_in_fuuro += 1,
                        _ => {}
                    }
                }
            }
        }

        let melds_for_calc: Vec<Meld> = if player_idx < self.melds.len() {
            self.melds[player_idx]
                .iter()
                .map(|m| Meld {
                    meld_type: m.meld_type,
                    tiles: m.tiles.iter().map(|&t| t / 4).collect(),
                    opened: m.opened,
                    from_who: m.from_who,
                    called_tile: m.called_tile.map(|t| t / 4),
                })
                .collect()
        } else {
            Vec::new()
        };

        let total_visible: u32 = tiles_seen.iter().map(|&c| c as u32).sum();
        let tiles_in_wall = (136u32).saturating_sub(total_visible);
        let tsumos_left = ((tiles_in_wall / 4) as usize).min(win_projection::MAX_TSUMOS_LEFT);
        let hand_tile_count: u32 = tehai.iter().map(|&c| c as u32).sum();
        let can_discard = hand_tile_count % 3 == 2;

        let calculator = win_projection::WinProjectionCalculator::new(
            tehai_len_div3,
            round_wind_tile,
            seat_wind_tile,
            is_menzen,
            is_oya,
            dora_indicators_34,
            num_doras_in_fuuro,
            num_aka_in_fuuro,
            melds_for_calc,
        );
        let candidates = calculator.calc(
            &tehai,
            &tiles_seen,
            &akas_in_hand,
            &akas_seen,
            tsumos_left,
            can_discard,
        );
        let sp_arr = win_projection::encode_win_projection_features(&candidates, can_discard);

        // Copy 89*34 floats into buf
        let start = ch_offset * 34;
        buf[start..start + 89 * 34].copy_from_slice(&sp_arr);
    }
}

#[pymethods]
impl Observation {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        player_id: u8,
        hands: Vec<Vec<u8>>,
        melds: Vec<Vec<Meld>>,
        discards: Vec<Vec<u8>>,
        dora_indicators: Vec<u8>,
        scores: Vec<i32>,
        riichi_declared: Vec<bool>,
        legal_actions: Vec<Action>,
        events: Vec<String>,
        honba: u8,
        riichi_sticks: u32,
        round_wind: u8,
        oya: u8,
        kyoku_index: u8,
        waits: Vec<u8>,
        is_tenpai: bool,
        riichi_sutehais: Vec<Option<u8>>,
        last_tedashis: Vec<Option<u8>>,
        last_discard: Option<u32>,
    ) -> Self {
        let hands_u32 = hands
            .iter()
            .map(|h| h.iter().map(|&x| x as u32).collect())
            .collect();
        let discards_u32 = discards
            .iter()
            .map(|d| d.iter().map(|&x| x as u32).collect())
            .collect();
        let dora_u32 = dora_indicators.iter().map(|&x| x as u32).collect();

        Self {
            player_id,
            hands: hands_u32,
            melds,
            discards: discards_u32,
            dora_indicators: dora_u32,
            scores,
            riichi_declared,
            _legal_actions: legal_actions,
            events,
            honba,
            riichi_sticks,
            round_wind,
            oya,
            kyoku_index,
            waits,
            is_tenpai,
            tsumogiri_flags: vec![vec![]; 4], // Initialize with empty vectors
            riichi_sutehais,
            last_tedashis,
            last_discard,
        }
    }

    #[getter]
    pub fn hand(&self) -> Vec<u32> {
        if (self.player_id as usize) < self.hands.len() {
            self.hands[self.player_id as usize].clone()
        } else {
            vec![]
        }
    }

    #[getter]
    pub fn events<'py>(&self, py: Python<'py>) -> PyResult<Vec<Py<PyAny>>> {
        let json = py.import("json")?;
        let loads = json.getattr("loads")?;
        let mut res = Vec::new();
        for s in &self.events {
            let obj = loads.call1((s,))?;
            res.push(obj.unbind());
        }
        Ok(res)
    }

    #[pyo3(name = "legal_actions")]
    pub fn legal_actions_method(&self) -> Vec<Action> {
        self._legal_actions.clone()
    }

    #[pyo3(name = "mask")]
    pub fn mask_method<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut mask = [0u8; 82];
        for action in &self._legal_actions {
            if let Ok(idx) = action.encode() {
                if (idx as usize) < mask.len() {
                    mask[idx as usize] = 1;
                }
            }
        }
        Ok(pyo3::types::PyBytes::new(py, &mask))
    }

    #[pyo3(signature = (action_id))]
    pub fn find_action(&self, action_id: usize) -> Option<Action> {
        self._legal_actions
            .iter()
            .find(|a| {
                if let Ok(idx) = a.encode() {
                    (idx as usize) == action_id
                } else {
                    false
                }
            })
            .cloned()
    }

    #[pyo3(signature = (mjai_data))]
    pub fn select_action_from_mjai(&self, mjai_data: &Bound<'_, PyAny>) -> Option<Action> {
        let (atype, tile_str) = if let Ok(s) = mjai_data.extract::<String>() {
            let v: serde_json::Value = serde_json::from_str(&s).ok()?;
            (
                v["type"].as_str()?.to_string(),
                v["pai"].as_str().unwrap_or("").to_string(),
            )
        } else if let Ok(dict) = mjai_data.cast::<PyDict>() {
            let type_str: String = dict
                .get_item("type")
                .ok()
                .flatten()
                .and_then(|x| x.extract::<String>().ok())
                .unwrap_or_default();
            let _args_list: Vec<String> = dict
                .get_item("args")
                .ok()
                .flatten()
                .and_then(|x| x.extract::<Vec<String>>().ok())
                .unwrap_or_default();
            let _who: i8 = dict
                .get_item("who")
                .ok()
                .flatten()
                .and_then(|x| x.extract::<i8>().ok())
                .unwrap_or(-1);
            // For now, tile is string "3m" etc.
            let tile_str: String = dict
                .get_item("pai")
                .ok()
                .flatten()
                .or_else(|| dict.get_item("tile").ok().flatten())
                .and_then(|x| x.extract::<String>().ok())
                .unwrap_or_default();
            (type_str, tile_str)
        } else {
            return None;
        };

        let target_type = match atype.as_str() {
            "dahai" => Some(crate::action::ActionType::Discard),
            "chi" => Some(crate::action::ActionType::Chi),
            "pon" => Some(crate::action::ActionType::Pon),
            "kakan" => Some(crate::action::ActionType::Kakan),
            "daiminkan" => Some(crate::action::ActionType::Daiminkan),
            "ankan" => Some(crate::action::ActionType::Ankan),
            "reach" => Some(crate::action::ActionType::Riichi),
            "hora" => None,
            "ryukyoku" => Some(crate::action::ActionType::KyushuKyuhai),
            _ => None,
        };

        if atype == "hora" {
            return self
                ._legal_actions
                .iter()
                .find(|a| {
                    a.action_type == crate::action::ActionType::Tsumo
                        || a.action_type == crate::action::ActionType::Ron
                })
                .cloned();
        }

        if let Some(tt) = target_type {
            return self
                ._legal_actions
                .iter()
                .find(|a| {
                    if a.action_type != tt {
                        return false;
                    }
                    if !tile_str.is_empty() {
                        if let Some(t) = a.tile {
                            let t_str = crate::parser::tid_to_mjai(t);
                            // MJAI sometimes uses "5p" for red 5 distinctively or "5pr"
                            // If mismatched, try to match the base.
                            if t_str == tile_str {
                                return true;
                            }
                            // Allow "5p" to match "5pr" if strict match failed?
                            // Or better: if mjai says "5p", it means non-red 5.
                            // If mjai says "5pr" (or however it denotes red), it means red 5.
                            // tid_to_mjai outputs "5mr" for red.
                            // If input is "5m", it shouldn't match "5mr".
                            return false;
                        } else {
                            // Action has no tile (e.g. Riichi), but tile_str provided?
                            // Should ideally not happen for Discard/Chi/Pon.
                            // But if it does, mismatch.
                            return false;
                        }
                    }
                    true
                })
                .cloned();
        }

        if atype == "none" {
            return self
                ._legal_actions
                .iter()
                .find(|a| a.action_type == crate::action::ActionType::Pass)
                .cloned();
        }

        None
    }

    pub fn new_events(&self) -> Vec<String> {
        self.events.clone()
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("player_id", self.player_id)?;
        dict.set_item("hands", self.hands.clone())?;

        let melds_py = pyo3::types::PyList::empty(py);
        for p_melds in &self.melds {
            let p_list = pyo3::types::PyList::new(
                py,
                p_melds.iter().map(|m| m.clone().into_pyobject(py).unwrap()),
            )?;
            melds_py.append(p_list)?;
        }
        dict.set_item("melds", melds_py)?;

        dict.set_item("discards", self.discards.clone())?;
        dict.set_item("dora_indicators", self.dora_indicators.clone())?;
        dict.set_item("scores", self.scores.clone())?;
        dict.set_item("riichi_declared", self.riichi_declared.clone())?;

        let actions_py = pyo3::types::PyList::empty(py);
        for a in &self._legal_actions {
            actions_py.append(a.to_dict(py)?)?;
        }
        dict.set_item("legal_actions", actions_py)?;

        dict.set_item("events", self.events.clone())?;
        dict.set_item("honba", self.honba)?;
        dict.set_item("riichi_sticks", self.riichi_sticks)?;
        dict.set_item("round_wind", self.round_wind)?;
        dict.set_item("oya", self.oya)?;

        Ok(dict.unbind().into())
    }

    /// Encode discard history with exponential decay weighting.
    ///
    /// Returns a (4, 34) array where:
    /// - Row 0: Self discard history with exponential decay
    /// - Row 1-3: Opponent discard history with exponential decay
    ///
    /// Each tile type gets a weighted sum of all discards of that type,
    /// with more recent discards weighted higher:
    /// weight = exp(-decay_rate × age)
    ///
    /// Parameters:
    /// - decay_rate: Rate of exponential decay (default 0.2, as used in Mortal)
    #[pyo3(name = "encode_discard_history_decay", signature = (decay_rate=None))]
    pub fn encode_discard_history_decay<'py>(
        &self,
        py: Python<'py>,
        decay_rate: Option<f32>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let decay_rate = decay_rate.unwrap_or(0.2);
        let mut arr = Array2::<f32>::zeros((4, 34));

        // Encode discard history for all 4 players
        for player_idx in 0..4 {
            if player_idx >= self.discards.len() {
                continue;
            }

            let discs = &self.discards[player_idx];
            let max_len = discs.len();

            if max_len == 0 {
                continue;
            }

            // Iterate through all discards, applying exponential decay
            for (turn, &tile) in discs.iter().enumerate() {
                let tile_idx = (tile as usize) / 4;
                if tile_idx < 34 {
                    // Age = how many turns ago this discard happened
                    // Most recent discard has age 0
                    let age = (max_len - 1 - turn) as f32;
                    let weight = (-decay_rate * age).exp();

                    // Add weighted value (accumulates if same tile discarded multiple times)
                    arr[[player_idx, tile_idx]] += weight;
                }
            }
        }

        // Convert to bytes
        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encode furiten-aware ron possibility based on tsumogiri patterns.
    ///
    /// Returns a (4, 21) array where:
    /// - Dimension 0 (4): Player index (0=self, 1-3=opponents)
    /// - Dimension 1 (21): Yaku type (same indices as encode_yaku_possibility)
    ///
    /// Values:
    /// - 1.0: Ron is possible (player is not in obvious furiten state for recent discards)
    /// - 0.0: Ron is likely impossible (player has been tsumogiri, suggesting no hand changes)
    ///
    /// Logic: If a player has been consistently doing tsumogiri (auto-discard), their hand
    /// hasn't changed, so they likely cannot ron on tiles in their own river (furiten rule).
    #[pyo3(name = "encode_furiten_ron_possibility")]
    pub fn encode_furiten_ron_possibility<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        const NUM_YAKU: usize = 21;
        let mut arr = Array2::<f32>::ones((4, NUM_YAKU));

        // For each player, check if they've been doing tsumogiri consecutively
        for player_idx in 0..4 {
            if player_idx >= self.tsumogiri_flags.len() {
                continue;
            }

            let flags = &self.tsumogiri_flags[player_idx];
            if flags.is_empty() {
                continue;
            }

            // Count consecutive tsumogiri from the end
            let mut consecutive_tsumogiri = 0;
            for &flag in flags.iter().rev() {
                if flag {
                    consecutive_tsumogiri += 1;
                } else {
                    break;
                }
            }

            // If 3+ consecutive tsumogiri, assume hand hasn't changed
            // This means ron on their own discards is likely furiten
            if consecutive_tsumogiri >= 3 {
                // Mark all yaku as having reduced ron possibility
                for yaku_idx in 0..NUM_YAKU {
                    arr[[player_idx, yaku_idx]] = 0.0;
                }
            }
        }

        // Convert to bytes
        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encode yaku (winning hand patterns) possibility for each player.
    ///
    /// Returns a (4, N, 2) array where:
    /// - Dimension 0 (4): Player index (0=self, 1-3=opponents)
    /// - Dimension 1 (N): Yaku type (see YAKU_INDICES below)
    /// - Dimension 2 (2): [Tsumo possibility, Ron possibility]
    ///
    /// Values:
    /// - 1.0: Yaku is possible or unknown (conservative estimate)
    /// - 0.0: Yaku is definitely impossible based on visible information
    ///
    /// Yaku indices:
    /// 0: Tanyao (all simples)
    /// 1-5: Yakuhai (honor tiles): White, Green, Red, Round wind, Seat wind
    /// 6: Honitsu (half flush)
    /// 7: Chinitsu (full flush)
    /// 8: Toitoi (all triplets)
    /// 9: Chiitoitsu (seven pairs)
    /// 10: Shousangen (small three dragons)
    /// 11: Daisangen (big three dragons)
    /// 12: Tsuuiisou (all honors)
    /// 13: Chinroutou (all terminals)
    /// 14: Honroutou (terminals and honors)
    /// 15: Kokushi (thirteen orphans)
    /// 16: Chanta (outside hand)
    /// 17: Junchan (pure outside hand)
    /// 18: Sanshoku doujun (three colored straight)
    /// 19: Iipeikou (pure double sequence)
    /// 20: Ittsu (straight)
    ///
    /// Total: 21 yaku types
    #[pyo3(name = "encode_yaku_possibility")]
    pub fn encode_yaku_possibility<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        const NUM_YAKU: usize = 21;
        let mut arr = Array3::<f32>::ones((4, NUM_YAKU, 2));

        // Combine all visible tiles for each player
        let mut all_discards: Vec<Vec<u32>> = Vec::with_capacity(4);
        for player_idx in 0..4 {
            if player_idx < self.discards.len() {
                all_discards.push(self.discards[player_idx].clone());
            } else {
                all_discards.push(Vec::new());
            }
        }

        // Encode for each player
        for player_idx in 0..4 {
            if player_idx >= self.melds.len() {
                continue;
            }

            let melds = &self.melds[player_idx];
            let discards = &all_discards[player_idx];

            // Yaku 0: Tanyao
            let tanyao = yaku_checker::check_tanyao(melds);
            arr[[player_idx, 0, 0]] = tanyao.to_f32();
            arr[[player_idx, 0, 1]] = tanyao.to_f32();

            // Yaku 1-3: Yakuhai (dragons: White=31, Green=32, Red=33)
            for (yaku_idx, &tile_type) in [31, 32, 33].iter().enumerate() {
                let yakuhai =
                    yaku_checker::check_yakuhai(tile_type, melds, discards, &self.dora_indicators);
                arr[[player_idx, 1 + yaku_idx, 0]] = yakuhai.to_f32();
                arr[[player_idx, 1 + yaku_idx, 1]] = yakuhai.to_f32();
            }

            // Yaku 4: Yakuhai (round wind)
            let round_wind_type = 27 + self.round_wind as usize;
            let yakuhai_round = yaku_checker::check_yakuhai(
                round_wind_type,
                melds,
                discards,
                &self.dora_indicators,
            );
            arr[[player_idx, 4, 0]] = yakuhai_round.to_f32();
            arr[[player_idx, 4, 1]] = yakuhai_round.to_f32();

            // Yaku 5: Yakuhai (seat wind)
            let seat = (player_idx as u8 + 4 - self.oya) % 4;
            let seat_wind_type = 27 + seat as usize;
            let yakuhai_seat =
                yaku_checker::check_yakuhai(seat_wind_type, melds, discards, &self.dora_indicators);
            arr[[player_idx, 5, 0]] = yakuhai_seat.to_f32();
            arr[[player_idx, 5, 1]] = yakuhai_seat.to_f32();

            // Yaku 6-7: Honitsu, Chinitsu
            let (honitsu, chinitsu) = yaku_checker::check_flush(melds);
            arr[[player_idx, 6, 0]] = honitsu.to_f32();
            arr[[player_idx, 6, 1]] = honitsu.to_f32();
            arr[[player_idx, 7, 0]] = chinitsu.to_f32();
            arr[[player_idx, 7, 1]] = chinitsu.to_f32();

            // Yaku 8: Toitoi
            let toitoi = yaku_checker::check_toitoi(melds);
            arr[[player_idx, 8, 0]] = toitoi.to_f32();
            arr[[player_idx, 8, 1]] = toitoi.to_f32();

            // Yaku 9: Chiitoitsu
            let chiitoitsu = yaku_checker::check_chiitoitsu(melds);
            arr[[player_idx, 9, 0]] = chiitoitsu.to_f32();
            arr[[player_idx, 9, 1]] = chiitoitsu.to_f32();

            // Yaku 10: Shousangen
            let shousangen = yaku_checker::check_shousangen(melds, discards, &self.dora_indicators);
            arr[[player_idx, 10, 0]] = shousangen.to_f32();
            arr[[player_idx, 10, 1]] = shousangen.to_f32();

            // Yaku 11: Daisangen
            let daisangen = yaku_checker::check_daisangen(melds, discards, &self.dora_indicators);
            arr[[player_idx, 11, 0]] = daisangen.to_f32();
            arr[[player_idx, 11, 1]] = daisangen.to_f32();

            // Yaku 12: Tsuuiisou
            let tsuuiisou = yaku_checker::check_tsuuiisou(melds);
            arr[[player_idx, 12, 0]] = tsuuiisou.to_f32();
            arr[[player_idx, 12, 1]] = tsuuiisou.to_f32();

            // Yaku 13: Chinroutou
            let chinroutou = yaku_checker::check_chinroutou(melds);
            arr[[player_idx, 13, 0]] = chinroutou.to_f32();
            arr[[player_idx, 13, 1]] = chinroutou.to_f32();

            // Yaku 14: Honroutou
            let honroutou = yaku_checker::check_honroutou(melds);
            arr[[player_idx, 14, 0]] = honroutou.to_f32();
            arr[[player_idx, 14, 1]] = honroutou.to_f32();

            // Yaku 15: Kokushi (thirteen orphans)
            let kokushi = yaku_checker::check_kokushi(melds, discards, &self.dora_indicators);
            arr[[player_idx, 15, 0]] = kokushi.to_f32();
            arr[[player_idx, 15, 1]] = kokushi.to_f32();

            // Yaku 16: Chanta (outside hand)
            let chanta = yaku_checker::check_chanta(melds);
            arr[[player_idx, 16, 0]] = chanta.to_f32();
            arr[[player_idx, 16, 1]] = chanta.to_f32();

            // Yaku 17: Junchan (pure outside hand)
            let junchan = yaku_checker::check_junchan(melds);
            arr[[player_idx, 17, 0]] = junchan.to_f32();
            arr[[player_idx, 17, 1]] = junchan.to_f32();

            // Yaku 18: Sanshoku doujun (three colored straight)
            let sanshoku = yaku_checker::check_sanshoku_doujun(melds);
            arr[[player_idx, 18, 0]] = sanshoku.to_f32();
            arr[[player_idx, 18, 1]] = sanshoku.to_f32();

            // Yaku 19: Iipeikou (pure double sequence)
            let iipeikou = yaku_checker::check_iipeikou(melds);
            arr[[player_idx, 19, 0]] = iipeikou.to_f32();
            arr[[player_idx, 19, 1]] = iipeikou.to_f32();

            // Yaku 20: Ittsu (straight)
            let ittsu = yaku_checker::check_ittsu(melds);
            arr[[player_idx, 20, 0]] = ittsu.to_f32();
            arr[[player_idx, 20, 1]] = ittsu.to_f32();
        }

        // Convert to bytes
        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    pub fn encode<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        // Total Channels (Expanded with Mortal features):
        // 0-3: Hand (1,2,3,4)
        // 4: Red (Hand)
        // 5-8: Melds (Self)
        // 9: Dora Indicators
        // 10-13: Discards (Self, Last 4) (History)
        // 14-25: Discards (Opponents, Last 4 each) (History)
        // 26-29: Discard Counts (All players, normalized)
        // 30: Tiles Left in Wall (normalized)
        // 31: Riichi (Self)
        // 32-34: Riichi (Opponents)
        // 35: Round Wind
        // 36: Self Wind
        // 37: Honba
        // 38: Riichi Sticks
        // 39-42: Scores (P0-P3, normalized 0-100000)
        // 43-46: Scores (P0-P3, normalized 0-30000)
        // 47: Waits (1 channel)
        // 48: Is Tenpai (1 channel)
        // 49-52: Rank (One-hot)
        // 53: Kyoku Index (Normalized)
        // 54: Round Progress (combined: (round_wind)*4 + kyoku_idx, normalized)
        // 55-58: Dora Count (per player, normalized)
        // 59-62: Melds Count (per player)
        // 63: Tiles Seen (Normalized)
        // 64-67: Extended discards (Self, 5-8)
        // 68-69: Extended discards (Opponent 1, 5-6)
        // 70-73: Tsumogiri flags (All players, last discard)
        // Total: 74 channels

        let num_channels = 74;
        let mut arr = Array2::<f32>::zeros((num_channels, 34));

        // 1. Hand (0-3), 2. Red (4)
        if (self.player_id as usize) < self.hands.len() {
            let mut counts = [0u8; 34];
            for &t in &self.hands[self.player_id as usize] {
                let idx = (t as usize) / 4;
                if idx < 34 {
                    counts[idx] += 1;
                    if t == 16 || t == 52 || t == 88 {
                        arr[[4, idx]] = 1.0;
                    }
                }
            }
            for i in 0..34 {
                let c = counts[i];
                if c >= 1 {
                    arr[[0, i]] = 1.0;
                }
                if c >= 2 {
                    arr[[1, i]] = 1.0;
                }
                if c >= 3 {
                    arr[[2, i]] = 1.0;
                }
                if c >= 4 {
                    arr[[3, i]] = 1.0;
                }
            }
        }

        // 3. Melds (Self) (5-8)
        if (self.player_id as usize) < self.melds.len() {
            for (m_idx, meld) in self.melds[self.player_id as usize].iter().enumerate() {
                if m_idx >= 4 {
                    break;
                }
                for &t in &meld.tiles {
                    let idx = (t as usize) / 4;
                    if idx < 34 {
                        arr[[5 + m_idx, idx]] = 1.0;
                    }
                }
            }
        }

        // 4. Dora Indicators (9)
        for &t in &self.dora_indicators {
            let idx = (t as usize) / 4;
            if idx < 34 {
                arr[[9, idx]] = 1.0;
            }
        }

        // 5. Discards (Self) (10-13)
        if (self.player_id as usize) < self.discards.len() {
            let discs = &self.discards[self.player_id as usize];
            for (i, &t) in discs.iter().rev().take(4).enumerate() {
                let idx = (t as usize) / 4;
                if idx < 34 {
                    arr[[10 + i, idx]] = 1.0;
                }
            }
        }

        // 6. Discards (Opponents) (14-25)
        for i in 1..4 {
            let opp_id = (self.player_id + i) % 4;
            if (opp_id as usize) < self.discards.len() {
                let discs = &self.discards[opp_id as usize];
                for (j, &t) in discs.iter().rev().take(4).enumerate() {
                    let idx = (t as usize) / 4;
                    if idx < 34 {
                        let ch_base = 14 + (i as usize - 1) * 4;
                        arr[[ch_base + j, idx]] = 1.0;
                    }
                }
            }
        }

        // 7. Discard Counts (All players, normalized) (26-29)
        for (player_idx, discs) in self.discards.iter().enumerate() {
            let count_norm = (discs.len() as f32) / 24.0; // Max ~24 discards
            for k in 0..34 {
                arr[[26 + player_idx, k]] = count_norm;
            }
        }

        // 8. Tiles Left in Wall (30)
        // Estimate tiles left: 136 total - discards - melds - hands visible
        let mut tiles_used = 0;
        for discs in &self.discards {
            tiles_used += discs.len();
        }
        for melds_list in &self.melds {
            for meld in melds_list {
                tiles_used += meld.tiles.len();
            }
        }
        // Visible hands (just self)
        if (self.player_id as usize) < self.hands.len() {
            tiles_used += self.hands[self.player_id as usize].len();
        }
        tiles_used += self.dora_indicators.len();
        let tiles_left = (136_i32 - tiles_used as i32).max(0) as f32;
        let tiles_left_norm = tiles_left / 70.0; // Max ~70 tiles left in wall
        for k in 0..34 {
            arr[[30, k]] = tiles_left_norm;
        }

        // 9. Riichi (31-34)
        if (self.player_id as usize) < self.riichi_declared.len()
            && self.riichi_declared[self.player_id as usize]
        {
            for i in 0..34 {
                arr[[31, i]] = 1.0;
            }
        }
        for i in 1..4 {
            let opp_id = (self.player_id + i) % 4;
            if (opp_id as usize) < self.riichi_declared.len()
                && self.riichi_declared[opp_id as usize]
            {
                for k in 0..34 {
                    arr[[32 + (i as usize - 1), k]] = 1.0;
                }
            }
        }

        // 10. Winds (35-36)
        let rw = self.round_wind as usize;
        if 27 + rw < 34 {
            arr[[35, 27 + rw]] = 1.0;
        }
        let seat = (self.player_id + 4 - self.oya) % 4;
        if 27 + (seat as usize) < 34 {
            arr[[36, 27 + (seat as usize)]] = 1.0;
        }

        // 11. Honba/Sticks (37-38)
        let honba_norm = (self.honba as f32) / 10.0;
        let sticks_norm = (self.riichi_sticks as f32) / 5.0;
        for i in 0..34 {
            arr[[37, i]] = honba_norm;
            arr[[38, i]] = sticks_norm;
        }

        // 12. Scores (39-42) normalized 0-100000
        for i in 0..4 {
            if i < self.scores.len() {
                let score_norm = (self.scores[i].clamp(0, 100000) as f32) / 100000.0;
                for k in 0..34 {
                    arr[[39 + i, k]] = score_norm;
                }
            }
        }

        // 13. Scores (43-46) normalized 0-30000
        for i in 0..4 {
            if i < self.scores.len() {
                let score_norm = (self.scores[i].clamp(0, 30000) as f32) / 30000.0;
                for k in 0..34 {
                    arr[[43 + i, k]] = score_norm;
                }
            }
        }

        // 14. Waits (47)
        for &t in &self.waits {
            if (t as usize) < 34 {
                arr[[47, t as usize]] = 1.0;
            }
        }

        // 15. Is Tenpai (48)
        let tenpai_val = if self.is_tenpai { 1.0 } else { 0.0 };
        for i in 0..34 {
            arr[[48, i]] = tenpai_val;
        }

        // 16. Rank (49-52)
        // Scores are sorted to find rank?
        // Rank 0 = Highest score.
        let my_score = self
            .scores
            .get(self.player_id as usize)
            .copied()
            .unwrap_or(0);
        let mut rank = 0;
        for &s in &self.scores {
            if s > my_score {
                rank += 1;
            }
        }
        // If tied, logic? Simple > check means same score = same rank (or lower rank if strict).
        // Let's assume strict > means we are 0 if we are max.
        // If tied, we might share rank.
        // Just broadcast 1 to channel (49 + rank).
        if rank < 4 {
            for i in 0..34 {
                arr[[49 + rank, i]] = 1.0;
            }
        }

        // 17. Kyoku (53)
        let k_norm = (self.kyoku_index as f32) / 8.0; // Approx max 8 (East 1-4, South 1-4).
        for i in 0..34 {
            arr[[53, i]] = k_norm;
        }

        // 18. Round Progress (54) - Combined: (round_wind) * 4 + kyoku_idx
        let round_progress = (self.round_wind as f32) * 4.0 + (self.kyoku_index as f32);
        let round_progress_norm = round_progress / 7.0; // 0-7 for East 1-4, South 1-4
        for i in 0..34 {
            arr[[54, i]] = round_progress_norm;
        }

        // 19. Dora Count (55-58) - per player, rescaled
        let mut dora_counts = [0u8; 4];
        // Count dora in each player's visible tiles (melds + discards)
        for (player_idx, dora_count) in dora_counts.iter_mut().enumerate() {
            // Melds
            if player_idx < self.melds.len() {
                for meld in &self.melds[player_idx] {
                    for &tile in &meld.tiles {
                        // Check if tile is dora
                        for &dora_ind in &self.dora_indicators {
                            let dora_tile = get_next_tile(dora_ind);
                            if (tile / 4) == (dora_tile / 4) {
                                *dora_count += 1;
                            }
                        }
                    }
                }
            }
            // Discards
            if player_idx < self.discards.len() {
                for &tile in &self.discards[player_idx] {
                    // tile is u32 from Vec<u32>
                    for &dora_ind in &self.dora_indicators {
                        let dora_tile = get_next_tile(dora_ind);
                        if ((tile / 4) as u8) == (dora_tile / 4) {
                            *dora_count += 1;
                        }
                    }
                }
            }
        }
        // Self hand
        if (self.player_id as usize) < self.hands.len() {
            for &tile in &self.hands[self.player_id as usize] {
                // tile is u32 from Vec<u32>
                for &dora_ind in &self.dora_indicators {
                    let dora_tile = get_next_tile(dora_ind);
                    if ((tile / 4) as u8) == (dora_tile / 4) {
                        dora_counts[self.player_id as usize] += 1;
                    }
                }
            }
        }
        for i in 0..4 {
            let dora_norm = (dora_counts[i] as f32) / 12.0; // Cap at 12
            for k in 0..34 {
                arr[[55 + i, k]] = dora_norm;
            }
        }

        // 20. Melds Count (59-62) - per player, normalized
        for (player_idx, melds_list) in self.melds.iter().enumerate() {
            let meld_count_norm = (melds_list.len() as f32) / 4.0; // Max 4 melds
            for k in 0..34 {
                arr[[59 + player_idx, k]] = meld_count_norm;
            }
        }

        // 21. Tiles Seen (63)
        let mut seen = [0u8; 34];
        // Hand
        if (self.player_id as usize) < self.hands.len() {
            for &t in &self.hands[self.player_id as usize] {
                seen[(t as usize) / 4] += 1;
            }
        }
        // Melds (All)
        for mlist in &self.melds {
            for m in mlist {
                for &t in &m.tiles {
                    seen[(t as usize) / 4] += 1;
                }
            }
        }
        // Discards (All)
        for dlist in &self.discards {
            for &t in dlist {
                seen[(t as usize) / 4] += 1;
            }
        }
        // Dora Indicators
        for &t in &self.dora_indicators {
            seen[(t as usize) / 4] += 1;
        }

        for i in 0..34 {
            // 4 visible max (usually). Red 5 counts as 5.
            let norm_seen = (seen[i] as f32) / 4.0;
            arr[[63, i]] = norm_seen;
        }

        // 22-24. Extended Discard History (64-69)
        // Self: Last 18 discards (64-67, 4 channels)
        if (self.player_id as usize) < self.discards.len() {
            let discs = &self.discards[self.player_id as usize];
            for (i, &t) in discs.iter().rev().skip(4).take(4).enumerate() {
                let idx = (t as usize) / 4;
                if idx < 34 {
                    arr[[64 + i, idx]] = 1.0;
                }
            }
        }

        // Opponents: Last 4-7 discards (68-69, 2 channels)
        // Simplified: only track 2 more channels for first opponent
        if self.discards.len() > ((self.player_id + 1) % 4) as usize {
            let opp_id = (self.player_id + 1) % 4;
            let discs = &self.discards[opp_id as usize];
            for (i, &t) in discs.iter().rev().skip(4).take(2).enumerate() {
                let idx = (t as usize) / 4;
                if idx < 34 {
                    arr[[68 + i, idx]] = 1.0;
                }
            }
        }

        // 25. Tsumogiri flags (70-73) - broadcast per player
        // 1.0 if last discard was tsumogiri (drawn and immediately discarded)
        for player_idx in 0..4 {
            if player_idx < self.tsumogiri_flags.len()
                && !self.tsumogiri_flags[player_idx].is_empty()
            {
                let last_tsumogiri = *self.tsumogiri_flags[player_idx].last().unwrap_or(&false);
                let val = if last_tsumogiri { 1.0 } else { 0.0 };
                for k in 0..34 {
                    arr[[70 + player_idx, k]] = val;
                }
            }
        }

        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encode shanten number and tile efficiency features.
    ///
    /// Returns a (4, 4) array where:
    /// - Dimension 0 (4): Player index (0=self, 1-3=opponents)
    /// - Dimension 1 (4): [shanten, effective_tiles, best_ukeire, normalized_turn]
    ///
    /// Features:
    /// - shanten: Normalized shanten number (/ 8.0, where 8 is max typical shanten)
    /// - effective_tiles: Number of tile types that reduce shanten (/ 34.0)
    /// - best_ukeire: Best ukeire count after optimal discard (/ 80.0)
    /// - normalized_turn: Current turn / max_turns (approx / 18.0)
    #[pyo3(name = "encode_shanten_efficiency")]
    pub fn encode_shanten_efficiency<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut arr = Array2::<f32>::zeros((4, 4));

        // Collect all visible tiles for ukire calculation
        let mut all_visible: Vec<u32> = Vec::new();
        for discs in &self.discards {
            all_visible.extend(discs.iter().copied());
        }
        for melds_list in &self.melds {
            for meld in melds_list {
                all_visible.extend(meld.tiles.iter().map(|&x| x as u32));
            }
        }
        all_visible.extend(self.dora_indicators.iter().copied());

        // Calculate features for each player
        for player_idx in 0..4 {
            if player_idx >= self.hands.len() {
                continue;
            }

            let hand = &self.hands[player_idx];

            // For self, we have full information
            // For opponents, we can only estimate based on visible info
            if player_idx == self.player_id as usize {
                // Self: full calculation
                let shanten = crate::shanten::calculate_shanten(hand);
                let effective = crate::shanten::calculate_effective_tiles(hand);
                let best_ukeire = crate::shanten::calculate_best_ukeire(hand, &all_visible);

                // Normalize features
                arr[[player_idx, 0]] = (shanten as f32).max(0.0) / 8.0;
                arr[[player_idx, 1]] = (effective as f32) / 34.0;
                arr[[player_idx, 2]] = (best_ukeire as f32) / 80.0;
            } else {
                // Opponents: estimate or use conservative values
                // We don't know their hand, so set to unknown (0.5)
                arr[[player_idx, 0]] = 0.5; // Unknown shanten
                arr[[player_idx, 1]] = 0.5; // Unknown effective tiles
                arr[[player_idx, 2]] = 0.5; // Unknown ukeire
            }

            // Turn count (same for all players)
            let turn_count = if player_idx < self.discards.len() {
                self.discards[player_idx].len() as f32
            } else {
                0.0
            };
            arr[[player_idx, 3]] = turn_count / 18.0; // Normalize by typical max turns
        }

        // Convert to bytes
        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encode kawa (discard pile) overview for all players
    /// Returns a (4, 7, 34) array: 4 players × 7 channels × 34 tile types
    /// Channels: [count1, count2, count3, count4, aka5m, aka5p, aka5s]
    #[pyo3(name = "encode_kawa_overview")]
    pub fn encode_kawa_overview<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut arr = Array3::<f32>::zeros((4, 7, 34));

        for (player_idx, discards) in self.discards.iter().enumerate() {
            if player_idx >= 4 {
                break;
            }

            // Count each tile type (up to 4 copies)
            let mut tile_counts = [0u8; 34];
            let mut aka_flags = [false; 3]; // aka 5m/5p/5s

            for &tile in discards {
                let tile_type = (tile / 4) as usize;
                if tile_type < 34 {
                    let count_idx = tile_counts[tile_type].min(3) as usize;
                    arr[[player_idx, count_idx, tile_type]] = 1.0;
                    tile_counts[tile_type] = tile_counts[tile_type].saturating_add(1);
                }

                // Check for aka tiles (20, 24, 28 are aka 5m/5p/5s in 136-tile encoding)
                match tile {
                    20 => aka_flags[0] = true, // aka 5m
                    24 => aka_flags[1] = true, // aka 5p
                    28 => aka_flags[2] = true, // aka 5s
                    _ => {}
                }
            }

            // Set aka flags
            for (i, &has_aka) in aka_flags.iter().enumerate() {
                if has_aka {
                    arr[[player_idx, 4 + i, 5 + i * 9]] = 1.0; // 5m=5, 5p=14, 5s=23
                }
            }
        }

        // Convert to bytes
        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encode fuuro (meld) overview for all players
    /// Returns a (4, 4, 5, 34) array: 4 players × 4 melds × 5 channels × 34 tile types
    /// Channels: [tile1, tile2, tile3, tile4, aka]
    #[pyo3(name = "encode_fuuro_overview")]
    pub fn encode_fuuro_overview<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut arr = Array4::<f32>::zeros((4, 4, 5, 34));

        for (player_idx, melds) in self.melds.iter().enumerate() {
            if player_idx >= 4 {
                break;
            }

            for (meld_idx, meld) in melds.iter().enumerate() {
                if meld_idx >= 4 {
                    break;
                }

                // Encode each tile in the meld
                for (tile_slot_idx, &tile) in meld.tiles.iter().enumerate() {
                    if tile_slot_idx >= 4 {
                        break;
                    }

                    let tile_type = (tile / 4) as usize;
                    if tile_type < 34 {
                        arr[[player_idx, meld_idx, tile_slot_idx, tile_type]] = 1.0;
                    }

                    // Check for aka (red five: 5m=16, 5p=52, 5s=88)
                    let is_aka = matches!(tile, 16 | 52 | 88);
                    if is_aka {
                        arr[[player_idx, meld_idx, 4, tile_type]] = 1.0;
                    }
                }
            }
        }

        // Convert to bytes
        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encode ankan (concealed kan) overview for all players
    /// Returns a (4, 34) array: 4 players × 34 tile types
    #[pyo3(name = "encode_ankan_overview")]
    pub fn encode_ankan_overview<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut arr = Array2::<f32>::zeros((4, 34));

        for (player_idx, melds) in self.melds.iter().enumerate() {
            if player_idx >= 4 {
                break;
            }

            for meld in melds {
                // Check if this is an ankan (concealed kan)
                if matches!(meld.meld_type, MeldType::Ankan) {
                    // Use the first tile to determine type
                    if let Some(&tile) = meld.tiles.first() {
                        let tile_type = (tile / 4) as usize;
                        if tile_type < 34 {
                            arr[[player_idx, tile_type]] = 1.0;
                        }
                    }
                }
            }
        }

        // Convert to bytes
        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encode action availability flags
    /// Returns a (11,) array: [can_riichi, can_chi_low, can_chi_mid, can_chi_high,
    ///                          can_pon, can_daiminkan, can_ankan, can_kakan,
    ///                          can_agari, can_ryukyoku, can_pass]
    #[pyo3(name = "encode_action_availability")]
    pub fn encode_action_availability<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut arr = Array1::<f32>::zeros(11);

        for action in &self._legal_actions {
            match action.action_type {
                ActionType::Riichi => arr[0] = 1.0,
                ActionType::Chi => {
                    // Determine chi type by consumed tiles
                    let tiles = &action.consume_tiles;
                    if tiles.len() == 2 {
                        let t0 = tiles[0] / 4;
                        let t1 = tiles[1] / 4;
                        let diff = (t1 as i32 - t0 as i32).abs();

                        if diff == 1 {
                            // Could be low or high
                            if t0 < t1 {
                                arr[1] = 1.0; // low (call tile is highest)
                            } else {
                                arr[3] = 1.0; // high (call tile is lowest)
                            }
                        } else if diff == 2 {
                            arr[2] = 1.0; // mid
                        }
                    }
                }
                ActionType::Pon => arr[4] = 1.0,
                ActionType::Daiminkan => arr[5] = 1.0,
                ActionType::Ankan => arr[6] = 1.0,
                ActionType::Kakan => arr[7] = 1.0,
                ActionType::Tsumo | ActionType::Ron => arr[8] = 1.0,
                ActionType::KyushuKyuhai => arr[9] = 1.0,
                ActionType::Pass => arr[10] = 1.0,
                _ => {}
            }
        }

        // Convert to bytes
        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encodes riichi sutehais (tiles discarded when declaring riichi) for opponents
    /// Returns: (3, 3) array
    /// - 3 opponents (excluding self)
    /// - 3 channels per opponent: [tile_type, is_aka, is_dora]
    #[pyo3(name = "encode_riichi_sutehais")]
    pub fn encode_riichi_sutehais<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut arr = Array2::<f32>::zeros((3, 3));

        let dora_tiles: Vec<u8> = self
            .dora_indicators
            .iter()
            .map(|&indicator| get_next_tile(indicator))
            .collect();

        let mut opponent_idx = 0;
        for player_id in 0..4 {
            if player_id == self.player_id as usize {
                continue;
            }

            if let Some(tile) = self.riichi_sutehais[player_id] {
                let tile_type = (tile / 4) as usize;

                // Channel 0: tile type (0-33)
                arr[[opponent_idx, 0]] = tile_type as f32 / 33.0;

                // Channel 1: is aka (red five: 5m=16, 5p=52, 5s=88)
                let is_aka = matches!(tile, 16 | 52 | 88);
                arr[[opponent_idx, 1]] = if is_aka { 1.0 } else { 0.0 };

                // Channel 2: is dora
                let is_dora = dora_tiles.contains(&tile);
                arr[[opponent_idx, 2]] = if is_dora { 1.0 } else { 0.0 };
            }

            opponent_idx += 1;
        }

        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encodes last tedashis (last hand discards, not tsumogiri) for opponents
    /// Returns: (3, 3) array
    /// - 3 opponents (excluding self)
    /// - 3 channels per opponent: [tile_type, is_aka, is_dora]
    #[pyo3(name = "encode_last_tedashis")]
    pub fn encode_last_tedashis<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut arr = Array2::<f32>::zeros((3, 3));

        let dora_tiles: Vec<u8> = self
            .dora_indicators
            .iter()
            .map(|&indicator| get_next_tile(indicator))
            .collect();

        let mut opponent_idx = 0;
        for player_id in 0..4 {
            if player_id == self.player_id as usize {
                continue;
            }

            if let Some(tile) = self.last_tedashis[player_id] {
                let tile_type = (tile / 4) as usize;

                // Channel 0: tile type (0-33)
                arr[[opponent_idx, 0]] = tile_type as f32 / 33.0;

                // Channel 1: is aka (red five: 5m=16, 5p=52, 5s=88)
                let is_aka = matches!(tile, 16 | 52 | 88);
                arr[[opponent_idx, 1]] = if is_aka { 1.0 } else { 0.0 };

                // Channel 2: is dora
                let is_dora = dora_tiles.contains(&tile);
                arr[[opponent_idx, 2]] = if is_dora { 1.0 } else { 0.0 };
            }

            opponent_idx += 1;
        }

        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encodes pass context (current offer tile for chi/pon/kan/ron)
    /// Returns: (3,) array: [tile_type, is_aka, is_dora]
    #[pyo3(name = "encode_pass_context")]
    pub fn encode_pass_context<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut arr = Array1::<f32>::zeros(3);

        if let Some(tile) = self.last_discard {
            let tile_type = (tile / 4) as usize;

            // Channel 0: tile type (0-33)
            arr[0] = tile_type as f32 / 33.0;

            // Channel 1: is aka (red five: 5m=16, 5p=52, 5s=88)
            let is_aka = matches!(tile, 16 | 52 | 88);
            arr[1] = if is_aka { 1.0 } else { 0.0 };

            // Channel 2: is dora
            let dora_tiles: Vec<u8> = self
                .dora_indicators
                .iter()
                .map(|&indicator| get_next_tile(indicator))
                .collect();
            let is_dora = dora_tiles.contains(&(tile as u8));
            arr[2] = if is_dora { 1.0 } else { 0.0 };
        }

        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encodes discard candidates detail
    /// Returns: (5,) array
    /// - Channel 0: number of tiles that can be discarded
    /// - Channel 1: number of discards that keep current shanten
    /// - Channel 2: number of discards that increase shanten
    /// - Channel 3: whether already in tenpai (unconditional tenpai check)
    /// - Channel 4: whether riichi is declared
    #[pyo3(name = "encode_discard_candidates")]
    pub fn encode_discard_candidates<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut arr = Array1::<f32>::zeros(5);

        let player_idx = self.player_id as usize;
        if player_idx >= self.hands.len() {
            let slice = arr.as_slice().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
            })?;
            let byte_len = std::mem::size_of_val(slice);
            let byte_slice =
                unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
            return Ok(pyo3::types::PyBytes::new(py, byte_slice));
        }

        let hand = &self.hands[player_idx];
        let current_shanten = shanten::calculate_shanten(hand);

        // Channel 0: number of tiles that can be discarded (normalized by 34)
        arr[0] = hand.len() as f32 / 34.0;

        // Analyze each possible discard
        let mut keep_shanten_count = 0;
        let mut increase_shanten_count = 0;

        for (idx, _tile) in hand.iter().enumerate() {
            let new_hand: Vec<u32> = hand
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(_, &t)| t)
                .collect();

            let new_shanten = shanten::calculate_shanten(&new_hand);

            if new_shanten == current_shanten {
                keep_shanten_count += 1;
            } else if new_shanten > current_shanten {
                increase_shanten_count += 1;
            }
        }

        // Channel 1: discards that keep shanten (normalized by hand size)
        if !hand.is_empty() {
            arr[1] = keep_shanten_count as f32 / hand.len() as f32;
        }

        // Channel 2: discards that increase shanten (normalized by hand size)
        if !hand.is_empty() {
            arr[2] = increase_shanten_count as f32 / hand.len() as f32;
        }

        // Channel 3: unconditional tenpai (shanten == -1)
        arr[3] = if current_shanten == -1 { 1.0 } else { 0.0 };

        // Channel 4: riichi declared
        arr[4] = if player_idx < self.riichi_declared.len() && self.riichi_declared[player_idx] {
            1.0
        } else {
            0.0
        };

        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encode win projection features as (89, 34) float array.
    #[pyo3(name = "encode_win_projection")]
    pub fn encode_win_projection<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        use crate::types::TILE_MAX;
        use crate::win_projection;

        let player_idx = self.player_id as usize;
        let num_channels = 89;
        let total_floats = num_channels * 34;

        // Default: return zeros if hand not available
        if player_idx >= self.hands.len() {
            let arr = vec![0.0f32; total_floats];
            let byte_len = arr.len() * std::mem::size_of::<f32>();
            let byte_slice =
                unsafe { std::slice::from_raw_parts(arr.as_ptr() as *const u8, byte_len) };
            return Ok(pyo3::types::PyBytes::new(py, byte_slice));
        }

        let hand_136 = &self.hands[player_idx];

        // Convert 136-tile hand to 34-tile counts
        let mut tehai = [0u8; TILE_MAX];
        let mut akas_in_hand = [false; 3];
        for &t in hand_136 {
            let tile_type = (t / 4) as usize;
            if tile_type < TILE_MAX {
                tehai[tile_type] += 1;
            }
            // Check for aka (red five): tile IDs 16, 52, 88 in 136-encoding
            match t {
                16 => akas_in_hand[0] = true, // aka 5m
                52 => akas_in_hand[1] = true, // aka 5p
                88 => akas_in_hand[2] = true, // aka 5s
                _ => {}
            }
        }

        // Build tiles_seen: all visible tiles (hand + melds + discards + dora indicators)
        let mut tiles_seen = [0u8; TILE_MAX];
        let mut akas_seen = [false; 3];

        // Hand tiles
        for &t in hand_136 {
            let tile_type = (t / 4) as usize;
            if tile_type < TILE_MAX {
                tiles_seen[tile_type] += 1;
            }
            match t {
                16 => akas_seen[0] = true,
                52 => akas_seen[1] = true,
                88 => akas_seen[2] = true,
                _ => {}
            }
        }

        // All melds (all players)
        for melds_list in &self.melds {
            for meld in melds_list {
                for &tile in &meld.tiles {
                    let tile_type = (tile / 4) as usize;
                    if tile_type < TILE_MAX {
                        tiles_seen[tile_type] += 1;
                    }
                    match tile {
                        16 => akas_seen[0] = true,
                        52 => akas_seen[1] = true,
                        88 => akas_seen[2] = true,
                        _ => {}
                    }
                }
            }
        }

        // All discards (all players)
        for discs in &self.discards {
            for &tile in discs {
                let tile_type = (tile / 4) as usize;
                if tile_type < TILE_MAX {
                    tiles_seen[tile_type] += 1;
                }
                match tile {
                    16 => akas_seen[0] = true,
                    52 => akas_seen[1] = true,
                    88 => akas_seen[2] = true,
                    _ => {}
                }
            }
        }

        // Dora indicators
        for &t in &self.dora_indicators {
            let tile_type = (t / 4) as usize;
            if tile_type < TILE_MAX {
                tiles_seen[tile_type] += 1;
            }
            match t {
                16 => akas_seen[0] = true,
                52 => akas_seen[1] = true,
                88 => akas_seen[2] = true,
                _ => {}
            }
        }

        // Clamp tiles_seen to max 4 per type
        for ts in tiles_seen.iter_mut().take(TILE_MAX) {
            *ts = (*ts).min(4);
        }

        // Determine game context
        let num_melds = if player_idx < self.melds.len() {
            self.melds[player_idx].len()
        } else {
            0
        };
        let tehai_len_div3 = (4 - num_melds) as u8; // 3n+2 tiles in hand = (4-melds)*3+2

        let is_menzen = if player_idx < self.melds.len() {
            self.melds[player_idx]
                .iter()
                .all(|m| matches!(m.meld_type, MeldType::Ankan))
        } else {
            true
        };

        let seat = (self.player_id + 4 - self.oya) % 4;
        let is_oya = seat == 0;
        let round_wind_tile = 27 + self.round_wind;
        let seat_wind_tile = 27 + seat;

        // Dora indicators in 34-tile format
        let dora_indicators_34: Vec<u8> = self
            .dora_indicators
            .iter()
            .map(|&t| (t / 4) as u8)
            .collect();

        // Count dora and aka in open melds
        let mut num_doras_in_fuuro = 0u8;
        let mut num_aka_in_fuuro = 0u8;
        if player_idx < self.melds.len() {
            for meld in &self.melds[player_idx] {
                for &tile in &meld.tiles {
                    // Check dora
                    for &ind in &dora_indicators_34 {
                        let dora = win_projection::next_dora_tile(ind);
                        if tile / 4 == dora {
                            num_doras_in_fuuro += 1;
                        }
                    }
                    // Check aka
                    match tile {
                        16 | 52 | 88 => num_aka_in_fuuro += 1,
                        _ => {}
                    }
                }
            }
        }

        // Clone melds for the calculator (convert tile IDs to 34-format)
        let melds_for_calc: Vec<crate::types::Meld> = if player_idx < self.melds.len() {
            self.melds[player_idx]
                .iter()
                .map(|m| crate::types::Meld {
                    meld_type: m.meld_type,
                    tiles: m.tiles.iter().map(|&t| t / 4).collect(),
                    opened: m.opened,
                    from_who: m.from_who,
                    called_tile: m.called_tile.map(|t| t / 4),
                })
                .collect()
        } else {
            Vec::new()
        };

        // Estimate tsumos_left from tiles remaining in wall
        let total_visible: u32 = tiles_seen.iter().map(|&c| c as u32).sum();
        let tiles_in_wall = (136u32).saturating_sub(total_visible);
        // Each of 4 players draws ~equally; we care about our remaining draws
        // Rough estimate: tiles_in_wall / 4, capped at MAX_TSUMOS_LEFT
        let tsumos_left = ((tiles_in_wall / 4) as usize).min(win_projection::MAX_TSUMOS_LEFT);

        // Determine if we can discard (hand has 3n+2 tiles, meaning we just drew)
        let hand_tile_count: u32 = tehai.iter().map(|&c| c as u32).sum();
        let can_discard = hand_tile_count % 3 == 2;

        // Create calculator and compute
        let calculator = win_projection::WinProjectionCalculator::new(
            tehai_len_div3,
            round_wind_tile,
            seat_wind_tile,
            is_menzen,
            is_oya,
            dora_indicators_34,
            num_doras_in_fuuro,
            num_aka_in_fuuro,
            melds_for_calc,
        );

        let candidates = calculator.calc(
            &tehai,
            &tiles_seen,
            &akas_in_hand,
            &akas_seen,
            tsumos_left,
            can_discard,
        );

        let arr = win_projection::encode_win_projection_features(&candidates, can_discard);

        // Convert to bytes
        let byte_len = arr.len() * std::mem::size_of::<f32>();
        let byte_slice = unsafe { std::slice::from_raw_parts(arr.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encode all 304 channels of Extended features in a single call.
    ///
    /// Channel layout:
    ///   0- 73: base encode (74ch spatial)
    ///  74- 77: discard history decay (4ch spatial)
    ///  78- 93: shanten efficiency (16ch broadcast)
    ///  94- 97: ankan overview (4ch spatial)
    ///  98-177: fuuro overview (80ch spatial)
    /// 178-188: action availability (11ch broadcast)
    /// 189-193: discard candidates (5ch broadcast)
    /// 194-196: pass context (3ch broadcast)
    /// 197-205: last tedashis (9ch broadcast)
    /// 206-214: riichi sutehais (9ch broadcast)
    /// 215-303: win projection (89ch)
    ///
    /// Returns PyBytes of 304 × 34 × 4 = 41,344 bytes.
    #[pyo3(name = "encode_extended_win_projection")]
    pub fn encode_extended_win_projection<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let total = 304 * 34;
        let mut buf = vec![0.0f32; total];

        self.encode_base_into(&mut buf, 0);
        self.encode_discard_decay_into(&mut buf, 74);
        self.encode_shanten_into(&mut buf, 78);
        self.encode_ankan_into(&mut buf, 94);
        self.encode_fuuro_into(&mut buf, 98);
        self.encode_action_avail_into(&mut buf, 178);
        self.encode_discard_cand_into(&mut buf, 189);
        self.encode_pass_ctx_into(&mut buf, 194);
        self.encode_last_ted_into(&mut buf, 197);
        self.encode_riichi_sute_into(&mut buf, 206);
        self.encode_win_projection_into(&mut buf, 215);

        let byte_len = total * std::mem::size_of::<f32>();
        let byte_slice = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encode all 215 channels of Extended features in a single call.
    /// Same as encode_extended_win_projection but without the 89ch win projection.
    #[pyo3(name = "encode_extended")]
    pub fn encode_extended<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let total = 215 * 34;
        let mut buf = vec![0.0f32; total];

        self.encode_base_into(&mut buf, 0);
        self.encode_discard_decay_into(&mut buf, 74);
        self.encode_shanten_into(&mut buf, 78);
        self.encode_ankan_into(&mut buf, 94);
        self.encode_fuuro_into(&mut buf, 98);
        self.encode_action_avail_into(&mut buf, 178);
        self.encode_discard_cand_into(&mut buf, 189);
        self.encode_pass_ctx_into(&mut buf, 194);
        self.encode_last_ted_into(&mut buf, 197);
        self.encode_riichi_sute_into(&mut buf, 206);

        let byte_len = total * std::mem::size_of::<f32>();
        let byte_slice = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Return diagnostic statistics from win projection calculation as a Python dict.
    #[pyo3(name = "win_projection_stats")]
    pub fn win_projection_stats<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        use crate::types::TILE_MAX;
        use crate::win_projection;

        let player_idx = self.player_id as usize;

        // Build tehai/tiles_seen (same as encode_win_projection_into)
        let mut tehai = [0u8; TILE_MAX];
        let mut akas_in_hand = [false; 3];
        if player_idx < self.hands.len() {
            for &t in &self.hands[player_idx] {
                let tile_type = (t / 4) as usize;
                if tile_type < TILE_MAX {
                    tehai[tile_type] += 1;
                }
                match t {
                    16 => akas_in_hand[0] = true,
                    52 => akas_in_hand[1] = true,
                    88 => akas_in_hand[2] = true,
                    _ => {}
                }
            }
        }

        let mut tiles_seen = [0u8; TILE_MAX];
        let mut akas_seen = [false; 3];
        if player_idx < self.hands.len() {
            for &t in &self.hands[player_idx] {
                let tt = (t / 4) as usize;
                if tt < TILE_MAX {
                    tiles_seen[tt] += 1;
                }
                match t {
                    16 => akas_seen[0] = true,
                    52 => akas_seen[1] = true,
                    88 => akas_seen[2] = true,
                    _ => {}
                }
            }
        }
        for melds_list in &self.melds {
            for meld in melds_list {
                for &tile in &meld.tiles {
                    let tt = (tile / 4) as usize;
                    if tt < TILE_MAX {
                        tiles_seen[tt] += 1;
                    }
                    match tile {
                        16 => akas_seen[0] = true,
                        52 => akas_seen[1] = true,
                        88 => akas_seen[2] = true,
                        _ => {}
                    }
                }
            }
        }
        for discs in &self.discards {
            for &tile in discs {
                let tt = (tile / 4) as usize;
                if tt < TILE_MAX {
                    tiles_seen[tt] += 1;
                }
                match tile {
                    16 => akas_seen[0] = true,
                    52 => akas_seen[1] = true,
                    88 => akas_seen[2] = true,
                    _ => {}
                }
            }
        }
        for &t in &self.dora_indicators {
            let tt = (t / 4) as usize;
            if tt < TILE_MAX {
                tiles_seen[tt] += 1;
            }
            match t {
                16 => akas_seen[0] = true,
                52 => akas_seen[1] = true,
                88 => akas_seen[2] = true,
                _ => {}
            }
        }
        // Subtract double-counted tiles: claimed tiles appear in both discards and melds
        for melds_list in &self.melds {
            for meld in melds_list {
                if let Some(ct) = meld.called_tile {
                    let tt = (ct as u32 / 4) as usize;
                    if tt < TILE_MAX && tiles_seen[tt] > 0 {
                        tiles_seen[tt] -= 1;
                    }
                }
            }
        }
        for ts in tiles_seen.iter_mut().take(TILE_MAX) {
            *ts = (*ts).min(4);
        }

        let num_melds = if player_idx < self.melds.len() {
            self.melds[player_idx].len()
        } else {
            0
        };
        let tehai_len_div3 = (4 - num_melds) as u8;
        let is_menzen = if player_idx < self.melds.len() {
            self.melds[player_idx]
                .iter()
                .all(|m| matches!(m.meld_type, MeldType::Ankan))
        } else {
            true
        };

        let seat = (self.player_id + 4 - self.oya) % 4;
        let is_oya = seat == 0;
        let round_wind_tile = 27 + self.round_wind;
        let seat_wind_tile = 27 + seat;

        let dora_indicators_34: Vec<u8> = self
            .dora_indicators
            .iter()
            .map(|&t| (t / 4) as u8)
            .collect();

        let mut num_doras_in_fuuro = 0u8;
        let mut num_aka_in_fuuro = 0u8;
        if player_idx < self.melds.len() {
            for meld in &self.melds[player_idx] {
                for &tile in &meld.tiles {
                    for &ind in &dora_indicators_34 {
                        if tile / 4 == win_projection::next_dora_tile(ind) {
                            num_doras_in_fuuro += 1;
                        }
                    }
                    match tile {
                        16 | 52 | 88 => num_aka_in_fuuro += 1,
                        _ => {}
                    }
                }
            }
        }

        let melds_for_calc: Vec<Meld> = if player_idx < self.melds.len() {
            self.melds[player_idx]
                .iter()
                .map(|m| Meld {
                    meld_type: m.meld_type,
                    tiles: m.tiles.iter().map(|&t| t / 4).collect(),
                    opened: m.opened,
                    from_who: m.from_who,
                    called_tile: m.called_tile.map(|t| t / 4),
                })
                .collect()
        } else {
            Vec::new()
        };

        let total_visible: u32 = tiles_seen.iter().map(|&c| c as u32).sum();
        let tiles_in_wall = (136u32).saturating_sub(total_visible);
        let tsumos_left = ((tiles_in_wall / 4) as usize).min(win_projection::MAX_TSUMOS_LEFT);
        let hand_tile_count: u32 = tehai.iter().map(|&c| c as u32).sum();
        let can_discard = hand_tile_count % 3 == 2;

        let calculator = win_projection::WinProjectionCalculator::new(
            tehai_len_div3,
            round_wind_tile,
            seat_wind_tile,
            is_menzen,
            is_oya,
            dora_indicators_34,
            num_doras_in_fuuro,
            num_aka_in_fuuro,
            melds_for_calc,
        );
        let (_, stats) = calculator.calc_with_stats(
            &tehai,
            &tiles_seen,
            &akas_in_hand,
            &akas_seen,
            tsumos_left,
            can_discard,
        );

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("shanten", stats.shanten)?;
        dict.set_item("tsumos_left", stats.tsumos_left)?;
        dict.set_item("can_discard", stats.can_discard)?;
        dict.set_item("num_candidates", stats.num_candidates)?;
        dict.set_item("draw_cache_hits", stats.draw_cache_hits)?;
        dict.set_item("draw_cache_misses", stats.draw_cache_misses)?;
        dict.set_item("discard_cache_hits", stats.discard_cache_hits)?;
        dict.set_item("discard_cache_misses", stats.discard_cache_misses)?;
        dict.set_item("get_score_calls", stats.get_score_calls)?;
        dict.set_item("draw_slow_calls", stats.draw_slow_calls)?;
        dict.set_item("discard_slow_calls", stats.discard_slow_calls)?;
        dict.set_item("rc_values_created", stats.rc_values_created)?;
        dict.set_item("draw_cache_size", stats.draw_cache_size)?;
        dict.set_item("discard_cache_size", stats.discard_cache_size)?;
        Ok(dict)
    }
}
