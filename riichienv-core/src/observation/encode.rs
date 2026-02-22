use crate::action::ActionType;
use crate::shanten;
use crate::types::MeldType;

use super::helpers::{add_val, broadcast_scalar, get_next_tile, set_val};
use super::Observation;

/// Internal (non-PyO3) methods that write features directly into a flat f32 buffer.
/// Buffer layout: channel-major, buf[(ch_offset + ch) * 34 + tile] = value.
impl Observation {
    /// Write 74 base encode channels into buf starting at ch_offset.
    pub(crate) fn encode_base_into(&self, buf: &mut [f32], ch_offset: usize) {
        // Hand (ch 0-3) + Red (ch 4)
        {
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
        {
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
        {
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
            let opp_id = ((self.player_id + i) % 4) as usize;
            let discs = &self.discards[opp_id];
            for (j, &t) in discs.iter().rev().take(4).enumerate() {
                let idx = (t as usize) / 4;
                if idx < 34 {
                    let ch = 14 + (i as usize - 1) * 4 + j;
                    set_val(buf, ch_offset, ch, idx, 1.0);
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
        tiles_used += self.hands[self.player_id as usize].len();
        tiles_used += self.dora_indicators.len();
        let tiles_left = (136_i32 - tiles_used as i32).max(0) as f32;
        broadcast_scalar(buf, ch_offset, 30, tiles_left / 70.0);

        // Riichi (ch 31-34)
        if self.riichi_declared[self.player_id as usize] {
            broadcast_scalar(buf, ch_offset, 31, 1.0);
        }
        for i in 1..4u8 {
            let opp_id = ((self.player_id + i) % 4) as usize;
            if self.riichi_declared[opp_id] {
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

        // Waits (ch 47)
        for &t in &self.waits {
            if (t as usize) < 34 {
                set_val(buf, ch_offset, 47, t as usize, 1.0);
            }
        }

        // Is Tenpai (ch 48)
        broadcast_scalar(buf, ch_offset, 48, if self.is_tenpai { 1.0 } else { 0.0 });

        // Rank (ch 49-52)
        let my_score = self.scores[self.player_id as usize];
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
            for &tile in &self.discards[player_idx] {
                for &dora_ind in &self.dora_indicators {
                    let dora_tile = get_next_tile(dora_ind);
                    if ((tile / 4) as u8) == (dora_tile / 4) {
                        *dora_count += 1;
                    }
                }
            }
        }
        for &tile in &self.hands[self.player_id as usize] {
            for &dora_ind in &self.dora_indicators {
                let dora_tile = get_next_tile(dora_ind);
                if ((tile / 4) as u8) == (dora_tile / 4) {
                    dora_counts[self.player_id as usize] += 1;
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
        for &t in &self.hands[self.player_id as usize] {
            seen[(t as usize) / 4] += 1;
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
        {
            let discs = &self.discards[self.player_id as usize];
            for (i, &t) in discs.iter().rev().skip(4).take(4).enumerate() {
                let idx = (t as usize) / 4;
                if idx < 34 {
                    set_val(buf, ch_offset, 64 + i, idx, 1.0);
                }
            }
        }

        // Extended discards opponent 1 (ch 68-69)
        {
            let opp1_id = ((self.player_id + 1) % 4) as usize;
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
            if !self.tsumogiri_flags[player_idx].is_empty() {
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
    pub(crate) fn encode_discard_decay_into(&self, buf: &mut [f32], ch_offset: usize) {
        let decay_rate = 0.2f32;
        for player_idx in 0..4 {
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
    pub(crate) fn encode_shanten_into(&self, buf: &mut [f32], ch_offset: usize) {
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

            let turn_count = self.discards[player_idx].len() as f32;
            broadcast_scalar(buf, ch_offset, base_ch + 3, (turn_count / 18.0).min(1.0));
        }
    }

    /// Write 4 ankan overview channels into buf starting at ch_offset.
    pub(crate) fn encode_ankan_into(&self, buf: &mut [f32], ch_offset: usize) {
        for (player_idx, melds) in self.melds.iter().enumerate() {
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
    pub(crate) fn encode_fuuro_into(&self, buf: &mut [f32], ch_offset: usize) {
        for (player_idx, melds) in self.melds.iter().enumerate() {
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
    pub(crate) fn encode_action_avail_into(&self, buf: &mut [f32], ch_offset: usize) {
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
    pub(crate) fn encode_discard_cand_into(&self, buf: &mut [f32], ch_offset: usize) {
        let player_idx = self.player_id as usize;
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
            if self.riichi_declared[player_idx] {
                1.0
            } else {
                0.0
            },
        );
    }

    /// Write 3 pass context channels (broadcast) into buf starting at ch_offset.
    pub(crate) fn encode_pass_ctx_into(&self, buf: &mut [f32], ch_offset: usize) {
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
    pub(crate) fn encode_last_ted_into(&self, buf: &mut [f32], ch_offset: usize) {
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
    pub(crate) fn encode_riichi_sute_into(&self, buf: &mut [f32], ch_offset: usize) {
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
}
