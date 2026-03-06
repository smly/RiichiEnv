use crate::action::ActionType;
use crate::shanten;
use crate::types::MeldType;

use super::Observation3P;
use super::helpers::{
    TILE_DIM_3P, add_val, broadcast_scalar, get_next_tile_sanma, set_val, tile34_to_compact,
};

const NP: usize = 3;
const TOTAL_TILES: u32 = 108;

/// Internal (non-PyO3) methods that write features directly into a flat f32 buffer.
/// Buffer layout: channel-major, buf[(ch_offset + ch) * TILE_DIM_3P + tile] = value.
impl Observation3P {
    /// Sanma dora next tile.
    fn dora_next(&self, tile: u32) -> u8 {
        get_next_tile_sanma(tile)
    }

    /// Write 74 base encode channels into buf starting at ch_offset.
    pub(crate) fn encode_base_into(&self, buf: &mut [f32], ch_offset: usize) {
        // Hand (ch 0-3) + Red (ch 4)
        {
            let mut counts = [0u8; TILE_DIM_3P];
            for &t in &self.hands[self.player_id as usize] {
                let idx34 = (t as usize) / 4;
                if let Some(idx) = tile34_to_compact(idx34) {
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
                    let idx34 = (t as usize) / 4;
                    if let Some(idx) = tile34_to_compact(idx34) {
                        set_val(buf, ch_offset, 5 + m_idx, idx, 1.0);
                    }
                }
            }
        }

        // Dora Indicators (ch 9)
        for &t in &self.dora_indicators {
            let idx34 = (t as usize) / 4;
            if let Some(idx) = tile34_to_compact(idx34) {
                set_val(buf, ch_offset, 9, idx, 1.0);
            }
        }

        // Self discards last 4 (ch 10-13)
        {
            let discs = &self.discards[self.player_id as usize];
            for (i, &t) in discs.iter().rev().take(4).enumerate() {
                let idx34 = (t as usize) / 4;
                if let Some(idx) = tile34_to_compact(idx34) {
                    set_val(buf, ch_offset, 10 + i, idx, 1.0);
                }
            }
        }

        // Opponents discards last 4 (ch 14-21 for 2 opponents)
        for i in 1..NP {
            let opp_id = (self.player_id as usize + i) % NP;
            {
                let discs = &self.discards[opp_id];
                for (j, &t) in discs.iter().rev().take(4).enumerate() {
                    let idx34 = (t as usize) / 4;
                    if let Some(idx) = tile34_to_compact(idx34) {
                        let ch = 14 + (i - 1) * 4 + j;
                        set_val(buf, ch_offset, ch, idx, 1.0);
                    }
                }
            }
        }

        // Discard counts (ch 26-28 for 3 players, relative order)
        for (ch_idx, &abs_idx) in self.rel_order().iter().enumerate() {
            let count_norm = (self.discards[abs_idx].len() as f32) / 24.0;
            broadcast_scalar(buf, ch_offset, 26 + ch_idx, count_norm);
        }

        // Tiles left in wall (ch 30)
        let mut tiles_used = 0;
        for discs in &self.discards {
            tiles_used += discs.len();
        }
        for melds_list in &self.melds {
            for meld in melds_list {
                tiles_used += meld.tiles.len();
                if meld.called_tile.is_some() {
                    tiles_used -= 1;
                }
            }
        }
        tiles_used += self.hands[self.player_id as usize].len();
        tiles_used += self.dora_indicators.len();
        let tiles_left = (TOTAL_TILES as i32 - tiles_used as i32).max(0) as f32;
        broadcast_scalar(buf, ch_offset, 30, tiles_left / 70.0);

        // Riichi (ch 31: self, ch 32-33: 2 opponents)
        if self.riichi_declared[self.player_id as usize] {
            broadcast_scalar(buf, ch_offset, 31, 1.0);
        }
        for i in 1..NP {
            let opp_id = (self.player_id as usize + i) % NP;
            if self.riichi_declared[opp_id] {
                broadcast_scalar(buf, ch_offset, 32 + (i - 1), 1.0);
            }
        }

        // Winds (ch 35-36)
        // tile34=27-30 (winds) → compact=20-23
        let rw = self.round_wind as usize;
        if let Some(compact_wind) = tile34_to_compact(27 + rw) {
            set_val(buf, ch_offset, 35, compact_wind, 1.0);
        }
        let seat = (self.player_id + NP as u8 - self.oya) % NP as u8;
        if let Some(compact_wind) = tile34_to_compact(27 + (seat as usize)) {
            set_val(buf, ch_offset, 36, compact_wind, 1.0);
        }

        // Honba/Sticks (ch 37-38)
        broadcast_scalar(buf, ch_offset, 37, (self.honba as f32) / 10.0);
        broadcast_scalar(buf, ch_offset, 38, (self.riichi_sticks as f32) / 5.0);

        // Scores (ch 39-44: 3 players x 2 normalizations, relative order)
        for (ch_idx, &abs_idx) in self.rel_order().iter().enumerate() {
            broadcast_scalar(
                buf,
                ch_offset,
                39 + ch_idx,
                (self.scores[abs_idx].clamp(0, 100000) as f32) / 100000.0,
            );
            broadcast_scalar(
                buf,
                ch_offset,
                43 + ch_idx,
                (self.scores[abs_idx].clamp(0, 30000) as f32) / 30000.0,
            );
        }

        // Waits (ch 47)
        for &t in &self.waits {
            if let Some(idx) = tile34_to_compact(t as usize) {
                set_val(buf, ch_offset, 47, idx, 1.0);
            }
        }

        // Is Tenpai (ch 48)
        broadcast_scalar(buf, ch_offset, 48, if self.is_tenpai { 1.0 } else { 0.0 });

        // Rank (ch 49-51 for 3 players)
        let my_score = self.scores[self.player_id as usize];
        let mut rank = 0;
        for &s in &self.scores {
            if s > my_score {
                rank += 1;
            }
        }
        if rank < NP {
            broadcast_scalar(buf, ch_offset, 49 + rank, 1.0);
        }

        // Kyoku (ch 53)
        broadcast_scalar(buf, ch_offset, 53, (self.kyoku_index as f32) / 8.0);

        // Round Progress (ch 54)
        let round_progress = (self.round_wind as f32) * 4.0 + (self.kyoku_index as f32);
        broadcast_scalar(buf, ch_offset, 54, round_progress / 7.0);

        // Dora Count (ch 55-57 for 3 players)
        let mut dora_counts = [0u8; NP];
        for (player_idx, dora_count) in dora_counts.iter_mut().enumerate() {
            for meld in &self.melds[player_idx] {
                for &tile in &meld.tiles {
                    for &dora_ind in &self.dora_indicators {
                        let dora_tile = self.dora_next(dora_ind);
                        if (tile / 4) == (dora_tile / 4) {
                            *dora_count += 1;
                        }
                    }
                }
            }
            for &tile in &self.discards[player_idx] {
                for &dora_ind in &self.dora_indicators {
                    let dora_tile = self.dora_next(dora_ind);
                    if ((tile / 4) as u8) == (dora_tile / 4) {
                        *dora_count += 1;
                    }
                }
            }
        }
        for &tile in &self.hands[self.player_id as usize] {
            for &dora_ind in &self.dora_indicators {
                let dora_tile = self.dora_next(dora_ind);
                if ((tile / 4) as u8) == (dora_tile / 4) {
                    dora_counts[self.player_id as usize] += 1;
                }
            }
        }
        for (ch_idx, &abs_idx) in self.rel_order().iter().enumerate() {
            broadcast_scalar(
                buf,
                ch_offset,
                55 + ch_idx,
                (dora_counts[abs_idx] as f32) / 12.0,
            );
        }

        // Melds Count (ch 59-61 for 3 players, relative order)
        for (ch_idx, &abs_idx) in self.rel_order().iter().enumerate() {
            broadcast_scalar(
                buf,
                ch_offset,
                59 + ch_idx,
                (self.melds[abs_idx].len() as f32) / 4.0,
            );
        }

        // Tiles Seen (ch 63)
        let mut seen = [0u8; TILE_DIM_3P];
        for &t in &self.hands[self.player_id as usize] {
            if let Some(idx) = tile34_to_compact((t as usize) / 4) {
                seen[idx] += 1;
            }
        }
        for mlist in &self.melds {
            for m in mlist {
                for &t in &m.tiles {
                    if let Some(idx) = tile34_to_compact((t as usize) / 4) {
                        seen[idx] += 1;
                    }
                }
            }
        }
        for dlist in &self.discards {
            for &t in dlist {
                if let Some(idx) = tile34_to_compact((t as usize) / 4) {
                    seen[idx] += 1;
                }
            }
        }
        for &t in &self.dora_indicators {
            if let Some(idx) = tile34_to_compact((t as usize) / 4) {
                seen[idx] += 1;
            }
        }
        for (i, &s) in seen.iter().enumerate() {
            set_val(buf, ch_offset, 63, i, (s as f32) / 4.0);
        }

        // Extended discards self (ch 64-67)
        {
            let discs = &self.discards[self.player_id as usize];
            for (i, &t) in discs.iter().rev().skip(4).take(4).enumerate() {
                let idx34 = (t as usize) / 4;
                if let Some(idx) = tile34_to_compact(idx34) {
                    set_val(buf, ch_offset, 64 + i, idx, 1.0);
                }
            }
        }

        // Extended discards opponent 1 (ch 68-69)
        {
            let opp1_id = (self.player_id as usize + 1) % NP;
            let discs = &self.discards[opp1_id];
            for (i, &t) in discs.iter().rev().skip(4).take(2).enumerate() {
                let idx34 = (t as usize) / 4;
                if let Some(idx) = tile34_to_compact(idx34) {
                    set_val(buf, ch_offset, 68 + i, idx, 1.0);
                }
            }
        }

        // Tsumogiri flags (ch 70-72 for 3 players, relative order)
        for (ch_idx, &abs_idx) in self.rel_order().iter().enumerate() {
            if !self.tsumogiri_flags[abs_idx].is_empty() {
                let last_tsumogiri = *self.tsumogiri_flags[abs_idx].last().unwrap_or(&false);
                broadcast_scalar(
                    buf,
                    ch_offset,
                    70 + ch_idx,
                    if last_tsumogiri { 1.0 } else { 0.0 },
                );
            }
        }
    }

    /// Write 3 discard history decay channels into buf starting at ch_offset.
    /// Channels are in relative seat order: [self, next, prev].
    pub(crate) fn encode_discard_decay_into(&self, buf: &mut [f32], ch_offset: usize) {
        let decay_rate = 0.2f32;
        for (ch_idx, &abs_idx) in self.rel_order().iter().enumerate() {
            let discs = &self.discards[abs_idx];
            let max_len = discs.len();
            if max_len == 0 {
                continue;
            }
            for (turn, &tile) in discs.iter().enumerate() {
                let tile34 = (tile as usize) / 4;
                if let Some(idx) = tile34_to_compact(tile34) {
                    let age = (max_len - 1 - turn) as f32;
                    let weight = (-decay_rate * age).exp();
                    add_val(buf, ch_offset, ch_idx, idx, weight);
                }
            }
        }
    }

    /// Write 12 shanten efficiency channels (broadcast) into buf starting at ch_offset.
    /// 3 players x 4 features = 12 channels, each broadcast to TILE_DIM_3P tiles.
    /// Channels are in relative seat order: [self, next, prev].
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

        for (ch_idx, &abs_idx) in self.rel_order().iter().enumerate() {
            let base_ch = ch_idx * 4;

            if abs_idx == self.player_id as usize {
                let hand = &self.hands[abs_idx];
                let shanten_val = shanten::calculate_shanten_3p(hand);
                let effective = shanten::calculate_effective_tiles_3p(hand);
                let best_ukeire = shanten::calculate_best_ukeire_3p(hand, &all_visible);

                broadcast_scalar(buf, ch_offset, base_ch, (shanten_val as f32).max(0.0) / 8.0);
                broadcast_scalar(buf, ch_offset, base_ch + 1, (effective as f32) / 27.0);
                broadcast_scalar(buf, ch_offset, base_ch + 2, (best_ukeire as f32) / 80.0);
            } else {
                broadcast_scalar(buf, ch_offset, base_ch, 0.5);
                broadcast_scalar(buf, ch_offset, base_ch + 1, 0.5);
                broadcast_scalar(buf, ch_offset, base_ch + 2, 0.5);
            }

            let turn_count = self.discards[abs_idx].len() as f32;
            broadcast_scalar(buf, ch_offset, base_ch + 3, (turn_count / 18.0).min(1.0));
        }
    }

    /// Write 3 ankan overview channels into buf starting at ch_offset.
    /// Channels are in relative seat order: [self, next, prev].
    pub(crate) fn encode_ankan_into(&self, buf: &mut [f32], ch_offset: usize) {
        for (ch_idx, &abs_idx) in self.rel_order().iter().enumerate() {
            for meld in &self.melds[abs_idx] {
                if matches!(meld.meld_type, MeldType::Ankan)
                    && let Some(&tile) = meld.tiles.first()
                {
                    let tile34 = (tile / 4) as usize;
                    if let Some(idx) = tile34_to_compact(tile34) {
                        set_val(buf, ch_offset, ch_idx, idx, 1.0);
                    }
                }
            }
        }
    }

    /// Write 60 fuuro overview channels into buf starting at ch_offset.
    /// Layout: player(3) x meld(4) x tile_slot(5) flattened = 60 channels, each spatial (TILE_DIM_3P).
    /// Players are in relative seat order: [self, next, prev].
    pub(crate) fn encode_fuuro_into(&self, buf: &mut [f32], ch_offset: usize) {
        for (ch_idx, &abs_idx) in self.rel_order().iter().enumerate() {
            for (meld_idx, meld) in self.melds[abs_idx].iter().enumerate() {
                if meld_idx >= 4 {
                    break;
                }
                for (tile_slot_idx, &tile) in meld.tiles.iter().enumerate() {
                    if tile_slot_idx >= 4 {
                        break;
                    }
                    let tile34 = (tile / 4) as usize;
                    if let Some(idx) = tile34_to_compact(tile34) {
                        let ch = ch_idx * 20 + meld_idx * 5 + tile_slot_idx;
                        set_val(buf, ch_offset, ch, idx, 1.0);
                    }
                    if matches!(tile, 16 | 52 | 88) {
                        let tile34 = (tile / 4) as usize;
                        if let Some(idx) = tile34_to_compact(tile34) {
                            let ch = ch_idx * 20 + meld_idx * 5 + 4;
                            set_val(buf, ch_offset, ch, idx, 1.0);
                        }
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
                    // Chi shouldn't happen in 3P, but handle gracefully
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
        let current_shanten = shanten::calculate_shanten_3p(hand);

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
            let new_shanten = shanten::calculate_shanten_3p(&new_hand);
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
            let tile34 = (tile / 4) as usize;
            if let Some(compact) = tile34_to_compact(tile34) {
                broadcast_scalar(buf, ch_offset, 0, compact as f32 / 26.0);
            }
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
                .map(|&ind| self.dora_next(ind))
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

    /// Write 6 last tedashis channels (broadcast) into buf starting at ch_offset.
    /// 2 opponents x 3 features = 6 channels.
    pub(crate) fn encode_last_ted_into(&self, buf: &mut [f32], ch_offset: usize) {
        let dora_tiles: Vec<u8> = self
            .dora_indicators
            .iter()
            .map(|&ind| self.dora_next(ind))
            .collect();

        let mut opp_idx = 0;
        for player_id in 0..NP {
            if player_id == self.player_id as usize {
                continue;
            }
            if let Some(tile) = self.last_tedashis[player_id] {
                let tile34 = (tile / 4) as usize;
                if let Some(compact) = tile34_to_compact(tile34) {
                    broadcast_scalar(buf, ch_offset, opp_idx * 3, compact as f32 / 26.0);
                }
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

    /// Write 6 riichi sutehais channels (broadcast) into buf starting at ch_offset.
    /// 2 opponents x 3 features = 6 channels.
    pub(crate) fn encode_riichi_sute_into(&self, buf: &mut [f32], ch_offset: usize) {
        let dora_tiles: Vec<u8> = self
            .dora_indicators
            .iter()
            .map(|&ind| self.dora_next(ind))
            .collect();

        let mut opp_idx = 0;
        for player_id in 0..NP {
            if player_id == self.player_id as usize {
                continue;
            }
            if let Some(tile) = self.riichi_sutehais[player_id] {
                let tile34 = (tile / 4) as usize;
                if let Some(compact) = tile34_to_compact(tile34) {
                    broadcast_scalar(buf, ch_offset, opp_idx * 3, compact as f32 / 26.0);
                }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Meld;

    /// Build a minimal Observation3P via the public constructor.
    fn make_obs(
        player_id: u8,
        discards: [Vec<u8>; 3],
        melds: [Vec<Meld>; 3],
    ) -> super::super::Observation3P {
        super::super::Observation3P::new(
            player_id,
            [
                // Sanma hands: use 1m,9m,1-9p,1-9s tiles only (no 2m-8m)
                vec![0, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76],
                vec![1, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77],
                vec![2, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78],
            ],
            melds,
            discards,
            vec![],       // dora_indicators
            [35000; 3],   // scores
            [false; 3],   // riichi_declared
            vec![],       // legal_actions
            vec![],       // events
            0,            // honba
            0,            // riichi_sticks
            27,           // round_wind
            0,            // oya
            0,            // kyoku_index
            vec![],       // waits
            false,        // is_tenpai
            [None; 3],    // riichi_sutehais
            [None; 3],    // last_tedashis
            None,         // last_discard
        )
    }

    fn empty_melds() -> [Vec<Meld>; 3] {
        [vec![], vec![], vec![]]
    }

    /// Helper: read a single value from the flat buffer at (ch, tile).
    fn read_val(buf: &[f32], ch: usize, tile: usize) -> f32 {
        buf[ch * TILE_DIM_3P + tile]
    }

    #[test]
    fn test_discard_decay_relative_order_3p() {
        // Player 0 discards tile 0 (1m, compact=0), player 1 discards tile 36 (1p, tile34=9, compact=2)
        let discards: [Vec<u8>; 3] = [vec![0], vec![36], vec![]];

        // From player 0: rel_order=[0,1,2], ch0=self, ch1=player1
        let obs0 = make_obs(0, discards.clone(), empty_melds());
        let mut buf0 = vec![0.0f32; 3 * TILE_DIM_3P];
        obs0.encode_discard_decay_into(&mut buf0, 0);
        assert!(read_val(&buf0, 0, 0) > 0.0, "ch0 should have player0's 1m");
        assert!(read_val(&buf0, 1, 2) > 0.0, "ch1 should have player1's 1p");

        // From player 1: rel_order=[1,2,0], ch0=self(player1), ch2=player0
        let obs1 = make_obs(1, discards, empty_melds());
        let mut buf1 = vec![0.0f32; 3 * TILE_DIM_3P];
        obs1.encode_discard_decay_into(&mut buf1, 0);
        assert!(
            read_val(&buf1, 0, 2) > 0.0,
            "ch0 should have player1's 1p (self)"
        );
        assert!(
            read_val(&buf1, 2, 0) > 0.0,
            "ch2 should have player0's 1m"
        );
    }

    #[test]
    fn test_shanten_relative_order_3p() {
        let discards: [Vec<u8>; 3] = [vec![0, 32], vec![36], vec![40, 44, 48]];

        let obs0 = make_obs(0, discards.clone(), empty_melds());
        let mut buf0 = vec![0.0f32; 12 * TILE_DIM_3P];
        obs0.encode_shanten_into(&mut buf0, 0);

        let obs1 = make_obs(1, discards, empty_melds());
        let mut buf1 = vec![0.0f32; 12 * TILE_DIM_3P];
        obs1.encode_shanten_into(&mut buf1, 0);

        // obs0: ch_idx=0 is self (player0), turn_count at base_ch+3 = ch3
        let obs0_self_turn = read_val(&buf0, 3, 0);
        // obs1: ch_idx=2 is player0 (prev), turn_count at 2*4+3 = ch11
        let obs1_prev_turn = read_val(&buf1, 2 * 4 + 3, 0);
        assert!(
            (obs0_self_turn - obs1_prev_turn).abs() < 1e-6,
            "Player 0's turn count should match across perspectives"
        );

        // Self shanten should differ from opponent placeholder 0.5
        let obs0_self_shanten = read_val(&buf0, 0, 0);
        let obs0_next_shanten = read_val(&buf0, 4, 0);
        assert_eq!(obs0_next_shanten, 0.5);
        assert!(
            (obs0_self_shanten - obs0_next_shanten).abs() > 1e-6,
            "self shanten should differ from opponent placeholder"
        );
    }

    #[test]
    fn test_ankan_relative_order_3p() {
        let mut melds = empty_melds();
        // Player 1 has an ankan of 1p (tiles 36,37,38,39; tile34=9, compact=2)
        melds[1] = vec![Meld {
            meld_type: MeldType::Ankan,
            tiles: vec![36, 37, 38, 39],
            opened: false,
            from_who: 0,
            called_tile: None,
        }];

        // From player 0: rel_order=[0,1,2], player1 at ch_idx=1
        let obs0 = make_obs(0, Default::default(), melds.clone());
        let mut buf0 = vec![0.0f32; 3 * TILE_DIM_3P];
        obs0.encode_ankan_into(&mut buf0, 0);
        assert_eq!(read_val(&buf0, 1, 2), 1.0, "player1's ankan at ch1");
        assert_eq!(read_val(&buf0, 0, 2), 0.0, "ch0 (self) should be empty");

        // From player 2: rel_order=[2,0,1], player1 at ch_idx=2
        let obs2 = make_obs(2, Default::default(), melds);
        let mut buf2 = vec![0.0f32; 3 * TILE_DIM_3P];
        obs2.encode_ankan_into(&mut buf2, 0);
        assert_eq!(read_val(&buf2, 2, 2), 1.0, "player1's ankan at ch2");
        assert_eq!(read_val(&buf2, 1, 2), 0.0, "ch1 should be empty");
    }

    #[test]
    fn test_fuuro_relative_order_3p() {
        let mut melds = empty_melds();
        // Player 2 has a pon of 1p (tiles 36,37,38; tile34=9, compact=2)
        melds[2] = vec![Meld {
            meld_type: MeldType::Pon,
            tiles: vec![36, 37, 38],
            opened: true,
            from_who: 0,
            called_tile: Some(36),
        }];

        // From player 0: rel_order=[0,1,2], player2 at ch_idx=2
        let obs0 = make_obs(0, Default::default(), melds.clone());
        let mut buf0 = vec![0.0f32; 60 * TILE_DIM_3P];
        obs0.encode_fuuro_into(&mut buf0, 0);
        // ch = 2*20 + 0*5 + 0 = 40
        assert_eq!(read_val(&buf0, 40, 2), 1.0, "player2 meld tile0 at ch40");

        // From player 1: rel_order=[1,2,0], player2 at ch_idx=1
        let obs1 = make_obs(1, Default::default(), melds);
        let mut buf1 = vec![0.0f32; 60 * TILE_DIM_3P];
        obs1.encode_fuuro_into(&mut buf1, 0);
        // ch = 1*20 + 0*5 + 0 = 20
        assert_eq!(read_val(&buf1, 20, 2), 1.0, "player2 meld tile0 at ch20");
    }

    #[test]
    fn test_self_channel_always_first_3p() {
        // tile 0=1m(tile34=0,compact=0), tile 36=1p(tile34=9,compact=2), tile 72=1s(tile34=18,compact=11)
        let discards: [Vec<u8>; 3] = [vec![0], vec![36], vec![72]];
        let compact_indices = [0usize, 2, 11];

        for pid in 0..3u8 {
            let obs = make_obs(pid, discards.clone(), empty_melds());
            let mut buf = vec![0.0f32; 3 * TILE_DIM_3P];
            obs.encode_discard_decay_into(&mut buf, 0);

            assert!(
                read_val(&buf, 0, compact_indices[pid as usize]) > 0.0,
                "pid={}: ch0 should have self discard",
                pid
            );
        }
    }
}
