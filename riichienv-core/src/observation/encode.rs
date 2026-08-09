use crate::action::ActionType;
use crate::drev::{self, DrevInput};
use crate::errors::{RiichiError, RiichiResult};
use crate::feature_context::FeatureContext;
use crate::shanten;
use crate::sp::{self, SpInput};
use crate::types::MeldType;

use super::Observation;
use super::helpers::{add_val, broadcast_scalar, get_next_tile, set_val};

/// Number of "extended observation" channels written by `encode_*_into` calls
/// at offsets 0..215. The full feature block is
/// `OBS_EXTENDED_CHANNELS + sp::SP_CHANNELS + drev::DREV_CHANNELS`.
pub const OBS_BASE_CHANNELS: usize = 74;
pub const OBS_EXTENDED_CHANNELS: usize = 215;
pub const OBS_TILE_TYPES: usize = 34;

/// Internal (non-PyO3) methods that write features directly into a flat f32 buffer.
/// Buffer layout: channel-major, buf[(ch_offset + ch) * 34 + tile] = value.
impl Observation {
    fn encode_base_into_with_context(
        &self,
        buf: &mut [f32],
        ch_offset: usize,
        context: &FeatureContext<'_>,
    ) {
        // Hand (ch 0-3) + Red (ch 4)
        {
            for (i, &c) in context.hand_counts().iter().enumerate() {
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
            for (red_index, present) in context.akas_in_hand().into_iter().enumerate() {
                if present {
                    set_val(buf, ch_offset, 4, 4 + red_index * 9, 1.0);
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

        // Discard counts (ch 26-29, relative order)
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

        // Scores (ch 39-46, relative order)
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
        for (ch_idx, &abs_idx) in self.rel_order().iter().enumerate() {
            broadcast_scalar(
                buf,
                ch_offset,
                55 + ch_idx,
                (dora_counts[abs_idx] as f32) / 12.0,
            );
        }

        // Melds Count (ch 59-62, relative order)
        for (ch_idx, &abs_idx) in self.rel_order().iter().enumerate() {
            broadcast_scalar(
                buf,
                ch_offset,
                59 + ch_idx,
                (self.melds[abs_idx].len() as f32) / 4.0,
            );
        }

        // Tiles Seen (ch 63)
        for (i, &s) in context.visible_counts_v0().iter().enumerate() {
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

        // Tsumogiri flags (ch 70-73, relative order)
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

    /// Write 4 discard history decay channels into buf starting at ch_offset.
    /// Channels are in relative seat order: [self, shimocha, toimen, kamicha].
    pub(crate) fn encode_discard_decay_into(&self, buf: &mut [f32], ch_offset: usize) {
        let decay_rate = 0.2f32;
        for (ch_idx, &abs_idx) in self.rel_order().iter().enumerate() {
            let discs = &self.discards[abs_idx];
            let max_len = discs.len();
            if max_len == 0 {
                continue;
            }
            for (turn, &tile) in discs.iter().enumerate() {
                let tile_idx = (tile as usize) / 4;
                if tile_idx < 34 {
                    let age = (max_len - 1 - turn) as f32;
                    let weight = (-decay_rate * age).exp();
                    add_val(buf, ch_offset, ch_idx, tile_idx, weight);
                }
            }
        }
    }

    /// Write 16 shanten efficiency channels (broadcast) into buf starting at ch_offset.
    /// 4 players × 4 features = 16 channels, each broadcast to 34 tiles.
    /// Channels are in relative seat order: [self, shimocha, toimen, kamicha].
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
                let shanten_val = shanten::calculate_shanten(hand);
                let effective = shanten::calculate_effective_tiles_with_discard(hand);
                let best_ukeire = shanten::calculate_best_ukeire(hand, &all_visible);

                broadcast_scalar(buf, ch_offset, base_ch, (shanten_val as f32).max(0.0) / 8.0);
                broadcast_scalar(buf, ch_offset, base_ch + 1, (effective as f32) / 34.0);
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

    /// Write 4 ankan overview channels into buf starting at ch_offset.
    /// Channels are in relative seat order: [self, shimocha, toimen, kamicha].
    pub(crate) fn encode_ankan_into(&self, buf: &mut [f32], ch_offset: usize) {
        for (ch_idx, &abs_idx) in self.rel_order().iter().enumerate() {
            for meld in &self.melds[abs_idx] {
                if matches!(meld.meld_type, MeldType::Ankan)
                    && let Some(&tile) = meld.tiles.first()
                {
                    let tile_type = (tile / 4) as usize;
                    if tile_type < 34 {
                        set_val(buf, ch_offset, ch_idx, tile_type, 1.0);
                    }
                }
            }
        }
    }

    /// Write 80 fuuro overview channels into buf starting at ch_offset.
    /// Layout: player(4) × meld(4) × tile_slot(5) flattened = 80 channels, each spatial (34).
    /// Players are in relative seat order: [self, shimocha, toimen, kamicha].
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
                    let tile_type = (tile / 4) as usize;
                    if tile_type < 34 {
                        // channel = rel_player*20 + meld*5 + slot
                        let ch = ch_idx * 20 + meld_idx * 5 + tile_slot_idx;
                        set_val(buf, ch_offset, ch, tile_type, 1.0);
                    }
                    // Check for aka (red five: 5m=16, 5p=52, 5s=88)
                    if matches!(tile, 16 | 52 | 88) {
                        let ch = ch_idx * 20 + meld_idx * 5 + 4;
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

    /// Write `sp::SP_CHANNELS` SP channels into buf starting at ch_offset.
    pub(crate) fn encode_sp_into(&self, buf: &mut [f32], ch_offset: usize) {
        let context = FeatureContext::new_unchecked(self);
        self.encode_sp_into_with_context(buf, ch_offset, &context);
    }

    fn encode_sp_into_with_context(
        &self,
        buf: &mut [f32],
        ch_offset: usize,
        context: &FeatureContext<'_>,
    ) {
        let input = SpInput::from_feature_context(context);
        let result = sp::calculate_sp(&input);
        sp::encode_sp_into(&result, buf, ch_offset);
    }

    /// Write `drev::DREV_CHANNELS` DREV channels into buf starting at
    /// ch_offset. DREV (Deal-in Risk EV) is a separate feature block from
    /// SP — orthogonal concept (opponent threat vs. own EV) so it has its
    /// own module and gets concatenated downstream.
    pub(crate) fn encode_drev_into(&self, buf: &mut [f32], ch_offset: usize) {
        let context = FeatureContext::new_unchecked(self);
        self.encode_drev_into_with_context(buf, ch_offset, &context);
    }

    fn encode_drev_into_with_context(
        &self,
        buf: &mut [f32],
        ch_offset: usize,
        context: &FeatureContext<'_>,
    ) {
        let input = DrevInput::from_feature_context(context);
        let result = drev::calculate_drev(&input);
        drev::encode_drev_into(&result, buf, ch_offset);
    }

    /// Encode the stable 74-channel player observation as channel-major
    /// `f32` values. This is the pure-Rust counterpart of Python's
    /// `Observation.encode()` method.
    pub fn encode_base_features(&self) -> RiichiResult<Vec<f32>> {
        self.validate()?;
        let mut buf = vec![0.0; OBS_BASE_CHANNELS * OBS_TILE_TYPES];
        self.encode_base_features_into_unchecked(&mut buf);
        Ok(buf)
    }

    /// Write the legacy 74-channel base-v0 layout into a caller-owned buffer.
    ///
    /// Base-v0 historically counts a called tile in both discards and melds
    /// for channel 30. Extended-v0 uses the corrected wall estimate in
    /// `encode_base_into`; the two frozen ABIs therefore intentionally differ
    /// after a call.
    pub fn encode_base_features_into(&self, buf: &mut [f32]) -> RiichiResult<()> {
        let expected = OBS_BASE_CHANNELS * OBS_TILE_TYPES;
        if buf.len() != expected {
            return Err(RiichiError::InvalidState {
                message: format!(
                    "base-4p v0 output has length {}; expected {expected}",
                    buf.len()
                ),
            });
        }
        self.validate()?;
        self.encode_base_features_into_unchecked(buf);
        Ok(())
    }

    pub(crate) fn encode_base_features_into_unchecked(&self, buf: &mut [f32]) {
        debug_assert_eq!(buf.len(), OBS_BASE_CHANNELS * OBS_TILE_TYPES);
        buf.fill(0.0);
        let context = FeatureContext::new_unchecked(self);
        self.encode_base_into_with_context(buf, 0, &context);

        let tiles_used = self.discards.iter().map(Vec::len).sum::<usize>()
            + self
                .melds
                .iter()
                .flatten()
                .map(|meld| meld.tiles.len())
                .sum::<usize>()
            + self.hands[self.player_id as usize].len()
            + self.dora_indicators.len();
        let tiles_left = (136_i32 - tiles_used as i32).max(0) as f32;
        broadcast_scalar(buf, 0, 30, tiles_left / 70.0);
    }

    /// Encode the 215-channel observation block used by current training
    /// pipelines. The returned layout is `[channel][tile_type]`.
    pub fn encode_extended_features(&self) -> RiichiResult<Vec<f32>> {
        self.validate()?;
        let mut buf = vec![0.0; OBS_EXTENDED_CHANNELS * OBS_TILE_TYPES];
        self.encode_extended_features_into_unchecked(&mut buf);
        Ok(buf)
    }

    /// Write extended features into a caller-owned buffer. Keeping this
    /// allocation-free entry point is important for batched self-play.
    ///
    pub fn encode_extended_features_into(&self, buf: &mut [f32]) -> RiichiResult<()> {
        let expected = OBS_EXTENDED_CHANNELS * OBS_TILE_TYPES;
        if buf.len() != expected {
            return Err(RiichiError::InvalidState {
                message: format!(
                    "extended-4p v0 output has length {}; expected {expected}",
                    buf.len()
                ),
            });
        }
        self.validate()?;
        self.encode_extended_features_into_unchecked(buf);
        Ok(())
    }

    pub(crate) fn encode_extended_features_into_unchecked(&self, buf: &mut [f32]) {
        let context = FeatureContext::new_unchecked(self);
        self.encode_extended_features_into_with_context(buf, &context);
    }

    fn encode_extended_features_into_with_context(
        &self,
        buf: &mut [f32],
        context: &FeatureContext<'_>,
    ) {
        debug_assert_eq!(buf.len(), OBS_EXTENDED_CHANNELS * OBS_TILE_TYPES);
        buf.fill(0.0);
        self.encode_base_into_with_context(buf, 0, context);
        self.encode_discard_decay_into(buf, 74);
        self.encode_shanten_into(buf, 78);
        self.encode_ankan_into(buf, 94);
        self.encode_fuuro_into(buf, 98);
        self.encode_action_avail_into(buf, 178);
        self.encode_discard_cand_into(buf, 189);
        self.encode_pass_ctx_into(buf, 194);
        self.encode_last_ted_into(buf, 197);
        self.encode_riichi_sute_into(buf, 206);
    }

    /// Encode only the SP feature block.
    pub fn encode_sp_features(&self) -> RiichiResult<Vec<f32>> {
        self.validate()?;
        let mut buf = vec![0.0; crate::sp::SP_CHANNELS * OBS_TILE_TYPES];
        self.encode_sp_into(&mut buf, 0);
        Ok(buf)
    }

    /// Encode only the DREV feature block.
    pub fn encode_drev_features(&self) -> RiichiResult<Vec<f32>> {
        self.validate()?;
        let mut buf = vec![0.0; crate::drev::DREV_CHANNELS * OBS_TILE_TYPES];
        self.encode_drev_into(&mut buf, 0);
        Ok(buf)
    }

    /// Encode extended, SP, and DREV features into one contiguous block.
    pub fn encode_extended_with_sp_features(&self) -> RiichiResult<Vec<f32>> {
        self.validate()?;
        let channels = OBS_EXTENDED_CHANNELS + crate::sp::SP_CHANNELS + crate::drev::DREV_CHANNELS;
        let mut buf = vec![0.0; channels * OBS_TILE_TYPES];
        self.encode_extended_with_sp_features_into_unchecked(&mut buf);
        Ok(buf)
    }

    /// Write the combined extended + SP + DREV v0 layout into caller memory.
    pub fn encode_extended_with_sp_features_into(&self, buf: &mut [f32]) -> RiichiResult<()> {
        let channels = OBS_EXTENDED_CHANNELS + crate::sp::SP_CHANNELS + crate::drev::DREV_CHANNELS;
        let expected = channels * OBS_TILE_TYPES;
        if buf.len() != expected {
            return Err(RiichiError::InvalidState {
                message: format!(
                    "extended-sp-drev-4p v1 output has length {}; expected {expected}",
                    buf.len()
                ),
            });
        }
        self.validate()?;
        self.encode_extended_with_sp_features_into_unchecked(buf);
        Ok(())
    }

    pub(crate) fn encode_extended_with_sp_features_into_unchecked(&self, buf: &mut [f32]) {
        debug_assert_eq!(
            buf.len(),
            (OBS_EXTENDED_CHANNELS + crate::sp::SP_CHANNELS + crate::drev::DREV_CHANNELS)
                * OBS_TILE_TYPES
        );
        let context = FeatureContext::new_unchecked(self);
        self.encode_extended_features_into_with_context(
            &mut buf[..OBS_EXTENDED_CHANNELS * OBS_TILE_TYPES],
            &context,
        );
        self.encode_sp_into_with_context(buf, OBS_EXTENDED_CHANNELS, &context);
        self.encode_drev_into_with_context(
            buf,
            OBS_EXTENDED_CHANNELS + crate::sp::SP_CHANNELS,
            &context,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::{Action, ActionType};
    use crate::drev::DREV_CHANNELS;
    use crate::sp::SP_CHANNELS;
    use crate::types::Meld;

    /// Build a minimal Observation via the public constructor.
    fn make_obs(player_id: u8, discards: [Vec<u8>; 4], melds: [Vec<Meld>; 4]) -> Observation {
        Observation::new(
            player_id,
            [
                vec![0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48],
                vec![1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49],
                vec![2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50],
                vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51],
            ],
            melds,
            discards,
            vec![],     // dora_indicators
            [25000; 4], // scores
            [false; 4], // riichi_declared
            vec![],     // legal_actions
            vec![],     // events
            0,          // honba
            0,          // riichi_sticks
            27,         // round_wind
            0,          // oya
            0,          // kyoku_index
            vec![],     // waits
            false,      // is_tenpai
            [None; 4],  // riichi_sutehais
            [None; 4],  // last_tedashis
            None,       // last_discard
            None,       // drawn_tile
        )
    }

    fn empty_melds() -> [Vec<Meld>; 4] {
        [vec![], vec![], vec![], vec![]]
    }

    fn tile_type_to_136(tile_type: u8, copy: u8) -> u8 {
        tile_type * 4 + copy
    }

    fn sp_obs() -> Observation {
        // 123456789m 12p 11s + extra 5s. Discarding 5s leaves a 3p wait.
        let hand = vec![
            tile_type_to_136(0, 0),
            tile_type_to_136(1, 0),
            tile_type_to_136(2, 0),
            tile_type_to_136(3, 0),
            tile_type_to_136(4, 1),
            tile_type_to_136(5, 0),
            tile_type_to_136(6, 0),
            tile_type_to_136(7, 0),
            tile_type_to_136(8, 0),
            tile_type_to_136(9, 0),
            tile_type_to_136(10, 0),
            tile_type_to_136(18, 0),
            tile_type_to_136(18, 1),
            tile_type_to_136(22, 1),
        ];
        let legal_actions = hand
            .iter()
            .map(|&tile| Action::new(ActionType::Discard, Some(tile), vec![], Some(0)))
            .collect();

        Observation::new(
            0,
            [hand, vec![], vec![], vec![]],
            empty_melds(),
            Default::default(),
            vec![],
            [25000; 4],
            [false; 4],
            legal_actions,
            vec![],
            0,
            0,
            27,
            0,
            0,
            vec![],
            false,
            [None; 4],
            [None; 4],
            None,
            None,
        )
    }

    /// Helper: read a single value from the flat buffer at (ch, tile).
    fn read_val(buf: &[f32], ch: usize, tile: usize) -> f32 {
        buf[ch * 34 + tile]
    }

    #[test]
    fn sp_tail_matches_standalone_encoding() {
        let obs = sp_obs();
        let mut sp_only = vec![0.0f32; SP_CHANNELS * 34];
        obs.encode_sp_into(&mut sp_only, 0);

        let total_channels = OBS_EXTENDED_CHANNELS + SP_CHANNELS + DREV_CHANNELS;
        let mut extended = vec![0.0f32; total_channels * 34];
        obs.encode_extended_with_sp_features_into_unchecked(&mut extended);

        let sp_start = OBS_EXTENDED_CHANNELS * 34;
        let sp_end = (OBS_EXTENDED_CHANNELS + SP_CHANNELS) * 34;
        assert_eq!(sp_only, extended[sp_start..sp_end]);
    }

    #[test]
    fn test_discard_decay_relative_order() {
        // Player 0 discards tile 0 (1m), player 2 discards tile 36 (1p=tile_type 9)
        let discards: [Vec<u8>; 4] = [vec![0], vec![], vec![36], vec![]];

        // From player 0's perspective: rel_order = [0, 1, 2, 3]
        // ch0=player0 (self), ch2=player2 (toimen)
        let obs0 = make_obs(0, discards.clone(), empty_melds());
        let mut buf0 = vec![0.0f32; 4 * 34];
        obs0.encode_discard_decay_into(&mut buf0, 0);
        assert!(read_val(&buf0, 0, 0) > 0.0, "ch0 should have player0's 1m");
        assert!(read_val(&buf0, 2, 9) > 0.0, "ch2 should have player2's 1p");

        // From player 2's perspective: rel_order = [2, 3, 0, 1]
        // ch0=player2 (self), ch2=player0 (toimen)
        let obs2 = make_obs(2, discards, empty_melds());
        let mut buf2 = vec![0.0f32; 4 * 34];
        obs2.encode_discard_decay_into(&mut buf2, 0);
        assert!(
            read_val(&buf2, 0, 9) > 0.0,
            "ch0 should have player2's 1p (self)"
        );
        assert!(
            read_val(&buf2, 2, 0) > 0.0,
            "ch2 should have player0's 1m (toimen)"
        );

        // Self channel (ch0) for obs0 should match self channel (ch0) for obs2
        // in that each has their own discards at channel 0
        assert_eq!(read_val(&buf0, 0, 0), read_val(&buf2, 0, 9));
    }

    #[test]
    fn test_shanten_relative_order() {
        let discards: [Vec<u8>; 4] = [vec![0, 4], vec![8], vec![12, 16, 20], vec![]];

        let obs0 = make_obs(0, discards.clone(), empty_melds());
        let mut buf0 = vec![0.0f32; 16 * 34];
        obs0.encode_shanten_into(&mut buf0, 0);

        let obs2 = make_obs(2, discards, empty_melds());
        let mut buf2 = vec![0.0f32; 16 * 34];
        obs2.encode_shanten_into(&mut buf2, 0);

        // For obs0: ch_idx=0 is self (player 0), turn_count channel at base_ch+3
        // turn_count for player 0 = 2 discards
        let obs0_self_turn = read_val(&buf0, 3, 0); // ch0*4+3=3, broadcast so tile=0
        // For obs2: ch_idx=2 is toimen (player 0), turn_count channel at base_ch+3
        let obs2_toimen_turn = read_val(&buf2, 2 * 4 + 3, 0); // ch8+3=11
        assert!(
            (obs0_self_turn - obs2_toimen_turn).abs() < 1e-6,
            "Player 0's turn count should appear at ch0 for obs0 and ch2 for obs2"
        );

        // Self always at channel 0-3: self shanten should differ from opponent placeholder
        let obs0_self_shanten = read_val(&buf0, 0, 0);
        let obs0_shimocha_shanten = read_val(&buf0, 4, 0);
        assert_eq!(obs0_shimocha_shanten, 0.5, "opponent shanten should be 0.5");
        assert!(
            (obs0_self_shanten - obs0_shimocha_shanten).abs() > 1e-6,
            "obs0 self shanten should differ from opponent placeholder"
        );

        let obs2_self_shanten = read_val(&buf2, 0, 0);
        let obs2_next_shanten = read_val(&buf2, 4, 0);
        assert_eq!(obs2_next_shanten, 0.5, "opponent shanten should be 0.5");
        assert!(
            (obs2_self_shanten - obs2_next_shanten).abs() > 1e-6,
            "obs2 self shanten should differ from opponent placeholder"
        );
    }

    #[test]
    fn test_ankan_relative_order() {
        let mut melds = empty_melds();
        // Player 1 has an ankan of 1m (tiles 0,1,2,3)
        melds[1] = vec![Meld {
            meld_type: MeldType::Ankan,
            tiles: vec![0, 1, 2, 3],
            opened: false,
            from_who: 0,
            called_tile: None,
        }];

        // From player 0: rel_order=[0,1,2,3], player1 is at ch_idx=1
        let obs0 = make_obs(0, Default::default(), melds.clone());
        let mut buf0 = vec![0.0f32; 4 * 34];
        obs0.encode_ankan_into(&mut buf0, 0);
        assert_eq!(
            read_val(&buf0, 1, 0),
            1.0,
            "player1's ankan at ch1 for obs0"
        );
        assert_eq!(read_val(&buf0, 0, 0), 0.0, "ch0 (self) should be empty");

        // From player 3: rel_order=[3,0,1,2], player1 is at ch_idx=2
        let obs3 = make_obs(3, Default::default(), melds);
        let mut buf3 = vec![0.0f32; 4 * 34];
        obs3.encode_ankan_into(&mut buf3, 0);
        assert_eq!(
            read_val(&buf3, 2, 0),
            1.0,
            "player1's ankan at ch2 for obs3"
        );
        assert_eq!(read_val(&buf3, 1, 0), 0.0, "ch1 should be empty for obs3");
    }

    #[test]
    fn test_fuuro_relative_order() {
        let mut melds = empty_melds();
        // Player 2 has a chi meld with tiles 0,4,8 (1m,2m,3m)
        melds[2] = vec![Meld {
            meld_type: MeldType::Chi,
            tiles: vec![0, 4, 8],
            opened: true,
            from_who: 1,
            called_tile: Some(0),
        }];

        // From player 0: rel_order=[0,1,2,3], player2 at ch_idx=2
        let obs0 = make_obs(0, Default::default(), melds.clone());
        let mut buf0 = vec![0.0f32; 80 * 34];
        obs0.encode_fuuro_into(&mut buf0, 0);
        // ch = ch_idx*20 + meld_idx*5 + tile_slot_idx = 2*20+0*5+0=40
        assert_eq!(
            read_val(&buf0, 40, 0),
            1.0,
            "player2 meld tile0 at ch40 for obs0"
        );
        assert_eq!(
            read_val(&buf0, 41, 1),
            1.0,
            "player2 meld tile1 at ch41 for obs0"
        );

        // From player 1: rel_order=[1,2,3,0], player2 at ch_idx=1
        let obs1 = make_obs(1, Default::default(), melds);
        let mut buf1 = vec![0.0f32; 80 * 34];
        obs1.encode_fuuro_into(&mut buf1, 0);
        // ch = 1*20+0*5+0=20
        assert_eq!(
            read_val(&buf1, 20, 0),
            1.0,
            "player2 meld tile0 at ch20 for obs1"
        );
        assert_eq!(
            read_val(&buf1, 21, 1),
            1.0,
            "player2 meld tile1 at ch21 for obs1"
        );
    }

    #[test]
    fn test_self_channel_always_first() {
        // Verify that regardless of player_id, the self player's data is always at channel 0
        let discards: [Vec<u8>; 4] = [vec![0], vec![4], vec![8], vec![12]];

        for pid in 0..4u8 {
            let obs = make_obs(pid, discards.clone(), empty_melds());
            let mut buf = vec![0.0f32; 4 * 34];
            obs.encode_discard_decay_into(&mut buf, 0);

            // Channel 0 should contain the self player's discard
            let self_tile_type = (discards[pid as usize][0] as usize) / 4;
            assert!(
                read_val(&buf, 0, self_tile_type) > 0.0,
                "pid={}: ch0 should have self discard at tile_type {}",
                pid,
                self_tile_type
            );
        }
    }
}
