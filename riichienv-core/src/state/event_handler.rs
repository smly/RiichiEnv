use crate::action::Phase;
use crate::parser::mjai_to_tid;
use crate::replay::{Action as LogAction, MjaiEvent};
use crate::state::GameState;
use crate::state::legal_actions::GameStateLegalActions;
use crate::types::{INITIAL_HAND_SIZE, Meld, MeldType, TILES_4P, Wind};

fn parse_mjai_tile(s: &str) -> u8 {
    mjai_to_tid(s).unwrap_or(0)
}

fn remove_mjai_tile(hand: &mut Vec<u8>, tile: u8) -> bool {
    let index = hand
        .iter()
        .position(|&held| held == tile)
        .or_else(|| hand.iter().position(|&held| held / 4 == tile / 4));
    if let Some(index) = index {
        hand.remove(index);
        true
    } else {
        false
    }
}

pub trait GameStateEventHandler {
    fn apply_mjai_event(&mut self, event: MjaiEvent);
    fn apply_log_action(&mut self, action: &LogAction);
}

impl GameStateEventHandler for GameState {
    fn apply_mjai_event(&mut self, event: MjaiEvent) {
        macro_rules! public_tile {
            ($value:expr) => {
                match mjai_to_tid($value) {
                    Some(tile) => tile,
                    None => {
                        self.public_history_valid = false;
                        return;
                    }
                }
            };
        }

        match event {
            MjaiEvent::StartGame { .. } => {
                // Clear stale state from constructor's reset() so that
                // get_observation() does not return stale legal actions.
                self.current_player = u8::MAX;
                self.active_players.clear();
            }
            MjaiEvent::StartKyoku {
                bakaze,
                kyoku,
                honba,
                kyoutaku,
                scores,
                dora_marker,
                tehais,
                oya,
                ..
            } => {
                // Replay starts before the dealer's opening draw.
                // Keep a placeholder wall sized to that pre-tsumo state.
                self.honba = honba;
                self.riichi_sticks = kyoutaku as u32;
                self.round_wind = match bakaze.as_str() {
                    "E" => Wind::East as u8,
                    "S" => Wind::South as u8,
                    "W" => Wind::West as u8,
                    "N" => Wind::North as u8,
                    _ => Wind::East as u8,
                };
                self.oya = oya;
                self.kyoku_idx = kyoku.saturating_sub(1);
                self.current_player = u8::MAX; // No active player until first tsumo
                self.turn_count = 0;
                self.is_done = false;
                self.needs_tsumo = true;
                self.needs_initialize_next_round = false;
                self.pending_oya_won = false;
                self.pending_is_draw = false;
                self.phase = Phase::WaitAct;
                self.active_players.clear();
                self.last_discard = None;
                self.current_claims.clear();
                self.pending_kan = None;
                self.is_rinshan_flag = false;
                self.is_first_turn = true;
                self.riichi_pending_acceptance = None;
                self.drawn_tile = None;
                self.win_results.clear();
                self.last_win_results.clear();
                self.round_end_scores = None;
                self.last_error = None;
                self.is_after_kan = false;
                self.riichi_sutehais = [None; 4];
                self.last_tedashis = [None; 4];
                self.discard_actor_history.clear();
                self.resolved_discard_count = 0;
                self.public_temporary_safe_masks = [0; 4];
                self.public_history_valid = true;
                self.wall.tiles = vec![0; TILES_4P - INITIAL_HAND_SIZE * 4];
                self.wall.dora_indicators = vec![public_tile!(&dora_marker)];
                self.wall.rinshan_draw_count = 0;
                self.wall.pending_kan_dora_count = 0;
                // Match _initialize_round: 14 = dead wall (rinshan + dora stacks).
                // Without this reset, drawable_count carries over from prior rounds
                // and can fall below riichi/kan thresholds (e.g. >= 4) on replay.
                self.wall.drawable_count = (self.wall.tiles.len() as u8) - 14;
                self.wall.wall_digest.clear();
                self.wall.salt.clear();

                for p in &mut self.players {
                    p.reset_round();
                }
                self.players.iter_mut().enumerate().for_each(|(i, p)| {
                    p.score = scores[i];
                });
                for (i, hand_strs) in tehais.iter().enumerate() {
                    let mut hand = Vec::new();
                    for tile_str in hand_strs {
                        hand.push(parse_mjai_tile(tile_str));
                    }
                    hand.sort();
                    self.players[i].hand = hand;
                }
            }
            MjaiEvent::Tsumo { actor, pai } => {
                self.mark_latest_public_discard_resolved();
                self.clear_public_temporary_safety(actor as u8);
                self.players[actor].missed_agari_doujun = false;
                let tile = parse_mjai_tile(&pai);
                self.current_player = actor as u8;
                self.drawn_tile = Some(tile);
                self.players[actor].hand.push(tile);
                self.players[actor].hand.sort();
                self.players[actor].forbidden_discards.clear();
                if !self.wall.tiles.is_empty() {
                    self.wall.tiles.pop();
                    self.wall.drawable_count = self.wall.drawable_count.saturating_sub(1);
                }
                self.phase = Phase::WaitAct;
                self.active_players = vec![actor as u8];
                self.needs_tsumo = false;
            }
            MjaiEvent::Dahai {
                actor,
                pai,
                tsumogiri,
            } => {
                let tile = public_tile!(&pai);
                if self.resolved_discard_count != self.discard_actor_history.len()
                    || !self.current_claims.is_empty()
                {
                    self.public_history_valid = false;
                }
                // A standalone reducer call may omit the preceding draw/call.
                // Clearing here is conservative and prevents stale temporary
                // safety from surviving a concealed-hand transition.
                self.clear_public_temporary_safety(actor as u8);
                self.players[actor].missed_agari_doujun = false;
                if self.players[actor].riichi_declared
                    && (!tsumogiri || self.drawn_tile.map(|drawn| drawn / 4) != Some(tile / 4))
                {
                    self.public_history_valid = false;
                }
                self.current_player = actor as u8;
                if !remove_mjai_tile(&mut self.players[actor].hand, tile) {
                    self.public_history_valid = false;
                }
                let is_riichi_discard = self.players[actor].riichi_stage;
                self.players[actor].discards.push(tile);
                self.players[actor].discard_from_hand.push(!tsumogiri);
                self.players[actor]
                    .discard_is_riichi
                    .push(is_riichi_discard);
                self.discard_actor_history.push(actor as u8);
                self.last_discard = Some((actor as u8, tile));
                self.drawn_tile = None;

                if !tsumogiri {
                    self.last_tedashis[actor] = Some(tile);
                }
                if is_riichi_discard {
                    self.riichi_sutehais[actor] = Some(tile);
                    self.players[actor].riichi_declaration_index =
                        Some(self.players[actor].discards.len() - 1);
                }

                if self.players[actor].riichi_stage {
                    self.players[actor].riichi_declared = true;
                    self.players[actor].riichi_stage = false;
                }

                // Populate current_claims for reaction phase (chi/pon/ron).
                // The same helper is used by log replay so resolving the
                // response window also applies missed-win furiten uniformly.
                let claim_active = self.record_public_discard_offers(actor as u8, tile);
                self.active_players.clear();
                if !claim_active.is_empty() {
                    self.phase = Phase::WaitResponse;
                    self.active_players = claim_active;
                } else {
                    self.phase = Phase::WaitAct;
                    self.active_players.clear();
                    self.current_player = u8::MAX;
                }
                self.needs_tsumo = true;
            }
            MjaiEvent::Pon {
                actor,
                pai,
                consumed,
                ..
            } => {
                if consumed.len() != 2 {
                    self.public_history_valid = false;
                    return;
                }
                let tile = public_tile!(&pai);
                let c1 = public_tile!(&consumed[0]);
                let c2 = public_tile!(&consumed[1]);
                self.mark_latest_public_discard_resolved();
                self.clear_public_temporary_safety(actor as u8);
                self.players[actor].missed_agari_doujun = false;
                if self.players[actor].riichi_declared {
                    self.public_history_valid = false;
                }
                self.current_player = actor as u8;
                let form_tiles = vec![tile, c1, c2];

                for t in &[c1, c2] {
                    if !remove_mjai_tile(&mut self.players[actor].hand, *t) {
                        self.public_history_valid = false;
                    }
                }

                self.players[actor].melds.push(Meld {
                    meld_type: MeldType::Pon,
                    tiles: form_tiles,
                    opened: true,
                    from_who: -1,
                    called_tile: Some(tile),
                });
                self.drawn_tile = None;
                self.phase = Phase::WaitAct;
                self.active_players = vec![actor as u8];
                self.needs_tsumo = false;
                // Kuikae (swap-calling) forbidden
                self.players[actor].forbidden_discards.clear();
                if self.rule.kuikae_forbidden {
                    self.players[actor].forbidden_discards.push(tile);
                }
            }
            MjaiEvent::Chi {
                actor,
                pai,
                consumed,
                ..
            } => {
                if consumed.len() != 2 {
                    self.public_history_valid = false;
                    return;
                }
                let tile = public_tile!(&pai);
                let c1 = public_tile!(&consumed[0]);
                let c2 = public_tile!(&consumed[1]);
                self.mark_latest_public_discard_resolved();
                self.clear_public_temporary_safety(actor as u8);
                self.players[actor].missed_agari_doujun = false;
                if self.players[actor].riichi_declared {
                    self.public_history_valid = false;
                }
                self.current_player = actor as u8;
                let form_tiles = vec![tile, c1, c2];

                for t in &[c1, c2] {
                    if !remove_mjai_tile(&mut self.players[actor].hand, *t) {
                        self.public_history_valid = false;
                    }
                }

                self.players[actor].melds.push(Meld {
                    meld_type: MeldType::Chi,
                    tiles: form_tiles,
                    opened: true,
                    from_who: -1,
                    called_tile: Some(tile),
                });
                self.drawn_tile = None;
                self.phase = Phase::WaitAct;
                self.active_players = vec![actor as u8];
                self.needs_tsumo = false;
                // Kuikae forbidden for chi: mirror claim-resolution logic
                self.players[actor].forbidden_discards.clear();
                if self.rule.kuikae_forbidden {
                    self.players[actor].forbidden_discards.push(tile);
                    let t34 = tile / 4;
                    let mut consumed_34 = [c1 / 4, c2 / 4];
                    consumed_34.sort();
                    if consumed_34[0] == t34 + 1 && consumed_34[1] == t34 + 2 {
                        if t34 % 9 <= 5 {
                            self.players[actor].forbidden_discards.push((t34 + 3) * 4);
                        }
                    } else if t34 >= 2
                        && consumed_34[1] == t34 - 1
                        && consumed_34[0] == t34 - 2
                        && t34 % 9 >= 3
                    {
                        self.players[actor].forbidden_discards.push((t34 - 3) * 4);
                    }
                }
            }
            MjaiEvent::Kan {
                actor,
                pai,
                consumed,
                ..
            } => {
                if consumed.len() != 3 {
                    self.public_history_valid = false;
                    return;
                }
                let tile = public_tile!(&pai);
                let mut tiles = vec![tile];
                for consumed_tile in &consumed {
                    tiles.push(public_tile!(consumed_tile));
                }
                self.mark_latest_public_discard_resolved();
                self.clear_public_temporary_safety(actor as u8);
                self.players[actor].missed_agari_doujun = false;
                if self.players[actor].riichi_declared {
                    self.public_history_valid = false;
                }
                self.current_player = actor as u8;

                for &tv in tiles.iter().skip(1) {
                    if !remove_mjai_tile(&mut self.players[actor].hand, tv) {
                        self.public_history_valid = false;
                    }
                }

                self.players[actor].melds.push(Meld {
                    meld_type: MeldType::Daiminkan,
                    tiles,
                    opened: true,
                    from_who: -1,
                    called_tile: Some(tile),
                });
                self.phase = Phase::WaitAct;
                self.active_players = vec![actor as u8];
                self.needs_tsumo = true;
            }
            MjaiEvent::Ankan { actor, consumed } => {
                if self.resolved_discard_count != self.discard_actor_history.len()
                    || !self.current_claims.is_empty()
                {
                    self.public_history_valid = false;
                    return;
                }
                let mut tiles = Vec::with_capacity(consumed.len());
                for consumed_tile in &consumed {
                    tiles.push(public_tile!(consumed_tile));
                }
                let tile_type = tiles.first().map(|tile| tile / 4);
                let payload_is_quad = tiles.len() == 4
                    && tile_type.is_some()
                    && tiles.iter().all(|tile| Some(tile / 4) == tile_type)
                    && self.players[actor]
                        .hand
                        .iter()
                        .filter(|&&held| Some(held / 4) == tile_type)
                        .count()
                        == 4;
                if !payload_is_quad {
                    self.public_history_valid = false;
                    return;
                }
                if self.players[actor].riichi_declared {
                    let legal =
                        self._get_legal_actions_internal(actor as u8)
                            .iter()
                            .any(|action| {
                                action.action_type == crate::action::ActionType::Ankan
                                    && action.tile.map(|tile| tile / 4) == tile_type
                            });
                    if !legal {
                        self.public_history_valid = false;
                    }
                }
                self.clear_public_temporary_safety(actor as u8);
                self.players[actor].missed_agari_doujun = false;
                for &tile in &tiles {
                    if !remove_mjai_tile(&mut self.players[actor].hand, tile) {
                        self.public_history_valid = false;
                    }
                }
                self.players[actor].melds.push(Meld {
                    meld_type: MeldType::Ankan,
                    tiles,
                    opened: false,
                    from_who: -1,
                    called_tile: None,
                });
                self.current_player = actor as u8;
                self.phase = Phase::WaitAct;
                self.active_players = vec![actor as u8];
                self.needs_tsumo = true;
            }
            MjaiEvent::Kakan { actor, pai } => {
                if self.resolved_discard_count != self.discard_actor_history.len()
                    || !self.current_claims.is_empty()
                {
                    self.public_history_valid = false;
                    return;
                }
                let tile = public_tile!(&pai);
                self.clear_public_temporary_safety(actor as u8);
                self.players[actor].missed_agari_doujun = false;
                if self.players[actor].riichi_declared {
                    self.public_history_valid = false;
                }
                if !remove_mjai_tile(&mut self.players[actor].hand, tile) {
                    self.public_history_valid = false;
                }
                for m in self.players[actor].melds.iter_mut() {
                    if m.meld_type == MeldType::Pon && m.tiles[0] / 4 == tile / 4 {
                        m.meld_type = MeldType::Kakan;
                        m.tiles.push(tile);
                        break;
                    }
                }
                self.current_player = actor as u8;
                self.phase = Phase::WaitAct;
                self.active_players = vec![actor as u8];
                self.needs_tsumo = true;
            }
            MjaiEvent::Reach { actor } => {
                self.players[actor].riichi_stage = true;
                // Keep current_player and phase so that the agent can
                // select a riichi discard tile via legal_actions.
            }
            MjaiEvent::ReachAccepted { actor } => {
                self.mark_latest_public_discard_resolved();
                self.players[actor].riichi_declared = true;
                self.riichi_sticks += 1;
                self.players[actor].score -= 1000;
            }
            MjaiEvent::Dora { dora_marker } => {
                let tile = public_tile!(&dora_marker);
                self.wall.dora_indicators.push(tile);
            }
            MjaiEvent::Kita { .. } => {
                // Kita is 3P only; ignored in 4P event handler
            }
            MjaiEvent::Hora { .. } | MjaiEvent::Ryukyoku { .. } | MjaiEvent::EndKyoku => {
                self.is_done = true;
            }
            _ => {}
        }
    }

    fn apply_log_action(&mut self, action: &LogAction) {
        self.apply_log_action_with_tsumogiri(action, None);
    }
}

impl GameState {
    /// Applies replay-only source metadata without exposing it through the
    /// public v0.4.8 `replay::Action` enum.
    pub(crate) fn apply_log_action_with_tsumogiri(
        &mut self,
        action: &LogAction,
        source_tsumogiri: Option<bool>,
    ) {
        self.apply_log_action_with_metadata(action, source_tsumogiri, None);
    }

    pub(crate) fn apply_log_action_with_metadata(
        &mut self,
        action: &LogAction,
        source_tsumogiri: Option<bool>,
        dora_snapshot: Option<&[u8]>,
    ) {
        match action {
            LogAction::DiscardTile {
                seat,
                tile,
                is_liqi,
                is_wliqi,
                doras,
            } => {
                self.sync_log_dora_snapshot(doras);
                let s = *seat;
                let t = *tile;
                if self.resolved_discard_count != self.discard_actor_history.len() {
                    self.public_history_valid = false;
                }
                self.clear_public_temporary_safety(s as u8);
                let is_tsumogiri = source_tsumogiri.unwrap_or_else(|| {
                    self.drawn_tile
                        .is_some_and(|drawn_tile| drawn_tile / 4 == t / 4)
                });
                if source_tsumogiri.is_none()
                    || (self.players[s].riichi_declared
                        && (!is_tsumogiri || self.drawn_tile.map(|drawn| drawn / 4) != Some(t / 4)))
                {
                    self.public_history_valid = false;
                }

                // Update progression cache (replay mode).
                #[cfg(feature = "python")]
                if self.enable_seq_caching {
                    use crate::observation::sequence_features::process_single_event_progression;
                    use crate::parser::tid_to_mjai;
                    use std::sync::Arc;

                    if *is_liqi || *is_wliqi {
                        let ev = serde_json::json!({"type": "reach", "actor": s});
                        if let Some(entry) = process_single_event_progression(
                            &ev,
                            &mut self.round_seq_prog_pending_reach,
                        ) {
                            Arc::make_mut(&mut self.round_seq_progression).push(entry);
                        }
                    }
                    let pai = tid_to_mjai(t);
                    let ev = serde_json::json!({
                        "type": "dahai",
                        "actor": s,
                        "pai": pai,
                        "tsumogiri": is_tsumogiri,
                    });
                    if let Some(entry) = process_single_event_progression(
                        &ev,
                        &mut self.round_seq_prog_pending_reach,
                    ) {
                        Arc::make_mut(&mut self.round_seq_progression).push(entry);
                    }
                }

                let remove_index = self.players[s]
                    .hand
                    .iter()
                    .position(|&held| held == t)
                    .or_else(|| {
                        self.players[s]
                            .hand
                            .iter()
                            .position(|&held| held / 4 == t / 4)
                    });
                if let Some(idx) = remove_index {
                    self.players[s].hand.remove(idx);
                } else {
                    self.public_history_valid = false;
                }
                self.players[s].hand.sort();
                self.players[s].discards.push(t);
                self.discard_actor_history.push(s as u8);
                self.players[s].discard_from_hand.push(!is_tsumogiri);
                self.players[s]
                    .discard_is_riichi
                    .push(*is_liqi || *is_wliqi);
                self.last_discard = Some((s as u8, t));
                self.drawn_tile = None;
                // Reset same-turn furiten after own discard.
                self.players[s].missed_agari_doujun = false;
                // Track nagashi eligibility: discard must be terminal/honor
                self.players[s].nagashi_eligible &= crate::types::is_terminal_tile(t);

                if *is_liqi || *is_wliqi {
                    if !self.players[s].riichi_declared {
                        self.players[s].riichi_declared = true;
                        if *is_wliqi {
                            self.players[s].double_riichi_declared = true;
                        }
                        // Defer the 1000 deposit; it gets voided if this
                        // discard is ronned, otherwise finalized on the next
                        // DealTile / ChiPengGang.
                        self.riichi_pending_acceptance = Some(s as u8);
                    }
                    self.players[s].riichi_declaration_index =
                        Some(self.players[s].discards.len() - 1);
                }
                let claim_active = self.record_public_discard_offers(s as u8, t);
                self.current_player = (s as u8 + 1) % 4;
                if claim_active.is_empty() {
                    self.phase = Phase::WaitAct;
                    self.active_players = vec![self.current_player];
                } else {
                    self.phase = Phase::WaitResponse;
                    self.active_players = claim_active;
                }
                self.needs_tsumo = true;
                self.is_first_turn = false;
                self.is_after_kan = false;
            }
            LogAction::DealTile {
                seat, tile, doras, ..
            } => {
                self.sync_log_dora_snapshot(doras);
                self.mark_latest_public_discard_resolved();
                self.clear_public_temporary_safety(*seat as u8);
                self.players[*seat].missed_agari_doujun = false;
                // Finalize pending riichi deposit (discard was not ronned)
                if let Some(rp) = self.riichi_pending_acceptance.take() {
                    self.players[rp as usize].score -= 1000;
                    self.riichi_sticks += 1;
                }
                self.players[*seat].hand.push(*tile);
                self.drawn_tile = Some(*tile);
                self.current_player = *seat as u8;
                self.phase = Phase::WaitAct;
                self.active_players = vec![self.current_player];
                self.is_rinshan_flag = self.is_after_kan && *seat == self.current_player as usize;
                self.needs_tsumo = false;
                self.is_after_kan = false;
                self.players[*seat].hand.sort();
                if !self.wall.tiles.is_empty() {
                    self.wall.tiles.pop();
                    self.wall.drawable_count = self.wall.drawable_count.saturating_sub(1);
                }
            }
            LogAction::ChiPengGang {
                seat,
                meld_type,
                tiles,
                froms,
            } => {
                self.mark_latest_public_discard_resolved();
                self.clear_public_temporary_safety(*seat as u8);
                self.players[*seat].missed_agari_doujun = false;
                if self.players[*seat].riichi_declared {
                    self.public_history_valid = false;
                }
                // Update progression cache (replay mode).
                #[cfg(feature = "python")]
                if self.enable_seq_caching {
                    use crate::observation::sequence_features::process_single_event_progression;
                    use crate::parser::tid_to_mjai;
                    use std::sync::Arc;

                    let mtype_str = match meld_type {
                        MeldType::Chi => "chi",
                        MeldType::Pon => "pon",
                        MeldType::Daiminkan => "daiminkan",
                        _ => "",
                    };
                    if !mtype_str.is_empty() {
                        let target = froms.iter().find(|&&f| f != *seat).copied().unwrap_or(0);
                        let called_tile: Option<u8> = tiles
                            .iter()
                            .zip(froms.iter())
                            .find(|&(_, &f)| f != *seat)
                            .map(|(&t, _)| t);
                        let consumed_tiles: Vec<u8> = tiles
                            .iter()
                            .zip(froms.iter())
                            .filter(|&(_, &f)| f == *seat)
                            .map(|(&t, _)| t)
                            .collect();
                        let pai_str = called_tile.map(tid_to_mjai).unwrap_or_default();
                        let consumed_strs: Vec<String> =
                            consumed_tiles.iter().map(|&t| tid_to_mjai(t)).collect();
                        let ev = serde_json::json!({
                            "type": mtype_str,
                            "actor": *seat,
                            "target": target,
                            "pai": pai_str,
                            "consumed": consumed_strs,
                        });
                        if let Some(entry) = process_single_event_progression(
                            &ev,
                            &mut self.round_seq_prog_pending_reach,
                        ) {
                            Arc::make_mut(&mut self.round_seq_progression).push(entry);
                        }
                    }
                }

                // Finalize pending riichi deposit (discard was claimed, not ronned)
                if let Some(rp) = self.riichi_pending_acceptance.take() {
                    self.players[rp as usize].score -= 1000;
                    self.riichi_sticks += 1;
                }
                // Discard was called -> discarder loses nagashi eligibility
                if let Some((discarder_pid, _)) = self.last_discard {
                    self.players[discarder_pid as usize].nagashi_eligible = false;
                }
                // Remove tiles from hand
                for (i, t) in tiles.iter().enumerate() {
                    if i < froms.len()
                        && froms[i] == *seat
                        && let Some(idx) = self.players[*seat].hand.iter().position(|&x| x == *t)
                    {
                        self.players[*seat].hand.remove(idx);
                    }
                }
                self.players[*seat].hand.sort();

                let from_who = froms
                    .iter()
                    .find(|&&f| f != *seat)
                    .map(|&f| f as i8)
                    .unwrap_or(-1);
                let ct = tiles
                    .iter()
                    .zip(froms.iter())
                    .find(|&(_, &f)| f != *seat)
                    .map(|(&t, _)| t);
                let discarder = from_who.max(0) as u8;
                self.players[*seat].melds.push(Meld {
                    meld_type: *meld_type,
                    tiles: tiles.clone(),
                    opened: true,
                    from_who,
                    called_tile: ct,
                });

                // PAO detection: daisangen (3 dragon melds) or daisuushii (4 wind melds)
                if (*meld_type == MeldType::Pon || *meld_type == MeldType::Daiminkan)
                    && let Some(&called) = ct.as_ref()
                {
                    let tile_val = called / 4;
                    if (31..=33).contains(&tile_val) {
                        let dragon_melds = self.players[*seat]
                            .melds
                            .iter()
                            .filter(|m| {
                                let t = m.tiles[0] / 4;
                                (31..=33).contains(&t) && m.meld_type != MeldType::Chi
                            })
                            .count();
                        if dragon_melds == 3 {
                            self.players[*seat].pao.insert(37, discarder);
                        }
                    } else if (27..=30).contains(&tile_val) {
                        let wind_melds = self.players[*seat]
                            .melds
                            .iter()
                            .filter(|m| {
                                let t = m.tiles[0] / 4;
                                (27..=30).contains(&t) && m.meld_type != MeldType::Chi
                            })
                            .count();
                        if wind_melds == 4 {
                            self.players[*seat].pao.insert(50, discarder);
                        }
                    }
                }

                self.current_player = *seat as u8;
                self.phase = Phase::WaitAct;
                self.active_players = vec![self.current_player];
                let is_gang = *meld_type == MeldType::Daiminkan;
                self.needs_tsumo = is_gang;
                self.is_first_turn = false;
                self.is_after_kan = is_gang;
            }
            LogAction::AnGangAddGang {
                seat,
                meld_type,
                tiles,
                doras,
                ..
            } => {
                if self.resolved_discard_count != self.discard_actor_history.len()
                    || !self.current_claims.is_empty()
                {
                    self.public_history_valid = false;
                    return;
                }
                if let Some(snapshot) = dora_snapshot {
                    self.sync_log_dora_slice(snapshot);
                } else {
                    self.sync_log_dora_snapshot(doras);
                }
                self.clear_public_temporary_safety(*seat as u8);
                self.players[*seat].missed_agari_doujun = false;
                if self.players[*seat].riichi_declared {
                    let action_type = if *meld_type == MeldType::Ankan {
                        crate::action::ActionType::Ankan
                    } else {
                        crate::action::ActionType::Kakan
                    };
                    let tile_type = tiles.first().map(|tile| tile / 4);
                    let legal =
                        self._get_legal_actions_internal(*seat as u8)
                            .iter()
                            .any(|action| {
                                action.action_type == action_type
                                    && action.tile.map(|tile| tile / 4) == tile_type
                            });
                    if !legal {
                        self.public_history_valid = false;
                    }
                }
                // Update progression cache (replay mode).
                #[cfg(feature = "python")]
                if self.enable_seq_caching {
                    use crate::observation::sequence_features::process_single_event_progression;
                    use crate::parser::tid_to_mjai;
                    use std::sync::Arc;

                    let ev = if *meld_type == MeldType::Ankan {
                        let t_val = tiles[0] / 4;
                        let consumed_tids =
                            [t_val * 4, t_val * 4 + 1, t_val * 4 + 2, t_val * 4 + 3];
                        let consumed_strs: Vec<String> =
                            consumed_tids.iter().map(|&t| tid_to_mjai(t)).collect();
                        serde_json::json!({
                            "type": "ankan",
                            "actor": *seat,
                            "consumed": consumed_strs,
                        })
                    } else {
                        // Kakan
                        let pai_str = tid_to_mjai(tiles[0]);
                        serde_json::json!({
                            "type": "kakan",
                            "actor": *seat,
                            "pai": pai_str,
                        })
                    };
                    if let Some(entry) = process_single_event_progression(
                        &ev,
                        &mut self.round_seq_prog_pending_reach,
                    ) {
                        Arc::make_mut(&mut self.round_seq_progression).push(entry);
                    }
                }

                if *meld_type == MeldType::Ankan {
                    let t_val = tiles[0] / 4;
                    for _ in 0..4 {
                        if let Some(idx) = self.players[*seat]
                            .hand
                            .iter()
                            .position(|&x| x / 4 == t_val)
                        {
                            self.players[*seat].hand.remove(idx);
                        }
                    }
                    let mut m_tiles = vec![t_val * 4, t_val * 4 + 1, t_val * 4 + 2, t_val * 4 + 3];
                    if t_val == 4 {
                        m_tiles = vec![16, 17, 18, 19];
                    } else if t_val == 13 {
                        m_tiles = vec![52, 53, 54, 55];
                    } else if t_val == 22 {
                        m_tiles = vec![88, 89, 90, 91];
                    }

                    self.players[*seat].melds.push(Meld {
                        meld_type: *meld_type,
                        tiles: m_tiles,
                        opened: false,
                        from_who: -1,
                        called_tile: None,
                    });
                } else {
                    // Kakan
                    let tile = tiles[0];
                    if let Some(idx) = self.players[*seat].hand.iter().position(|&x| x == tile) {
                        self.players[*seat].hand.remove(idx);
                    }
                    for m in self.players[*seat].melds.iter_mut() {
                        if m.meld_type == MeldType::Pon && m.tiles[0] / 4 == tile / 4 {
                            m.meld_type = MeldType::Kakan;
                            m.tiles.push(tile);
                            m.tiles.sort();
                            break;
                        }
                    }
                    // Set last_discard so chankan ron targets the kakan player
                    self.last_discard = Some((*seat as u8, tile));
                }
                self.players[*seat].hand.sort();
                self.current_player = *seat as u8;
                self.phase = Phase::WaitAct;
                self.active_players = vec![self.current_player];
                self.needs_tsumo = true;
                self.is_first_turn = false;
                self.is_after_kan = true;
                // Also record ankan for kokushi chankan (kokushi can ron on closed kan)
                if *meld_type == MeldType::Ankan {
                    self.last_discard = Some((*seat as u8, tiles[0]));
                }
            }
            LogAction::Dora { dora_marker } => {
                self.wall.dora_indicators.push(*dora_marker);
            }
            LogAction::Hule { hules } => {
                // If a riichi deposit is pending and this is a ron, the deposit
                // is voided (MjSoul does not deduct it when the discard is ronned).
                let first_is_ron = hules.first().is_some_and(|h| !h.zimo);
                if first_is_ron {
                    self.riichi_pending_acceptance = None;
                }

                let honba = self.honba;
                let riichi_on_table = self.riichi_sticks;
                let mut honba_taken = false;

                for h in hules {
                    let winner = h.seat;
                    let is_tsumo = h.zimo;

                    if is_tsumo {
                        let is_oya = (winner as u8) == self.oya;

                        // Check PAO (sekinin barai): for yakuman tsumo, the PAO
                        // player pays the full amount for all non-winning
                        // players. We detect PAO from the player's pao map
                        // which was populated when dragon/wind melds were claimed.
                        let mut pao_payer = None;
                        let mut pao_yakuman_val: i32 = 0;
                        let mut total_yakuman_val: i32 = 0;

                        if h.yiman {
                            // Daisangen = yaku 37, Daisuushii = yaku 50
                            // Double yakuman IDs: 47, 48, 49, 50
                            for &yid in &h.fans {
                                let val: i32 = if [47, 48, 49, 50].contains(&yid) {
                                    2
                                } else {
                                    1
                                };
                                total_yakuman_val += val;
                                if let Some(&liable) = self.players[winner].pao.get(&(yid as u8)) {
                                    pao_yakuman_val += val;
                                    pao_payer = Some(liable);
                                }
                            }
                        }

                        if pao_yakuman_val > 0 {
                            // PAO: liable player pays the PAO portion entirely
                            let unit: i32 = if is_oya { 48000 } else { 32000 };
                            let pao_amt = pao_yakuman_val * unit;
                            let non_pao_yakuman_val = total_yakuman_val - pao_yakuman_val;
                            let non_pao_amt = non_pao_yakuman_val * unit;

                            if let Some(pp) = pao_payer {
                                self.players[pp as usize].score -= pao_amt;
                                self.players[winner].score += pao_amt;
                            }

                            // Non-PAO part split normally
                            if non_pao_amt > 0 {
                                if is_oya {
                                    let share = non_pao_amt / 3;
                                    for i in 0..4 {
                                        if i != winner {
                                            self.players[i].score -= share;
                                            self.players[winner].score += share;
                                        }
                                    }
                                } else {
                                    for i in 0..4 {
                                        if i != winner {
                                            let share = if (i as u8) == self.oya {
                                                non_pao_amt / 2
                                            } else {
                                                non_pao_amt / 4
                                            };
                                            self.players[i].score -= share;
                                            self.players[winner].score += share;
                                        }
                                    }
                                }
                            }

                            // Add honba bonus (paid by PAO player)
                            if let Some(pp) = pao_payer {
                                let honba_total = honba as i32 * 300;
                                self.players[pp as usize].score -= honba_total;
                                self.players[winner].score += honba_total;
                            }
                        } else {
                            // Standard tsumo distribution
                            for i in 0..4 {
                                if i != winner {
                                    let base_pay = if is_oya {
                                        h.point_zimo_xian
                                    } else if (i as u8) == self.oya {
                                        h.point_zimo_qin
                                    } else {
                                        h.point_zimo_xian
                                    };
                                    let pay = base_pay as i32 + honba as i32 * 100;
                                    self.players[i].score -= pay;
                                    self.players[winner].score += pay;
                                }
                            }
                        }
                    } else if let Some((discarder, _)) = self.last_discard {
                        // Only the first ron winner gets the honba bonus
                        let ron_honba = if !honba_taken {
                            honba_taken = true;
                            honba
                        } else {
                            0
                        };

                        // Check PAO for ron yakuman: target pays half,
                        // PAO player pays the other half.
                        let mut pao_payer_ron: Option<u8> = None;
                        if h.yiman {
                            for &yid in &h.fans {
                                if let Some(&liable) = self.players[winner].pao.get(&(yid as u8)) {
                                    pao_payer_ron = Some(liable);
                                    break;
                                }
                            }
                        }

                        if let Some(pp) = pao_payer_ron {
                            let half = h.point_rong as i32 / 2;
                            let honba_pts = ron_honba as i32 * 300;
                            self.players[pp as usize].score -= half + honba_pts;
                            self.players[discarder as usize].score -= half;
                            self.players[winner].score += h.point_rong as i32 + honba_pts;
                        } else {
                            let pay = h.point_rong as i32 + ron_honba as i32 * 300;
                            self.players[discarder as usize].score -= pay;
                            self.players[winner].score += pay;
                        }
                    }
                }

                // Distribute riichi sticks to first winner
                if !hules.is_empty() {
                    let winner = hules[0].seat;
                    self.players[winner].score += riichi_on_table as i32 * 1000;
                    self.riichi_sticks = 0;
                }

                self.is_done = true;
            }
            LogAction::NoTile => {
                // Finalize pending riichi deposit (exhaustive draw, not ronned)
                if let Some(rp) = self.riichi_pending_acceptance.take() {
                    self.players[rp as usize].score -= 1000;
                    self.riichi_sticks += 1;
                }

                // Check for nagashi mangan first
                let np = 4usize;
                let mut nagashi_winners = Vec::new();
                for (i, p) in self.players.iter().enumerate() {
                    if p.nagashi_eligible {
                        nagashi_winners.push(i as u8);
                    }
                }

                if !nagashi_winners.is_empty() {
                    // Nagashi mangan: apply mangan tsumo payment (no honba)
                    for &w in &nagashi_winners {
                        let is_oya = w == self.oya;
                        let score_res =
                            crate::score::calculate_score(5, 30, is_oya, true, 0, np as u8);
                        if is_oya {
                            for i in 0..np {
                                if i as u8 != w {
                                    self.players[i].score -= score_res.pay_tsumo_ko as i32;
                                    self.players[w as usize].score += score_res.pay_tsumo_ko as i32;
                                }
                            }
                        } else {
                            for i in 0..np {
                                if i as u8 != w {
                                    let pay = if i as u8 == self.oya {
                                        score_res.pay_tsumo_oya as i32
                                    } else {
                                        score_res.pay_tsumo_ko as i32
                                    };
                                    self.players[i].score -= pay;
                                    self.players[w as usize].score += pay;
                                }
                            }
                        }
                    }
                } else {
                    // Compute tenpai/noten payments
                    let mut tenpai = [false; 4];
                    for (i, p) in self.players.iter().enumerate() {
                        if i < 4 {
                            let calc = crate::hand_evaluator::HandEvaluator::new(
                                p.hand.clone(),
                                p.melds.clone(),
                            );
                            tenpai[i] = calc.is_tenpai();
                        }
                    }
                    let num_tp = tenpai.iter().filter(|&&t| t).count();
                    if num_tp > 0 && num_tp < 4 {
                        let pk = 3000 / num_tp as i32;
                        let pn = 3000 / (4 - num_tp) as i32;
                        for (i, tp) in tenpai.iter().enumerate() {
                            let delta = if *tp { pk } else { -pn };
                            self.players[i].score += delta;
                        }
                    }
                }
                self.is_done = true;
            }
            LogAction::LiuJu { .. } => {
                // Finalize pending riichi deposit
                if let Some(rp) = self.riichi_pending_acceptance.take() {
                    self.players[rp as usize].score -= 1000;
                    self.riichi_sticks += 1;
                }
                // Abortive draw - no score changes
                self.is_done = true;
            }
            _ => {}
        }
    }

    fn sync_log_dora_snapshot(&mut self, doras: &Option<Vec<u8>>) {
        let Some(doras) = doras else {
            return;
        };
        self.sync_log_dora_slice(doras);
    }

    fn sync_log_dora_slice(&mut self, doras: &[u8]) {
        if doras.is_empty() || doras.len() > 5 || doras.iter().any(|&tile| tile >= 136) {
            self.public_history_valid = false;
            return;
        }
        if doras.len() < self.wall.dora_indicators.len()
            || !doras.starts_with(&self.wall.dora_indicators)
        {
            // Mahjong Soul emits cumulative indicator snapshots.  Accepting
            // a shorter list or a changed prefix would silently rewrite
            // already-public history and corrupt DREV/feature values.
            self.public_history_valid = false;
            return;
        }
        self.wall.dora_indicators.clear();
        self.wall.dora_indicators.extend_from_slice(doras);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::ActionType;
    use crate::rule::GameRule;

    fn chiitoitsu_9m_wait() -> Vec<u8> {
        vec![
            32, // 9m singleton
            40, 41, // 2p
            44, 45, // 3p
            52, 53, // 5p
            68, 69, // 9p
            112, 113, // South
            124, 125, // White
        ]
    }

    #[test]
    fn direct_log_discard_records_ron_offer_until_resolution() {
        let mut state = GameState::new(2, true, Some(701), 0, GameRule::default());
        state.players[1].hand = chiitoitsu_9m_wait();
        state.players[1].melds.clear();
        state.players[1].discards.clear();
        state.players[1].riichi_declared = true;
        state.players[0].hand.push(33);

        state.apply_log_action_with_tsumogiri(
            &LogAction::DiscardTile {
                seat: 0,
                tile: 33,
                is_liqi: false,
                is_wliqi: false,
                doras: None,
            },
            Some(false),
        );

        assert!(
            state.current_claims[&1]
                .iter()
                .any(|action| action.action_type == ActionType::Ron)
        );
        assert_eq!(state.resolved_discard_count, 0);

        state.apply_log_action(&LogAction::DealTile {
            seat: 2,
            tile: 0,
            doras: None,
            left_tile_count: None,
        });

        assert_eq!(state.resolved_discard_count, 1);
        assert!(state.players[1].missed_agari_riichi);
        assert_ne!(state.public_temporary_safe_masks[1] & (1 << 8), 0);
    }

    #[test]
    fn mjai_discard_falls_back_by_tile_type_and_rejects_invalid_text() {
        let mut state = GameState::new(2, true, Some(703), 0, GameRule::default());
        state.players[0].hand = vec![53]; // normal 5p
        state.drawn_tile = Some(53);
        state.players[0].riichi_declared = true;
        state.public_temporary_safe_masks[0] = 1 << 8;

        state.apply_mjai_event(MjaiEvent::Dahai {
            actor: 0,
            pai: "5pr".to_string(),
            tsumogiri: true,
        });

        assert!(state.players[0].hand.is_empty());
        assert!(state.public_history_valid);
        assert_eq!(state.public_temporary_safe_masks[0], 0);
        let history_len = state.discard_actor_history.len();

        state.apply_mjai_event(MjaiEvent::Dahai {
            actor: 1,
            pai: "not-a-tile".to_string(),
            tsumogiri: false,
        });

        assert!(!state.public_history_valid);
        assert_eq!(state.discard_actor_history.len(), history_len);
    }

    #[test]
    fn second_direct_mjai_discard_without_resolution_fails_closed() {
        let mut state = GameState::new(2, true, Some(705), 0, GameRule::default());
        state.players[0].hand = vec![0];
        state.players[1].hand = vec![4];

        state.apply_mjai_event(MjaiEvent::Dahai {
            actor: 0,
            pai: "1m".to_string(),
            tsumogiri: false,
        });
        assert!(state.public_history_valid);
        assert_eq!(state.resolved_discard_count, 0);

        state.apply_mjai_event(MjaiEvent::Dahai {
            actor: 1,
            pai: "2m".to_string(),
            tsumogiri: false,
        });

        assert!(!state.public_history_valid);
        assert_eq!(state.discard_actor_history, vec![0, 1]);
        assert_eq!(state.resolved_discard_count, 0);
    }

    #[test]
    fn unresolved_discard_blocks_direct_ankan_transitions_without_mutation() {
        let mut base = GameState::new(2, true, Some(707), 0, GameRule::default());
        base.players[0].hand = vec![108];
        base.players[1].hand = vec![32, 33, 34, 35];
        base.apply_mjai_event(MjaiEvent::Dahai {
            actor: 0,
            pai: "E".to_string(),
            tsumogiri: false,
        });
        assert_eq!(base.resolved_discard_count, 0);
        let hand_before = base.players[1].hand.clone();

        let mut mjai_state = base.clone();
        mjai_state.apply_mjai_event(MjaiEvent::Ankan {
            actor: 1,
            consumed: vec!["9m".to_string(); 4],
        });
        assert!(!mjai_state.public_history_valid);
        assert_eq!(mjai_state.players[1].hand, hand_before);
        assert!(mjai_state.players[1].melds.is_empty());

        let mut log_state = base;
        log_state.apply_log_action(&LogAction::AnGangAddGang {
            seat: 1,
            meld_type: MeldType::Ankan,
            tiles: vec![32, 33, 34, 35],
            tile_raw_id: 0,
            doras: None,
        });
        assert!(!log_state.public_history_valid);
        assert_eq!(log_state.players[1].hand, hand_before);
        assert!(log_state.players[1].melds.is_empty());
    }

    #[test]
    fn mixed_post_riichi_ankan_payload_fails_closed_without_mutation() {
        let mut state = GameState::new(2, true, Some(709), 0, GameRule::default());
        state.players[1].hand = vec![
            36, 37, 38, 39, // 1111p
            40, 44, 48, // 234p
            52, 56, 60, // 567p
            96, 100, 104, // 789s
            76,  // 2s
        ];
        state.players[1].riichi_declared = true;
        state.drawn_tile = Some(39);
        let hand_before = state.players[1].hand.clone();

        state.apply_mjai_event(MjaiEvent::Ankan {
            actor: 1,
            consumed: vec![
                "1p".to_string(),
                "2p".to_string(),
                "3p".to_string(),
                "2s".to_string(),
            ],
        });

        assert!(!state.public_history_valid);
        assert_eq!(state.players[1].hand, hand_before);
        assert!(state.players[1].melds.is_empty());
    }

    #[test]
    fn direct_draw_and_start_kyoku_clear_temporary_safety() {
        let mut state = GameState::new(2, true, Some(719), 0, GameRule::default());
        state.public_temporary_safe_masks[1] = 1 << 8;
        state.apply_mjai_event(MjaiEvent::Tsumo {
            actor: 1,
            pai: "1p".to_string(),
        });
        assert_eq!(state.public_temporary_safe_masks[1], 0);

        state.public_temporary_safe_masks[1] = 1 << 8;
        state.apply_mjai_event(MjaiEvent::StartKyoku {
            bakaze: "E".to_string(),
            kyoku: 1,
            honba: 0,
            kyoutaku: 0,
            oya: 0,
            scores: vec![25_000; 4],
            dora_marker: "1m".to_string(),
            tehais: vec![vec![]; 4],
        });
        assert_eq!(state.public_temporary_safe_masks, [0; 4]);
    }

    #[test]
    fn replay_kan_dora_snapshot_reaches_the_next_observation() {
        let mut state = GameState::new(2, true, Some(727), 0, GameRule::default());
        state.players[0].hand = vec![
            36, 37, 38, 39, // 1111p
            40, 44, 48, 52, 56, 60, 64, 68, 72, 76,
        ];
        state.wall.dora_indicators = vec![0];
        state.apply_log_action(&LogAction::AnGangAddGang {
            seat: 0,
            meld_type: MeldType::Ankan,
            tiles: vec![36, 37, 38, 39],
            tile_raw_id: 9,
            doras: Some(vec![0, 4]),
        });

        assert_eq!(state.wall.dora_indicators, vec![0, 4]);
        assert_eq!(state.get_observation(0).dora_indicators, vec![0, 4]);
    }

    #[test]
    fn cumulative_dora_snapshot_rejects_shrink_and_prefix_rewrite_without_mutation() {
        let mut state = GameState::new(2, true, Some(733), 0, GameRule::default());
        state.wall.dora_indicators = vec![0, 4];

        for malformed in [&[0][..], &[8, 4][..]] {
            state.public_history_valid = true;
            state.sync_log_dora_slice(malformed);
            assert!(!state.public_history_valid);
            assert_eq!(state.wall.dora_indicators, vec![0, 4]);
        }

        state.public_history_valid = true;
        state.sync_log_dora_slice(&[0, 4, 8]);
        assert!(state.public_history_valid);
        assert_eq!(state.wall.dora_indicators, vec![0, 4, 8]);
    }
}
