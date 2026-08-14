use crate::action::Phase;
use crate::hand_evaluator_3p::HandEvaluator3P;
use crate::parser::mjai_to_tid;
use crate::replay::{Action as LogAction, MjaiEvent};
use crate::state_3p::GameState3P;
use crate::state_3p::legal_actions::GameState3PLegalActions;
use crate::types::{INITIAL_HAND_SIZE, Meld, MeldType, TILES_3P, Wind};

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

pub trait GameState3PEventHandler {
    fn apply_mjai_event(&mut self, event: MjaiEvent);
    fn apply_log_action(&mut self, action: &LogAction);
}

impl GameState3PEventHandler for GameState3P {
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
                self.riichi_sutehais = [None; 3];
                self.last_tedashis = [None; 3];
                self.discard_actor_history.clear();
                self.resolved_discard_count = 0;
                self.public_temporary_safe_masks = [0; 3];
                self.public_history_valid = true;
                self.wall.tiles = vec![0; TILES_3P - INITIAL_HAND_SIZE * 3];
                self.wall.dora_indicators = vec![public_tile!(&dora_marker)];
                self.wall.rinshan_draw_count = 0;
                self.wall.pending_kan_dora_count = 0;
                // Match _initialize_round: 14 = dead wall (rinshan + dora stacks).
                // Without this reset, drawable_count carries over from prior rounds
                // and can fall below riichi/kan legality thresholds on replay.
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

                // Populate current_claims for reaction phase (pon/ron).
                let claim_active = self.record_public_discard_offers(actor as u8, tile);
                self.active_players.clear();
                if !claim_active.is_empty() {
                    self.phase = Phase::WaitResponse;
                    self.active_players = claim_active;
                } else {
                    // No reactions possible; nobody should act until next tsumo
                    self.phase = Phase::WaitAct;
                    self.active_players.clear();
                    self.current_player = u8::MAX; // sentinel: no active player
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
                // Kuikae (swap-calling) forbidden: cannot discard the
                // same tile type that was just pon'd.
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
                // Chi shouldn't happen in 3P, but handle gracefully
                self.clear_public_temporary_safety(actor as u8);
                self.players[actor].missed_agari_doujun = false;
                self.public_history_valid = false;
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
                self.current_player = actor as u8;
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
            MjaiEvent::Kita { actor } => {
                if self.resolved_discard_count != self.discard_actor_history.len()
                    || !self.current_claims.is_empty()
                {
                    self.public_history_valid = false;
                    return;
                }
                let north_id = 30;
                let mut kita_tile: Option<u8> = None;
                let wait_is_fixed = self.players[actor].riichi_declared;
                let drawn_north = self
                    .drawn_tile
                    .filter(|tile| *tile / 4 == north_id)
                    .and_then(|tile| {
                        self.players[actor]
                            .hand
                            .iter()
                            .position(|&held| held == tile)
                    });
                if wait_is_fixed && drawn_north.is_none() {
                    // A post-riichi Kita that removes an older concealed North
                    // can change the wait.  Keep replay compatibility, but make
                    // the public-history proof sidecar fail closed.
                    self.public_history_valid = false;
                }
                let kita_index = if wait_is_fixed {
                    drawn_north
                } else {
                    self.players[actor]
                        .hand
                        .iter()
                        .position(|&tile| tile / 4 == north_id)
                };
                if let Some(idx) = kita_index {
                    self.clear_public_temporary_safety(actor as u8);
                    self.players[actor].missed_agari_doujun = false;
                    let tile = self.players[actor].hand.remove(idx);
                    self.players[actor].kita_tiles.push(tile);
                    kita_tile = Some(tile);
                }
                // Kita can be ronned; populate current_claims for reaction phase
                let np = self.players.len() as u8;
                self.current_player = actor as u8;
                self.current_claims.clear();
                self.active_players.clear();
                if let Some(tile) = kita_tile {
                    let mut claim_active = Vec::new();
                    for i in 0..np {
                        if i == actor as u8 {
                            continue;
                        }
                        // For kita, only ron is possible (no pon/kan)
                        let (legals, _missed) =
                            self._get_claim_actions_for_player(i, actor as u8, tile);
                        let ron_only: Vec<_> = legals
                            .into_iter()
                            .filter(|a| a.action_type == crate::action::ActionType::Ron)
                            .collect();
                        if !ron_only.is_empty() {
                            claim_active.push(i);
                            self.current_claims.insert(i, ron_only);
                        }
                    }
                    if !claim_active.is_empty() {
                        self.phase = Phase::WaitResponse;
                        self.active_players = claim_active;
                    } else {
                        self.phase = Phase::WaitAct;
                        self.active_players.clear();
                        self.current_player = u8::MAX;
                    }
                } else {
                    self.phase = Phase::WaitAct;
                    self.active_players.clear();
                    self.current_player = u8::MAX;
                }
                self.needs_tsumo = true;
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

impl GameState3P {
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
        let np: u8 = 3;
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
                if self.resolved_discard_count != self.discard_actor_history.len()
                    || !self.current_claims.is_empty()
                {
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

                self.players[s].riichi_declared =
                    self.players[s].riichi_declared || *is_liqi || *is_wliqi;
                if *is_wliqi {
                    self.players[s].double_riichi_declared = true;
                }
                if *is_liqi || *is_wliqi {
                    self.players[s].riichi_declaration_index =
                        Some(self.players[s].discards.len() - 1);
                    self.riichi_pending_acceptance = Some(s as u8);
                }
                // Track nagashi eligibility: discard must be terminal/honor
                self.players[s].nagashi_eligible &= crate::types::is_terminal_tile(t);
                let claim_active = self.record_public_discard_offers(s as u8, t);
                self.current_player = (s as u8 + 1) % np;
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
                // Accept pending riichi deposit (discard was not ronned)
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
                // Accept pending riichi deposit (discard was not ronned)
                if let Some(rp) = self.riichi_pending_acceptance.take() {
                    self.players[rp as usize].score -= 1000;
                    self.riichi_sticks += 1;
                }
                // Discard was called → discarder loses nagashi eligibility
                if let Some((discarder_pid, _)) = self.last_discard {
                    self.players[discarder_pid as usize].nagashi_eligible = false;
                }
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
                }
                self.players[*seat].hand.sort();
                self.current_player = *seat as u8;
                self.phase = Phase::WaitAct;
                self.active_players = vec![self.current_player];
                self.needs_tsumo = true;
                self.is_first_turn = false;
                self.is_after_kan = true;
                // Record as last_discard so chankan ron (Hule) can identify
                // the kan declarer as the payer (e.g. kokushi chankan on ankan).
                self.last_discard = Some((*seat as u8, tiles[0]));
            }
            LogAction::Dora { dora_marker } => {
                self.wall.dora_indicators.push(*dora_marker);
            }
            LogAction::BaBei { seat, moqie } => {
                if self.resolved_discard_count != self.discard_actor_history.len()
                    || !self.current_claims.is_empty()
                {
                    self.public_history_valid = false;
                    return;
                }
                if let Some(snapshot) = dora_snapshot {
                    self.sync_log_dora_slice(snapshot);
                }
                self.clear_public_temporary_safety(*seat as u8);
                self.players[*seat].missed_agari_doujun = false;
                // Accept pending riichi deposit (discard was not ronned)
                if let Some(rp) = self.riichi_pending_acceptance.take() {
                    self.players[rp as usize].score -= 1000;
                    self.riichi_sticks += 1;
                }
                // Remove a North tile from hand and add to kita_tiles
                let north_34: u8 = 30; // 4z = North
                let wait_is_fixed = self.players[*seat].riichi_declared;
                let drawn_north = self
                    .drawn_tile
                    .filter(|tile| *tile / 4 == north_34)
                    .and_then(|tile| {
                        self.players[*seat]
                            .hand
                            .iter()
                            .position(|&held| held == tile)
                    });
                if wait_is_fixed && (!*moqie || drawn_north.is_none()) {
                    self.public_history_valid = false;
                }
                let kita_index = if wait_is_fixed {
                    drawn_north
                } else {
                    self.players[*seat]
                        .hand
                        .iter()
                        .position(|&tile| tile / 4 == north_34)
                };
                if let Some(idx) = kita_index {
                    let tile = self.players[*seat].hand.remove(idx);
                    self.players[*seat].kita_tiles.push(tile);
                    // Record as last_discard so ron-on-kita (Hule) can identify
                    // the kita declarer as the payer.
                    self.last_discard = Some((*seat as u8, tile));
                }
                self.players[*seat].hand.sort();
                self.current_player = *seat as u8;
                self.phase = Phase::WaitAct;
                self.active_players = vec![self.current_player];
                self.needs_tsumo = true;
                self.is_first_turn = false;
                self.is_after_kan = true; // Rinshan draw follows Kita, like after Kan
            }
            LogAction::Hule { hules } => {
                // If a riichi deposit is pending and this is a ron, the deposit
                // is voided (MjSoul does not deduct it when the discard is ronned).
                // Also detect chankan-on-Kita: mjsoul marks it as zimo=true,
                // but the winner is not the current player.
                let first_is_ron = hules
                    .first()
                    .is_some_and(|h| !h.zimo || h.seat as u8 != self.current_player);
                if first_is_ron {
                    self.riichi_pending_acceptance = None;
                }

                let honba = self.honba;
                let riichi_on_table = self.riichi_sticks;
                let mut honba_taken = false;

                for h in hules {
                    let winner = h.seat;
                    // Detect chankan-on-Kita: zimo=true but winner != current player
                    let is_tsumo = h.zimo && h.seat as u8 == self.current_player;

                    if is_tsumo {
                        let is_oya = (winner as u8) == self.oya;

                        // Check PAO (sekinin barai) for yakuman tsumo
                        let mut pao_payer = None;
                        let mut pao_yakuman_val: i32 = 0;
                        let mut total_yakuman_val: i32 = 0;

                        if h.yiman {
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
                            // PAO tsumo: pao payer pays the full tsumo total
                            // for the liable portion.
                            let tsumo_total: i32 = if is_oya {
                                // Oya: each ko pays xian, total = xian * (np-1)
                                h.point_zimo_xian as i32 * (np as i32 - 1)
                            } else {
                                // Ko: oya pays qin, each other ko pays xian
                                h.point_zimo_qin as i32 + h.point_zimo_xian as i32 * (np as i32 - 2)
                            };
                            let pao_amt = if total_yakuman_val > 0 {
                                tsumo_total * pao_yakuman_val / total_yakuman_val
                            } else {
                                tsumo_total
                            };
                            let non_pao_amt = tsumo_total - pao_amt;

                            if let Some(pp) = pao_payer {
                                self.players[pp as usize].score -= pao_amt;
                                self.players[winner].score += pao_amt;
                            }

                            // Non-PAO part split normally
                            if non_pao_amt > 0 {
                                for i in 0..np as usize {
                                    if i != winner {
                                        let share = if is_oya {
                                            non_pao_amt / (np as i32 - 1)
                                        } else if (i as u8) == self.oya {
                                            // Approximate: qin share
                                            h.point_zimo_qin as i32 * non_pao_amt / tsumo_total
                                        } else {
                                            h.point_zimo_xian as i32 * non_pao_amt / tsumo_total
                                        };
                                        self.players[i].score -= share;
                                        self.players[winner].score += share;
                                    }
                                }
                            }

                            // Honba paid by pao payer
                            if let Some(pp) = pao_payer {
                                let honba_total = honba as i32 * (np as i32 - 1) * 100;
                                self.players[pp as usize].score -= honba_total;
                                self.players[winner].score += honba_total;
                            }
                        } else {
                            // Standard tsumo distribution (no pao)
                            for i in 0..np as usize {
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
                        // Ron
                        let ron_honba = if !honba_taken {
                            honba_taken = true;
                            honba
                        } else {
                            0
                        };

                        // Check PAO for ron
                        let mut pao_payer = None;
                        let mut pao_yakuman_val: i32 = 0;
                        let mut total_yakuman_val: i32 = 0;

                        if h.yiman {
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
                            let pp = pao_payer.unwrap_or(discarder);
                            let ron_total = h.point_rong as i32;
                            let pao_amt = ron_total * pao_yakuman_val / total_yakuman_val;
                            let honba_pts = ron_honba as i32 * (np as i32 - 1) * 100;

                            // PAO ron: split between pao payer and discarder
                            let pao_share = pao_amt / 2 + honba_pts;
                            let discarder_share = ron_total - pao_amt / 2;

                            self.players[pp as usize].score -= pao_share;
                            self.players[discarder as usize].score -= discarder_share;
                            self.players[winner].score += pao_share + discarder_share;
                        } else {
                            // Standard ron
                            let pay =
                                h.point_rong as i32 + ron_honba as i32 * (np as i32 - 1) * 100;
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
                        let score_res = crate::score::calculate_score(5, 30, is_oya, true, 0, np);
                        if is_oya {
                            for i in 0..np as usize {
                                if i as u8 != w {
                                    self.players[i].score -= score_res.pay_tsumo_ko as i32;
                                    self.players[w as usize].score += score_res.pay_tsumo_ko as i32;
                                }
                            }
                        } else {
                            for i in 0..np as usize {
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
                    // Regular tenpai/noten payments (pool = 2000 in 3P)
                    let mut tenpai = [false; 3];
                    for (i, p) in self.players.iter().enumerate() {
                        if i < 3 {
                            let calc = HandEvaluator3P::new(p.hand.clone(), p.melds.clone());
                            tenpai[i] = calc.is_tenpai();
                        }
                    }
                    let num_tp = tenpai.iter().filter(|&&t| t).count();
                    if num_tp > 0 && num_tp < 3 {
                        let pk = 2000 / num_tp as i32;
                        let pn = 2000 / (3 - num_tp) as i32;
                        for (i, tp) in tenpai.iter().enumerate() {
                            let delta = if *tp { pk } else { -pn };
                            self.players[i].score += delta;
                        }
                    }
                }
                self.is_done = true;
            }
            LogAction::LiuJu { .. } => {
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
        let invalid = doras.is_empty()
            || doras.len() > 5
            || doras.iter().any(|&tile| {
                let tile_type = tile / 4;
                tile >= 136 || (1..=7).contains(&tile_type)
            });
        if invalid {
            self.public_history_valid = false;
            return;
        }
        if doras.len() < self.wall.dora_indicators.len()
            || !doras.starts_with(&self.wall.dora_indicators)
        {
            // Provider snapshots are cumulative; never let a malformed
            // update erase or rewrite an indicator already visible publicly.
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
    fn direct_log_discard_3p_records_ron_offer_until_resolution() {
        let mut state = GameState3P::new(5, true, Some(811), 0, GameRule::default());
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
    fn mjai_discard_3p_falls_back_by_tile_type_and_rejects_invalid_text() {
        let mut state = GameState3P::new(5, true, Some(813), 0, GameRule::default());
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
    fn second_direct_mjai_discard_3p_without_resolution_fails_closed() {
        let mut state = GameState3P::new(5, true, Some(815), 0, GameRule::default());
        state.players[0].hand = vec![0];
        state.players[1].hand = vec![36];

        state.apply_mjai_event(MjaiEvent::Dahai {
            actor: 0,
            pai: "1m".to_string(),
            tsumogiri: false,
        });
        assert!(state.public_history_valid);
        assert_eq!(state.resolved_discard_count, 0);

        state.apply_mjai_event(MjaiEvent::Dahai {
            actor: 1,
            pai: "1p".to_string(),
            tsumogiri: false,
        });

        assert!(!state.public_history_valid);
        assert_eq!(state.discard_actor_history, vec![0, 1]);
        assert_eq!(state.resolved_discard_count, 0);
    }

    #[test]
    fn unresolved_discard_blocks_direct_3p_ankan_and_kita_without_mutation() {
        let mut base = GameState3P::new(5, true, Some(816), 0, GameRule::default());
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

        let mut log_state = base.clone();
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

        let mut kita_state = base;
        kita_state.players[1].hand = vec![120];
        kita_state.drawn_tile = Some(120);
        kita_state.apply_log_action(&LogAction::BaBei {
            seat: 1,
            moqie: true,
        });
        assert!(!kita_state.public_history_valid);
        assert_eq!(kita_state.players[1].hand, vec![120]);
        assert!(kita_state.players[1].kita_tiles.is_empty());
    }

    #[test]
    fn mixed_post_riichi_ankan_payload_3p_fails_closed_without_mutation() {
        let mut state = GameState3P::new(5, true, Some(817), 0, GameRule::default());
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
    fn direct_draw_and_start_kyoku_3p_clear_temporary_safety() {
        let mut state = GameState3P::new(5, true, Some(821), 0, GameRule::default());
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
            scores: vec![35_000; 3],
            dora_marker: "1p".to_string(),
            tehais: vec![vec![]; 3],
        });
        assert_eq!(state.public_temporary_safe_masks, [0; 3]);
    }

    #[test]
    fn replay_3p_kan_dora_snapshot_reaches_the_next_observation() {
        let mut state = GameState3P::new(5, true, Some(829), 0, GameRule::default());
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
            doras: Some(vec![0, 36]),
        });

        assert_eq!(state.wall.dora_indicators, vec![0, 36]);
        assert_eq!(state.get_observation(0).dora_indicators, vec![0, 36]);
    }

    #[test]
    fn cumulative_3p_dora_snapshot_rejects_shrink_and_prefix_rewrite_without_mutation() {
        let mut state = GameState3P::new(5, true, Some(839), 0, GameRule::default());
        state.wall.dora_indicators = vec![0, 36];

        for malformed in [&[0][..], &[72, 36][..]] {
            state.public_history_valid = true;
            state.sync_log_dora_slice(malformed);
            assert!(!state.public_history_valid);
            assert_eq!(state.wall.dora_indicators, vec![0, 36]);
        }

        state.public_history_valid = true;
        state.sync_log_dora_slice(&[0, 36, 72]);
        assert!(state.public_history_valid);
        assert_eq!(state.wall.dora_indicators, vec![0, 36, 72]);
    }
}
