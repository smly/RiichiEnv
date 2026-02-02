use crate::action::Phase;
use crate::replay::{Action as LogAction, MjaiEvent};
use crate::state::GameState;
use crate::types::{Meld, MeldType, Wind};

pub trait GameStateEventHandler {
    fn apply_mjai_event(&mut self, event: MjaiEvent);
    fn apply_log_action(&mut self, action: &LogAction);
}

impl GameStateEventHandler for GameState {
    fn apply_mjai_event(&mut self, event: MjaiEvent) {
        match event {
            MjaiEvent::StartKyoku {
                bakaze,
                honba,
                kyoutaku,
                scores,
                dora_marker,
                tehais,
                oya,
                ..
            } => {
                // Initialize round state from event
                self.honba = honba;
                self.riichi_sticks = kyoutaku as u32;
                self.players.iter_mut().enumerate().for_each(|(i, p)| {
                    p.score = scores[i];
                });
                self.round_wind = match bakaze.as_str() {
                    "E" => Wind::East as u8,
                    "S" => Wind::South as u8,
                    "W" => Wind::West as u8,
                    "N" => Wind::North as u8,
                    _ => Wind::East as u8,
                };
                self.oya = oya;
                self.wall.dora_indicators =
                    vec![crate::replay::TileConverter::parse_tile_136(&dora_marker)];

                // Set hands
                for (i, hand_strs) in tehais.iter().enumerate() {
                    let mut hand = Vec::new();
                    for tile_str in hand_strs {
                        hand.push(crate::replay::TileConverter::parse_tile_136(tile_str));
                    }
                    hand.sort();
                    self.players[i].hand = hand;
                }

                // Clear other state
                for p in &mut self.players {
                    p.discards.clear();
                    p.melds.clear();
                    p.riichi_declared = false;
                    p.riichi_stage = false;
                }
                self.drawn_tile = None;
                self.current_player = self.oya; // Oya starts
                self.needs_tsumo = true;
                self.is_done = false;
            }
            MjaiEvent::Tsumo { actor, pai } => {
                let tile = crate::replay::TileConverter::parse_tile_136(&pai);
                self.current_player = actor as u8;
                self.drawn_tile = Some(tile);
                self.players[actor].hand.push(tile);
                self.players[actor].hand.sort();
                if !self.wall.tiles.is_empty() {
                    self.wall.tiles.pop();
                }
                self.needs_tsumo = false;
            }
            MjaiEvent::Dahai { actor, pai, .. } => {
                let tile = crate::replay::TileConverter::parse_tile_136(&pai);
                self.current_player = actor as u8;
                if let Some(idx) = self.players[actor].hand.iter().position(|&t| t == tile) {
                    self.players[actor].hand.remove(idx);
                }
                self.players[actor].discards.push(tile);
                self.last_discard = Some((actor as u8, tile));
                self.drawn_tile = None;

                if self.players[actor].riichi_stage {
                    self.players[actor].riichi_declared = true;
                }
                self.needs_tsumo = true;
            }
            MjaiEvent::Pon {
                actor,
                pai,
                consumed,
                ..
            } => {
                let tile = crate::replay::TileConverter::parse_tile_136(&pai);
                self.current_player = actor as u8;
                let c1 = crate::replay::TileConverter::parse_tile_136(&consumed[0]);
                let c2 = crate::replay::TileConverter::parse_tile_136(&consumed[1]);
                let form_tiles = vec![tile, c1, c2];

                for t in &[c1, c2] {
                    if let Some(idx) = self.players[actor].hand.iter().position(|&x| x == *t) {
                        self.players[actor].hand.remove(idx);
                    }
                }

                self.players[actor].melds.push(Meld {
                    meld_type: MeldType::Peng,
                    tiles: form_tiles,
                    opened: true,
                    from_who: -1,
                });
                self.drawn_tile = None;
                self.needs_tsumo = false;
            }
            MjaiEvent::Chi {
                actor,
                pai,
                consumed,
                ..
            } => {
                let tile = crate::replay::TileConverter::parse_tile_136(&pai);
                self.current_player = actor as u8;
                let c1 = crate::replay::TileConverter::parse_tile_136(&consumed[0]);
                let c2 = crate::replay::TileConverter::parse_tile_136(&consumed[1]);
                let form_tiles = vec![tile, c1, c2];

                for t in &[c1, c2] {
                    if let Some(idx) = self.players[actor].hand.iter().position(|&x| x == *t) {
                        self.players[actor].hand.remove(idx);
                    }
                }

                self.players[actor].melds.push(Meld {
                    meld_type: MeldType::Chi,
                    tiles: form_tiles,
                    opened: true,
                    from_who: -1,
                });
                self.drawn_tile = None;
                self.needs_tsumo = false;
            }
            MjaiEvent::Kan {
                actor,
                pai,
                consumed,
                ..
            } => {
                let tile = crate::replay::TileConverter::parse_tile_136(&pai);
                self.current_player = actor as u8;
                let mut tiles = vec![tile];
                for c in &consumed {
                    tiles.push(crate::replay::TileConverter::parse_tile_136(c));
                }

                for c in &consumed {
                    let tv = crate::replay::TileConverter::parse_tile_136(c);
                    if let Some(idx) = self.players[actor].hand.iter().position(|&x| x == tv) {
                        self.players[actor].hand.remove(idx);
                    }
                }

                self.players[actor].melds.push(Meld {
                    meld_type: MeldType::Gang,
                    tiles,
                    opened: true,
                    from_who: -1,
                });
                self.needs_tsumo = true;
            }
            MjaiEvent::Ankan { actor, consumed } => {
                let mut tiles = Vec::new();
                for c in &consumed {
                    let t = crate::replay::TileConverter::parse_tile_136(c);
                    tiles.push(t);
                    if let Some(idx) = self.players[actor].hand.iter().position(|&x| x == t) {
                        self.players[actor].hand.remove(idx);
                    }
                }
                self.players[actor].melds.push(Meld {
                    meld_type: MeldType::Angang,
                    tiles,
                    opened: false,
                    from_who: -1,
                });
                self.needs_tsumo = true;
            }
            MjaiEvent::Kakan { actor, pai } => {
                let tile = crate::replay::TileConverter::parse_tile_136(&pai);
                if let Some(idx) = self.players[actor].hand.iter().position(|&x| x == tile) {
                    self.players[actor].hand.remove(idx);
                }
                for m in self.players[actor].melds.iter_mut() {
                    if m.meld_type == MeldType::Peng && m.tiles[0] / 4 == tile / 4 {
                        m.meld_type = MeldType::Addgang;
                        m.tiles.push(tile);
                        break;
                    }
                }
                self.needs_tsumo = true;
            }
            MjaiEvent::Reach { actor } => {
                self.players[actor].riichi_stage = true;
            }
            MjaiEvent::ReachAccepted { actor } => {
                self.players[actor].riichi_declared = true;
                self.riichi_sticks += 1;
                self.players[actor].score -= 1000;
            }
            MjaiEvent::Dora { dora_marker } => {
                let tile = crate::replay::TileConverter::parse_tile_136(&dora_marker);
                self.wall.dora_indicators.push(tile);
            }
            MjaiEvent::Hora { .. } | MjaiEvent::Ryukyoku { .. } | MjaiEvent::EndKyoku => {
                self.is_done = true;
            }
            _ => {}
        }
    }

    fn apply_log_action(&mut self, action: &LogAction) {
        match action {
            LogAction::DiscardTile {
                seat,
                tile,
                is_liqi,
                is_wliqi,
                ..
            } => {
                let s = *seat;
                let t = *tile;
                let is_tsumogiri = if let Some(dt) = self.drawn_tile {
                    dt == t
                } else {
                    false
                };

                if let Some(idx) = self.players[s].hand.iter().position(|&x| x == t) {
                    self.players[s].hand.remove(idx);
                }
                self.players[s].hand.sort();
                self.players[s].discards.push(t);
                self.players[s].discard_from_hand.push(!is_tsumogiri);
                self.players[s]
                    .discard_is_riichi
                    .push(*is_liqi || *is_wliqi);
                self.last_discard = Some((s as u8, t));
                self.drawn_tile = None;

                self.players[s].riichi_declared = self.players[s].riichi_declared || *is_liqi;
                if *is_liqi {
                    self.players[s].riichi_declaration_index =
                        Some(self.players[s].discards.len() - 1);
                }
                self.current_player = (s as u8 + 1) % 4;
                self.phase = Phase::WaitAct;
                self.active_players = vec![self.current_player];
                self.needs_tsumo = true;
                self.is_first_turn = false;
                self.is_after_kan = false;
            }
            LogAction::DealTile { seat, tile, .. } => {
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
                }
            }
            LogAction::ChiPengGang {
                seat,
                meld_type,
                tiles,
                froms,
            } => {
                // Remove tiles from hand
                for (i, t) in tiles.iter().enumerate() {
                    if i < froms.len() && froms[i] == *seat {
                        if let Some(idx) = self.players[*seat].hand.iter().position(|&x| x == *t) {
                            self.players[*seat].hand.remove(idx);
                        }
                    }
                }
                self.players[*seat].hand.sort();

                let from_who = froms
                    .iter()
                    .find(|&&f| f != *seat)
                    .map(|&f| f as i8)
                    .unwrap_or(-1);
                self.players[*seat].melds.push(Meld {
                    meld_type: *meld_type,
                    tiles: tiles.clone(),
                    opened: true,
                    from_who,
                });
                self.current_player = *seat as u8;
                self.phase = Phase::WaitAct;
                self.active_players = vec![self.current_player];
                let is_gang = *meld_type == MeldType::Gang;
                self.needs_tsumo = is_gang;
                self.is_first_turn = false;
                self.is_after_kan = is_gang;
            }
            LogAction::AnGangAddGang {
                seat,
                meld_type,
                tiles,
                ..
            } => {
                if *meld_type == MeldType::Angang {
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
                    });
                } else {
                    let tile = tiles[0];
                    if let Some(idx) = self.players[*seat].hand.iter().position(|&x| x == tile) {
                        self.players[*seat].hand.remove(idx);
                    }
                    for m in self.players[*seat].melds.iter_mut() {
                        if m.meld_type == MeldType::Peng && m.tiles[0] / 4 == tile / 4 {
                            m.meld_type = MeldType::Addgang;
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
            }
            LogAction::Dora { dora_marker } => {
                self.wall.dora_indicators.push(*dora_marker);
            }
            _ => {}
        }
    }
}
