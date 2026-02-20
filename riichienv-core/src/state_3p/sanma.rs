use crate::action::{Action, ActionType};
use crate::parser::tid_to_mjai;
use crate::types::{Conditions, Wind};
use serde_json::Value;

use super::GameState3P;

impl GameState3P {
    pub fn handle_kita(&mut self, pid: u8, act: &Action) {
        let tile = act.tile.unwrap_or_else(|| act.consume_tiles.first().copied().unwrap_or(0));
        let p_idx = pid as usize;

        // Remove North tile from hand
        if let Some(idx) = self.players[p_idx].hand.iter().position(|&t| t == tile) {
            self.players[p_idx].hand.remove(idx);
        }

        // Add to kita_tiles
        self.players[p_idx].kita_tiles.push(tile);

        // Log kita event
        if !self.skip_mjai_logging {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("kita".to_string()));
            ev.insert("actor".to_string(), Value::Number(pid.into()));
            ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
            self._push_mjai_event(Value::Object(ev));
        }

        // Check other players for chankan-style ron on kita
        let np: u8 = 3;
        let mut chankan_ronners = Vec::new();
        for i in 0..np {
            if i == pid {
                continue;
            }
            let hand = &self.players[i as usize].hand;
            let melds = &self.players[i as usize].melds;

            // Furiten check
            let calc = crate::hand_evaluator::HandEvaluator::new(hand.clone(), melds.clone());
            let waits = calc.get_waits_u8();
            let mut is_furiten = false;
            for &w in &waits {
                if self.players[i as usize]
                    .discards
                    .iter()
                    .any(|&d| d / 4 == w)
                {
                    is_furiten = true;
                    break;
                }
            }
            if self.players[i as usize].missed_agari_riichi
                || self.players[i as usize].missed_agari_doujun
            {
                is_furiten = true;
            }

            if is_furiten {
                continue;
            }

            let p_wind = (i + np - self.oya) % np;
            let cond = Conditions {
                tsumo: false,
                riichi: self.players[i as usize].riichi_declared,
                double_riichi: self.players[i as usize].double_riichi_declared,
                ippatsu: self.players[i as usize].ippatsu_cycle,
                chankan: true,
                player_wind: Wind::from(p_wind),
                round_wind: Wind::from(self.round_wind),
                riichi_sticks: self.riichi_sticks,
                honba: self.honba as u32,
                is_sanma: true,
                num_players: np,
                ..Default::default()
            };

            let res = calc.calc(
                tile,
                self.wall.dora_indicators.clone(),
                vec![],
                Some(cond),
            );

            if res.is_win && (res.yakuman || res.han >= 1) {
                chankan_ronners.push(i);
                self.current_claims.entry(i).or_default().push(Action::new(
                    ActionType::Ron,
                    Some(tile),
                    vec![],
                    Some(i),
                ));
            }
        }

        if !chankan_ronners.is_empty() {
            // Offer ron to opponents (chankan-style)
            self.phase = crate::action::Phase::WaitResponse;
            self.active_players = chankan_ronners;
            self.last_discard = Some((pid, tile));
            // Store kita as pending kan for resolution
            self.pending_kan = Some((pid, act.clone()));
        } else {
            // No ron - draw from rinshan
            self.resolve_kita_rinshan(pid);
        }
    }

    pub fn get_kita_legal_actions(&self, pid: u8) -> Vec<Action> {
        // Must have a drawn tile (it's player's turn to act)
        if self.drawn_tile.is_none() {
            return Vec::new();
        }

        // Must have tiles left in wall (enough for rinshan draw)
        if self.wall.tiles.len() <= 14 {
            return Vec::new();
        }

        let p_idx = pid as usize;
        let mut actions = Vec::new();

        // Find North tiles (type 30, IDs 120-123) in hand
        for &tile in &self.players[p_idx].hand {
            if tile / 4 == 30 {
                // North wind
                actions.push(Action::new(ActionType::Kita, Some(tile), vec![], Some(pid)));
            }
        }

        actions
    }

    pub fn resolve_kita_rinshan(&mut self, pid: u8) {
        let p_idx = pid as usize;

        if self.wall.tiles.len() > 14 {
            // Draw from rinshan (front of wall vector)
            let t = self.wall.tiles.remove(0);
            self.players[p_idx].hand.push(t);
            self.drawn_tile = Some(t);
            self.wall.rinshan_draw_count += 1;
            self.is_rinshan_flag = true;

            // NO new dora indicator for kita (confirmed Tenhou rule)

            if !self.skip_mjai_logging {
                let mut t_ev = serde_json::Map::new();
                t_ev.insert("type".to_string(), Value::String("tsumo".to_string()));
                t_ev.insert("actor".to_string(), Value::Number(pid.into()));
                t_ev.insert(
                    "pai".to_string(),
                    Value::String(tid_to_mjai(t)),
                );
                self._push_mjai_event(Value::Object(t_ev));
            }

            self.phase = crate::action::Phase::WaitAct;
            self.active_players = vec![pid];
        }
    }
}
