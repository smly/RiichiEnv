use crate::action::{Action, ActionType};
use crate::parser::tid_to_mjai;
use crate::types::{Conditions, Wind};
use serde_json::Value;

use super::GameState3P;

impl GameState3P {
    pub fn handle_kita(&mut self, pid: u8, act: &Action) {
        let p_idx = pid as usize;

        if self.resolved_discard_count != self.discard_actor_history.len()
            || !self.current_claims.is_empty()
        {
            self.public_history_valid = false;
            return;
        }

        // Resolve the actual North tile before mutating anything. After riichi,
        // only the just-drawn physical North is legal; accepting an older held
        // North can change the fixed wait and make prior pass evidence unsound.
        let requested_tile = act.tile.or_else(|| act.consume_tiles.first().copied());
        let tile = if self.players[p_idx].riichi_declared {
            let Some(drawn_north) = self.drawn_tile.filter(|tile| tile / 4 == 30) else {
                self.public_history_valid = false;
                return;
            };
            if requested_tile.is_some_and(|requested| requested != drawn_north) {
                self.public_history_valid = false;
                return;
            }
            drawn_north
        } else if let Some(requested) = requested_tile.filter(|tile| tile / 4 == 30) {
            requested
        } else if let Some(held_north) = self.players[p_idx]
            .hand
            .iter()
            .find(|&&tile| tile / 4 == 30)
            .copied()
        {
            held_north
        } else {
            self.public_history_valid = false;
            return;
        };

        let Some(index) = self.players[p_idx]
            .hand
            .iter()
            .position(|&held| held == tile)
        else {
            self.public_history_valid = false;
            return;
        };
        self.clear_public_temporary_safety(pid);
        self.players[p_idx].missed_agari_doujun = false;
        self.players[p_idx].hand.remove(index);

        // Add to kita_tiles
        self.players[p_idx].kita_tiles.push(tile);

        // Kita declaration breaks first-turn status (invalidates Tenhou/Chiihou)
        self.is_first_turn = false;

        // NOTE: Don't break ippatsu here (before chankan ron check).
        // MjSoul awards ippatsu for ron on kita tiles, so the check below
        // must use the pre-kita ippatsu state.  Ippatsu is broken later:
        //  - in the "no ron" branch below (before rinshan draw), or
        //  - when "all pass" resolves pending_kan (mod.rs).

        // Log kita event
        if !self.skip_mjai_logging {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("kita".to_string()));
            ev.insert("actor".to_string(), Value::Number(pid.into()));
            ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
            self._push_mjai_event(Value::Object(ev));
        }

        // Reveal any pending kan dora (e.g. from a prior kakan/daiminkan) before
        // checking for ron on kita.  Kita acts like a discard for dora timing
        // purposes, so the pending dora must be active when evaluating and
        // scoring the ron.  Without this, a kakan→kita→ron sequence would
        // miss the kakan dora in the scoring.
        while self.wall.pending_kan_dora_count > 0 {
            self.wall.pending_kan_dora_count -= 1;
            self._reveal_kan_dora();
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
            let calc = crate::hand_evaluator_3p::HandEvaluator3P::new(hand.clone(), melds.clone());
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
                chankan: false, // Kita does not award chankan yaku
                player_wind: Wind::from(p_wind),
                round_wind: Wind::from(self.round_wind),
                riichi_sticks: self.riichi_sticks,
                honba: self.honba as u32,
                is_sanma: true,
                num_players: np,
                kita_count: self.players[i as usize].kita_tiles.len() as u8,
                ..Default::default()
            };

            let res = calc.calc(tile, self.wall.dora_indicators.clone(), vec![], Some(cond));

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
            // No ron - break ippatsu for all players, then draw from rinshan
            for p in &mut self.players {
                p.ippatsu_cycle = false;
            }
            self.resolve_kita_rinshan(pid);
        }
    }

    pub fn get_kita_legal_actions(&self, pid: u8) -> Vec<Action> {
        // Must have a drawn tile (it's player's turn to act)
        if self.drawn_tile.is_none() {
            return Vec::new();
        }

        // Must have tiles left in wall (enough for rinshan draw)
        if self.wall.drawable_count == 0 {
            return Vec::new();
        }

        let p_idx = pid as usize;
        if self.players[p_idx].riichi_stage {
            // A separately signalled reach must be completed by its tenpai-
            // preserving declaration discard; Kita is not an alternative.
            return Vec::new();
        }
        let mut actions = Vec::new();

        // Find North tiles (type 30, IDs 120-123) in hand
        for &tile in &self.players[p_idx].hand {
            if tile / 4 == 30 {
                // After riichi, only a just-drawn North may be set aside. This
                // returns the concealed hand to the exact fixed-wait state;
                // extracting a North that was already held could change waits
                // and invalidate permanent post-riichi safety evidence.
                if self.players[p_idx].riichi_declared && self.drawn_tile != Some(tile) {
                    continue;
                }
                // North wind
                actions.push(Action::new(ActionType::Kita, Some(tile), vec![], Some(pid)));
            }
        }

        actions
    }

    pub fn resolve_kita_rinshan(&mut self, pid: u8) {
        let p_idx = pid as usize;

        if self.wall.drawable_count > 0 {
            // Reveal any pending kan dora (e.g. from a prior kakan/daiminkan)
            while self.wall.pending_kan_dora_count > 0 {
                self.wall.pending_kan_dora_count -= 1;
                self._reveal_kan_dora();
            }

            // Draw from rinshan (dead wall front via remove(0)).
            let t = self
                .wall
                .draw_rinshan_tile()
                .expect("rinshan draw failed despite drawable_count > 0");
            self.players[p_idx].hand.push(t);
            self.drawn_tile = Some(t);
            self.wall.rinshan_draw_count += 1;
            self.is_rinshan_flag = true;

            // NO new dora indicator for kita (confirmed Tenhou rule)

            if !self.skip_mjai_logging {
                let mut t_ev = serde_json::Map::new();
                t_ev.insert("type".to_string(), Value::String("tsumo".to_string()));
                t_ev.insert("actor".to_string(), Value::Number(pid.into()));
                t_ev.insert("pai".to_string(), Value::String(tid_to_mjai(t)));
                self._push_mjai_event(Value::Object(t_ev));
            }

            self.phase = crate::action::Phase::WaitAct;
            self.active_players = vec![pid];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::GameRule;

    #[test]
    fn post_riichi_kita_only_allows_the_drawn_north() {
        let mut state = GameState3P::new(5, true, Some(7), 0, GameRule::default());
        state.players[0].hand = vec![120, 121];
        state.drawn_tile = Some(121);
        state.players[0].riichi_declared = true;
        state.wall.drawable_count = 1;

        let actions = state.get_kita_legal_actions(0);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].tile, Some(121));

        state.players[0].riichi_stage = true;
        assert!(state.get_kita_legal_actions(0).is_empty());
    }

    #[test]
    fn direct_post_riichi_kita_rejects_a_held_north_without_mutation() {
        let mut state = GameState3P::new(5, true, Some(11), 0, GameRule::default());
        state.players[0].hand = vec![120, 121];
        state.drawn_tile = Some(121);
        state.players[0].riichi_declared = true;
        state.public_temporary_safe_masks[0] = 1 << 19;
        let hand_before = state.players[0].hand.clone();

        state.handle_kita(
            0,
            &Action::new(ActionType::Kita, Some(120), vec![], Some(0)),
        );

        assert!(!state.public_history_valid);
        assert_eq!(state.players[0].hand, hand_before);
        assert!(state.players[0].kita_tiles.is_empty());
        assert_eq!(state.public_temporary_safe_masks[0], 1 << 19);
    }

    #[test]
    fn direct_kita_rejects_an_unresolved_discard_without_mutation() {
        let mut state = GameState3P::new(5, true, Some(12), 0, GameRule::default());
        state.players[0].hand = vec![120];
        state.discard_actor_history.push(1);
        let hand_before = state.players[0].hand.clone();

        state.handle_kita(
            0,
            &Action::new(ActionType::Kita, Some(120), vec![], Some(0)),
        );

        assert!(!state.public_history_valid);
        assert_eq!(state.players[0].hand, hand_before);
        assert!(state.players[0].kita_tiles.is_empty());
    }

    #[test]
    fn generic_post_riichi_kita_removes_drawn_north_and_clears_temporary_safety() {
        let mut state = GameState3P::new(5, true, Some(13), 0, GameRule::default());
        state.players[0].hand = vec![120, 121];
        state.drawn_tile = Some(121);
        state.players[0].riichi_declared = true;
        state.public_temporary_safe_masks[0] = 1 << 19;

        state.handle_kita(0, &Action::new(ActionType::Kita, None, vec![], Some(0)));

        assert!(state.public_history_valid);
        assert_eq!(state.players[0].kita_tiles, vec![121]);
        assert!(state.players[0].hand.contains(&120));
        assert_eq!(state.public_temporary_safe_masks[0], 0);
    }
}
