use crate::action::{Action, ActionType, Phase};
use crate::state_3p::GameState3P;
use crate::types::{Conditions, Meld, MeldType, Wind, is_terminal_tile};

pub trait GameState3PLegalActions {
    fn _get_legal_actions_internal(&self, pid: u8) -> Vec<Action>;
    fn _get_claim_actions_for_player(&self, i: u8, pid: u8, tile: u8) -> (Vec<Action>, bool);
}

impl GameState3PLegalActions for GameState3P {
    fn _get_legal_actions_internal(&self, pid: u8) -> Vec<Action> {
        let mut legals = Vec::new();
        let pid_us = pid as usize;
        let mut hand = self.players[pid_us].hand.clone();
        hand.sort();
        let np: u8 = 3;

        if self.is_done {
            return legals;
        }

        if self.phase == Phase::WaitAct {
            if pid != self.current_player {
                return legals;
            }

            // 1. Tsumo
            if let Some(tile) = self.drawn_tile
                && !self.players[pid_us].riichi_stage
            {
                let cond = Conditions {
                    tsumo: true,
                    riichi: self.players[pid_us].riichi_declared,
                    double_riichi: self.players[pid_us].double_riichi_declared,
                    ippatsu: self.players[pid_us].ippatsu_cycle,
                    player_wind: Wind::from((pid + np - self.oya) % np),
                    round_wind: Wind::from(self.round_wind),
                    chankan: false,
                    haitei: self.wall.drawable_count == 0 && !self.is_rinshan_flag,
                    houtei: false,
                    rinshan: self.is_rinshan_flag,
                    tsumo_first_turn: self.is_first_turn
                        && self.players[pid_us].discards.is_empty(),
                    riichi_sticks: self.riichi_sticks,
                    honba: self.honba as u32,
                    is_sanma: true,
                    num_players: np,
                    ..Default::default()
                };
                let mut hand = self.players[pid_us].hand.clone();
                if let Some(idx) = hand.iter().rposition(|&t| t == tile) {
                    hand.remove(idx);
                }
                let calc = crate::hand_evaluator_3p::HandEvaluator3P::new(
                    hand,
                    self.players[pid_us].melds.clone(),
                );
                let res = calc.calc(tile, self.wall.dora_indicators.clone(), vec![], Some(cond));
                if res.is_win && (res.yakuman || res.han >= 1) {
                    legals.push(Action::new(
                        ActionType::Tsumo,
                        Some(tile),
                        vec![],
                        Some(pid),
                    ));
                }
            }

            // 2. Discard / Riichi
            if self.players[pid_us].riichi_declared {
                // Post-riichi: only tsumogiri is allowed.
                if let Some(dt) = self.drawn_tile {
                    legals.push(Action::new(
                        ActionType::Discard,
                        Some(dt),
                        vec![],
                        Some(pid),
                    ));
                }
            } else if self.players[pid_us].riichi_stage {
                // Riichi declared this turn but not yet committed (the Riichi
                // action came in without a tile, e.g. from an mjai `reach`
                // event). The follow-up discard must keep the hand tenpai.
                for &t in self.players[pid_us].hand.iter() {
                    let is_forbidden = self.players[pid_us]
                        .forbidden_discards
                        .iter()
                        .any(|&f| f / 4 == t / 4);
                    if is_forbidden {
                        continue;
                    }
                    let mut temp_hand = self.players[pid_us].hand.clone();
                    if let Some(idx) = temp_hand.iter().position(|&x| x == t) {
                        temp_hand.remove(idx);
                    }
                    let calc = crate::hand_evaluator_3p::HandEvaluator3P::new(
                        temp_hand,
                        self.players[pid_us].melds.clone(),
                    );
                    if calc.is_tenpai() {
                        legals.push(Action::new(ActionType::Discard, Some(t), vec![], Some(pid)));
                    }
                }
            } else {
                for &t in self.players[pid_us].hand.iter() {
                    let is_forbidden = self.players[pid_us]
                        .forbidden_discards
                        .iter()
                        .any(|&f| f / 4 == t / 4);
                    if !is_forbidden {
                        legals.push(Action::new(ActionType::Discard, Some(t), vec![], Some(pid)));
                    }
                }

                if self.players[pid_us].score >= 1000
                    && self.wall.drawable_count > 0
                    && self.players[pid_us].melds.iter().all(|m| !m.opened)
                {
                    let mut can_riichi = false;
                    for skip_idx in 0..self.players[pid_us].hand.len() {
                        let mut temp_hand = self.players[pid_us].hand.clone();
                        temp_hand.remove(skip_idx);
                        let calc = crate::hand_evaluator_3p::HandEvaluator3P::new(
                            temp_hand,
                            self.players[pid_us].melds.clone(),
                        );
                        if calc.is_tenpai() {
                            can_riichi = true;
                            break;
                        }
                    }
                    if can_riichi {
                        legals.push(Action::new(ActionType::Riichi, None, vec![], Some(pid)));
                    }
                }
            }

            // 3. Kan (Ankan / Kakan)
            if self.wall.drawable_count > 0 && self.drawn_tile.is_some() {
                let mut counts = [0; 34];
                for &t in &self.players[pid_us].hand {
                    let idx = t as usize / 4;
                    counts[idx] += 1;
                }

                if !self.players[pid_us].riichi_declared && !self.players[pid_us].riichi_stage {
                    // Ankan
                    for (t_val, &c) in counts.iter().enumerate() {
                        if c == 4 {
                            let lowest = (t_val * 4) as u8;
                            let consume = vec![lowest, lowest + 1, lowest + 2, lowest + 3];
                            legals.push(Action::new(
                                ActionType::Ankan,
                                Some(lowest),
                                consume,
                                Some(pid),
                            ));
                        }
                    }
                    // Kakan
                    for m in &self.players[pid_us].melds {
                        if m.meld_type == MeldType::Pon {
                            let target = m.tiles[0] / 4;
                            for &t in &self.players[pid_us].hand {
                                if t / 4 == target {
                                    legals.push(Action::new(
                                        ActionType::Kakan,
                                        Some(t),
                                        m.tiles.clone(),
                                        Some(pid),
                                    ));
                                }
                            }
                        }
                    }
                } else if self.players[pid_us].riichi_declared
                    && let Some(t) = self.drawn_tile
                {
                    let t34 = t / 4;
                    if counts[t34 as usize] == 4 {
                        let mut hand_pre = self.players[pid_us].hand.clone();
                        if let Some(pos) = hand_pre.iter().position(|&x| x == t) {
                            hand_pre.remove(pos);
                        }
                        let calc_pre = crate::hand_evaluator_3p::HandEvaluator3P::new(
                            hand_pre,
                            self.players[pid_us].melds.clone(),
                        );
                        let mut waits_pre = calc_pre.get_waits();
                        waits_pre.sort();

                        let mut hand_post = self.players[pid_us].hand.clone();
                        hand_post.retain(|&x| x / 4 != t34);
                        let mut melds_post = self.players[pid_us].melds.clone();
                        let lowest = t34 * 4;
                        melds_post.push(Meld::new(
                            MeldType::Ankan,
                            vec![lowest, lowest + 1, lowest + 2, lowest + 3],
                            false,
                            -1,
                            None,
                        ));
                        let calc_post =
                            crate::hand_evaluator_3p::HandEvaluator3P::new(hand_post, melds_post);
                        let mut waits_post = calc_post.get_waits();
                        waits_post.sort();

                        if waits_pre == waits_post && !waits_pre.is_empty() {
                            let consume = vec![lowest, lowest + 1, lowest + 2, lowest + 3];
                            legals.push(Action::new(
                                ActionType::Ankan,
                                Some(lowest),
                                consume,
                                Some(pid),
                            ));
                        }
                    }
                }
            }

            // 4. Kyushu Kyuhai
            let no_calls = self.players.iter().all(|p| p.melds.is_empty());

            if self.is_first_turn && no_calls && !self.players[pid_us].riichi_stage {
                let mut distinct_terminals = std::collections::HashSet::new();
                for &t in &self.players[pid_us].hand {
                    if is_terminal_tile(t) {
                        distinct_terminals.insert(t / 4);
                    }
                }
                if distinct_terminals.len() >= 9 {
                    legals.push(Action::new(
                        ActionType::KyushuKyuhai,
                        None,
                        vec![],
                        Some(pid),
                    ));
                }
            }

            // 5. Kita (3P always allows kita)
            let kita_actions = self.get_kita_legal_actions(pid);
            legals.extend(kita_actions);
        } else if self.phase == Phase::WaitResponse {
            if let Some(acts) = self.current_claims.get(&pid) {
                legals.extend(acts.clone());
            }
            legals.push(Action::new(ActionType::Pass, None, vec![], Some(pid)));
        }
        legals
    }

    fn _get_claim_actions_for_player(&self, i: u8, _pid: u8, tile: u8) -> (Vec<Action>, bool) {
        let mut legals = Vec::new();
        let mut missed_agari = false;
        let i_us = i as usize;
        let hand = &self.players[i_us].hand;
        let melds = &self.players[i_us].melds;
        let np: u8 = 3;

        // 1. Ron
        let tile_class = tile / 4;
        let in_discards = self.players[i_us]
            .discards
            .iter()
            .any(|&d| d / 4 == tile_class);
        let in_missed = self.players[i_us].missed_agari_doujun
            || (self.players[i_us].riichi_declared && self.players[i_us].missed_agari_riichi);

        if !in_discards && !in_missed {
            let calc = crate::hand_evaluator_3p::HandEvaluator3P::new(hand.clone(), melds.clone());
            let p_wind = (i + np - self.oya) % np;
            let cond = Conditions {
                tsumo: false,
                riichi: self.players[i_us].riichi_declared,
                double_riichi: self.players[i_us].double_riichi_declared,
                ippatsu: self.players[i_us].ippatsu_cycle,
                player_wind: Wind::from(p_wind),
                round_wind: Wind::from(self.round_wind),
                chankan: false,
                haitei: false,
                houtei: self.wall.drawable_count == 0 && !self.is_rinshan_flag,
                rinshan: false,
                tsumo_first_turn: false,
                riichi_sticks: self.riichi_sticks,
                honba: self.honba as u32,
                is_sanma: true,
                num_players: np,
                ..Default::default()
            };

            let mut is_furiten = false;
            let waits = calc.get_waits_u8();
            for &w in &waits {
                if self.players[i_us].discards.iter().any(|&d| d / 4 == w) {
                    is_furiten = true;
                    break;
                }
            }
            if self.players[i_us].missed_agari_riichi || self.players[i_us].missed_agari_doujun {
                is_furiten = true;
            }

            if !is_furiten {
                let res = calc.calc(tile, self.wall.dora_indicators.clone(), vec![], Some(cond));
                if res.is_win {
                    legals.push(Action::new(ActionType::Ron, Some(tile), vec![], Some(i)));
                } else if res.has_win_shape {
                    missed_agari = true;
                }
            }
        }

        // 2. Pon / Kan (no Chi in 3P)
        if !self.players[i_us].riichi_declared && self.wall.drawable_count > 0 {
            let count = hand.iter().filter(|&&t| t / 4 == tile / 4).count();
            if count >= 2 && hand.len() >= 3 {
                let check_pon_kuikae = |consumes: &Vec<u8>| -> bool {
                    let mut forbidden_34 = Vec::new();
                    if self.rule.kuikae_forbidden {
                        forbidden_34.push(tile / 4);
                    }
                    let mut used_consumes = vec![false; consumes.len()];
                    for &t in hand.iter() {
                        let mut consumed_this = false;
                        for (idx, &c) in consumes.iter().enumerate() {
                            if !used_consumes[idx] && c == t {
                                used_consumes[idx] = true;
                                consumed_this = true;
                                break;
                            }
                        }
                        if consumed_this {
                            continue;
                        }
                        if !forbidden_34.contains(&(t / 4)) {
                            return true;
                        }
                    }
                    false
                };

                // Generate all distinct pon consume pairs.
                // When a player has 3 copies of a tile (e.g. red 5m + 5m + 5m),
                // we need separate pon options with and without the red five.
                let matching: Vec<u8> = hand
                    .iter()
                    .filter(|&&t| t / 4 == tile / 4)
                    .cloned()
                    .collect();
                let mut seen_pairs: Vec<(u8, u8)> = Vec::new();
                for a in 0..matching.len() {
                    for b in (a + 1)..matching.len() {
                        let pair = (matching[a], matching[b]);
                        if !seen_pairs.contains(&pair) {
                            seen_pairs.push(pair);
                            let consumes = vec![pair.0, pair.1];
                            if check_pon_kuikae(&consumes) {
                                legals.push(Action::new(
                                    ActionType::Pon,
                                    Some(tile),
                                    consumes,
                                    Some(i),
                                ));
                            }
                        }
                    }
                }
            }
            if count >= 3 {
                let consumes: Vec<u8> = hand
                    .iter()
                    .filter(|&&t| t / 4 == tile / 4)
                    .take(3)
                    .cloned()
                    .collect();
                legals.push(Action::new(
                    ActionType::Daiminkan,
                    Some(tile),
                    consumes,
                    Some(i),
                ));
            }
        }

        // No Chi in 3P

        (legals, missed_agari)
    }
}
