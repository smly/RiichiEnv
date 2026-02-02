use crate::action::{Action, ActionType, Phase};
use crate::state::GameState;
use crate::types::{is_terminal_tile, Conditions, Meld, MeldType, Wind};

pub trait GameStateLegalActions {
    fn _get_legal_actions_internal(&self, pid: u8) -> Vec<Action>;
    fn _get_claim_actions_for_player(&self, i: u8, pid: u8, tile: u8) -> (Vec<Action>, bool);
}

impl GameStateLegalActions for GameState {
    fn _get_legal_actions_internal(&self, pid: u8) -> Vec<Action> {
        let mut legals = Vec::new();
        let pid_us = pid as usize;
        let mut hand = self.players[pid_us].hand.clone();
        hand.sort();

        if self.is_done {
            return legals;
        }

        if self.phase == Phase::WaitAct {
            if pid != self.current_player {
                return legals;
            }

            // 1. Tsumo
            if let Some(tile) = self.drawn_tile {
                if !self.players[pid_us].riichi_stage {
                    let cond = Conditions {
                        tsumo: true,
                        riichi: self.players[pid_us].riichi_declared,
                        double_riichi: self.players[pid_us].double_riichi_declared,
                        ippatsu: self.players[pid_us].ippatsu_cycle,
                        player_wind: Wind::from((pid + 4 - self.oya) % 4),
                        round_wind: Wind::from(self.round_wind),
                        chankan: false,
                        haitei: self.wall.tiles.len() <= 14 && !self.is_rinshan_flag,
                        houtei: false,
                        rinshan: self.is_rinshan_flag,
                        tsumo_first_turn: self.is_first_turn
                            && self.players[pid_us].discards.is_empty(),
                        kyoutaku: self.riichi_sticks,
                        tsumi: self.honba as u32,
                    };
                    let mut hand = self.players[pid_us].hand.clone();
                    if let Some(idx) = hand.iter().rposition(|&t| t == tile) {
                        hand.remove(idx);
                    }
                    let calc = crate::agari_calculator::AgariCalculator::new(
                        hand,
                        self.players[pid_us].melds.clone(),
                    );
                    let res =
                        calc.calc(tile, self.wall.dora_indicators.clone(), vec![], Some(cond));
                    if res.agari && (res.yakuman || res.han >= 1) {
                        legals.push(Action::new(ActionType::Tsumo, Some(tile), vec![]));
                    }
                }
            }

            // 2. Discard / Riichi
            let declaration_turn = if self.players[pid_us].riichi_declared {
                if let Some(idx) = self.players[pid_us].riichi_declaration_index {
                    self.players[pid_us].discards.len() <= idx
                } else {
                    false
                }
            } else {
                false
            };

            if !self.players[pid_us].riichi_declared || declaration_turn {
                for &t in self.players[pid_us].hand.iter() {
                    let is_forbidden = self.players[pid_us]
                        .forbidden_discards
                        .iter()
                        .any(|&f| f / 4 == t / 4);
                    if !is_forbidden {
                        legals.push(Action::new(ActionType::Discard, Some(t), vec![]));
                    }
                }

                // Riichi check (Only if not already declared)
                if !self.players[pid_us].riichi_declared
                    && self.players[pid_us].score >= 1000
                    && self.wall.tiles.len() >= 18
                    && self.players[pid_us].melds.iter().all(|m| !m.opened)
                    && !self.players[pid_us].riichi_stage
                {
                    let indices: Vec<usize> = (0..self.players[pid_us].hand.len()).collect();
                    let mut can_riichi = false;

                    for &skip_idx in &indices {
                        let mut temp_hand = self.players[pid_us].hand.clone();
                        temp_hand.remove(skip_idx);
                        let calc = crate::agari_calculator::AgariCalculator::new(
                            temp_hand,
                            self.players[pid_us].melds.clone(),
                        );
                        if calc.is_tenpai() {
                            can_riichi = true;
                            break;
                        }
                    }
                    if can_riichi {
                        legals.push(Action::new(ActionType::Riichi, None, vec![]));
                    }
                }
            } else if let Some(dt) = self.drawn_tile {
                legals.push(Action::new(ActionType::Discard, Some(dt), vec![]));
            }

            // 3. Kan (Ankan / Kakan)
            if self.wall.tiles.len() > 14 && self.drawn_tile.is_some() {
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
                            legals.push(Action::new(ActionType::Ankan, Some(lowest), consume));
                        }
                    }
                    // Kakan
                    for m in &self.players[pid_us].melds {
                        if m.meld_type == MeldType::Peng {
                            let target = m.tiles[0] / 4;
                            for &t in &self.players[pid_us].hand {
                                if t / 4 == target {
                                    legals.push(Action::new(
                                        ActionType::Kakan,
                                        Some(t),
                                        m.tiles.clone(),
                                    ));
                                }
                            }
                        }
                    }
                } else if let Some(t) = self.drawn_tile {
                    let t34 = t / 4;
                    if counts[t34 as usize] == 4 {
                        // Check waits
                        let mut hand_pre = self.players[pid_us].hand.clone();
                        if let Some(pos) = hand_pre.iter().position(|&x| x == t) {
                            hand_pre.remove(pos);
                        }
                        let calc_pre = crate::agari_calculator::AgariCalculator::new(
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
                            MeldType::Angang,
                            vec![lowest, lowest + 1, lowest + 2, lowest + 3],
                            false,
                            -1,
                        ));
                        let calc_post =
                            crate::agari_calculator::AgariCalculator::new(hand_post, melds_post);
                        let mut waits_post = calc_post.get_waits();
                        waits_post.sort();

                        if waits_pre == waits_post && !waits_pre.is_empty() {
                            let consume = vec![lowest, lowest + 1, lowest + 2, lowest + 3];
                            legals.push(Action::new(ActionType::Ankan, Some(lowest), consume));
                        }
                    }
                }
            }

            // 4. Kyushu Kyuhai (Abortive Draw)
            // Simplified check: Check if all melds of all players are empty? No, Kyusyu Kyuhai is usually only valid if NO ONE has called.
            // But here we emulate generic rules.
            // Original code: if self.is_first_turn && self.melds.iter().all(|m| m.is_empty()) -> This meant check all players' melds?
            // In original GameState, melds was [Vec<Meld>; 4]. so self.melds.iter().all... checked all 4 vectors.
            let no_calls = self.players.iter().all(|p| p.melds.is_empty());

            if self.is_first_turn && no_calls && !self.players[pid_us].riichi_stage {
                let mut distinct_terminals = std::collections::HashSet::new();
                for &t in &self.players[pid_us].hand {
                    if is_terminal_tile(t) {
                        distinct_terminals.insert(t / 4);
                    }
                }
                if distinct_terminals.len() >= 9 {
                    legals.push(Action::new(ActionType::KyushuKyuhai, None, vec![]));
                }
            }
        } else if self.phase == Phase::WaitResponse {
            if let Some(acts) = self.current_claims.get(&pid) {
                legals.extend(acts.clone());
            }
            // Always offer Pass
            legals.push(Action::new(ActionType::Pass, None, vec![]));
        }
        legals
    }

    fn _get_claim_actions_for_player(&self, i: u8, pid: u8, tile: u8) -> (Vec<Action>, bool) {
        let mut legals = Vec::new();
        let mut missed_agari = false;
        let i_us = i as usize;
        let hand = &self.players[i_us].hand;
        let melds = &self.players[i_us].melds;

        // 1. Ron
        let tile_class = tile / 4;
        let in_discards = self.players[i_us]
            .discards
            .iter()
            .any(|&d| d / 4 == tile_class);
        let in_missed = self.players[i_us].missed_agari_doujun
            || (self.players[i_us].riichi_declared && self.players[i_us].missed_agari_riichi);

        if !in_discards && !in_missed {
            let calc = crate::agari_calculator::AgariCalculator::new(hand.clone(), melds.clone());
            let p_wind = (i + 4 - self.oya) % 4;
            let cond = Conditions {
                tsumo: false,
                riichi: self.players[i_us].riichi_declared,
                double_riichi: self.players[i_us].double_riichi_declared,
                ippatsu: self.players[i_us].ippatsu_cycle,
                player_wind: Wind::from(p_wind),
                round_wind: Wind::from(self.round_wind),
                chankan: false,
                haitei: false,
                houtei: self.wall.tiles.len() <= 14 && !self.is_rinshan_flag,
                rinshan: false,
                tsumo_first_turn: false,
                kyoutaku: self.riichi_sticks,
                tsumi: self.honba as u32,
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
                if res.agari {
                    legals.push(Action::new(ActionType::Ron, Some(tile), vec![]));
                } else if res.has_agari_shape {
                    missed_agari = true;
                }
            }
        }

        // 2. Pon / Kan
        if !self.players[i_us].riichi_declared && self.wall.tiles.len() > 14 {
            let count = hand.iter().filter(|&&t| t / 4 == tile / 4).count();
            if count >= 2 && hand.len() >= 3 {
                let check_pon_kuikae = |consumes: &Vec<u8>| -> bool {
                    let mut forbidden_34 = Vec::new();
                    if !matches!(self.rule.kuikae_mode, crate::rule::KuikaeMode::None) {
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

                let consumes: Vec<u8> = hand
                    .iter()
                    .filter(|&&t| t / 4 == tile / 4)
                    .take(2)
                    .cloned()
                    .collect();

                if check_pon_kuikae(&consumes) {
                    legals.push(Action::new(ActionType::Pon, Some(tile), consumes));
                }
            }
            if count >= 3 {
                let consumes: Vec<u8> = hand
                    .iter()
                    .filter(|&&t| t / 4 == tile / 4)
                    .take(3)
                    .cloned()
                    .collect();
                legals.push(Action::new(ActionType::Daiminkan, Some(tile), consumes));
            }
        }

        // 3. Chi
        let is_shimocha = i == (pid + 1) % 4;
        if !self.players[i_us].riichi_declared
            && self.wall.tiles.len() > 14
            && is_shimocha
            && hand.len() >= 3
        {
            let t_val = tile / 4;
            if t_val < 27 {
                let check_chi_kuikae = |c1: u8, c2: u8| -> bool {
                    let mut forbidden_34 = Vec::new();
                    if !matches!(self.rule.kuikae_mode, crate::rule::KuikaeMode::None) {
                        forbidden_34.push(t_val);
                        if self.rule.kuikae_mode == crate::rule::KuikaeMode::StrictFlank {
                            let mut cons_34 = [c1 / 4, c2 / 4];
                            cons_34.sort();
                            if cons_34[0] == t_val + 1 && cons_34[1] == t_val + 2 {
                                if t_val % 9 <= 5 {
                                    forbidden_34.push(t_val + 3);
                                }
                            } else if t_val >= 2
                                && cons_34[1] == t_val - 1
                                && cons_34[0] == t_val - 2
                                && t_val % 9 >= 3
                            {
                                forbidden_34.push(t_val - 3);
                            }
                        }
                    }
                    let mut used_c1 = false;
                    let mut used_c2 = false;
                    for &t in hand.iter() {
                        if !used_c1 && t == c1 {
                            used_c1 = true;
                            continue;
                        }
                        if !used_c2 && t == c2 {
                            used_c2 = true;
                            continue;
                        }
                        if !forbidden_34.contains(&(t / 4)) {
                            return true;
                        }
                    }
                    false
                };

                // Pattern 1: t-2, t-1, t
                if t_val % 9 >= 2 {
                    let c1_opts: Vec<u8> = hand
                        .iter()
                        .filter(|&&t| t / 4 == t_val - 2)
                        .copied()
                        .collect();
                    let c2_opts: Vec<u8> = hand
                        .iter()
                        .filter(|&&t| t / 4 == t_val - 1)
                        .copied()
                        .collect();
                    for &c1 in &c1_opts {
                        for &c2 in &c2_opts {
                            if check_chi_kuikae(c1, c2) {
                                legals.push(Action::new(ActionType::Chi, Some(tile), vec![c1, c2]));
                            }
                        }
                    }
                }
                // Pattern 2: t-1, t, t+1
                if t_val % 9 >= 1 && t_val % 9 <= 7 {
                    let c1_opts: Vec<u8> = hand
                        .iter()
                        .filter(|&&t| t / 4 == t_val - 1)
                        .copied()
                        .collect();
                    let c2_opts: Vec<u8> = hand
                        .iter()
                        .filter(|&&t| t / 4 == t_val + 1)
                        .copied()
                        .collect();
                    for &c1 in &c1_opts {
                        for &c2 in &c2_opts {
                            if check_chi_kuikae(c1, c2) {
                                legals.push(Action::new(ActionType::Chi, Some(tile), vec![c1, c2]));
                            }
                        }
                    }
                }
                // Pattern 3: t, t+1, t+2
                if t_val % 9 <= 6 {
                    let c1_opts: Vec<u8> = hand
                        .iter()
                        .filter(|&&t| t / 4 == t_val + 1)
                        .copied()
                        .collect();
                    let c2_opts: Vec<u8> = hand
                        .iter()
                        .filter(|&&t| t / 4 == t_val + 2)
                        .copied()
                        .collect();
                    for &c1 in &c1_opts {
                        for &c2 in &c2_opts {
                            if check_chi_kuikae(c1, c2) {
                                legals.push(Action::new(ActionType::Chi, Some(tile), vec![c1, c2]));
                            }
                        }
                    }
                }
            }
        }

        (legals, missed_agari)
    }
}
