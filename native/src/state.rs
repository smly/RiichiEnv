use pyo3::prelude::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use serde_json::Value;
use sha2::Digest;
use std::collections::HashMap;

use crate::action::{Action, ActionType, Phase};
use crate::observation::Observation;
use crate::parser::tid_to_mjai;
use crate::replay::Action as LogAction;
use crate::replay::MjaiEvent;
use crate::rule::GameRule;
use crate::types::{Agari, Conditions, Meld, MeldType, Wind};

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn is_terminal_tile(t: u8) -> bool {
    let t_type = t / 4;
    let rank = t_type % 9;
    let suit = t_type / 9;
    suit == 3 || rank == 0 || rank == 8
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct GameState {
    pub wall: Vec<u8>,
    pub hands: [Vec<u8>; 4],
    pub melds: [Vec<Meld>; 4],
    pub discards: [Vec<u8>; 4],
    pub discard_from_hand: [Vec<bool>; 4],
    pub discard_is_riichi: [Vec<bool>; 4],
    pub riichi_declaration_index: [Option<usize>; 4],
    pub current_player: u8,
    pub turn_count: u32,
    pub is_done: bool,
    pub needs_tsumo: bool,
    pub needs_initialize_next_round: bool,
    pub pending_oya_won: bool,
    pub pending_is_draw: bool,
    pub scores: [i32; 4],
    pub score_deltas: [i32; 4],
    pub riichi_sticks: u32,
    pub riichi_declared: [bool; 4],
    pub riichi_stage: [bool; 4],
    pub double_riichi_declared: [bool; 4],

    pub phase: Phase,
    pub active_players: Vec<u8>,
    pub last_discard: Option<(u8, u8)>,
    pub current_claims: HashMap<u8, Vec<Action>>,

    pub pending_kan: Option<(u8, Action)>,

    pub oya: u8,
    pub honba: u8,
    pub kyoku_idx: u8,
    pub round_wind: u8,
    pub dora_indicators: Vec<u8>,
    pub rinshan_draw_count: u8,
    pub pending_kan_dora_count: u8,

    pub is_rinshan_flag: bool,
    pub is_first_turn: bool,
    pub missed_agari_riichi: [bool; 4],
    pub missed_agari_doujun: [bool; 4],
    pub riichi_pending_acceptance: Option<u8>,
    pub nagashi_eligible: [bool; 4],
    pub drawn_tile: Option<u8>,
    pub ippatsu_cycle: [bool; 4],

    pub wall_digest: String,
    pub salt: String,
    pub agari_results: HashMap<u8, Agari>,
    pub last_agari_results: HashMap<u8, Agari>,
    pub round_end_scores: Option<[i32; 4]>,
    pub forbidden_discards: [Vec<u8>; 4],

    pub mjai_log: Vec<String>,
    pub mjai_log_per_player: [Vec<String>; 4],
    pub player_event_counts: [usize; 4],

    pub game_mode: u8,
    pub skip_mjai_logging: bool,
    pub seed: Option<u64>,
    pub(crate) hand_index: u64,
    pub rule: GameRule,
    pub pao: [HashMap<u8, u8>; 4],
    pub last_error: Option<String>,
    pub is_after_kan: bool,
}

impl GameState {
    pub fn new(
        game_mode: u8,
        skip_mjai_logging: bool,
        seed: Option<u64>,
        round_wind: u8,
        rule: GameRule,
    ) -> Self {
        let mut state = Self {
            wall: Vec::new(),
            hands: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            melds: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            discards: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            discard_from_hand: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            discard_is_riichi: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            riichi_declaration_index: [None; 4],
            current_player: 0,
            turn_count: 0,
            is_done: false,
            needs_tsumo: false,
            needs_initialize_next_round: false,
            pending_oya_won: false,
            pending_is_draw: false,
            scores: [25000; 4],
            score_deltas: [0; 4],
            riichi_sticks: 0,
            riichi_declared: [false; 4],
            riichi_stage: [false; 4],
            double_riichi_declared: [false; 4],
            phase: Phase::WaitAct,
            active_players: vec![0],
            last_discard: None,
            current_claims: HashMap::new(),
            pending_kan: None,
            oya: 0,
            honba: 0,
            kyoku_idx: 0,
            dora_indicators: Vec::new(),
            rinshan_draw_count: 0,
            pending_kan_dora_count: 0,
            is_rinshan_flag: false,
            is_first_turn: true,
            missed_agari_riichi: [false; 4],
            missed_agari_doujun: [false; 4],
            riichi_pending_acceptance: None,
            nagashi_eligible: [true; 4],
            drawn_tile: None,
            wall_digest: String::new(),
            salt: String::new(),
            agari_results: HashMap::new(),
            last_agari_results: HashMap::new(),
            round_end_scores: None,
            mjai_log: Vec::new(),
            mjai_log_per_player: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            player_event_counts: [0; 4],
            round_wind,
            ippatsu_cycle: [false; 4],
            game_mode,
            skip_mjai_logging,
            seed,
            hand_index: 0,
            forbidden_discards: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            rule,
            pao: [
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
            ],
            last_error: None,
            is_after_kan: false,
        };
        // Initial setup
        state._initialize_round(0, round_wind, 0, 0, None, None);
        state
    }

    pub fn reset(&mut self) {
        self.mjai_log = Vec::new();
        self.mjai_log_per_player = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        if !self.skip_mjai_logging {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("start_game".to_string()));
            self._push_mjai_event(Value::Object(ev));
        }
    }

    pub fn get_observation(&mut self, player_id: u8) -> Observation {
        let pid = player_id as usize;

        let mut masked_hands = Vec::new();
        for (i, h) in self.hands.iter().enumerate() {
            if i == pid {
                masked_hands.push(h.clone());
            } else {
                masked_hands.push(Vec::new());
            }
        }

        let legal_actions = if self.is_done {
            Vec::new()
        } else if (self.phase == Phase::WaitAct && self.current_player == player_id)
            || (self.phase == Phase::WaitResponse && self.active_players.contains(&player_id))
        {
            self._get_legal_actions_internal(player_id)
        } else {
            Vec::new()
        };

        let old_count = self.player_event_counts[pid];
        let full_log_len = self.mjai_log_per_player[pid].len();
        let new_events = if old_count < full_log_len {
            self.mjai_log_per_player[pid][old_count..].to_vec()
        } else {
            Vec::new()
        };
        self.player_event_counts[pid] = full_log_len;

        let calc = crate::agari_calculator::AgariCalculator::new(
            self.hands[pid].clone(),
            self.melds[pid].clone(),
        );
        let waits = calc.get_waits_u8();
        let is_tenpai = !waits.is_empty();

        Observation::new(
            player_id,
            masked_hands,
            self.melds.clone().to_vec(),
            self.discards.clone().to_vec(),
            self.dora_indicators.clone(),
            self.scores.to_vec(),
            self.riichi_declared.to_vec(),
            legal_actions,
            new_events,
            self.honba,
            self.riichi_sticks,
            self.round_wind,
            self.oya,
            self.kyoku_idx,
            waits,
            is_tenpai,
        )
    }

    pub fn get_observation_for_replay(
        &mut self,
        pid: u8,
        env_action: &Action,
        log_action_str: &str,
    ) -> PyResult<Observation> {
        // 1. Save state that might be modified
        let original_phase = self.phase;
        let original_active_players = self.active_players.clone();
        let original_claims = self.current_claims.clone();
        let original_riichi = self.riichi_declared[pid as usize];

        // 2. Adjust state based on action type to match environment expectations
        match env_action.action_type {
            ActionType::Ron | ActionType::Chi | ActionType::Pon | ActionType::Daiminkan => {
                // These actions occur in response to another player's discard
                self.phase = Phase::WaitResponse;
                self.active_players = vec![pid];
                // Ensure the expert action is available as a claim for get_observation to see
                self.current_claims
                    .entry(pid)
                    .or_default()
                    .push(env_action.clone());
            }
            _ => {}
        }

        // 3. Generate observation
        let mut obs = self.get_observation(pid);

        // 4. Handle known desync cases (e.g. Riichi state mismatch)
        // Check if the expert action exists in simulator's legal_actions
        let mut exists = obs
            ._legal_actions
            .iter()
            .any(|a| a.action_type == env_action.action_type && a.tile == env_action.tile);

        if !exists
            && env_action.action_type == ActionType::Discard
            && self.riichi_declared[pid as usize]
        {
            // Try unsetting riichi to see if it makes the discard legal (likely a desync)
            self.riichi_declared[pid as usize] = false;
            let new_obs = self.get_observation(pid);
            let is_legal_retry = new_obs
                ._legal_actions
                .iter()
                .any(|a| a.action_type == ActionType::Discard && a.tile == env_action.tile);

            if is_legal_retry {
                obs = new_obs;
                exists = true;
            } else {
                self.riichi_declared[pid as usize] = original_riichi;
            }
        }

        // 5. Restore other states
        self.phase = original_phase;
        self.active_players = original_active_players;
        self.current_claims = original_claims;

        // 6. Final Validation
        if !exists {
            let legal_summaries: Vec<String> = obs
                ._legal_actions
                .iter()
                .map(|a| format!("{:?}", a))
                .collect();
            let mut hand_tiles = Vec::new();
            for &t in &self.hands[pid as usize] {
                hand_tiles.push(t);
            }
            let is_tenpai = obs.is_tenpai;
            let opened = self.melds[pid as usize].iter().any(|m| m.opened);
            let melds = &self.melds[pid as usize];
            let jikaze = (pid + 4 - self.oya) % 4;
            let mut yaku_info = String::from("None");
            if let Some(tile) = env_action.tile {
                let cond = Conditions {
                    tsumo: env_action.action_type == ActionType::Tsumo,
                    riichi: self.riichi_declared[pid as usize],
                    double_riichi: self.double_riichi_declared[pid as usize],
                    ippatsu: self.ippatsu_cycle[pid as usize],
                    player_wind: Wind::from(jikaze),
                    round_wind: Wind::from(self.round_wind),
                    chankan: env_action.action_type == ActionType::Ron
                        && self.phase == Phase::WaitAct, // Simplified
                    haitei: self.wall.len() <= 14
                        && env_action.action_type == ActionType::Tsumo
                        && !self.is_rinshan_flag,
                    houtei: self.wall.len() <= 14
                        && env_action.action_type == ActionType::Ron
                        && !self.is_rinshan_flag,
                    rinshan: self.is_rinshan_flag,
                    tsumo_first_turn: self.is_first_turn && self.discards[pid as usize].is_empty(),
                    kyoutaku: self.riichi_sticks,
                    tsumi: self.honba as u32,
                };
                let mut hand = self.hands[pid as usize].clone();
                if let Some(idx) = hand.iter().position(|&t| t == tile) {
                    hand.remove(idx);
                }
                let calc = crate::agari_calculator::AgariCalculator::new(
                    hand,
                    self.melds[pid as usize].clone(),
                );
                let res = calc.calc(tile, self.dora_indicators.clone(), vec![], Some(cond));
                yaku_info = format!(
                    "Agari={}, Han={}, Fu={}, YakuIDs={:?}",
                    res.agari, res.han, res.fu, res.yaku
                );
            }

            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Replay desync: Action {:?} is not legal for player {} in simulator.\n\
                 Log Action: {}\n\
                 State: Score={}, WallLen={}, Tenpai={}, Opened={}, Phase={:?}, Rinshan={}, Bakaze={}, Jikaze={}\n\
                 DoraIndicators: {:?}\n\
                 Hand: {:?}\n\
                 Melds: {:?}\n\
                 YakuCalc: {}\n\
                 Legal actions: [{}]",
                env_action,
                pid,
                log_action_str,
                self.scores[pid as usize],
                self.wall.len(),
                is_tenpai,
                opened,
                self.phase,
                self.is_rinshan_flag,
                self.round_wind,
                jikaze,
                self.dora_indicators,
                hand_tiles,
                melds,
                yaku_info,
                legal_summaries.join(", ")
            )));
        }

        Ok(obs)
    }

    pub fn apply_log_action(&mut self, action: &LogAction) {
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

                if let Some(idx) = self.hands[s].iter().position(|&x| x == t) {
                    self.hands[s].remove(idx);
                }
                self.hands[s].sort();
                self.discards[s].push(t);
                self.discard_from_hand[s].push(!is_tsumogiri);
                self.discard_is_riichi[s].push(*is_liqi || *is_wliqi);
                self.last_discard = Some((s as u8, t));
                self.drawn_tile = None;

                self.riichi_declared[s] = self.riichi_declared[s] || *is_liqi;
                if *is_liqi {
                    self.riichi_declaration_index[s] = Some(self.discards[s].len() - 1);
                }
                self.current_player = (s as u8 + 1) % 4;
                self.phase = Phase::WaitAct;
                self.active_players = vec![self.current_player];
                self.needs_tsumo = true;
                self.is_first_turn = false;
                self.is_after_kan = false;
            }
            LogAction::DealTile { seat, tile, .. } => {
                self.hands[*seat].push(*tile);
                self.drawn_tile = Some(*tile);
                self.current_player = *seat as u8;
                self.phase = Phase::WaitAct;
                self.active_players = vec![self.current_player];
                self.is_rinshan_flag = self.is_after_kan && *seat == self.current_player as usize;
                self.needs_tsumo = false;
                self.is_after_kan = false;
                self.hands[*seat].sort();
                if !self.wall.is_empty() {
                    self.wall.pop();
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
                        if let Some(idx) = self.hands[*seat].iter().position(|&x| x == *t) {
                            self.hands[*seat].remove(idx);
                        }
                    }
                }
                self.hands[*seat].sort();

                let from_who = froms
                    .iter()
                    .find(|&&f| f != *seat)
                    .map(|&f| f as i8)
                    .unwrap_or(-1);
                self.melds[*seat].push(Meld {
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
                        if let Some(idx) = self.hands[*seat].iter().position(|&x| x / 4 == t_val) {
                            self.hands[*seat].remove(idx);
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

                    self.melds[*seat].push(Meld {
                        meld_type: *meld_type,
                        tiles: m_tiles,
                        opened: false,
                        from_who: -1,
                    });
                } else {
                    let tile = tiles[0];
                    if let Some(idx) = self.hands[*seat].iter().position(|&x| x == tile) {
                        self.hands[*seat].remove(idx);
                    }
                    for m in self.melds[*seat].iter_mut() {
                        if m.meld_type == MeldType::Peng && m.tiles[0] / 4 == tile / 4 {
                            m.meld_type = MeldType::Addgang;
                            m.tiles.push(tile);
                            m.tiles.sort();
                            break;
                        }
                    }
                }
                self.hands[*seat].sort();
                self.current_player = *seat as u8;
                self.phase = Phase::WaitAct;
                self.active_players = vec![self.current_player];
                self.needs_tsumo = true;
                self.is_first_turn = false;
                self.is_after_kan = true;
            }
            LogAction::Dora { dora_marker } => {
                self.dora_indicators.push(*dora_marker);
            }
            _ => {}
        }
    }

    pub fn _get_legal_actions_internal(&self, pid: u8) -> Vec<Action> {
        let mut legals = Vec::new();
        let mut hand = self.hands[pid as usize].clone();
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
                let cond = Conditions {
                    tsumo: true,
                    riichi: self.riichi_declared[pid as usize],
                    double_riichi: self.double_riichi_declared[pid as usize],
                    ippatsu: self.ippatsu_cycle[pid as usize],
                    player_wind: Wind::from((pid + 4 - self.oya) % 4),
                    round_wind: Wind::from(self.round_wind),
                    chankan: false,
                    haitei: self.wall.len() <= 14 && !self.is_rinshan_flag,
                    houtei: false,
                    rinshan: self.is_rinshan_flag,
                    tsumo_first_turn: self.is_first_turn && self.discards[pid as usize].is_empty(),
                    kyoutaku: self.riichi_sticks,
                    tsumi: self.honba as u32,
                };
                let mut hand = self.hands[pid as usize].clone();
                if let Some(idx) = hand.iter().rposition(|&t| t == tile) {
                    hand.remove(idx);
                }
                let calc = crate::agari_calculator::AgariCalculator::new(
                    hand,
                    self.melds[pid as usize].clone(),
                );
                let res = calc.calc(tile, self.dora_indicators.clone(), vec![], Some(cond));
                if res.agari && (res.yakuman || res.han >= 1) {
                    legals.push(Action::new(ActionType::Tsumo, Some(tile), vec![]));
                }
            }

            // 2. Discard / Riichi
            let declaration_turn = if self.riichi_declared[pid as usize] {
                if let Some(idx) = self.riichi_declaration_index[pid as usize] {
                    self.discards[pid as usize].len() <= idx
                } else {
                    false
                }
            } else {
                false
            };

            if !self.riichi_declared[pid as usize] || declaration_turn {
                for &t in self.hands[pid as usize].iter() {
                    let is_forbidden = self.forbidden_discards[pid as usize]
                        .iter()
                        .any(|&f| f / 4 == t / 4);
                    if !is_forbidden {
                        legals.push(Action::new(ActionType::Discard, Some(t), vec![]));
                    }
                }

                // Riichi check (Only if not already declared)
                if !self.riichi_declared[pid as usize]
                    && self.scores[pid as usize] >= 1000
                    && self.wall.len() > 14
                    && self.melds[pid as usize].iter().all(|m| !m.opened)
                    && !self.riichi_stage[pid as usize]
                {
                    let indices: Vec<usize> = (0..self.hands[pid as usize].len()).collect();
                    let mut can_riichi = false;

                    for &skip_idx in &indices {
                        let mut temp_hand = self.hands[pid as usize].clone();
                        temp_hand.remove(skip_idx);
                        let calc = crate::agari_calculator::AgariCalculator::new(
                            temp_hand,
                            self.melds[pid as usize].clone(),
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
            if self.wall.len() > 14 && self.drawn_tile.is_some() {
                let mut counts = [0; 34];
                for &t in &self.hands[pid as usize] {
                    let idx = t as usize / 4;
                    if idx < 34 {
                        counts[idx] += 1;
                    }
                }

                if !self.riichi_declared[pid as usize] {
                    // Ankan
                    for (t_val, &c) in counts.iter().enumerate() {
                        if c == 4 {
                            let lowest = (t_val * 4) as u8;
                            let consume = vec![lowest, lowest + 1, lowest + 2, lowest + 3];
                            legals.push(Action::new(ActionType::Ankan, Some(lowest), consume));
                        }
                    }
                    // Kakan
                    for m in &self.melds[pid as usize] {
                        if m.meld_type == MeldType::Peng {
                            let target = m.tiles[0] / 4;
                            for &t in &self.hands[pid as usize] {
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
                        let mut hand_pre = self.hands[pid as usize].clone();
                        if let Some(pos) = hand_pre.iter().position(|&x| x == t) {
                            hand_pre.remove(pos);
                        }
                        let calc_pre = crate::agari_calculator::AgariCalculator::new(
                            hand_pre,
                            self.melds[pid as usize].clone(),
                        );
                        let mut waits_pre = calc_pre.get_waits();
                        waits_pre.sort();

                        let mut hand_post = self.hands[pid as usize].clone();
                        hand_post.retain(|&x| x / 4 != t34);
                        let mut melds_post = self.melds[pid as usize].clone();
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
            if self.is_first_turn && self.melds.iter().all(|m| m.is_empty()) {
                let mut distinct_terminals = std::collections::HashSet::new();
                for &t in &self.hands[pid as usize] {
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

    pub fn apply_mjai_event(&mut self, event: MjaiEvent) {
        match event {
            MjaiEvent::StartKyoku {
                bakaze,
                kyoku: _,
                honba,
                kyoutaku,
                scores,
                dora_marker,
                tehais,
                oya,
            } => {
                // Initialize round state from event
                self.honba = honba;
                self.riichi_sticks = kyoutaku as u32;
                self.scores = scores.try_into().unwrap_or([25000; 4]);
                self.round_wind = match bakaze.as_str() {
                    "E" => Wind::East as u8,
                    "S" => Wind::South as u8,
                    "W" => Wind::West as u8,
                    "N" => Wind::North as u8,
                    _ => Wind::East as u8,
                };
                self.oya = oya;
                self.dora_indicators =
                    vec![crate::replay::TileConverter::parse_tile_136(&dora_marker)];

                // Set hands
                for (i, hand_strs) in tehais.iter().enumerate() {
                    let mut hand = Vec::new();
                    for tile_str in hand_strs {
                        hand.push(crate::replay::TileConverter::parse_tile_136(tile_str));
                    }
                    hand.sort();
                    self.hands[i] = hand;
                }

                // Clear other state
                self.discards = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
                self.melds = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
                self.riichi_declared = [false; 4];
                self.riichi_stage = [false; 4];
                self.drawn_tile = None;
                self.current_player = self.oya; // Oya starts
                self.needs_tsumo = true;
                self.is_done = false;
            }
            MjaiEvent::Tsumo { actor, pai } => {
                let tile = crate::replay::TileConverter::parse_tile_136(&pai);
                self.current_player = actor as u8;
                self.drawn_tile = Some(tile);
                self.hands[actor].push(tile);
                self.hands[actor].sort();
                if !self.wall.is_empty() {
                    self.wall.pop();
                }
                self.needs_tsumo = false;
            }
            MjaiEvent::Dahai { actor, pai, .. } => {
                let tile = crate::replay::TileConverter::parse_tile_136(&pai);
                self.current_player = actor as u8;
                if let Some(idx) = self.hands[actor].iter().position(|&t| t == tile) {
                    self.hands[actor].remove(idx);
                }
                self.discards[actor].push(tile);
                self.last_discard = Some((actor as u8, tile));
                self.drawn_tile = None;

                if self.riichi_stage[actor] {
                    self.riichi_declared[actor] = true;
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
                    if let Some(idx) = self.hands[actor].iter().position(|&x| x == *t) {
                        self.hands[actor].remove(idx);
                    }
                }

                self.melds[actor].push(Meld {
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
                    if let Some(idx) = self.hands[actor].iter().position(|&x| x == *t) {
                        self.hands[actor].remove(idx);
                    }
                }

                self.melds[actor].push(Meld {
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
                    if let Some(idx) = self.hands[actor].iter().position(|&x| x == tv) {
                        self.hands[actor].remove(idx);
                    }
                }

                self.melds[actor].push(Meld {
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
                    if let Some(idx) = self.hands[actor].iter().position(|&x| x == t) {
                        self.hands[actor].remove(idx);
                    }
                }
                self.melds[actor].push(Meld {
                    meld_type: MeldType::Angang,
                    tiles,
                    opened: false,
                    from_who: -1,
                });
                self.needs_tsumo = true;
            }
            MjaiEvent::Kakan { actor, pai } => {
                let tile = crate::replay::TileConverter::parse_tile_136(&pai);
                if let Some(idx) = self.hands[actor].iter().position(|&x| x == tile) {
                    self.hands[actor].remove(idx);
                }
                for m in self.melds[actor].iter_mut() {
                    if m.meld_type == MeldType::Peng && m.tiles[0] / 4 == tile / 4 {
                        m.meld_type = MeldType::Addgang;
                        m.tiles.push(tile);
                        break;
                    }
                }
                self.needs_tsumo = true;
            }
            MjaiEvent::Reach { actor } => {
                self.riichi_stage[actor] = true;
            }
            MjaiEvent::ReachAccepted { actor } => {
                self.riichi_declared[actor] = true;
                self.riichi_sticks += 1;
                self.scores[actor] -= 1000;
            }
            MjaiEvent::Dora { dora_marker } => {
                let tile = crate::replay::TileConverter::parse_tile_136(&dora_marker);
                self.dora_indicators.push(tile);
            }
            MjaiEvent::Hora { .. } | MjaiEvent::Ryukyoku { .. } | MjaiEvent::EndKyoku => {
                self.is_done = true;
            }
            _ => {}
        }
    }

    pub fn step(&mut self, actions: &HashMap<u8, Action>) {
        if self.is_done {
            return;
        }

        if self.needs_initialize_next_round {
            self._initialize_next_round(self.pending_oya_won, self.pending_is_draw);
            return;
        }
        // Validation
        for pid in 0..4 {
            if let Some(act) = actions.get(&(pid as u8)) {
                let legals = self._get_legal_actions_internal(pid as u8);
                let is_valid = legals.iter().any(|l| {
                    if l.action_type != act.action_type {
                        return false;
                    }

                    let tiles_match = l.tile == act.tile;
                    let consumes_match = l.consume_tiles == act.consume_tiles;

                    if tiles_match {
                        if consumes_match {
                            return true;
                        }
                        // Allow empty consume for Kakan
                        if act.consume_tiles.is_empty() && l.action_type == ActionType::Kakan {
                            return true;
                        }
                        // Allow empty consume for Discard, Riichi, Tsumo, Ron, Pass
                        if act.consume_tiles.is_empty()
                            && matches!(
                                l.action_type,
                                ActionType::Discard
                                    | ActionType::Riichi
                                    | ActionType::Tsumo
                                    | ActionType::Ron
                                    | ActionType::Pass
                            )
                        {
                            return true;
                        }
                    }

                    if consumes_match
                        && matches!(l.action_type, ActionType::Ankan | ActionType::Kakan)
                    {
                        return true;
                    }

                    // Allow None from python for context-implied actions
                    if act.tile.is_none() {
                        return matches!(
                            l.action_type,
                            ActionType::Tsumo
                                | ActionType::Ron
                                | ActionType::Riichi
                                | ActionType::KyushuKyuhai
                        );
                    }
                    false
                });

                if !is_valid {
                    let reason = format!("Error: Illegal Action by Player {}", pid);
                    self.last_error = Some(reason.clone());
                    self._trigger_ryukyoku(&reason);
                    return;
                }
            }
        }

        // --- Phase: WaitAct (Discards, Riichi, Tsumo, Kan) ---
        if self.phase == Phase::WaitAct {
            let pid = self.current_player;
            if let Some(act) = actions.get(&pid) {
                match act.action_type {
                    ActionType::Discard => {
                        if let Some(tile) = act.tile {
                            let mut tsumogiri = false;
                            let mut valid = false;
                            if let Some(dt) = self.drawn_tile {
                                if dt == tile {
                                    tsumogiri = true;
                                    valid = true;
                                }
                            }
                            if let Some(idx) =
                                self.hands[pid as usize].iter().position(|&t| t == tile)
                            {
                                self.hands[pid as usize].remove(idx);
                                self.hands[pid as usize].sort();
                                valid = true;
                                if let Some(dt) = self.drawn_tile {
                                    if dt == tile {
                                        tsumogiri = true;
                                    }
                                }
                            }
                            if valid {
                                self._resolve_discard(pid, tile, tsumogiri);
                            }
                        }
                    }
                    ActionType::KyushuKyuhai => {
                        self._trigger_ryukyoku("kyushu_kyuhai");
                    }
                    ActionType::Riichi => {
                        // Declare Riichi
                        if self.scores[pid as usize] >= 1000
                            && self.wall.len() > 14
                            && !self.riichi_declared[pid as usize]
                        {
                            self.riichi_stage[pid as usize] = true;
                            if !self.skip_mjai_logging {
                                let mut ev = serde_json::Map::new();
                                ev.insert("type".to_string(), Value::String("reach".to_string()));
                                ev.insert("actor".to_string(), Value::Number(pid.into()));
                                self._push_mjai_event(Value::Object(ev));
                            }
                            if let Some(t) = act.tile {
                                let mut tsumogiri = false;
                                if let Some(dt) = self.drawn_tile {
                                    if dt == t {
                                        tsumogiri = true;
                                    }
                                }
                                if let Some(idx) =
                                    self.hands[pid as usize].iter().position(|&x| x == t)
                                {
                                    self.hands[pid as usize].remove(idx);
                                    self.hands[pid as usize].sort();
                                }
                                self._resolve_discard(pid, t, tsumogiri);
                            }
                        }
                    }
                    ActionType::Ankan => {
                        // Ankan Logic
                        let tile = act.tile.or(act.consume_tiles.first().copied()).unwrap_or(0);
                        let mut chankan_ronners = Vec::new();
                        if self.rule.allows_ron_on_ankan_for_kokushi_musou {
                            for i in 0..4 {
                                if i == pid {
                                    continue;
                                }

                                // Check Kokushi Only
                                let hand = &self.hands[i as usize];
                                let melds = &self.melds[i as usize];

                                // Furiten check
                                let tile_class = tile / 4;
                                let in_discards = self.discards[i as usize]
                                    .iter()
                                    .any(|&d| d / 4 == tile_class);
                                if in_discards {
                                    continue;
                                }

                                let p_wind = (i + 4 - self.oya) % 4;
                                let cond = Conditions {
                                    tsumo: false,
                                    riichi: self.riichi_declared[i as usize],
                                    chankan: true,
                                    player_wind: Wind::from(p_wind),
                                    round_wind: Wind::from(self.round_wind),
                                    ..Default::default()
                                };
                                let calc = crate::agari_calculator::AgariCalculator::new(
                                    hand.clone(),
                                    melds.clone(),
                                );
                                let res = calc.calc(
                                    tile,
                                    self.dora_indicators.clone(),
                                    vec![],
                                    Some(cond),
                                );

                                // 42=Kokushi, 49=Kokushi13
                                if res.agari && (res.yaku.contains(&42) || res.yaku.contains(&49)) {
                                    chankan_ronners.push(i);
                                    self.current_claims.entry(i).or_default().push(Action::new(
                                        ActionType::Ron,
                                        Some(tile),
                                        vec![],
                                    ));
                                }
                            }
                        }

                        if !chankan_ronners.is_empty() {
                            self.pending_kan = Some((pid, act.clone()));
                            self.phase = Phase::WaitResponse;
                            self.active_players = chankan_ronners;
                            self.last_discard = Some((pid, tile));
                        } else {
                            self._resolve_kan(pid, act.clone());
                        }
                    }
                    ActionType::Kakan => {
                        // Log Kakan immediately (before Chankan check)
                        if !self.skip_mjai_logging {
                            let mut ev = serde_json::Map::new();
                            ev.insert("type".to_string(), Value::String("kakan".to_string()));
                            ev.insert("actor".to_string(), Value::Number(pid.into()));
                            let tile = act.tile.unwrap_or(0);
                            ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
                            let cons: Vec<String> =
                                act.consume_tiles.iter().map(|&t| tid_to_mjai(t)).collect();
                            ev.insert("consumed".to_string(), serde_json::to_value(cons).unwrap());
                            self._push_mjai_event(Value::Object(ev));
                        }

                        // Kakan Logic
                        // Check Chankan
                        let tile = act.tile.or(act.consume_tiles.first().copied()).unwrap_or(0);
                        let mut chankan_ronners = Vec::new();
                        for i in 0..4 {
                            if i == pid {
                                continue;
                            }
                            // Check Agari
                            let hand = &self.hands[i as usize];
                            let melds = &self.melds[i as usize];
                            let p_wind = (i + 4 - self.oya) % 4;
                            let cond = Conditions {
                                tsumo: false,
                                riichi: self.riichi_declared[i as usize],
                                double_riichi: self.double_riichi_declared[i as usize],
                                ippatsu: self.ippatsu_cycle[i as usize],
                                player_wind: Wind::from(p_wind),
                                round_wind: Wind::from(self.round_wind),
                                chankan: true,
                                haitei: false,
                                houtei: false,
                                rinshan: false,
                                tsumo_first_turn: false,
                                kyoutaku: self.riichi_sticks,
                                tsumi: self.honba as u32,
                            };
                            let calc = crate::agari_calculator::AgariCalculator::new(
                                hand.clone(),
                                melds.clone(),
                            );

                            // Check Furiten
                            let mut is_furiten = false;
                            let waits = calc.get_waits_u8();
                            for &w in &waits {
                                if self.discards[i as usize].iter().any(|&d| d / 4 == w) {
                                    is_furiten = true;
                                    break;
                                }
                            }
                            if self.missed_agari_riichi[i as usize]
                                || self.missed_agari_doujun[i as usize]
                            {
                                is_furiten = true;
                            }

                            // If valid:
                            let res = if !is_furiten {
                                calc.calc(tile, self.dora_indicators.clone(), vec![], Some(cond))
                            } else {
                                crate::types::Agari::new(
                                    false,
                                    false,
                                    0,
                                    0,
                                    0,
                                    vec![],
                                    0,
                                    0,
                                    None,
                                    false,
                                )
                            };

                            if res.agari && (res.yakuman || res.han >= 1) {
                                // Add Ron action offer
                                chankan_ronners.push(i);
                                self.current_claims.entry(i).or_default().push(Action::new(
                                    ActionType::Ron,
                                    Some(tile),
                                    vec![],
                                ));
                            }
                        }

                        if !chankan_ronners.is_empty() {
                            self.pending_kan = Some((pid, act.clone()));
                            self.phase = Phase::WaitResponse;
                            self.active_players = chankan_ronners;
                            self.last_discard = Some((pid, tile)); // Treat Kakan tile as discard for Ron targeting
                        } else {
                            self._resolve_kan(pid, act.clone());
                        }
                    }
                    ActionType::Tsumo => {
                        let hand = &self.hands[pid as usize];
                        let melds = &self.melds[pid as usize];
                        let p_wind = (pid + 4 - self.oya) % 4;
                        let cond = Conditions {
                            tsumo: true,
                            riichi: self.riichi_declared[pid as usize],
                            double_riichi: self.double_riichi_declared[pid as usize],
                            ippatsu: self.ippatsu_cycle[pid as usize],
                            haitei: self.wall.len() <= 14 && !self.is_rinshan_flag,
                            rinshan: self.is_rinshan_flag,
                            tsumo_first_turn: self.is_first_turn
                                && self.melds.iter().all(|m| m.is_empty()),
                            player_wind: Wind::from(p_wind),
                            round_wind: Wind::from(self.round_wind),
                            kyoutaku: self.riichi_sticks,
                            tsumi: self.honba as u32,
                            ..Default::default()
                        };
                        let calc = crate::agari_calculator::AgariCalculator::new(
                            hand.clone(),
                            melds.clone(),
                        );
                        let win_tile = self.drawn_tile.unwrap_or(0);
                        let res =
                            calc.calc(win_tile, self.dora_indicators.clone(), vec![], Some(cond));

                        if std::env::var("DEBUG").is_ok() {
                            eprintln!(
                                "DEBUG RUST: Tsumo Agari Check result: agari={} han={} yaku={:?}",
                                res.agari, res.han, res.yaku
                            );
                        }
                        if res.agari {
                            let mut deltas = [0; 4];
                            let mut total_win = 0;

                            // Check Pao
                            let mut pao_payer = None;
                            let mut pao_yakuman_val = 0;
                            let mut total_yakuman_val = 0;

                            if res.yakuman {
                                for &yid in &res.yaku {
                                    let val = if [47, 48, 49, 50].contains(&yid) {
                                        2
                                    } else {
                                        1
                                    };
                                    total_yakuman_val += val;
                                    if let Some(liable) = self.pao[pid as usize].get(&(yid as u8)) {
                                        pao_yakuman_val += val;
                                        pao_payer = Some(*liable);
                                    }
                                }
                            }

                            if pao_yakuman_val > 0 {
                                let unit = if pid == self.oya { 48000 } else { 32000 };
                                let pao_amt = pao_yakuman_val * unit;
                                let non_pao_yakuman_val = total_yakuman_val - pao_yakuman_val;
                                let non_pao_amt = non_pao_yakuman_val * unit;

                                // Pao Payer pays for Pao Part
                                if let Some(pp) = pao_payer {
                                    deltas[pp as usize] -= pao_amt;
                                    total_win += pao_amt;
                                }

                                // Non-Pao Part split normally
                                if non_pao_amt > 0 {
                                    if pid == self.oya {
                                        let share = non_pao_amt / 3;
                                        for i in 0..4 {
                                            if i != pid {
                                                deltas[i as usize] -= share;
                                                total_win += share;
                                            }
                                        }
                                    } else {
                                        let oya_share = non_pao_amt / 2;
                                        let ko_share = non_pao_amt / 4;
                                        for i in 0..4 {
                                            if i != pid {
                                                if i == self.oya {
                                                    deltas[i as usize] -= oya_share;
                                                    total_win += oya_share;
                                                } else {
                                                    deltas[i as usize] -= ko_share;
                                                    total_win += ko_share;
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                // Standard Scoring
                                if pid == self.oya {
                                    for i in 0..4 {
                                        if i != pid {
                                            deltas[i as usize] = -(res.tsumo_agari_ko as i32);
                                            total_win += res.tsumo_agari_ko as i32;
                                        }
                                    }
                                } else {
                                    for i in 0..4 {
                                        if i != pid {
                                            if i == self.oya {
                                                deltas[i as usize] = -(res.tsumo_agari_oya as i32);
                                                total_win += res.tsumo_agari_oya as i32;
                                            } else {
                                                deltas[i as usize] = -(res.tsumo_agari_ko as i32);
                                                total_win += res.tsumo_agari_ko as i32;
                                            }
                                        }
                                    }
                                }
                            }

                            total_win += (self.riichi_sticks * 1000) as i32;
                            self.riichi_sticks = 0;

                            deltas[pid as usize] += total_win;

                            self.score_deltas.copy_from_slice(&deltas);
                            for (score, delta) in self.scores.iter_mut().zip(deltas.iter()) {
                                *score += delta;
                            }

                            let mut val = res;
                            for (&yid, &liable) in &self.pao[pid as usize] {
                                if val.yaku.contains(&(yid as u32)) {
                                    val.pao_payer = Some(liable);
                                    break;
                                }
                            }
                            self.agari_results.insert(pid, val);

                            if !self.skip_mjai_logging {
                                let mut ev = serde_json::Map::new();
                                ev.insert("type".to_string(), Value::String("hora".to_string()));
                                ev.insert("actor".to_string(), Value::Number(pid.into()));
                                ev.insert("target".to_string(), Value::Number(pid.into()));
                                ev.insert(
                                    "deltas".to_string(),
                                    serde_json::to_value(deltas).unwrap(),
                                );
                                ev.insert("tsumo".to_string(), Value::Bool(true));

                                let mut ura_markers = Vec::new();
                                if self.riichi_declared[pid as usize] {
                                    ura_markers = self._get_ura_markers();
                                }
                                ev.insert(
                                    "ura_markers".to_string(),
                                    serde_json::to_value(&ura_markers).unwrap(),
                                );

                                self._push_mjai_event(Value::Object(ev));
                            }

                            self._initialize_next_round(pid == self.oya, false);
                        } else {
                            self.current_player = (self.current_player + 1) % 4;
                            self._deal_next();
                        }
                    }
                    _ => {}
                }
            }
        } else if self.phase == Phase::WaitResponse {
            // Check Missed Agari for all who could Ron but didn't
            for (&pid, legals) in &self.current_claims {
                if legals.iter().any(|a| a.action_type == ActionType::Ron) {
                    let mut roned = false;
                    if let Some(act) = actions.get(&pid) {
                        if act.action_type == ActionType::Ron {
                            roned = true;
                        }
                    }
                    if !roned {
                        self.missed_agari_doujun[pid as usize] = true;
                        if self.riichi_declared[pid as usize] {
                            self.missed_agari_riichi[pid as usize] = true;
                        }
                    }
                }
            }

            let mut ron_claims = Vec::new();
            let mut call_claim: Option<(u8, Action)> = None;

            for &pid in &self.active_players {
                if let Some(act) = actions.get(&pid) {
                    if act.action_type == ActionType::Ron {
                        ron_claims.push(pid);
                    } else if act.action_type == ActionType::Pon
                        || act.action_type == ActionType::Daiminkan
                        || act.action_type == ActionType::Chi
                    {
                        if let Some((_old_pid, old_act)) = &call_claim {
                            let old_is_pon = old_act.action_type == ActionType::Pon
                                || old_act.action_type == ActionType::Daiminkan;
                            let new_is_pon = act.action_type == ActionType::Pon
                                || act.action_type == ActionType::Daiminkan;
                            if !old_is_pon && new_is_pon {
                                call_claim = Some((pid, act.clone()));
                            }
                        } else {
                            call_claim = Some((pid, act.clone()));
                        }
                    }
                }
            }

            if !ron_claims.is_empty() {
                let (target_pid, win_tile) = self.last_discard.unwrap_or((self.current_player, 0));

                ron_claims.sort_by_key(|&pid| (pid + 4 - target_pid) % 4);

                let winners = if self.rule.allow_double_ron {
                    ron_claims
                } else {
                    vec![ron_claims[0]]
                };

                let mut total_deltas = [0; 4];
                let mut oya_won = false;
                let mut deposit_taken = false;

                for &w_pid in &winners {
                    let hand = &self.hands[w_pid as usize];
                    let melds = &self.melds[w_pid as usize];
                    let p_wind = (w_pid + 4 - self.oya) % 4;
                    let is_chankan = self.pending_kan.is_some();

                    let cond = Conditions {
                        tsumo: false,
                        riichi: self.riichi_declared[w_pid as usize],
                        double_riichi: self.double_riichi_declared[w_pid as usize],
                        ippatsu: self.ippatsu_cycle[w_pid as usize],
                        haitei: false,
                        houtei: self.wall.len() <= 14 && !self.is_rinshan_flag,
                        rinshan: false,
                        chankan: is_chankan,
                        tsumo_first_turn: false,
                        player_wind: Wind::from(p_wind),
                        round_wind: Wind::from(self.round_wind),
                        kyoutaku: self.riichi_sticks,
                        tsumi: self.honba as u32,
                    };

                    let calc =
                        crate::agari_calculator::AgariCalculator::new(hand.clone(), melds.clone());
                    let res = calc.calc(win_tile, self.dora_indicators.clone(), vec![], Some(cond));

                    if res.agari {
                        let score = res.ron_agari as i32;

                        let mut pao_payer = target_pid;
                        let mut pao_amt = 0;

                        if res.yakuman {
                            let mut pao_yakuman_val = 0;
                            // Calculate values
                            for &yid in &res.yaku {
                                let val = if [47, 48, 49, 50].contains(&yid) {
                                    2
                                } else {
                                    1
                                };
                                if let Some(liable) = self.pao[w_pid as usize].get(&(yid as u8)) {
                                    pao_yakuman_val += val;
                                    pao_payer = *liable;
                                }
                            }

                            if pao_yakuman_val > 0 {
                                let unit = if w_pid == self.oya { 48000 } else { 32000 };
                                let honba_pts = (self.honba as i32) * 300;
                                pao_amt = pao_yakuman_val * unit / 2 + honba_pts as u32;
                            }
                        }

                        let mut this_deltas = [0; 4];
                        this_deltas[w_pid as usize] += score;
                        this_deltas[pao_payer as usize] -= pao_amt as i32;
                        this_deltas[target_pid as usize] -= score - pao_amt as i32;

                        total_deltas[w_pid as usize] += score;
                        total_deltas[pao_payer as usize] -= pao_amt as i32;
                        total_deltas[target_pid as usize] -= score - pao_amt as i32;

                        if !deposit_taken {
                            let stick_pts = (self.riichi_sticks * 1000) as i32;
                            total_deltas[w_pid as usize] += stick_pts;
                            this_deltas[w_pid as usize] += stick_pts;
                            self.riichi_sticks = 0;
                            deposit_taken = true;
                        }

                        let mut val = res;
                        for (&yid, &liable) in &self.pao[w_pid as usize] {
                            if val.yaku.contains(&(yid as u32)) {
                                val.pao_payer = Some(liable);
                                break;
                            }
                        }
                        self.agari_results.insert(w_pid, val);

                        if w_pid == self.oya {
                            oya_won = true;
                        }

                        if !self.skip_mjai_logging {
                            let mut ev = serde_json::Map::new();
                            ev.insert("type".to_string(), Value::String("hora".to_string()));
                            ev.insert("actor".to_string(), Value::Number(w_pid.into()));
                            ev.insert("target".to_string(), Value::Number(target_pid.into()));
                            ev.insert(
                                "deltas".to_string(),
                                serde_json::to_value(this_deltas).unwrap(),
                            );

                            let mut ura_markers = Vec::new();
                            if self.riichi_declared[w_pid as usize] {
                                ura_markers = self._get_ura_markers();
                            }
                            ev.insert(
                                "ura_markers".to_string(),
                                serde_json::to_value(&ura_markers).unwrap(),
                            );

                            self._push_mjai_event(Value::Object(ev));
                        }
                    }
                }

                self.score_deltas.copy_from_slice(&total_deltas);
                for (score, delta) in self.scores.iter_mut().zip(total_deltas.iter()) {
                    *score += delta;
                }

                self._initialize_next_round(oya_won, false);
            } else if let Some((claimer, action)) = call_claim {
                if let Some(rp) = self.riichi_pending_acceptance {
                    self.scores[rp as usize] -= 1000;
                    self.score_deltas[rp as usize] -= 1000;
                    self.riichi_sticks += 1;
                    self.riichi_declared[rp as usize] = true;
                    self.ippatsu_cycle[rp as usize] = true;
                    if !self.skip_mjai_logging {
                        let mut ev = serde_json::Map::new();
                        ev.insert(
                            "type".to_string(),
                            Value::String("reach_accepted".to_string()),
                        );
                        ev.insert("actor".to_string(), Value::Number(rp.into()));
                        self._push_mjai_event(Value::Object(ev));
                    }
                    self.riichi_pending_acceptance = None;
                }
                self.is_rinshan_flag = false;
                self.is_first_turn = false;

                for p in 0..4 {
                    self.ippatsu_cycle[p] = false;
                }

                for &t in &action.consume_tiles {
                    if let Some(idx) = self.hands[claimer as usize].iter().position(|&x| x == t) {
                        self.hands[claimer as usize].remove(idx);
                    }
                }
                let (discarder, tile) = self.last_discard.unwrap();
                let mut tiles = action.consume_tiles.clone();
                tiles.push(tile);
                tiles.sort();
                let meld_type = match action.action_type {
                    ActionType::Pon => MeldType::Peng,
                    ActionType::Chi => MeldType::Chi,
                    ActionType::Daiminkan => MeldType::Gang,
                    _ => MeldType::Chi,
                };
                self.melds[claimer as usize].push(Meld {
                    meld_type,
                    tiles: tiles.clone(),
                    opened: true,
                    from_who: discarder as i8,
                });

                if !self.skip_mjai_logging {
                    let type_str = match action.action_type {
                        ActionType::Pon => Some("pon"),
                        ActionType::Chi => Some("chi"),
                        _ => None,
                    };
                    if let Some(s) = type_str {
                        let mut ev = serde_json::Map::new();
                        ev.insert("type".to_string(), serde_json::Value::String(s.to_string()));
                        ev.insert(
                            "actor".to_string(),
                            serde_json::Value::Number(claimer.into()),
                        );
                        ev.insert(
                            "target".to_string(),
                            serde_json::Value::Number(discarder.into()),
                        );
                        ev.insert(
                            "pai".to_string(),
                            serde_json::Value::String(tid_to_mjai(tile)),
                        );
                        let cons_strs: Vec<String> = action
                            .consume_tiles
                            .iter()
                            .map(|&t| tid_to_mjai(t))
                            .collect();
                        ev.insert(
                            "consumed".to_string(),
                            serde_json::to_value(cons_strs).unwrap(),
                        );
                        self._push_mjai_event(serde_json::Value::Object(ev));
                    }
                }

                // PAO implementation
                if meld_type == MeldType::Peng
                    || meld_type == MeldType::Gang
                    || meld_type == MeldType::Addgang
                {
                    let tile_val = tile / 4;
                    if (31..=33).contains(&tile_val) {
                        let dragon_melds = self.melds[claimer as usize]
                            .iter()
                            .filter(|m| {
                                let t = m.tiles[0] / 4;
                                (31..=33).contains(&t) && (m.meld_type != MeldType::Chi)
                            })
                            .count();
                        if dragon_melds == 3 {
                            self.pao[claimer as usize].insert(37, discarder);
                        }
                    } else if (27..=30).contains(&tile_val) {
                        let wind_melds = self.melds[claimer as usize]
                            .iter()
                            .filter(|m| {
                                let t = m.tiles[0] / 4;
                                (27..=30).contains(&t) && (m.meld_type != MeldType::Chi)
                            })
                            .count();
                        if wind_melds == 4 {
                            self.pao[claimer as usize].insert(50, discarder);
                        }
                    }
                }

                self.current_player = claimer;
                self.phase = Phase::WaitAct;
                self.active_players = vec![claimer];
                self.forbidden_discards[claimer as usize].clear();

                if action.action_type == ActionType::Pon {
                    self.forbidden_discards[claimer as usize].push(tile);
                } else if action.action_type == ActionType::Chi {
                    self.forbidden_discards[claimer as usize].push(tile);
                    let t34 = tile / 4;
                    let mut consumed_34: Vec<u8> =
                        action.consume_tiles.iter().map(|&x| x / 4).collect();
                    consumed_34.sort();
                    if consumed_34[0] == t34 + 1 && consumed_34[1] == t34 + 2 {
                        if t34 % 9 <= 5 {
                            self.forbidden_discards[claimer as usize].push((t34 + 3) * 4);
                        }
                    } else if t34 >= 2
                        && consumed_34[1] == t34 - 1
                        && consumed_34[0] == t34 - 2
                        && t34 % 9 >= 3
                    {
                        self.forbidden_discards[claimer as usize].push((t34 - 3) * 4);
                    }
                }

                if action.action_type == ActionType::Daiminkan {
                    self._resolve_kan(claimer, action.clone());
                } else {
                    self.needs_tsumo = false;
                    self.drawn_tile = None;
                }
            } else {
                // All Pass
                if let Some((pk_pid, pk_act)) = self.pending_kan.take() {
                    // Resume Kan
                    self._resolve_kan(pk_pid, pk_act);
                } else {
                    if let Some(rp) = self.riichi_pending_acceptance {
                        self.scores[rp as usize] -= 1000;
                        self.score_deltas[rp as usize] -= 1000;
                        self.riichi_sticks += 1;
                        self.riichi_declared[rp as usize] = true;
                        self.ippatsu_cycle[rp as usize] = true;
                        if !self.skip_mjai_logging {
                            let mut ev = serde_json::Map::new();
                            ev.insert(
                                "type".to_string(),
                                Value::String("reach_accepted".to_string()),
                            );
                            ev.insert("actor".to_string(), Value::Number(rp.into()));
                            self._push_mjai_event(Value::Object(ev));
                        }
                        self.riichi_pending_acceptance = None;
                    }
                    self.current_player = (self.current_player + 1) % 4;
                    self._deal_next();
                }
            }
        }
    }

    fn _deal_next(&mut self) {
        self.is_rinshan_flag = false;
        if self.wall.len() <= 14 {
            self._trigger_ryukyoku("exhaustive_draw");
            return;
        }

        if let Some(t) = self.wall.pop() {
            let pid = self.current_player;
            self.hands[pid as usize].push(t);
            self.drawn_tile = Some(t);
            self.needs_tsumo = false;
            self.phase = Phase::WaitAct;
            self.active_players = vec![pid];

            if !self.skip_mjai_logging {
                let mut ev = serde_json::Map::new();
                ev.insert("type".to_string(), Value::String("tsumo".to_string()));
                ev.insert("actor".to_string(), Value::Number(pid.into()));
                ev.insert("pai".to_string(), Value::String(tid_to_mjai(t)));
                self._push_mjai_event(Value::Object(ev));
            }
            self.forbidden_discards[pid as usize].clear();
        }
    }

    pub fn _initialize_next_round(&mut self, oya_won: bool, is_draw: bool) {
        if self.is_done {
            return;
        }

        let mut next_honba = self.honba;
        let mut next_oya = self.oya;
        let mut next_round_wind = self.round_wind;

        if oya_won {
            next_honba = next_honba.saturating_add(1);
        } else if is_draw {
            next_honba = next_honba.saturating_add(1);
            next_oya = (next_oya + 1) % 4;
            if next_oya == 0 {
                next_round_wind += 1;
            }
        } else {
            next_honba = 0;
            next_oya = (next_oya + 1) % 4;
            if next_oya == 0 {
                next_round_wind += 1;
            }
        }

        match self.game_mode {
            1 | 4 => {
                let max_score = self.scores.iter().cloned().max().unwrap_or(0);
                if next_round_wind >= 1 && (max_score >= 30000 || next_round_wind > 1) {
                    self._process_end_game();
                    return;
                }
            }
            2 | 5 => {
                let max_score = self.scores.iter().cloned().max().unwrap_or(0);
                if next_round_wind >= 2 && (max_score >= 30000 || next_round_wind > 2) {
                    self._process_end_game();
                    return;
                }
            }
            0 | 3 => {
                self._process_end_game();
                return;
            }
            _ => {
                if next_round_wind >= 1 {
                    self._process_end_game();
                    return;
                }
            }
        }

        if !self.skip_mjai_logging {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("end_kyoku".to_string()));
            self._push_mjai_event(Value::Object(ev));
        }

        let next_scores = self.scores;
        let next_sticks = self.riichi_sticks;
        self._initialize_round(
            next_oya,
            next_round_wind,
            next_honba,
            next_sticks,
            None,
            Some(next_scores),
        );
    }

    pub fn _initialize_round(
        &mut self,
        oya: u8,
        bakaze: u8,
        honba: u8,
        kyotaku: u32,
        wall: Option<Vec<u8>>,
        scores: Option<[i32; 4]>,
    ) {
        self.oya = oya;
        self.kyoku_idx = oya;
        self.current_player = oya;
        self.honba = honba;
        self.riichi_sticks = kyotaku;
        self.round_wind = bakaze;
        self.hands = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        self.melds = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        self.discards = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        self.discard_from_hand = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        self.discard_is_riichi = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        self.riichi_declaration_index = [None; 4];
        self.is_done = false;
        self.current_claims = HashMap::new();
        self.pending_kan = None;
        self.riichi_declared = [false; 4];
        self.riichi_stage = [false; 4];
        self.double_riichi_declared = [false; 4];
        self.is_rinshan_flag = false;
        self.rinshan_draw_count = 0;
        self.is_first_turn = true;
        self.missed_agari_riichi = [false; 4];
        self.missed_agari_doujun = [false; 4];
        self.riichi_pending_acceptance = None;
        self.nagashi_eligible = [true; 4];
        self.score_deltas = [0; 4];
        self.turn_count = 0;
        self.needs_tsumo = true;
        self.needs_initialize_next_round = false;
        self.pending_oya_won = false;
        self.pending_is_draw = false;
        self.ippatsu_cycle = [false; 4];

        if let Some(s) = scores {
            self.scores = s;
        }

        if let Some(mut w) = wall {
            w.reverse();
            self.wall = w;
        } else {
            let mut w: Vec<u8> = (0..136).collect();
            let mut rng = if let Some(episode_seed) = self.seed {
                let hand_seed = splitmix64(episode_seed.wrapping_add(self.hand_index));
                self.hand_index = self.hand_index.wrapping_add(1);
                StdRng::seed_from_u64(hand_seed)
            } else {
                self.hand_index = self.hand_index.wrapping_add(1);
                StdRng::from_entropy()
            };
            w.shuffle(&mut rng);
            self.wall = w;
        }

        if self.salt.is_empty() {
            let mut rng = if let Some(s) = self.seed {
                StdRng::seed_from_u64(s)
            } else {
                StdRng::from_entropy()
            };
            let chars: Vec<u8> = (0..8).map(|_| rng.gen()).collect();
            self.salt = hex::encode(chars);
        }
        let wall_str = self
            .wall
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<String>>()
            .join(",");
        let input = format!("{}{}", self.salt, wall_str);
        let mut hasher = sha2::Sha256::new();
        hasher.update(input);
        let result = hasher.finalize();
        self.wall_digest = hex::encode(result);

        self.dora_indicators = vec![self.wall[4]];

        for i in 0..4 {
            self.hands[i] = Vec::new();
            self.forbidden_discards[i] = Vec::new();
        }

        for _ in 0..3 {
            for idx in 0..4 {
                let p = (idx + oya as usize) % 4;
                for _ in 0..4 {
                    if let Some(t) = self.wall.pop() {
                        self.hands[p].push(t);
                    }
                }
            }
        }
        for idx in 0..4 {
            let p = (idx + oya as usize) % 4;
            if let Some(t) = self.wall.pop() {
                self.hands[p].push(t);
            }
        }
        for pid in 0..4 {
            self.hands[pid].sort();
        }

        if !self.skip_mjai_logging {
            let bakaze_str = match bakaze % 4 {
                0 => "E",
                1 => "S",
                2 => "W",
                _ => "N",
            };
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("start_kyoku".to_string()));
            ev.insert("bakaze".to_string(), Value::String(bakaze_str.to_string()));
            ev.insert("kyoku".to_string(), Value::Number((oya + 1).into()));
            ev.insert("honba".to_string(), Value::Number(honba.into()));
            ev.insert("kyotaku".to_string(), Value::Number(kyotaku.into()));
            ev.insert("oya".to_string(), Value::Number(oya.into()));
            ev.insert(
                "scores".to_string(),
                serde_json::to_value(self.scores).unwrap(),
            );
            ev.insert(
                "dora_marker".to_string(),
                Value::String(tid_to_mjai(self.dora_indicators[0])),
            );

            let mut tehais = Vec::new();
            for hand in &self.hands {
                let hand_strs: Vec<String> = hand.iter().map(|&t| tid_to_mjai(t)).collect();
                tehais.push(hand_strs);
            }
            ev.insert("tehais".to_string(), serde_json::to_value(tehais).unwrap());

            self._push_mjai_event(Value::Object(ev));
        }

        self.current_player = self.oya;
        self.phase = Phase::WaitAct;
        self.active_players = vec![self.oya];

        // Draw 14th tile for Oya immediately
        if let Some(t) = self.wall.pop() {
            self.hands[self.oya as usize].push(t);
            self.drawn_tile = Some(t);
            self.needs_tsumo = false;

            if !self.skip_mjai_logging {
                let mut ev = serde_json::Map::new();
                ev.insert("type".to_string(), Value::String("tsumo".to_string()));
                ev.insert("actor".to_string(), Value::Number(self.oya.into()));
                ev.insert("pai".to_string(), Value::String(tid_to_mjai(t)));
                self._push_mjai_event(Value::Object(ev));
            }
        } else {
            // Should not happen
            self.needs_tsumo = true;
            self.drawn_tile = None;
        }
    }

    pub fn _trigger_ryukyoku(&mut self, reason: &str) {
        self._accept_riichi();

        let mut tenpai = [false; 4];
        let mut final_reason = reason.to_string();
        let mut nagashi_winners = Vec::new();

        if reason == "exhaustive_draw" {
            for (i, (hand, melds)) in self.hands.iter().zip(self.melds.iter()).enumerate() {
                let calc =
                    crate::agari_calculator::AgariCalculator::new(hand.clone(), melds.clone());
                if calc.is_tenpai() {
                    tenpai[i] = true;
                }
            }
            for (i, &eligible) in self.nagashi_eligible.iter().enumerate() {
                if eligible {
                    nagashi_winners.push(i as u8);
                }
            }

            if !nagashi_winners.is_empty() {
                final_reason = "nagashimangan".to_string();
            } else {
                let num_tp = tenpai.iter().filter(|&&t| t).count();
                if (1..=3).contains(&num_tp) {
                    let pk = 3000 / num_tp as i32;
                    let pn = 3000 / (4 - num_tp) as i32;
                    for (i, tp) in tenpai.iter().enumerate() {
                        let delta = if *tp { pk } else { -pn };
                        self.scores[i] += delta;
                        self.score_deltas[i] = delta;
                    }
                }
            }
        } else if let Some(stripped) = reason.strip_prefix("Error: Illegal Action by Player ") {
            if let Ok(pid) = stripped.parse::<usize>() {
                if pid < 4 {
                    let is_offender_oya = (pid as u8) == self.oya;
                    if is_offender_oya {
                        for i in 0..4 {
                            if i == pid {
                                self.scores[i] -= 12000;
                                self.score_deltas[i] = -12000;
                            } else {
                                self.scores[i] += 4000;
                                self.score_deltas[i] = 4000;
                            }
                        }
                    } else {
                        for i in 0..4 {
                            if i == pid {
                                self.scores[i] -= 8000;
                                self.score_deltas[i] = -8000;
                            } else if (i as u8) == self.oya {
                                self.scores[i] += 4000;
                                self.score_deltas[i] = 4000;
                            } else {
                                self.scores[i] += 2000;
                                self.score_deltas[i] = 2000;
                            }
                        }
                    }
                }
            }
        }

        let is_renchan = if final_reason == "exhaustive_draw" {
            tenpai[self.oya as usize]
        } else if final_reason == "nagashimangan" {
            nagashi_winners.contains(&self.oya)
        } else {
            true // Other abortive draws or penalties usually Renchan
        };

        if !self.skip_mjai_logging {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("ryukyoku".to_string()));
            ev.insert("reason".to_string(), Value::String(final_reason.clone()));
            ev.insert(
                "deltas".to_string(),
                serde_json::to_value(self.score_deltas).unwrap(),
            );
            self._push_mjai_event(Value::Object(ev));
        }

        self._initialize_next_round(is_renchan, true);
    }

    pub(crate) fn _trigger_abortive_draw(&mut self, reason: &str) {
        self._trigger_ryukyoku(reason);
    }

    fn _process_end_game(&mut self) {
        self.is_done = true;
        if !self.skip_mjai_logging {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("end_game".to_string()));
            self._push_mjai_event(Value::Object(ev));
        }
    }

    pub(crate) fn abort_game(&mut self, reason: &str) {
        self._trigger_ryukyoku(reason);
    }

    fn check_abortive_draw(&mut self) -> bool {
        // 1. Sufuurenta (Four Winds)
        let turns_ok = self.discards.iter().all(|d| d.len() == 1);
        let melds_empty = self.melds.iter().all(|m| m.is_empty());

        if turns_ok && melds_empty {
            if let Some(first_tile) = self.discards[0].first() {
                let first = first_tile / 4;
                if (27..=30).contains(&first)
                    && self
                        .discards
                        .iter()
                        .all(|d| d.first().map(|&t| t / 4) == Some(first))
                {
                    self._trigger_abortive_draw("sufuurenta");
                    return true;
                }
            }
        }

        // 2. Suukansansen (4 Kans)
        // 1 initial + 4 kans = 5 indicators
        let mut kan_owners = Vec::new();
        for (pid, meld_list) in self.melds.iter().enumerate() {
            for m in meld_list {
                if m.meld_type == crate::types::MeldType::Gang
                    || m.meld_type == crate::types::MeldType::Angang
                    || m.meld_type == crate::types::MeldType::Addgang
                {
                    kan_owners.push(pid);
                }
            }
        }

        if self.dora_indicators.len() == 5 || kan_owners.len() == 4 {
            if kan_owners.len() == 4 {
                let first_owner = kan_owners[0];
                if !kan_owners.iter().all(|&o| o == first_owner) {
                    self.abort_game("suukansansen");
                    return true;
                }
            } else if self.dora_indicators.len() == 5 {
                // Check if dora based logic is needed if owners < 4 (e.g. malformed test?)
                // Ignoring for clear count logic above.
            }
        }

        // 3. Suucha Riichi (Four Riichis)
        if self.riichi_declared.iter().all(|&x| x) {
            self.abort_game("suucha_riichi");
            return true;
        }

        false
    }

    fn _push_mjai_event(&mut self, event: Value) {
        if self.skip_mjai_logging {
            return;
        }
        let json_str = serde_json::to_string(&event).unwrap();
        self.mjai_log.push(json_str.clone());

        let type_str = event["type"].as_str().unwrap_or("");
        let actor = event["actor"].as_u64().map(|a| a as usize);

        for pid in 0..4 {
            let should_push = true;
            let mut final_json = json_str.clone();

            if type_str == "start_kyoku" {
                if let Some(tehais_val) = event.get("tehais").and_then(|v| v.as_array()) {
                    let mut masked_tehais = Vec::new();
                    for (i, hand_val) in tehais_val.iter().enumerate() {
                        if i == pid {
                            masked_tehais.push(hand_val.clone());
                        } else {
                            let len = hand_val.as_array().map(|a| a.len()).unwrap_or(13);
                            let masked = vec!["?".to_string(); len];
                            masked_tehais.push(serde_json::to_value(masked).unwrap());
                        }
                    }
                    let mut masked_event = event.as_object().unwrap().clone();
                    masked_event.insert("tehais".to_string(), Value::Array(masked_tehais));
                    final_json = serde_json::to_string(&Value::Object(masked_event)).unwrap();
                }
            } else if type_str == "tsumo" {
                if let Some(act_id) = actor {
                    if act_id != pid {
                        let mut masked_event = event.as_object().unwrap().clone();
                        masked_event.insert("pai".to_string(), Value::String("?".to_string()));
                        final_json = serde_json::to_string(&Value::Object(masked_event)).unwrap();
                    }
                }
            }

            if should_push {
                self.mjai_log_per_player[pid].push(final_json);
            }
        }
    }

    fn _accept_riichi(&mut self) {
        if let Some(p) = self.riichi_pending_acceptance {
            self.scores[p as usize] -= 1000;
            self.score_deltas[p as usize] -= 1000;
            self.riichi_sticks += 1;
            self.riichi_pending_acceptance = None;
        }
    }

    pub(crate) fn _end_kyoku_ryukyoku(&mut self, is_renchan: bool, is_draw: bool) {
        self.round_end_scores = Some(self.scores);
        let mut ev = serde_json::Map::new();
        ev.insert("type".to_string(), Value::String("end_kyoku".to_string()));
        self._push_mjai_event(Value::Object(ev));

        if self._is_game_over() {
            self._process_end_game();
            self.phase = Phase::WaitAct;
            self.active_players = vec![];
        } else if self.last_error.is_none() {
            self._initialize_next_round(is_renchan, is_draw);
        }
    }

    fn _is_game_over(&self) -> bool {
        self.scores.iter().any(|&s| s < 0)
    }

    pub fn _resolve_kan(&mut self, pid: u8, action: Action) {
        let p_idx = pid as usize;

        // Remove tiles from hand
        if action.action_type == ActionType::Kakan {
            // Remove the added tile
            if let Some(tile) = action.tile {
                if let Some(idx) = self.hands[p_idx].iter().position(|&x| x == tile) {
                    self.hands[p_idx].remove(idx);
                }
            }
        } else {
            // Ankan / Daiminkan
            for &t in &action.consume_tiles {
                if let Some(idx) = self.hands[p_idx].iter().position(|&x| x == t) {
                    self.hands[p_idx].remove(idx);
                }
            }
        }

        // Update Melds
        if action.action_type == ActionType::Ankan {
            self.melds[p_idx].push(Meld {
                meld_type: MeldType::Angang,
                tiles: action.consume_tiles.clone(),
                opened: false,
                from_who: -1,
            });
        } else if action.action_type == ActionType::Kakan {
            let tile = action.tile.unwrap_or(0); // The added tile
            for m in self.melds[p_idx].iter_mut() {
                if m.meld_type == MeldType::Peng && m.tiles[0] / 4 == tile / 4 {
                    m.meld_type = MeldType::Addgang;
                    m.tiles.push(tile);
                    m.tiles.sort();
                    break;
                }
            }
        }

        // Draw Rinshan
        self.is_first_turn = false;
        for p in 0..4 {
            self.ippatsu_cycle[p] = false;
        }

        if self.wall.len() > 14 {
            let t = self.wall.pop().unwrap();
            self.hands[p_idx].push(t);
            self.drawn_tile = Some(t);
            self.rinshan_draw_count += 1;
            self.is_rinshan_flag = true;

            if !self.skip_mjai_logging {
                // Determine type and log if NOT Kakan (Kakan is logged in step)
                let type_str = match action.action_type {
                    ActionType::Ankan => Some("ankan"),
                    ActionType::Kakan => None, // Already logged
                    ActionType::Daiminkan => Some("daiminkan"),
                    _ => Some("kan"),
                };

                if let Some(ts) = type_str {
                    let mut ev = serde_json::Map::new();
                    ev.insert("type".to_string(), Value::String(ts.to_string()));
                    ev.insert("actor".to_string(), Value::Number(pid.into()));
                    if action.action_type == ActionType::Daiminkan {
                        let (discarder, tile) = self.last_discard.unwrap();
                        ev.insert("target".to_string(), Value::Number(discarder.into()));
                        ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
                    }
                    let cons: Vec<String> = action
                        .consume_tiles
                        .iter()
                        .map(|&t| tid_to_mjai(t))
                        .collect();
                    ev.insert("consumed".to_string(), serde_json::to_value(cons).unwrap());
                    self._push_mjai_event(Value::Object(ev));
                }

                // PUSH TSUMO EVENT FOR RINSHAN
                let mut t_ev = serde_json::Map::new();
                t_ev.insert("type".to_string(), Value::String("tsumo".to_string()));
                t_ev.insert("actor".to_string(), Value::Number(pid.into()));
                t_ev.insert("pai".to_string(), Value::String(tid_to_mjai(t)));
                self._push_mjai_event(Value::Object(t_ev));
            }
        } else {
            self._trigger_ryukyoku("exhaustive_draw");
            return;
        }

        self.current_player = pid;
        self.phase = Phase::WaitAct;
        self.active_players = vec![pid];
        self.needs_tsumo = false;
    }

    pub fn _reveal_kan_dora(&mut self) {
        let count = self.dora_indicators.len();
        if count < 5 {
            let base_idx = 4 + 2 * count;
            let idx = base_idx as i32 - self.rinshan_draw_count as i32;
            if idx >= 0 && (idx as usize) < self.wall.len() {
                self.dora_indicators.push(self.wall[idx as usize]);
            }
        }
    }

    pub fn _get_ura_markers(&self) -> Vec<String> {
        let mut results = Vec::new();
        // Ura markers are at 5, 7, 9, 11, 13 in original reversed wall.
        for i in 0..self.dora_indicators.len() {
            let idx = (5 + 2 * i) as i32 - self.rinshan_draw_count as i32;
            if idx >= 0 && (idx as usize) < self.wall.len() {
                results.push(tid_to_mjai(self.wall[idx as usize]));
            }
        }
        results
    }

    fn _get_claim_actions_for_player(&self, i: u8, pid: u8, tile: u8) -> (Vec<Action>, bool) {
        let mut legals = Vec::new();
        let mut missed_agari = false;
        let hand = &self.hands[i as usize];
        let melds = &self.melds[i as usize];

        // 1. Ron
        let tile_class = tile / 4;
        let in_discards = self.discards[i as usize]
            .iter()
            .any(|&d| d / 4 == tile_class);
        let in_missed = self.missed_agari_doujun[i as usize]
            || (self.riichi_declared[i as usize] && self.missed_agari_riichi[i as usize]);

        if !in_discards && !in_missed {
            let calc = crate::agari_calculator::AgariCalculator::new(hand.clone(), melds.clone());
            let p_wind = (i + 4 - self.oya) % 4;
            let cond = Conditions {
                tsumo: false,
                riichi: self.riichi_declared[i as usize],
                double_riichi: self.double_riichi_declared[i as usize],
                ippatsu: self.ippatsu_cycle[i as usize],
                player_wind: Wind::from(p_wind),
                round_wind: Wind::from(self.round_wind),
                chankan: false,
                haitei: false,
                houtei: self.wall.len() <= 14 && !self.is_rinshan_flag,
                rinshan: false,
                tsumo_first_turn: false,
                kyoutaku: self.riichi_sticks,
                tsumi: self.honba as u32,
            };

            let mut is_furiten = false;
            let waits = calc.get_waits_u8();
            for &w in &waits {
                if self.discards[i as usize].iter().any(|&d| d / 4 == w) {
                    is_furiten = true;
                    break;
                }
            }
            if self.missed_agari_riichi[i as usize] || self.missed_agari_doujun[i as usize] {
                is_furiten = true;
            }

            if !is_furiten {
                let res = calc.calc(tile, self.dora_indicators.clone(), vec![], Some(cond));
                if res.agari {
                    legals.push(Action::new(ActionType::Ron, Some(tile), vec![]));
                } else if res.has_agari_shape {
                    // Hand shape complete but no Yaku (and not Yakuman).
                    missed_agari = true;
                }
            }
        }

        // 2. Pon / Kan
        if !self.riichi_declared[i as usize] && self.wall.len() > 14 {
            let count = hand.iter().filter(|&&t| t / 4 == tile / 4).count();
            if count >= 2 && hand.len() >= 3 {
                // Check if taking this Pon leaves any valid discard (Kuikae check)
                let check_pon_kuikae = |consumes: &Vec<u8>| -> bool {
                    let mut forbidden_34 = Vec::new();
                    if !matches!(self.rule.kuikae_mode, crate::rule::KuikaeMode::None) {
                        forbidden_34.push(tile / 4);
                    }

                    // Simulation: Check if any tile in remaining hand is NOT forbidden
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
                // Daiminkan
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
        if !self.riichi_declared[i as usize]
            && self.wall.len() > 14
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
                                // Taken Low: t, t+1, t+2. Forbidden: t, t+3
                                if t_val % 9 <= 5 {
                                    forbidden_34.push(t_val + 3);
                                }
                            } else if t_val >= 2
                                && cons_34[1] == t_val - 1
                                && cons_34[0] == t_val - 2
                            {
                                // Taken High: t-2, t-1, t. Forbidden: t, t-3
                                if t_val % 9 >= 3 {
                                    forbidden_34.push(t_val - 3);
                                }
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

    fn _resolve_discard(&mut self, pid: u8, tile: u8, tsumogiri: bool) {
        self.discards[pid as usize].push(tile);
        self.last_discard = Some((pid, tile));
        self.discard_from_hand[pid as usize].push(!tsumogiri);
        let is_riichi_decl = self.riichi_stage[pid as usize];
        self.discard_is_riichi[pid as usize].push(is_riichi_decl);
        if is_riichi_decl {
            self.riichi_declaration_index[pid as usize] =
                Some(self.discards[pid as usize].len() - 1);
        }

        self.drawn_tile = None;
        self.last_discard = Some((pid, tile));

        if !self.skip_mjai_logging {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("dahai".to_string()));
            ev.insert("actor".to_string(), Value::Number(pid.into()));
            ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
            ev.insert("tsumogiri".to_string(), Value::Bool(tsumogiri));
            self._push_mjai_event(Value::Object(ev));
        }

        self.missed_agari_doujun[pid as usize] = false;
        self.nagashi_eligible[pid as usize] &= is_terminal_tile(tile);

        // Riichi Handling
        if self.riichi_stage[pid as usize] {
            self.riichi_stage[pid as usize] = false;
            self.riichi_pending_acceptance = Some(pid);
        }

        // Check Claims
        let mut has_claims = false;
        let mut claim_active = Vec::new();
        self.current_claims.clear();

        for i in 0..4 {
            if i == pid {
                continue;
            }

            let (legals, missed_agari) = self._get_claim_actions_for_player(i, pid, tile);

            if missed_agari {
                self.missed_agari_doujun[i as usize] = true;
            }

            if !legals.is_empty() {
                has_claims = true;
                claim_active.push(i);
                self.current_claims.insert(i, legals);
            }
        }

        if has_claims {
            self.phase = Phase::WaitResponse;
            self.active_players = claim_active;
        } else {
            if let Some(rp) = self.riichi_pending_acceptance {
                self.scores[rp as usize] -= 1000;
                self.score_deltas[rp as usize] -= 1000;
                self.riichi_sticks += 1;
                self.riichi_declared[rp as usize] = true;
                self.ippatsu_cycle[rp as usize] = true;
                if !self.skip_mjai_logging {
                    let mut ev = serde_json::Map::new();
                    ev.insert(
                        "type".to_string(),
                        Value::String("reach_accepted".to_string()),
                    );
                    ev.insert("actor".to_string(), Value::Number(rp.into()));
                    self._push_mjai_event(Value::Object(ev));
                }
                self.riichi_pending_acceptance = None;
            }
            if !self.check_abortive_draw() {
                self.current_player = (pid + 1) % 4;
                self._deal_next();
                if self.turn_count >= 4 {
                    self.is_first_turn = false;
                }
            }
        }
    }
}
