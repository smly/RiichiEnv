use std::collections::HashMap;

use serde_json::Value;

use crate::action::{Action, ActionType, Phase};
use crate::errors::{RiichiError, RiichiResult};
use crate::observation_3p::Observation3P;
use crate::parser::tid_to_mjai;
use crate::replay::Action as LogAction;
use crate::replay::MjaiEvent;
use crate::rule::GameRule;
use crate::types::{Conditions, Meld, MeldType, WinResult, Wind};

pub mod event_handler;
pub mod game_mode;
pub mod legal_actions;
pub mod player;
pub mod sanma;
pub mod wall;
use event_handler::GameState3PEventHandler;
use game_mode::GameSubMode3P;
use legal_actions::GameState3PLegalActions;
use player::PlayerState3P;
use wall::WallState3P;

const NP: usize = 3;

#[derive(Debug, Clone)]
pub struct GameState3P {
    pub wall: WallState3P,
    pub players: [PlayerState3P; NP],

    pub current_player: u8,
    pub turn_count: u32,
    pub is_done: bool,
    pub needs_tsumo: bool,
    pub needs_initialize_next_round: bool,
    pub pending_oya_won: bool,
    pub pending_is_draw: bool,

    pub riichi_sticks: u32,
    pub phase: Phase,
    pub active_players: Vec<u8>,
    pub last_discard: Option<(u8, u8)>,
    pub current_claims: HashMap<u8, Vec<Action>>,
    pub pending_kan: Option<(u8, Action)>,

    pub oya: u8,
    pub honba: u8,
    pub kyoku_idx: u8,
    pub round_wind: u8,

    pub is_rinshan_flag: bool,
    pub is_first_turn: bool,
    pub riichi_pending_acceptance: Option<u8>,
    pub drawn_tile: Option<u8>,

    pub win_results: HashMap<u8, WinResult>,
    pub last_win_results: HashMap<u8, WinResult>,
    pub round_end_scores: Option<Vec<i32>>,

    pub mjai_log: Vec<String>,
    pub player_event_counts: [usize; NP],
    pub mjai_log_per_player: [Vec<String>; NP],

    pub sub_mode: GameSubMode3P,
    pub game_mode: u8,
    pub skip_mjai_logging: bool,
    pub seed: Option<u64>,
    pub rule: GameRule,
    pub last_error: Option<String>,
    pub is_after_kan: bool,

    pub riichi_sutehais: [Option<u8>; NP],
    pub last_tedashis: [Option<u8>; NP],
}

impl GameState3P {
    pub fn np(&self) -> usize {
        NP
    }

    pub fn new(
        game_mode: u8,
        skip_mjai_logging: bool,
        seed: Option<u64>,
        round_wind: u8,
        rule: GameRule,
    ) -> Self {
        let sub_mode = GameSubMode3P::from_game_mode(game_mode);
        let players = [(); NP].map(|_| PlayerState3P::new(game_mode::starting_score()));

        let wall = WallState3P::new(seed);

        let mut state = Self {
            wall,
            players,
            current_player: 0,
            turn_count: 0,
            is_done: false,
            needs_tsumo: false,
            needs_initialize_next_round: false,
            pending_oya_won: false,
            pending_is_draw: false,
            riichi_sticks: 0,
            phase: Phase::WaitAct,
            active_players: Vec::new(),
            last_discard: None,
            current_claims: HashMap::new(),
            pending_kan: None,
            oya: 0,
            honba: 0,
            kyoku_idx: 0,
            round_wind,
            is_rinshan_flag: false,
            is_first_turn: true,
            riichi_pending_acceptance: None,
            drawn_tile: None,
            win_results: HashMap::new(),
            last_win_results: HashMap::new(),
            round_end_scores: None,
            mjai_log: Vec::new(),
            player_event_counts: [0; NP],
            mjai_log_per_player: Default::default(),
            sub_mode,
            game_mode,
            skip_mjai_logging,
            seed,
            rule,
            last_error: None,
            is_after_kan: false,
            riichi_sutehais: [None; NP],
            last_tedashis: [None; NP],
        };

        if !state.skip_mjai_logging {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("start_game".to_string()));
            state._push_mjai_event(Value::Object(ev));
        }

        state._initialize_round(0, round_wind, 0, 0, None, None);
        state
    }

    pub fn reset(&mut self) {
        self.mjai_log = Vec::new();
        self.mjai_log_per_player = Default::default();
        self.player_event_counts = [0; NP];

        if !self.skip_mjai_logging {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("start_game".to_string()));
            self._push_mjai_event(Value::Object(ev));
        }
    }

    pub fn get_observation(&mut self, player_id: u8) -> Observation3P {
        let pid = player_id as usize;

        let masked_hands: [Vec<u8>; 3] = std::array::from_fn(|i| {
            if i == pid {
                self.players[i].hand.clone()
            } else {
                Vec::new()
            }
        });

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

        let calc = crate::hand_evaluator_3p::HandEvaluator3P::new(
            self.players[pid].hand.clone(),
            self.players[pid].melds.clone(),
        );
        let waits = calc.get_waits_u8();
        let is_tenpai = !waits.is_empty();

        let melds: [Vec<Meld>; 3] = std::array::from_fn(|i| self.players[i].melds.clone());
        let discards: [Vec<u8>; 3] = std::array::from_fn(|i| self.players[i].discards.clone());
        let scores: [i32; 3] = std::array::from_fn(|i| self.players[i].score);
        let riichi_declared: [bool; 3] = std::array::from_fn(|i| self.players[i].riichi_declared);

        Observation3P::new(
            player_id,
            masked_hands,
            melds,
            discards,
            self.wall.dora_indicators.clone(),
            scores,
            riichi_declared,
            legal_actions,
            new_events,
            self.honba,
            self.riichi_sticks,
            self.round_wind,
            self.oya,
            self.kyoku_idx,
            waits,
            is_tenpai,
            self.riichi_sutehais,
            self.last_tedashis,
            self.last_discard.map(|(tile, _pid)| tile as u32),
        )
    }

    pub fn get_observation_for_replay(
        &mut self,
        pid: u8,
        env_action: &Action,
        log_action_str: &str,
    ) -> RiichiResult<Observation3P> {
        let original_phase = self.phase;
        let original_active_players = self.active_players.clone();
        let original_claims = self.current_claims.clone();
        let original_riichi = self.players[pid as usize].riichi_declared;

        match env_action.action_type {
            ActionType::Ron | ActionType::Chi | ActionType::Pon | ActionType::Daiminkan => {
                self.phase = Phase::WaitResponse;
                self.active_players = vec![pid];
                self.current_claims
                    .entry(pid)
                    .or_default()
                    .push(env_action.clone());
            }
            _ => {}
        }

        let mut obs = self.get_observation(pid);

        let mut exists = obs
            ._legal_actions
            .iter()
            .any(|a| a.action_type == env_action.action_type && a.tile == env_action.tile);

        if !exists
            && env_action.action_type == ActionType::Discard
            && self.players[pid as usize].riichi_declared
        {
            self.players[pid as usize].riichi_declared = false;
            let new_obs = self.get_observation(pid);
            let is_legal_retry = new_obs
                ._legal_actions
                .iter()
                .any(|a| a.action_type == ActionType::Discard && a.tile == env_action.tile);

            if is_legal_retry {
                obs = new_obs;
                exists = true;
            } else {
                self.players[pid as usize].riichi_declared = original_riichi;
            }
        }

        self.phase = original_phase;
        self.active_players = original_active_players;
        self.current_claims = original_claims;

        if !exists {
            return Err(RiichiError::InvalidState {
                message: format!(
                    "Replay desync:\n  Env action: {:?}\n  Log action: {}\n  Self state:\n    phase: {:?}\n    drawn: {:?}",
                    env_action,
                    log_action_str,
                    self.phase,
                    self.drawn_tile
                ),
            });
        }

        Ok(obs)
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
        for pid in 0..NP {
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
                        if act.consume_tiles.is_empty() && l.action_type == ActionType::Kakan {
                            return true;
                        }
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

                    if act.tile.is_none() {
                        return matches!(
                            l.action_type,
                            ActionType::Tsumo
                                | ActionType::Ron
                                | ActionType::Riichi
                                | ActionType::KyushuKyuhai
                                | ActionType::Kita
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
                            if let Some(idx) = self.players[pid as usize]
                                .hand
                                .iter()
                                .position(|&t| t == tile)
                            {
                                self.players[pid as usize].hand.remove(idx);
                                self.players[pid as usize].hand.sort();
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
                        if self.players[pid as usize].score >= 1000
                            && self.wall.tiles.len() > 14
                            && !self.players[pid as usize].riichi_declared
                        {
                            self.players[pid as usize].riichi_stage = true;
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
                                self.riichi_sutehais[pid as usize] = Some(t);
                                if !tsumogiri {
                                    self.last_tedashis[pid as usize] = Some(t);
                                }
                                if let Some(idx) =
                                    self.players[pid as usize].hand.iter().position(|&x| x == t)
                                {
                                    self.players[pid as usize].hand.remove(idx);
                                    self.players[pid as usize].hand.sort();
                                }
                                self._resolve_discard(pid, t, tsumogiri);
                            }
                        }
                    }
                    ActionType::Ankan => {
                        let tile = act.tile.or(act.consume_tiles.first().copied()).unwrap_or(0);
                        let mut chankan_ronners = Vec::new();
                        if self.rule.allows_ron_on_ankan_for_kokushi_musou {
                            for i in 0..NP as u8 {
                                if i == pid {
                                    continue;
                                }
                                let hand = &self.players[i as usize].hand;
                                let melds = &self.players[i as usize].melds;
                                let tile_class = tile / 4;
                                let in_discards = self.players[i as usize]
                                    .discards
                                    .iter()
                                    .any(|&d| d / 4 == tile_class);
                                if in_discards {
                                    continue;
                                }
                                let p_wind = (i + NP as u8 - self.oya) % NP as u8;
                                let cond = Conditions {
                                    tsumo: false,
                                    riichi: self.players[i as usize].riichi_declared,
                                    chankan: true,
                                    player_wind: Wind::from(p_wind),
                                    round_wind: Wind::from(self.round_wind),
                                    is_sanma: true,
                                    num_players: NP as u8,
                                    ..Default::default()
                                };
                                let calc = crate::hand_evaluator_3p::HandEvaluator3P::new(
                                    hand.clone(),
                                    melds.clone(),
                                );
                                let res = calc.calc(
                                    tile,
                                    self.wall.dora_indicators.clone(),
                                    vec![],
                                    Some(cond),
                                );
                                if res.is_win && (res.yaku.contains(&42) || res.yaku.contains(&49))
                                {
                                    chankan_ronners.push(i);
                                    self.current_claims.entry(i).or_default().push(Action::new(
                                        ActionType::Ron,
                                        Some(tile),
                                        vec![],
                                        Some(i),
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
                        let tile = act.tile.or(act.consume_tiles.first().copied()).unwrap_or(0);
                        let p_idx = pid as usize;

                        if let Some(idx) = self.players[p_idx].hand.iter().position(|&x| x == tile)
                        {
                            self.players[p_idx].hand.remove(idx);
                        }
                        for m in self.players[p_idx].melds.iter_mut() {
                            if m.meld_type == MeldType::Pon && m.tiles[0] / 4 == tile / 4 {
                                m.meld_type = MeldType::Kakan;
                                m.tiles.push(tile);
                                m.tiles.sort();
                                break;
                            }
                        }

                        if !self.skip_mjai_logging {
                            let mut ev = serde_json::Map::new();
                            ev.insert("type".to_string(), Value::String("kakan".to_string()));
                            ev.insert("actor".to_string(), Value::Number(pid.into()));
                            ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
                            let cons: Vec<String> =
                                act.consume_tiles.iter().map(|&t| tid_to_mjai(t)).collect();
                            ev.insert("consumed".to_string(), serde_json::to_value(cons).unwrap());
                            self._push_mjai_event(Value::Object(ev));
                        }

                        // Reveal any pending kan doras from previous kans
                        while self.wall.pending_kan_dora_count > 0 {
                            self.wall.pending_kan_dora_count -= 1;
                            self._reveal_kan_dora();
                        }

                        let tile = act.tile.or(act.consume_tiles.first().copied()).unwrap_or(0);
                        let mut chankan_ronners = Vec::new();
                        for i in 0..NP as u8 {
                            if i == pid {
                                continue;
                            }
                            let hand = &self.players[i as usize].hand;
                            let melds = &self.players[i as usize].melds;
                            let p_wind = (i + NP as u8 - self.oya) % NP as u8;
                            let cond = Conditions {
                                tsumo: false,
                                riichi: self.players[i as usize].riichi_declared,
                                double_riichi: self.players[i as usize].double_riichi_declared,
                                ippatsu: self.players[i as usize].ippatsu_cycle,
                                player_wind: Wind::from(p_wind),
                                round_wind: Wind::from(self.round_wind),
                                chankan: true,
                                haitei: false,
                                houtei: false,
                                rinshan: false,
                                tsumo_first_turn: false,
                                riichi_sticks: self.riichi_sticks,
                                honba: self.honba as u32,
                                is_sanma: true,
                                num_players: NP as u8,
                                ..Default::default()
                            };
                            let calc = crate::hand_evaluator_3p::HandEvaluator3P::new(
                                hand.clone(),
                                melds.clone(),
                            );

                            let mut is_furiten = false;
                            let waits = calc.get_waits_u8();
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

                            let res = if !is_furiten {
                                calc.calc(
                                    tile,
                                    self.wall.dora_indicators.clone(),
                                    vec![],
                                    Some(cond),
                                )
                            } else {
                                WinResult::new(false, false, 0, 0, 0, vec![], 0, 0, None, false)
                            };

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
                            self.pending_kan = Some((pid, act.clone()));
                            self.phase = Phase::WaitResponse;
                            self.active_players = chankan_ronners;
                            self.last_discard = Some((pid, tile));
                        } else {
                            self._resolve_kan(pid, act.clone());
                        }
                    }
                    ActionType::Tsumo => {
                        let hand = &self.players[pid as usize].hand;
                        let melds = &self.players[pid as usize].melds;
                        let p_wind = (pid + NP as u8 - self.oya) % NP as u8;
                        let cond = Conditions {
                            tsumo: true,
                            riichi: self.players[pid as usize].riichi_declared,
                            double_riichi: self.players[pid as usize].double_riichi_declared,
                            ippatsu: self.players[pid as usize].ippatsu_cycle,
                            haitei: self.wall.tiles.len() <= 14 && !self.is_rinshan_flag,
                            rinshan: self.is_rinshan_flag,
                            tsumo_first_turn: self.is_first_turn
                                && self.players.iter().all(|p| p.melds.is_empty()),
                            player_wind: Wind::from(p_wind),
                            round_wind: Wind::from(self.round_wind),
                            riichi_sticks: self.riichi_sticks,
                            honba: self.honba as u32,
                            kita_count: self.players[pid as usize].kita_tiles.len() as u8,
                            is_sanma: true,
                            num_players: NP as u8,
                            ..Default::default()
                        };
                        let calc = crate::hand_evaluator_3p::HandEvaluator3P::new(
                            hand.clone(),
                            melds.clone(),
                        );
                        let win_tile = self.drawn_tile.unwrap_or(0);
                        let ura_indicators = if self.players[pid as usize].riichi_declared {
                            self._get_ura_indicators()
                        } else {
                            vec![]
                        };
                        let mut res = calc.calc(
                            win_tile,
                            self.wall.dora_indicators.clone(),
                            ura_indicators,
                            Some(cond.clone()),
                        );

                        // Cap double yakuman patterns when not enabled per rule flags
                        if res.yakuman && res.han > 13 {
                            let mut cap = 0u32;
                            for &y in &res.yaku {
                                match y {
                                    47 if !self.rule.is_junsei_chuurenpoutou_double => cap += 13,
                                    48 if !self.rule.is_suuankou_tanki_double => cap += 13,
                                    49 if !self.rule.is_kokushi_musou_13machi_double => cap += 13,
                                    50 if !self.rule.is_daisuushii_double => cap += 13,
                                    _ => {}
                                }
                            }
                            if cap > 0 {
                                res.han = res.han.saturating_sub(cap).max(13);
                                let capped = crate::score::calculate_score(
                                    res.han as u8,
                                    0,
                                    pid == self.oya,
                                    cond.tsumo,
                                    cond.honba,
                                    NP as u8,
                                );
                                res.ron_agari = capped.pay_ron;
                                res.tsumo_agari_oya = capped.pay_tsumo_oya;
                                res.tsumo_agari_ko = capped.pay_tsumo_ko;
                            }
                        }

                        if res.is_win {
                            let mut deltas = vec![0i32; NP];
                            let mut total_win = 0;

                            let mut pao_payer = None;
                            let mut pao_yakuman_val = 0;
                            let mut total_yakuman_val = 0;

                            if res.yakuman {
                                for &yid in &res.yaku {
                                    let val = match yid {
                                        47 if self.rule.is_junsei_chuurenpoutou_double => 2,
                                        48 if self.rule.is_suuankou_tanki_double => 2,
                                        49 if self.rule.is_kokushi_musou_13machi_double => 2,
                                        50 if self.rule.is_daisuushii_double => 2,
                                        _ => 1,
                                    };
                                    total_yakuman_val += val;
                                    if let Some(liable) =
                                        self.players[pid as usize].pao.get(&(yid as u8))
                                    {
                                        pao_yakuman_val += val;
                                        pao_payer = Some(*liable);
                                    }
                                }
                            }

                            if pao_yakuman_val > 0 {
                                // Per-yakuman tsumo total depends on player count.
                                // Yakuman base = 8000; pay_oya = 16000, pay_ko = 8000.
                                let np = NP as i32;
                                let unit = if pid == self.oya {
                                    (np - 1) * 16000 // oya tsumo: each ko pays 16000
                                } else {
                                    16000 + (np - 2) * 8000 // ko tsumo: oya pays 16000 + (np-2) ko pay 8000
                                };
                                let honba_total = self.honba as i32 * (np - 1) * 100;

                                if let Some(pp) = pao_payer {
                                    if self.rule.yakuman_pao_is_liability_only {
                                        // Majsoul: PAO pays PAO portion only, non-PAO split normally
                                        let pao_amt = pao_yakuman_val * unit + honba_total;
                                        let non_pao_yakuman_val =
                                            total_yakuman_val - pao_yakuman_val;

                                        deltas[pp as usize] -= pao_amt;
                                        total_win += pao_amt;

                                        if non_pao_yakuman_val > 0 {
                                            if pid == self.oya {
                                                let share = non_pao_yakuman_val * 16000;
                                                for i in 0..NP as u8 {
                                                    if i != pid {
                                                        deltas[i as usize] -= share;
                                                        total_win += share;
                                                    }
                                                }
                                            } else {
                                                let oya_pay = non_pao_yakuman_val * 16000;
                                                let ko_pay = non_pao_yakuman_val * 8000;
                                                for i in 0..NP as u8 {
                                                    if i != pid {
                                                        if i == self.oya {
                                                            deltas[i as usize] -= oya_pay;
                                                            total_win += oya_pay;
                                                        } else {
                                                            deltas[i as usize] -= ko_pay;
                                                            total_win += ko_pay;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        // Tenhou: PAO pays ALL yakuman (full amount)
                                        let full_amt = total_yakuman_val * unit + honba_total;
                                        deltas[pp as usize] -= full_amt;
                                        total_win += full_amt;
                                    }
                                }
                            } else if pid == self.oya {
                                for i in 0..NP as u8 {
                                    if i != pid {
                                        deltas[i as usize] = -(res.tsumo_agari_ko as i32);
                                        total_win += res.tsumo_agari_ko as i32;
                                    }
                                }
                            } else {
                                for i in 0..NP as u8 {
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

                            total_win += (self.riichi_sticks * 1000) as i32;
                            self.riichi_sticks = 0;
                            deltas[pid as usize] += total_win;

                            self.players[pid as usize].score_delta = deltas[pid as usize];
                            for (i, p) in self.players.iter_mut().enumerate() {
                                p.score += deltas[i];
                                p.score_delta = deltas[i];
                            }

                            let mut val = res;
                            for (&yid, &liable) in &self.players[pid as usize].pao {
                                if val.yaku.contains(&(yid as u32)) {
                                    val.pao_payer = Some(liable);
                                    break;
                                }
                            }
                            self.win_results.insert(pid, val);

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
                                if self.players[pid as usize].riichi_declared {
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
                            self.current_player = (self.current_player + 1) % NP as u8;
                            self._deal_next();
                        }
                    }
                    ActionType::Kita => {
                        self.handle_kita(pid, act);
                    }
                    _ => {}
                }
            }
        } else if self.phase == Phase::WaitResponse {
            // Check Missed WinResult
            for (&pid, legals) in &self.current_claims {
                if legals.iter().any(|a| a.action_type == ActionType::Ron) {
                    let mut roned = false;
                    if let Some(act) = actions.get(&pid) {
                        if act.action_type == ActionType::Ron {
                            roned = true;
                        }
                    }
                    if !roned {
                        self.players[pid as usize].missed_agari_doujun = true;
                        if self.players[pid as usize].riichi_declared {
                            self.players[pid as usize].missed_agari_riichi = true;
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
                ron_claims.sort_by_key(|&pid| (pid + NP as u8 - target_pid) % NP as u8);

                let winners = ron_claims;

                let mut total_deltas = [0i32; NP];
                let mut oya_won = false;
                let mut deposit_taken = false;
                let mut honba_taken = false;

                for &w_pid in &winners {
                    let hand = &self.players[w_pid as usize].hand;
                    let melds = &self.players[w_pid as usize].melds;
                    let p_wind = (w_pid + NP as u8 - self.oya) % NP as u8;
                    // Chankan yaku applies to kakan/ankan, but NOT to kita (BaBei).
                    // MjSoul allows ron on kita tiles but does not award chankan yaku.
                    let is_chankan = self
                        .pending_kan
                        .as_ref()
                        .is_some_and(|(_, act)| act.action_type != ActionType::Kita);

                    // Only the first winner (closest to discarder) gets honba
                    let ron_honba = if !honba_taken {
                        honba_taken = true;
                        self.honba as u32
                    } else {
                        0
                    };

                    let cond = Conditions {
                        tsumo: false,
                        riichi: self.players[w_pid as usize].riichi_declared,
                        double_riichi: self.players[w_pid as usize].double_riichi_declared,
                        ippatsu: self.players[w_pid as usize].ippatsu_cycle,
                        haitei: false,
                        houtei: self.wall.tiles.len() <= 14 && !self.is_rinshan_flag,
                        rinshan: false,
                        chankan: is_chankan,
                        tsumo_first_turn: false,
                        player_wind: Wind::from(p_wind),
                        round_wind: Wind::from(self.round_wind),
                        riichi_sticks: self.riichi_sticks,
                        honba: ron_honba,
                        kita_count: self.players[w_pid as usize].kita_tiles.len() as u8,
                        is_sanma: true,
                        num_players: NP as u8,
                    };

                    let calc =
                        crate::hand_evaluator_3p::HandEvaluator3P::new(hand.clone(), melds.clone());
                    let ura_indicators = if self.players[w_pid as usize].riichi_declared {
                        self._get_ura_indicators()
                    } else {
                        vec![]
                    };
                    let mut res = calc.calc(
                        win_tile,
                        self.wall.dora_indicators.clone(),
                        ura_indicators,
                        Some(cond),
                    );

                    // Cap double yakuman patterns when not enabled per rule flags
                    if res.yakuman && res.han > 13 {
                        let mut cap = 0u32;
                        for &y in &res.yaku {
                            match y {
                                47 if !self.rule.is_junsei_chuurenpoutou_double => cap += 13,
                                48 if !self.rule.is_suuankou_tanki_double => cap += 13,
                                49 if !self.rule.is_kokushi_musou_13machi_double => cap += 13,
                                50 if !self.rule.is_daisuushii_double => cap += 13,
                                _ => {}
                            }
                        }
                        if cap > 0 {
                            res.han = res.han.saturating_sub(cap).max(13);
                            let capped = crate::score::calculate_score(
                                res.han as u8,
                                0,
                                w_pid == self.oya,
                                false,
                                ron_honba,
                                NP as u8,
                            );
                            res.ron_agari = capped.pay_ron;
                            res.tsumo_agari_oya = capped.pay_tsumo_oya;
                            res.tsumo_agari_ko = capped.pay_tsumo_ko;
                        }
                    }

                    if res.is_win {
                        let score = res.ron_agari as i32;
                        let mut pao_payer = target_pid;
                        let mut pao_amt = 0;

                        if res.yakuman {
                            let mut has_pao = false;
                            let mut total_yakuman_val = 0i32;
                            let mut pao_yakuman_val = 0i32;
                            for &yid in &res.yaku {
                                let val: i32 = match yid {
                                    47 if self.rule.is_junsei_chuurenpoutou_double => 2,
                                    48 if self.rule.is_suuankou_tanki_double => 2,
                                    49 if self.rule.is_kokushi_musou_13machi_double => 2,
                                    50 if self.rule.is_daisuushii_double => 2,
                                    _ => 1,
                                };
                                total_yakuman_val += val;
                                if let Some(liable) =
                                    self.players[w_pid as usize].pao.get(&(yid as u8))
                                {
                                    has_pao = true;
                                    pao_payer = *liable;
                                    pao_yakuman_val += val;
                                }
                            }
                            if has_pao {
                                // Ron with PAO: split between PAO player and deal-in player.
                                // yakuman_pao_is_liability_only controls the split base:
                                //   true  (MjSoul): only PAO-triggering yakuman portion split 50/50
                                //   false (Tenhou): total yakuman split 50/50
                                let is_oya = w_pid == self.oya;
                                let unit: i32 = if is_oya { 48000 } else { 32000 };
                                let honba_ron = ron_honba as i32 * (NP as i32 - 1) * 100;
                                let split_base = if self.rule.yakuman_pao_is_liability_only {
                                    pao_yakuman_val * unit
                                } else {
                                    total_yakuman_val * unit
                                };
                                pao_amt = (split_base / 2 + honba_ron) as usize;
                            }
                        }

                        let mut this_deltas = vec![0i32; NP];
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
                        for (&yid, &liable) in &self.players[w_pid as usize].pao {
                            if val.yaku.contains(&(yid as u32)) {
                                val.pao_payer = Some(liable);
                                break;
                            }
                        }
                        self.win_results.insert(w_pid, val);

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
                            if self.players[w_pid as usize].riichi_declared {
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

                for (i, p) in self.players.iter_mut().enumerate() {
                    p.score += total_deltas[i];
                    p.score_delta = total_deltas[i];
                }

                self._initialize_next_round(oya_won, false);
            } else if let Some((claimer, action)) = call_claim {
                self._accept_riichi();
                self.is_rinshan_flag = false;
                self.is_first_turn = false;
                self.players[claimer as usize].missed_agari_doujun = false;

                // Discard was called → discarder loses nagashi eligibility
                if let Some((discarder_pid, _)) = self.last_discard {
                    self.players[discarder_pid as usize].nagashi_eligible = false;
                }

                for p in 0..NP {
                    self.players[p].ippatsu_cycle = false;
                }

                if action.action_type == ActionType::Daiminkan {
                    self.current_player = claimer;
                    self.active_players = vec![claimer];
                    self.players[claimer as usize].forbidden_discards.clear();
                    self._resolve_kan(claimer, action.clone());
                    return;
                }

                for &t in &action.consume_tiles {
                    if let Some(idx) = self.players[claimer as usize]
                        .hand
                        .iter()
                        .position(|&x| x == t)
                    {
                        self.players[claimer as usize].hand.remove(idx);
                    }
                }
                let (discarder, tile) = self.last_discard.unwrap();
                let mut tiles = action.consume_tiles.clone();
                tiles.push(tile);
                tiles.sort();
                let meld_type = match action.action_type {
                    ActionType::Pon => MeldType::Pon,
                    _ => MeldType::Pon,
                };
                self.players[claimer as usize].melds.push(Meld {
                    meld_type,
                    tiles: tiles.clone(),
                    opened: true,
                    from_who: discarder as i8,
                    called_tile: Some(tile),
                });

                if !self.skip_mjai_logging {
                    let type_str = match action.action_type {
                        ActionType::Pon => Some("pon"),
                        ActionType::Daiminkan => Some("daiminkan"),
                        _ => None,
                    };
                    if let Some(s) = type_str {
                        let mut ev = serde_json::Map::new();
                        ev.insert("type".to_string(), Value::String(s.to_string()));
                        ev.insert("actor".to_string(), Value::Number(claimer.into()));
                        ev.insert("target".to_string(), Value::Number(discarder.into()));
                        ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
                        let cons_strs: Vec<String> = action
                            .consume_tiles
                            .iter()
                            .map(|&t| tid_to_mjai(t))
                            .collect();
                        ev.insert(
                            "consumed".to_string(),
                            serde_json::to_value(cons_strs).unwrap(),
                        );
                        self._push_mjai_event(Value::Object(ev));
                    }
                }

                // PAO implementation
                if meld_type == MeldType::Pon
                    || meld_type == MeldType::Daiminkan
                    || meld_type == MeldType::Kakan
                {
                    let tile_val = tile / 4;
                    if (31..=33).contains(&tile_val) {
                        let dragon_melds = self.players[claimer as usize]
                            .melds
                            .iter()
                            .filter(|m| {
                                let t = m.tiles[0] / 4;
                                (31..=33).contains(&t) && (m.meld_type != MeldType::Chi)
                            })
                            .count();
                        if dragon_melds == 3 {
                            self.players[claimer as usize].pao.insert(37, discarder);
                        }
                    } else if (27..=30).contains(&tile_val) {
                        let wind_melds = self.players[claimer as usize]
                            .melds
                            .iter()
                            .filter(|m| {
                                let t = m.tiles[0] / 4;
                                (27..=30).contains(&t) && (m.meld_type != MeldType::Chi)
                            })
                            .count();
                        if wind_melds == 4 {
                            self.players[claimer as usize].pao.insert(50, discarder);
                        }
                    }
                }

                self.current_player = claimer;
                self.phase = Phase::WaitAct;
                self.active_players = vec![claimer];
                self.players[claimer as usize].forbidden_discards.clear();

                if action.action_type == ActionType::Pon {
                    self.players[claimer as usize].forbidden_discards.push(tile);
                }

                if action.action_type == ActionType::Daiminkan {
                    self._resolve_kan(claimer, action.clone());
                } else {
                    self.needs_tsumo = false;
                    self.drawn_tile = None;
                }
            } else {
                // All Pass
                self.current_claims.clear();
                self.active_players.clear();

                if let Some((pk_pid, pk_act)) = self.pending_kan.take() {
                    if pk_act.action_type == ActionType::Kita {
                        // All players passed on kita ron — break ippatsu now
                        for p in &mut self.players {
                            p.ippatsu_cycle = false;
                        }
                        self.resolve_kita_rinshan(pk_pid);
                    } else {
                        self._resolve_kan(pk_pid, pk_act);
                    }
                } else {
                    self._accept_riichi();
                    self.turn_count += 1;
                    self.current_player = (self.current_player + 1) % NP as u8;
                    self._deal_next();
                    if self.turn_count >= NP as u32 {
                        self.is_first_turn = false;
                    }
                }
            }
        }
    }

    fn _resolve_discard(&mut self, pid: u8, tile: u8, tsumogiri: bool) {
        // A normal discard is never chankan, so clear any stale pending_kan
        // to prevent false chankan detection on subsequent ron claims.
        self.pending_kan = None;

        // After a discard the rinshan context is over. Clearing here ensures
        // that houtei (last-discard win) is correctly detected even when the
        // discard comes after a kan draw.
        self.is_rinshan_flag = false;

        // Clear ippatsu for the discarding player. When a riichi player discards
        // without tsumo winning, their ippatsu window is over. Note: the riichi
        // declaration discard won't wrongly clear it because _accept_riichi() runs
        // AFTER this and sets ippatsu_cycle = true.
        self.players[pid as usize].ippatsu_cycle = false;
        self.players[pid as usize].discards.push(tile);
        self.last_discard = Some((pid, tile));
        self.drawn_tile = None;
        self.players[pid as usize]
            .discard_from_hand
            .push(!tsumogiri);
        let riichi_stage = self.players[pid as usize].riichi_stage;
        self.players[pid as usize]
            .discard_is_riichi
            .push(riichi_stage);

        if !tsumogiri {
            self.last_tedashis[pid as usize] = Some(tile);
        }

        self.needs_tsumo = true;

        if self.players[pid as usize].riichi_stage {
            self.players[pid as usize].riichi_declared = true;
            if self.is_first_turn {
                self.players[pid as usize].double_riichi_declared = true;
            }
            self.players[pid as usize].riichi_declaration_index =
                Some(self.players[pid as usize].discards.len() - 1);
            self.players[pid as usize].riichi_stage = false;
            self.riichi_pending_acceptance = Some(pid);
        }

        // Reveal pending kan doras before dahai event
        while self.wall.pending_kan_dora_count > 0 {
            self.wall.pending_kan_dora_count -= 1;
            self._reveal_kan_dora();
        }

        if !self.skip_mjai_logging {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("dahai".to_string()));
            ev.insert("actor".to_string(), Value::Number(pid.into()));
            ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
            ev.insert("tsumogiri".to_string(), Value::Bool(tsumogiri));
            self._push_mjai_event(Value::Object(ev));
        }

        self.players[pid as usize].missed_agari_doujun = false;
        self.players[pid as usize].nagashi_eligible &= crate::types::is_terminal_tile(tile);

        self.current_claims.clear();
        self.active_players.clear();
        let mut has_claims = false;
        let mut claim_active = Vec::new();

        for i in 0..NP as u8 {
            if i == pid {
                continue;
            }
            let (legals, missed_agari) = self._get_claim_actions_for_player(i, pid, tile);
            if missed_agari {
                self.players[i as usize].missed_agari_doujun = true;
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
            if let Some(_rp) = self.riichi_pending_acceptance {
                self._accept_riichi();
            }
            if !self.check_abortive_draw() {
                self.turn_count += 1;
                self.current_player = (pid + 1) % NP as u8;
                self._deal_next();
                if self.turn_count >= NP as u32 {
                    self.is_first_turn = false;
                }
            }
        }
    }

    pub fn _resolve_kan(&mut self, pid: u8, action: Action) {
        let p_idx = pid as usize;
        if action.action_type == ActionType::Kakan {
            // Already updated in step()
        } else {
            for &t in &action.consume_tiles {
                if let Some(idx) = self.players[p_idx].hand.iter().position(|&x| x == t) {
                    self.players[p_idx].hand.remove(idx);
                }
            }
            let (m_type, tiles, from_who, ct) = if action.action_type == ActionType::Ankan {
                (MeldType::Ankan, action.consume_tiles.clone(), -1i8, None)
            } else {
                let (discarder, tile) = self.last_discard.unwrap();
                let mut t_vec = action.consume_tiles.clone();
                t_vec.push(tile);
                t_vec.sort();
                (MeldType::Daiminkan, t_vec, discarder as i8, Some(tile))
            };
            self.players[p_idx].melds.push(Meld {
                meld_type: m_type,
                tiles,
                opened: m_type == MeldType::Daiminkan,
                from_who,
                called_tile: ct,
            });

            // PAO check for Daiminkan
            if action.action_type == ActionType::Daiminkan {
                let (discarder, tile) = self.last_discard.unwrap();
                let tile_val = tile / 4;
                if (31..=33).contains(&tile_val) {
                    let dragon_melds = self.players[p_idx]
                        .melds
                        .iter()
                        .filter(|m| {
                            let t = m.tiles[0] / 4;
                            (31..=33).contains(&t) && (m.meld_type != MeldType::Chi)
                        })
                        .count();
                    if dragon_melds == 3 {
                        self.players[p_idx].pao.insert(37, discarder);
                    }
                } else if (27..=30).contains(&tile_val) {
                    let wind_melds = self.players[p_idx]
                        .melds
                        .iter()
                        .filter(|m| {
                            let t = m.tiles[0] / 4;
                            (27..=30).contains(&t) && (m.meld_type != MeldType::Chi)
                        })
                        .count();
                    if wind_melds == 4 {
                        self.players[p_idx].pao.insert(50, discarder);
                    }
                }
            }
        }

        self.is_first_turn = false;
        for p in &mut self.players {
            p.ippatsu_cycle = false;
        }

        if self.wall.tiles.len() > 14 {
            let t = self.wall.tiles.remove(0);
            self.players[p_idx].hand.push(t);
            self.drawn_tile = Some(t);
            self.wall.rinshan_draw_count += 1;
            self.is_rinshan_flag = true;

            if !self.skip_mjai_logging {
                let m_type = match action.action_type {
                    ActionType::Ankan => Some("ankan"),
                    ActionType::Daiminkan => Some("daiminkan"),
                    ActionType::Kakan => None,
                    _ => None,
                };
                if let Some(s) = m_type {
                    let mut ev = serde_json::Map::new();
                    ev.insert("type".to_string(), Value::String(s.to_string()));
                    ev.insert("actor".to_string(), Value::Number(pid.into()));
                    if action.action_type == ActionType::Ankan {
                        let tile = action.tile.unwrap_or_else(|| action.consume_tiles[0]);
                        ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
                    } else if action.action_type == ActionType::Daiminkan {
                        if let Some((target, tile)) = self.last_discard {
                            ev.insert("target".to_string(), Value::Number(target.into()));
                            ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
                        }
                    }
                    let cons_strs: Vec<String> = action
                        .consume_tiles
                        .iter()
                        .map(|&t| tid_to_mjai(t))
                        .collect();
                    ev.insert(
                        "consumed".to_string(),
                        serde_json::to_value(cons_strs).unwrap(),
                    );
                    self._push_mjai_event(Value::Object(ev));
                }
            }

            // Reveal any pending doras from previous kans
            while self.wall.pending_kan_dora_count > 0 {
                self.wall.pending_kan_dora_count -= 1;
                self._reveal_kan_dora();
            }

            // Ankan: always reveal dora immediately (before rinshan tsumo)
            // Daiminkan/Kakan: defer dora reveal to after discard
            if action.action_type == ActionType::Ankan {
                self._reveal_kan_dora();
            } else {
                self.wall.pending_kan_dora_count += 1;
            }

            if !self.skip_mjai_logging {
                let mut t_ev = serde_json::Map::new();
                t_ev.insert("type".to_string(), Value::String("tsumo".to_string()));
                t_ev.insert("actor".to_string(), Value::Number(pid.into()));
                t_ev.insert("pai".to_string(), Value::String(tid_to_mjai(t)));
                self._push_mjai_event(Value::Object(t_ev));
            }
            self.phase = Phase::WaitAct;
            self.active_players = vec![pid];
        }
    }

    fn _accept_riichi(&mut self) {
        if let Some(p) = self.riichi_pending_acceptance {
            self.players[p as usize].score -= 1000;
            self.players[p as usize].score_delta -= 1000;
            self.riichi_sticks += 1;
            self.players[p as usize].riichi_declared = true;
            self.players[p as usize].ippatsu_cycle = true;
            if !self.skip_mjai_logging {
                let mut ev = serde_json::Map::new();
                ev.insert(
                    "type".to_string(),
                    Value::String("reach_accepted".to_string()),
                );
                ev.insert("actor".to_string(), Value::Number(p.into()));
                self._push_mjai_event(Value::Object(ev));
            }
            self.riichi_pending_acceptance = None;
        }
    }

    pub fn _deal_next(&mut self) {
        self.is_rinshan_flag = false;
        if self.wall.tiles.len() <= 14 {
            self._trigger_ryukyoku("exhaustive_draw");
            return;
        }
        if let Some(t) = self.wall.tiles.pop() {
            let pid = self.current_player;
            self.players[pid as usize].hand.push(t);
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
            self.players[pid as usize].forbidden_discards.clear();
        }
    }

    pub fn _initialize_next_round(&mut self, oya_won: bool, is_draw: bool) {
        if self.is_done {
            return;
        }

        let np: u8 = NP as u8;

        if self.players.iter().any(|p| p.score < 0) {
            self._process_end_game();
            return;
        }

        let mut next_honba = self.honba;
        let mut next_oya = self.oya;
        let mut next_round_wind = self.round_wind;

        if oya_won {
            next_honba = next_honba.saturating_add(1);
        } else if is_draw {
            next_honba = next_honba.saturating_add(1);
            next_oya = (next_oya + 1) % np;
            if next_oya == 0 {
                next_round_wind += 1;
            }
        } else {
            next_honba = 0;
            next_oya = (next_oya + 1) % np;
            if next_oya == 0 {
                next_round_wind += 1;
            }
        }

        match self.game_mode {
            4 => {
                // 3p-red-east
                let max_score = self.players.iter().map(|p| p.score).max().unwrap_or(0);
                if next_round_wind >= 1 && (max_score >= 30000 || next_round_wind > 1) {
                    self._process_end_game();
                    return;
                }
            }
            5 => {
                // 3p-red-half
                let max_score = self.players.iter().map(|p| p.score).max().unwrap_or(0);
                if next_round_wind >= 2 && (max_score >= 30000 || next_round_wind > 2) {
                    self._process_end_game();
                    return;
                }
            }
            3 => {
                // 3p-red-single
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

        let next_scores: Vec<i32> = self.players.iter().map(|p| p.score).collect();
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
        round_wind: u8,
        honba: u8,
        kyotaku: u32,
        wall: Option<Vec<u8>>,
        scores: Option<Vec<i32>>,
    ) {
        self.oya = oya;
        self.kyoku_idx = oya;
        self.current_player = oya;
        self.honba = honba;
        self.riichi_sticks = kyotaku;
        self.round_wind = round_wind;

        for p in &mut self.players {
            p.reset_round();
        }
        self.is_done = false;
        self.current_claims = HashMap::new();
        self.pending_kan = None;
        self.is_rinshan_flag = false;
        self.wall.rinshan_draw_count = 0;
        self.wall.pending_kan_dora_count = 0;
        self.is_first_turn = true;
        self.riichi_pending_acceptance = None;
        self.turn_count = 0;
        self.needs_tsumo = true;
        self.needs_initialize_next_round = false;
        self.pending_oya_won = false;
        self.pending_is_draw = false;
        self.last_discard = None;
        self.win_results.clear();
        self.last_win_results.clear();
        self.round_end_scores = None;
        self.riichi_sutehais = [None; NP];
        self.last_tedashis = [None; NP];

        if let Some(s) = scores {
            for (i, &sc) in s.iter().enumerate() {
                if i < NP {
                    self.players[i].score = sc;
                }
            }
        }

        if let Some(w) = wall {
            self.wall.load_wall(w);
        } else {
            self.wall.shuffle();
        }

        // Deal logic
        for _ in 0..3 {
            for idx in 0..NP {
                let p = (idx + oya as usize) % NP;
                for _ in 0..4 {
                    if let Some(t) = self.wall.tiles.pop() {
                        self.players[p].hand.push(t);
                    }
                }
            }
        }
        for idx in 0..NP {
            let p = (idx + oya as usize) % NP;
            if let Some(t) = self.wall.tiles.pop() {
                self.players[p].hand.push(t);
            }
        }
        for p in &mut self.players {
            p.hand.sort();
        }

        if !self.skip_mjai_logging {
            let wind_str = match round_wind % 4 {
                0 => "E",
                1 => "S",
                2 => "W",
                _ => "N",
            };
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("start_kyoku".to_string()));
            ev.insert("bakaze".to_string(), Value::String(wind_str.to_string()));
            ev.insert("kyoku".to_string(), Value::Number((oya + 1).into()));
            ev.insert("honba".to_string(), Value::Number(honba.into()));
            ev.insert("kyotaku".to_string(), Value::Number(kyotaku.into()));
            ev.insert("oya".to_string(), Value::Number(oya.into()));
            let scores_vec: Vec<i32> = self.players.iter().map(|p| p.score).collect();
            ev.insert(
                "scores".to_string(),
                serde_json::to_value(scores_vec).unwrap(),
            );
            ev.insert(
                "dora_marker".to_string(),
                Value::String(tid_to_mjai(self.wall.dora_indicators[0])),
            );
            let mut tehais = Vec::new();
            for p in &self.players {
                let hand_strs: Vec<String> = p.hand.iter().map(|&t| tid_to_mjai(t)).collect();
                tehais.push(hand_strs);
            }
            ev.insert("tehais".to_string(), serde_json::to_value(tehais).unwrap());
            self._push_mjai_event(Value::Object(ev));
        }

        self.current_player = self.oya;
        self.phase = Phase::WaitAct;
        self.active_players = vec![self.oya];

        if let Some(t) = self.wall.tiles.pop() {
            self.players[self.oya as usize].hand.push(t);
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
            self.needs_tsumo = true;
            self.drawn_tile = None;
        }
    }

    pub fn _trigger_ryukyoku(&mut self, reason: &str) {
        self._accept_riichi();

        let mut tenpai = [false; NP];
        let mut final_reason = reason.to_string();
        let mut nagashi_winners = Vec::new();

        if reason == "exhaustive_draw" {
            for (i, p) in self.players.iter().enumerate() {
                let calc =
                    crate::hand_evaluator_3p::HandEvaluator3P::new(p.hand.clone(), p.melds.clone());
                if calc.is_tenpai() {
                    tenpai[i] = true;
                }
            }
            for (i, p) in self.players.iter().enumerate() {
                if p.nagashi_eligible {
                    nagashi_winners.push(i as u8);
                }
            }

            if !nagashi_winners.is_empty() {
                final_reason = "nagashimangan".to_string();
                // Apply mangan tsumo payment for each nagashi winner (no honba)
                for &w in &nagashi_winners {
                    let is_oya = w == self.oya;
                    let score_res = crate::score::calculate_score(5, 30, is_oya, true, 0, NP as u8);
                    if is_oya {
                        for i in 0..NP {
                            if i as u8 != w {
                                self.players[i].score -= score_res.pay_tsumo_ko as i32;
                                self.players[i].score_delta -= score_res.pay_tsumo_ko as i32;
                                self.players[w as usize].score += score_res.pay_tsumo_ko as i32;
                                self.players[w as usize].score_delta +=
                                    score_res.pay_tsumo_ko as i32;
                            }
                        }
                    } else {
                        for i in 0..NP {
                            if i as u8 != w {
                                let pay = if i as u8 == self.oya {
                                    score_res.pay_tsumo_oya as i32
                                } else {
                                    score_res.pay_tsumo_ko as i32
                                };
                                self.players[i].score -= pay;
                                self.players[i].score_delta -= pay;
                                self.players[w as usize].score += pay;
                                self.players[w as usize].score_delta += pay;
                            }
                        }
                    }
                }
            } else {
                let tenpai_pool = game_mode::tenpai_pool();
                let num_tp = tenpai.iter().filter(|&&t| t).count();
                if num_tp > 0 && num_tp < NP {
                    let pk = tenpai_pool / num_tp as i32;
                    let pn = tenpai_pool / (NP - num_tp) as i32;
                    for (i, tp) in tenpai.iter().enumerate() {
                        let delta = if *tp { pk } else { -pn };
                        self.players[i].score += delta;
                        self.players[i].score_delta = delta;
                    }
                }
            }
        } else if let Some(stripped) = reason.strip_prefix("Error: Illegal Action by Player ") {
            if let Ok(pid) = stripped.parse::<usize>() {
                if pid < NP {
                    let is_offender_oya = (pid as u8) == self.oya;
                    if is_offender_oya {
                        let penalty = 4000 * (NP as i32 - 1);
                        let each_get = penalty / (NP as i32 - 1);
                        for i in 0..NP {
                            if i == pid {
                                self.players[i].score -= penalty;
                                self.players[i].score_delta = -penalty;
                            } else {
                                self.players[i].score += each_get;
                                self.players[i].score_delta = each_get;
                            }
                        }
                    } else {
                        let total_penalty = 4000 + 2000 * (NP as i32 - 2);
                        for i in 0..NP {
                            if i == pid {
                                self.players[i].score -= total_penalty;
                                self.players[i].score_delta = -total_penalty;
                            } else if (i as u8) == self.oya {
                                self.players[i].score += 4000;
                                self.players[i].score_delta = 4000;
                            } else {
                                self.players[i].score += 2000;
                                self.players[i].score_delta = 2000;
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
            true
        };

        if !self.skip_mjai_logging {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("ryukyoku".to_string()));
            ev.insert("reason".to_string(), Value::String(final_reason.clone()));
            let deltas: Vec<i32> = self.players.iter().map(|p| p.score_delta).collect();
            ev.insert("deltas".to_string(), serde_json::to_value(deltas).unwrap());
            self._push_mjai_event(Value::Object(ev));
        }

        self._initialize_next_round(is_renchan, true);
    }

    fn check_abortive_draw(&mut self) -> bool {
        // 1. Sufuurenta (Four Winds) - disabled in 3P
        // Sufuurenta requires all 4 players to discard the same wind tile.
        // With only 3 players this rule does not apply (MjSoul 3P confirmed).

        // 2. Suukansansen (4 Kans)
        let mut kan_owners = Vec::new();
        for (pid, p) in self.players.iter().enumerate() {
            for m in &p.melds {
                if m.meld_type == MeldType::Daiminkan
                    || m.meld_type == MeldType::Ankan
                    || m.meld_type == MeldType::Kakan
                {
                    kan_owners.push(pid);
                }
            }
        }

        if kan_owners.len() == 4 {
            let first_owner = kan_owners[0];
            if !kan_owners.iter().all(|&o| o == first_owner) {
                self._trigger_ryukyoku("suukansansen");
                return true;
            }
        }

        // 3. Suucha Riichi (All Riichis) - disabled in 3P
        // Suucha riichi requires all 4 players to declare riichi.
        // With only 3 players this rule does not apply (MjSoul 3P confirmed).

        false
    }

    pub fn _reveal_kan_dora(&mut self) {
        let count = self.wall.dora_indicators.len();
        if count < 5 {
            self.wall
                .dora_indicators
                .push(self.wall.dora_indicator_tiles[count]);
            if !self.skip_mjai_logging {
                let mut ev = serde_json::Map::new();
                ev.insert("type".to_string(), Value::String("dora".to_string()));
                ev.insert(
                    "dora_marker".to_string(),
                    Value::String(tid_to_mjai(
                        self.wall.dora_indicators.last().copied().unwrap(),
                    )),
                );
                self._push_mjai_event(Value::Object(ev));
            }
        }
    }

    pub fn _get_ura_markers(&self) -> Vec<String> {
        let mut markers = Vec::new();
        for i in 0..self.wall.dora_indicators.len() {
            markers.push(tid_to_mjai(self.wall.ura_indicator_tiles[i]));
        }
        markers
    }

    fn _get_ura_indicators(&self) -> Vec<u8> {
        let mut indicators = Vec::new();
        for i in 0..self.wall.dora_indicators.len() {
            indicators.push(self.wall.ura_indicator_tiles[i]);
        }
        indicators
    }

    pub(crate) fn _process_end_game(&mut self) {
        self.is_done = true;
        if !self.skip_mjai_logging {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("end_game".to_string()));
            self._push_mjai_event(Value::Object(ev));
        }
    }

    pub fn apply_mjai_event(&mut self, event: MjaiEvent) {
        <Self as GameState3PEventHandler>::apply_mjai_event(self, event)
    }

    pub fn apply_log_action(&mut self, action: &LogAction) {
        <Self as GameState3PEventHandler>::apply_log_action(self, action)
    }
}

impl GameState3P {
    pub fn _push_mjai_event(&mut self, event: Value) {
        if self.skip_mjai_logging {
            return;
        }
        let json_str = serde_json::to_string(&event).unwrap();
        self.mjai_log.push(json_str.clone());

        let type_str = event["type"].as_str().unwrap_or("");
        let actor = event["actor"].as_u64().map(|a| a as usize);

        for pid in 0..NP {
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
}
