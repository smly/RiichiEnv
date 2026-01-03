#![allow(clippy::useless_conversion)]
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyListMethods};
use pyo3::{pyclass, pymethods, IntoPy, PyErr, PyObject, PyResult, Python, ToPyObject};
// IntoPy might be needed for .into_py() calls if I revert?
// I used .to_object() which needs ToPyObject.
use rand::prelude::*;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use crate::parser::tid_to_mjai;
use crate::types::{Agari, Conditions, Meld, MeldType, Wind};

// --- Enums ---

#[pyclass(eq, eq_int)]
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Phase {
    WaitAct = 0,
    WaitResponse = 1,
}

#[pymethods]
impl Phase {
    fn __hash__(&self) -> i32 {
        *self as i32
    }
}

#[pyclass(eq, eq_int)]
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionType {
    Discard = 0,
    Chi = 1,
    Pon = 2,
    Daiminkan = 3,
    Ron = 4,
    Riichi = 5,
    Tsumo = 6,
    Pass = 7,
    Ankan = 8,
    Kakan = 9,
    KyushuKyuhai = 10,
}

#[pymethods]
impl ActionType {
    fn __hash__(&self) -> i32 {
        *self as i32
    }
}

// --- Structs ---

#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    #[pyo3(get, set, name = "type")]
    pub action_type: ActionType,
    #[pyo3(get, set)]
    pub tile: Option<u8>,
    #[pyo3(get, set)]
    pub consume_tiles: Vec<u8>,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Observation {
    #[pyo3(get)]
    pub player_id: u8,
    #[pyo3(get)]
    pub hand: Vec<u8>,
    pub events_json: Vec<String>,
    #[pyo3(get)]
    pub prev_events_size: usize,
    #[pyo3(get)]
    pub legal_actions: Vec<Action>,
}

// ... impl Observation ...

#[pyclass]
#[derive(Debug, Clone)]
pub struct RiichiEnv {
    // Game State
    #[pyo3(get)]
    pub wall: Vec<u8>,
    #[pyo3(get)]
    pub hands: [Vec<u8>; 4],
    #[pyo3(get, set)]
    pub melds: [Vec<Meld>; 4],
    #[pyo3(get, set)]
    pub discards: [Vec<u8>; 4],
    #[pyo3(get, set)]
    pub current_player: u8,
    #[pyo3(get, set)]
    pub turn_count: u32,
    #[pyo3(get, set)]
    pub is_done: bool,
    #[pyo3(get, set)]
    pub needs_tsumo: bool,
    #[pyo3(get, set)]
    pub scores: [i32; 4],
    #[pyo3(get, set)]
    pub score_deltas: [i32; 4],
    // riichi_sticks: u32 vs u8 mismatch in error logs?
    // Error log 1: riichi_sticks: u8 in new() vs u32 here?
    // Let's check reset uses u32 for kyotaku.
    // Python usually treats as int.
    #[pyo3(get, set)]
    pub riichi_sticks: u32,
    #[pyo3(get, set)]
    pub riichi_declared: [bool; 4],
    #[pyo3(get, set)]
    pub riichi_stage: [bool; 4],
    #[pyo3(get, set)]
    pub double_riichi_declared: [bool; 4],

    // Phases
    #[pyo3(get, set)]
    pub phase: Phase,
    #[pyo3(get, set)]
    pub active_players: Vec<u8>,
    #[pyo3(get, set)]
    pub last_discard: Option<(u8, u8)>,

    #[pyo3(get, set)]
    pub current_claims: HashMap<u8, Vec<Action>>,

    // pending_kan tuple?
    // ...

    // ...
    #[pyo3(get, set)]
    pub pending_kan: Option<Action>,

    #[pyo3(get, set)]
    pub oya: u8,
    #[pyo3(get, set)]
    pub honba: u8, // Added
    #[pyo3(get, set)]
    pub kyoku_idx: u8,
    #[pyo3(get)]
    pub round_wind: u8,
    // ...
    #[pyo3(get, set)]
    pub dora_indicators: Vec<u8>,
    #[pyo3(get, set)]
    pub rinshan_draw_count: u8,
    #[pyo3(get, set)]
    pub pending_kan_dora_count: u8,

    #[pyo3(get, set)]
    pub is_rinshan_flag: bool,
    #[pyo3(get, set)]
    pub is_first_turn: bool,
    #[pyo3(get, set)]
    pub missed_agari_riichi: [bool; 4],
    #[pyo3(get, set)]
    pub missed_agari_doujun: [bool; 4],
    #[pyo3(get, set)]
    pub riichi_pending_acceptance: Option<u8>,
    #[pyo3(get, set)]
    pub nagashi_eligible: [bool; 4],
    #[pyo3(get, set)]
    pub drawn_tile: Option<u8>,

    #[pyo3(get)]
    pub wall_digest: String,
    #[pyo3(get)]
    pub salt: String,
    #[pyo3(get)]
    pub agari_results: HashMap<u8, Agari>,
    #[pyo3(get)]
    pub last_agari_results: HashMap<u8, Agari>,
    #[pyo3(get)]
    pub round_end_scores: Option<[i32; 4]>,

    pub mjai_log: Vec<String>,
    #[pyo3(get)]
    pub player_event_counts: [usize; 4],

    // Config
    #[pyo3(get)]
    pub game_type: u8,
    #[pyo3(get)]
    pub mjai_mode: bool,
    #[pyo3(get)]
    pub seed: Option<u64>,
}

impl RiichiEnv {
    fn _trigger_ryukyoku(&mut self, reason: &str) {
        self._accept_riichi(); // Ensure riichi sticks are collected if pending
        let mut tenpai = [false; 4];
        if reason == "exhaustive_draw" {
            for (i, tp) in tenpai.iter_mut().enumerate() {
                let hand = &self.hands[i];
                let melds = &self.melds[i];
                let calc =
                    crate::agari_calculator::AgariCalculator::new(hand.clone(), melds.clone());
                if calc.is_tenpai() {
                    *tp = true;
                }
            }

            // Nagashi Mangan
            let mut nagashi_winners = Vec::new();
            for (i, &eligible) in self.nagashi_eligible.iter().enumerate() {
                if eligible {
                    nagashi_winners.push(i as u8);
                }
            }

            if !nagashi_winners.is_empty() {
                for &winner in &nagashi_winners {
                    let is_oya = winner == self.oya;
                    let mut deltas = [0; 4];
                    if is_oya {
                        for (i, d) in deltas.iter_mut().enumerate() {
                            *d = if i as u8 == winner { 12000 } else { -4000 };
                        }
                    } else {
                        for (i, d) in deltas.iter_mut().enumerate() {
                            if i as u8 == winner {
                                *d = 8000;
                            } else if i as u8 == self.oya {
                                *d = -4000;
                            } else {
                                *d = -2000;
                            }
                        }
                    }
                    for (i, d) in deltas.iter().enumerate() {
                        self.scores[i] += d;
                        self.score_deltas[i] += d;
                    }
                }
                let mut ev = serde_json::Map::new();
                ev.insert("type".to_string(), Value::String("ryukyoku".to_string()));
                ev.insert(
                    "reason".to_string(),
                    Value::String("nagashimangan".to_string()),
                );
                self.mjai_log.push(Value::Object(ev).to_string());
            } else {
                let num_tp = tenpai.iter().filter(|&&t| t).count();
                if (1..=3).contains(&num_tp) {
                    let pk = 3000 / num_tp as i32;
                    let pn = 3000 / (4 - num_tp) as i32;
                    for (i, tp) in tenpai.iter().enumerate() {
                        let delta = if *tp { pk } else { -pn };
                        self.scores[i] += delta;
                        self.score_deltas[i] += delta;
                    }
                }
            }
        }

        let mut ev = serde_json::Map::new();
        ev.insert("type".to_string(), Value::String("ryukyoku".to_string()));
        ev.insert("reason".to_string(), Value::String(reason.to_string()));
        self.mjai_log.push(Value::Object(ev).to_string());

        let mut is_renchan = false;
        if reason == "exhaustive_draw" {
            is_renchan = tenpai[self.oya as usize];
        } else if ["kyushu_kyuhai", "suurechi", "suukansansen", "sufuurenta"].contains(&reason) {
            is_renchan = true;
        }

        self._end_kyoku_ryukyoku(is_renchan, true);
    }

    fn _end_kyoku_win(
        &mut self,
        winners: Vec<u8>,
        is_tsumo: bool,
        loser: Option<u8>,
        agaris: HashMap<u8, Agari>,
    ) {
        self.agari_results = agaris;
        let mut total_deltas = [0; 4];

        // Calculate deltas for all winners
        for &w in &winners {
            if let Some(agari) = self.agari_results.get(&w).cloned() {
                let d = self._calculate_deltas(&agari, w, is_tsumo, loser, true);
                for i in 0..4 {
                    total_deltas[i] += d[i];
                }
            }
        }

        for (i, d) in total_deltas.iter().enumerate() {
            self.scores[i] += d;
            self.score_deltas[i] += d;
        }

        self.last_agari_results = self.agari_results.clone();
        self.round_end_scores = Some(self.scores);
        self.is_done = true;
    }

    fn _end_kyoku_ryukyoku(&mut self, is_renchan: bool, is_draw: bool) {
        let mut ev = serde_json::Map::new();
        ev.insert("type".to_string(), Value::String("end_kyoku".to_string()));
        self.mjai_log.push(Value::Object(ev).to_string());

        // Simple Game Over check (match OneKyokuRule behavior if that's the default)
        // For now, if game_type is IKKYOKU, we finish.
        // Actually, Python's is_game_over(OneKyokuRule) returns True always.
        if self.game_type == 0 || self.game_type == 1 {
            // Assuming 0, 1 are IKKYOKU
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("end_game".to_string()));
            self.mjai_log.push(Value::Object(ev).to_string());
            self.is_done = true;
            return;
        }

        // Standard Transition logic (Simplified)
        let next_honba = if is_renchan || is_draw {
            self.honba + 1
        } else {
            0
        };
        let next_oya = if is_renchan {
            self.oya
        } else {
            (self.oya + 1) % 4
        };

        // If oya moved and next_oya is 0, bakaze changes (simplified)
        // ... (This needs a bakaze field which I don't see in struct yet,
        // oh wait there is oya, but where is bakaze?)
        // Ah, current Python implementation of get_next_kyoku_params uses round_wind.
        // I need to add that to the struct.

        self.honba = next_honba;
        self.oya = next_oya;
        self.is_done = true; // For now, end kyoku just ends the loop in step()
                             // But next round should be initialized.
                             // Python calls _initialize_round.
                             // Since I'm essentially porting functionality, let's keep is_done = true for now
                             // and let the user decide how to handle multiple kyokus in one session if they want.
                             // Existing tests usually use one Kyoku per reset().
    }

    fn _accept_riichi(&mut self) {
        if let Some(p) = self.riichi_pending_acceptance {
            self.scores[p as usize] -= 1000;
            self.score_deltas[p as usize] -= 1000;
            self.riichi_sticks += 1;

            let mut deltas = [0; 4];
            deltas[p as usize] = -1000;

            let mut ev = serde_json::Map::new();
            ev.insert(
                "type".to_string(),
                Value::String("reach_accepted".to_string()),
            );
            ev.insert("actor".to_string(), Value::Number(p.into()));
            ev.insert("deltas".to_string(), serde_json::to_value(deltas).unwrap());
            self.mjai_log.push(Value::Object(ev).to_string());

            self.riichi_pending_acceptance = None;
        }
    }
}

// --- Helpers ---

// --- Implementations ---

#[pymethods]
impl Action {
    #[new]
    #[pyo3(signature = (action_type, tile=None, consume_tiles=vec![]))]
    pub fn new(action_type: ActionType, tile: Option<u8>, consume_tiles: Vec<u8>) -> Self {
        Self {
            action_type,
            tile,
            consume_tiles,
        }
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("type", self.action_type as i32)?;
        dict.set_item("tile", self.tile)?;

        let cons: Vec<u32> = self.consume_tiles.iter().map(|&x| x as u32).collect();
        dict.set_item("consume_tiles", cons)?;
        Ok(dict.to_object(py))
    }

    pub fn to_mjai(&self) -> PyResult<String> {
        let type_str = match self.action_type {
            ActionType::Discard => "dahai",
            ActionType::Chi => "chi",
            ActionType::Pon => "pon",
            ActionType::Daiminkan => "daiminkan",
            ActionType::Ankan => "ankan",
            ActionType::Kakan => "kakan",
            ActionType::Riichi => "reach",
            ActionType::Tsumo | ActionType::Ron => "hora",
            ActionType::KyushuKyuhai => "ryukyoku",
            ActionType::Pass => "none",
        };

        let mut data = serde_json::Map::new();
        data.insert("type".to_string(), Value::String(type_str.to_string()));

        if let Some(t) = self.tile {
            if self.action_type != ActionType::Tsumo
                && self.action_type != ActionType::Ron
                && self.action_type != ActionType::Riichi
            {
                data.insert("pai".to_string(), Value::String(tid_to_mjai(t)));
            }
        }

        if !self.consume_tiles.is_empty() {
            let cons: Vec<String> = self.consume_tiles.iter().map(|&t| tid_to_mjai(t)).collect();
            data.insert("consumed".to_string(), serde_json::to_value(cons).unwrap());
        }

        Ok(Value::Object(data).to_string())
    }
}

#[pymethods]
impl Observation {
    #[new]
    pub fn new(
        player_id: u8,
        hand: Vec<u8>,
        events_json: Vec<String>,
        prev_events_size: usize,
        legal_actions: Vec<Action>,
    ) -> Self {
        Self {
            player_id,
            hand,
            events_json,
            prev_events_size,
            legal_actions,
        }
    }

    #[getter]
    fn events(&self, py: Python) -> PyResult<PyObject> {
        let json = py.import_bound("json")?;
        let loads = json.getattr("loads")?;
        let list = pyo3::types::PyList::empty_bound(py);
        for s in &self.events_json {
            list.append(loads.call1((s,))?)?;
        }
        Ok(list.into_py(py))
    }

    pub fn new_events(&self, py: Python) -> PyResult<PyObject> {
        let json = py.import_bound("json")?;
        let loads = json.getattr("loads")?;
        let list = pyo3::types::PyList::empty_bound(py);
        for s in &self.events_json[self.prev_events_size..] {
            list.append(loads.call1((s,))?)?;
        }
        Ok(list.into_py(py))
    }

    pub fn legal_actions(&self) -> Vec<Action> {
        self.legal_actions.clone()
    }
}

fn _tid_to_mjai_hand(hand: &[u8]) -> Vec<String> {
    hand.iter().map(|&t| tid_to_mjai(t)).collect()
}

#[pymethods]
impl RiichiEnv {
    #[new]
    #[pyo3(signature = (game_type=4, mjai_mode=false, seed=None, round_wind=None))]
    pub fn new(game_type: u8, mjai_mode: bool, seed: Option<u64>, round_wind: Option<u8>) -> Self {
        RiichiEnv {
            wall: Vec::new(),
            hands: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            melds: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            discards: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            current_player: 0,
            turn_count: 0,
            is_done: false,
            needs_tsumo: false,
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
            player_event_counts: [0; 4],
            round_wind: round_wind.unwrap_or(0),
            game_type,
            mjai_mode,
            seed,
        }
    }

    #[pyo3(signature = (oya=None, wall=None, bakaze=None, scores=None, honba=None, kyotaku=None, seed=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn reset<'py>(
        &mut self,
        py: Python<'py>,
        oya: Option<u8>,
        wall: Option<Vec<u8>>,
        bakaze: Option<u8>,
        scores: Option<Vec<i32>>,
        honba: Option<u8>,
        kyotaku: Option<u32>,
        seed: Option<u64>,
    ) -> PyResult<PyObject> {
        // ... existing reset impl ...
        if let Some(s) = seed {
            self.seed = Some(s);
        }

        let initial_scores = if let Some(sc) = scores {
            let mut s = [0; 4];
            s.copy_from_slice(&sc[..4]);
            s
        } else {
            [25000; 4]
        };

        if self.mjai_log.is_empty() && self.mjai_mode {
            let mut start_game = serde_json::Map::new();
            start_game.insert("type".to_string(), Value::String("start_game".to_string()));
            start_game.insert("id".to_string(), Value::Number(0.into()));
            // names skipped for brevity
            self.mjai_log.push(Value::Object(start_game).to_string());
        }

        self.agari_results = HashMap::new();
        self.last_agari_results = HashMap::new();
        self.round_end_scores = None;

        self._initialize_round(
            oya.unwrap_or(0),
            bakaze.unwrap_or(0),
            honba.unwrap_or(0),
            kyotaku.unwrap_or(0),
            wall,
            Some(initial_scores),
        );

        self.step(py, HashMap::new())
    }

    #[pyo3(signature = (players=None))]
    fn get_obs_py<'py>(&mut self, py: Python<'py>, players: Option<Vec<u8>>) -> PyResult<PyObject> {
        Ok(self.get_observations(players).into_py(py))
    }

    #[pyo3(signature = (players=None))]
    pub fn get_observations(&mut self, players: Option<Vec<u8>>) -> HashMap<u8, Observation> {
        let targets = players.unwrap_or_else(|| (0..4).collect());
        let mut obs_map = HashMap::new();
        let current_log_len = self.mjai_log.len();

        for pid in targets {
            obs_map.insert(pid, self._get_obs(pid));
            self.player_event_counts[pid as usize] = current_log_len;
        }
        obs_map
    }

    pub fn scores(&self) -> [i32; 4] {
        self.scores
    }

    #[getter]
    pub fn _custom_honba(&self) -> u8 {
        self.honba
    }

    #[getter]
    pub fn _custom_round_wind(&self) -> u8 {
        self.round_wind
    }

    #[getter]
    pub fn mjai_log(&self, py: Python) -> PyResult<PyObject> {
        let json = py.import_bound("json")?;
        let loads = json.getattr("loads")?;
        let list = pyo3::types::PyList::empty_bound(py);
        for s in &self.mjai_log {
            list.append(loads.call1((s,))?)?;
        }
        Ok(list.into_py(py))
    }

    pub fn done(&self) -> bool {
        self.is_done
    }

    pub fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: HashMap<u8, Action>,
    ) -> PyResult<PyObject> {
        while !self.is_done {
            if self.needs_tsumo {
                // Handle leading tsumo draw
                // Midway draws logic TODO
                // Exhaustive draw check
                if self.wall.len() <= 14 {
                    self._trigger_ryukyoku("exhaustive_draw");
                    return self.get_obs_py(py, None);
                }

                if self.is_rinshan_flag {
                    if !self.wall.is_empty() {
                        self.drawn_tile = Some(self.wall.remove(0));
                        self.is_rinshan_flag = false;
                        self.rinshan_draw_count += 1;
                    }
                } else if !self.wall.is_empty() {
                    self.drawn_tile = self.wall.pop();
                }

                if let Some(t) = self.drawn_tile {
                    // Log
                    // Log
                    let mut ev = serde_json::Map::new();
                    ev.insert("type".to_string(), Value::String("tsumo".to_string()));
                    ev.insert(
                        "actor".to_string(),
                        Value::Number(self.current_player.into()),
                    );
                    ev.insert("pai".to_string(), Value::String(tid_to_mjai(t)));
                    self.mjai_log.push(Value::Object(ev).to_string());

                    // DO NOT add to hand here. Keep it in self.drawn_tile.
                } else {
                    // Should have triggered ryukyoku?
                }

                self.needs_tsumo = false;
                self.missed_agari_doujun[self.current_player as usize] = false;
                self.active_players = vec![self.current_player];

                if self.mjai_mode || !self.riichi_declared[self.current_player as usize] {
                    return self.get_obs_py(py, Some(self.active_players.clone()));
                }
                continue;
            }

            // Phase handling
            if self.active_players.is_empty() {
                return self.get_obs_py(py, Some(self.active_players.clone()));
            }

            // Check if actions provided match active_players
            // For now assume yes or partial return?

            // If WaitAct
            if self.phase == Phase::WaitAct {
                if let Some(act) = actions.get(&self.current_player) {
                    if act.action_type == ActionType::Discard {
                        let is_tsumogiri = act.tile == self.drawn_tile;
                        self._perform_discard(self.current_player, act.tile.unwrap(), is_tsumogiri);
                        if !self.active_players.is_empty() {
                            return self.get_obs_py(py, Some(self.active_players.clone()));
                        }
                        continue;
                    }
                    if act.action_type == ActionType::Tsumo {
                        // Tsumo
                        // Calculate Score
                        let winner = self.current_player;
                        let tile = self.drawn_tile.unwrap();

                        let cond = Conditions {
                            tsumo: true,
                            riichi: self.riichi_declared[winner as usize],
                            double_riichi: self.double_riichi_declared[winner as usize],
                            ippatsu: false,
                            player_wind: Wind::from((winner + 4 - self.oya) % 4),
                            round_wind: Wind::from(self.round_wind),
                            chankan: false,
                            haitei: self.wall.len() <= 14,
                            houtei: false,
                            rinshan: self.is_rinshan_flag,
                            tsumo_first_turn: self.is_first_turn
                                && self.discards[winner as usize].is_empty(),
                            kyoutaku: self.riichi_sticks,
                            tsumi: self.honba as u32,
                        };

                        let mut hand_for_calc = self.hands[winner as usize].clone();
                        hand_for_calc.push(tile);
                        let calc = crate::agari_calculator::AgariCalculator::new(
                            hand_for_calc,
                            self.melds[winner as usize].clone(),
                        );
                        let agari =
                            calc.calc(tile, self.dora_indicators.clone(), vec![], Some(cond));

                        let deltas =
                            self._calculate_deltas(&agari, winner, true, Some(winner), true);

                        // Log Hora
                        let mut ev = serde_json::Map::new();
                        ev.insert("type".to_string(), Value::String("hora".to_string()));
                        ev.insert("actor".to_string(), Value::Number(winner.into()));
                        ev.insert("target".to_string(), Value::Number(winner.into()));
                        ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
                        ev.insert("deltas".to_string(), serde_json::to_value(deltas).unwrap());
                        ev.insert("ura_markers".to_string(), Value::Array(vec![])); // TODO Ura
                        self.mjai_log.push(Value::Object(ev).to_string());

                        let mut agaris = HashMap::new();
                        agaris.insert(winner, agari);
                        self._end_kyoku_win(vec![winner], true, Some(winner), agaris);
                        return self.get_obs_py(py, None);
                    }

                    if act.action_type == ActionType::Riichi {
                        if self.scores[self.current_player as usize] < 1000 {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                "Not enough points for Riichi",
                            ));
                        }
                        self.riichi_stage[self.current_player as usize] = true;
                        let mut ev = serde_json::Map::new();
                        ev.insert("type".to_string(), Value::String("reach".to_string()));
                        ev.insert(
                            "actor".to_string(),
                            Value::Number(self.current_player.into()),
                        );
                        self.mjai_log.push(Value::Object(ev).to_string());
                        return self.get_obs_py(py, Some(vec![self.current_player]));
                    }
                }
            } else if self.phase == Phase::WaitResponse {
                // 1. Check missed agari
                for (pid, act) in &actions {
                    if let Some(claims) = self.current_claims.get(pid) {
                        let has_ron = claims.iter().any(|c| c.action_type == ActionType::Ron);
                        let chosen_ron = act.action_type == ActionType::Ron;
                        if has_ron && !chosen_ron {
                            if self.riichi_declared[*pid as usize] {
                                self.missed_agari_riichi[*pid as usize] = true;
                            } else {
                                self.missed_agari_doujun[*pid as usize] = true;
                            }
                        }
                    }
                }

                // 2. Filter actions
                let valid_actions: HashMap<u8, Action> = actions
                    .iter()
                    .filter(|(_, a)| a.action_type != ActionType::Pass)
                    .map(|(&k, v)| (k, v.clone()))
                    .collect();

                // 3. Ron Resolution
                let ronners: Vec<u8> = valid_actions
                    .iter()
                    .filter(|(_, a)| a.action_type == ActionType::Ron)
                    .map(|(&k, _)| k)
                    .collect();

                if !ronners.is_empty() {
                    let discarder = self.current_player;
                    let mut sorted_ronners = ronners.clone();
                    sorted_ronners.sort_by_key(|&p| (p + 4 - discarder) % 4);

                    let tile = self.last_discard.unwrap().1;
                    let mut agaris = HashMap::new();
                    for (idx, &winner) in sorted_ronners.iter().enumerate() {
                        let cond = Conditions {
                            tsumo: false,
                            riichi: self.riichi_declared[winner as usize],
                            double_riichi: self.double_riichi_declared[winner as usize],
                            ippatsu: false,
                            player_wind: Wind::from((winner + 4 - self.oya) % 4),
                            round_wind: Wind::from(self.round_wind),
                            chankan: self.pending_kan.is_some(),
                            haitei: false,
                            houtei: self.wall.len() <= 14, // Standard houtei is last discard?
                            // Actually wall.len() <= 14 in exhaust.
                            // Houtei usually means last tile from live wall.
                            rinshan: false,
                            tsumo_first_turn: false,
                            kyoutaku: self.riichi_sticks,
                            tsumi: self.honba as u32,
                        };

                        let calc = crate::agari_calculator::AgariCalculator::new(
                            self.hands[winner as usize].clone(),
                            self.melds[winner as usize].clone(),
                        );
                        let agari =
                            calc.calc(tile, self.dora_indicators.clone(), vec![], Some(cond));

                        // Calculate Deltas
                        let deltas = self._calculate_deltas(
                            &agari,
                            winner,
                            false,
                            Some(discarder),
                            idx == 0,
                        );

                        // Log Hora
                        let mut ev = serde_json::Map::new();
                        ev.insert("type".to_string(), Value::String("hora".to_string()));
                        ev.insert("actor".to_string(), Value::Number(winner.into()));
                        ev.insert("target".to_string(), Value::Number(discarder.into()));
                        ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
                        ev.insert("deltas".to_string(), serde_json::to_value(deltas).unwrap());
                        ev.insert("ura_markers".to_string(), Value::Array(vec![])); // TODO Ura
                        self.mjai_log.push(Value::Object(ev).to_string());

                        agaris.insert(winner, agari);
                    }

                    self._end_kyoku_win(sorted_ronners, false, Some(discarder), agaris);
                    return self.get_obs_py(py, None);
                }

                // 4. Pon / Daiminkan
                let ponners: Vec<u8> = valid_actions
                    .iter()
                    .filter(|(_, a)| {
                        a.action_type == ActionType::Pon || a.action_type == ActionType::Daiminkan
                    })
                    .map(|(&k, _)| k)
                    .collect();

                if let Some(&claimer) = ponners.first() {
                    self._execute_claim(
                        claimer,
                        valid_actions[&claimer].clone(),
                        Some(self.current_player),
                    );
                    self.current_player = claimer;
                    self.phase = Phase::WaitAct;
                    self.active_players = vec![claimer];

                    if valid_actions[&claimer].action_type == ActionType::Daiminkan {
                        self.is_rinshan_flag = true;
                        self.needs_tsumo = true;
                        self.active_players = vec![];
                    } else {
                        self.is_rinshan_flag = false;
                        self.drawn_tile = None;
                        self.needs_tsumo = false;
                    }

                    return self.get_obs_py(py, Some(self.active_players.clone()));
                }

                // 5. Chi
                let chiers: Vec<u8> = valid_actions
                    .iter()
                    .filter(|(_, a)| a.action_type == ActionType::Chi)
                    .map(|(&k, _)| k)
                    .collect();

                if let Some(&claimer) = chiers.first() {
                    self._execute_claim(
                        claimer,
                        valid_actions[&claimer].clone(),
                        Some(self.current_player),
                    );
                    self.current_player = claimer;
                    self.phase = Phase::WaitAct;
                    self.active_players = vec![claimer];
                    self.is_rinshan_flag = false;
                    self.drawn_tile = None;
                    self.needs_tsumo = false;

                    return self.get_obs_py(py, Some(self.active_players.clone()));
                }

                // 6. All Passed -> Next Player
                let discarder = self.current_player;

                // If riichi_stage, it's accepted
                if self.riichi_stage[discarder as usize] {
                    self.riichi_stage[discarder as usize] = false;
                    self.riichi_declared[discarder as usize] = true;
                    self.scores[discarder as usize] -= 1000;
                    self.riichi_sticks += 1;

                    let mut ev = serde_json::Map::new();
                    ev.insert(
                        "type".to_string(),
                        Value::String("reach_accepted".to_string()),
                    );
                    ev.insert("actor".to_string(), Value::Number(discarder.into()));
                    self.mjai_log.push(Value::Object(ev).to_string());
                }

                self.current_player = (discarder + 1) % 4;
                self.phase = Phase::WaitAct;
                self.needs_tsumo = true;
                self.active_players = vec![];
                continue; // Loop back for tsumo
            }

            // Fallback to break loop
            break;
        }
        self.get_obs_py(py, None)
    }

    pub fn _reveal_kan_dora(&mut self) {
        let target_idx =
            (4 + 2 * self.dora_indicators.len()) as isize - self.rinshan_draw_count as isize;
        if target_idx >= 0 && (target_idx as usize) < self.wall.len() {
            self.dora_indicators.push(self.wall[target_idx as usize]);
        }
    }

    pub fn _get_ura_markers(&self) -> Vec<String> {
        let mut uras = Vec::new();
        for i in 0..self.dora_indicators.len() {
            let target_idx = (5 + 2 * i) as isize - self.rinshan_draw_count as isize;
            if target_idx >= 0 && (target_idx as usize) < self.wall.len() {
                uras.push(tid_to_mjai(self.wall[target_idx as usize]));
            }
        }
        uras
    }
}

impl RiichiEnv {
    fn _perform_discard(&mut self, pid: u8, tile: u8, is_tsumogiri: bool) {
        let mut tsumogiri = is_tsumogiri;
        // If not tsumogiri, remove from hand and add current drawn tile to hand
        if !tsumogiri {
            // Remove tile from hand
            if let Some(pos) = self.hands[pid as usize].iter().position(|&t| t == tile) {
                self.hands[pid as usize].remove(pos);
                // Add drawn tile to hand
                if let Some(dt) = self.drawn_tile {
                    self.hands[pid as usize].push(dt);
                }
                self.hands[pid as usize].sort();
            } else {
                // Might be tsumogiri if hand doesn't have it?
                tsumogiri = true;
            }
        }

        self.discards[pid as usize].push(tile);
        self.drawn_tile = None;
        self.last_discard = Some((pid, tile));

        let mut ev = serde_json::Map::new();
        ev.insert("type".to_string(), Value::String("dahai".to_string()));
        ev.insert("actor".to_string(), Value::Number(pid.into()));
        ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
        ev.insert("tsumogiri".to_string(), Value::Bool(tsumogiri));
        self.mjai_log.push(Value::Object(ev).to_string());

        self.nagashi_eligible[pid as usize] &= is_terminal_tile(tile);

        self._update_claims(pid, tile);

        if !self.current_claims.is_empty() {
            self.phase = Phase::WaitResponse;
            self.active_players = self.current_claims.keys().cloned().collect();
            self.active_players.sort();
            self.needs_tsumo = false;
        } else {
            // If riichi_stage, it's accepted (No claims detected)
            if self.riichi_stage[pid as usize] {
                self.riichi_stage[pid as usize] = false;
                self.riichi_declared[pid as usize] = true;
                self.scores[pid as usize] -= 1000;
                self.riichi_sticks += 1;

                let mut ev = serde_json::Map::new();
                ev.insert(
                    "type".to_string(),
                    Value::String("reach_accepted".to_string()),
                );
                ev.insert("actor".to_string(), Value::Number(pid.into()));
                self.mjai_log.push(Value::Object(ev).to_string());
            }

            self.current_player = (pid + 1) % 4;
            self.phase = Phase::WaitAct;
            self.needs_tsumo = true;
            self.active_players = vec![];
        }
    }

    fn _update_claims(&mut self, discarded_pid: u8, tile: u8) {
        self.current_claims.clear();

        // 1. Ron Check
        for pid in 0..4 {
            if pid == discarded_pid {
                continue;
            }
            if self.is_done {
                continue;
            }

            if self.missed_agari_doujun[pid as usize] || self.missed_agari_riichi[pid as usize] {
                continue;
            }

            let is_furiten = self.discards[pid as usize].contains(&tile);
            if is_furiten {
                continue;
            }

            if self._check_ron(pid, tile, discarded_pid) {
                self.current_claims
                    .entry(pid)
                    .or_default()
                    .push(Action::new(ActionType::Ron, Some(tile), vec![]));
            }
        }

        // 2. Pon / Kan
        if self.wall.len() > 14 {
            // Houtei check (ignoring if python uses <=14 to forbid)

            // Python checks: if len(wall) <= 14: return (cannot call).
            // Except Ron.

            for pid in 0..4 {
                if pid == discarded_pid {
                    continue;
                }
                if self.riichi_declared[pid as usize] {
                    continue;
                }

                let hand = &self.hands[pid as usize];
                let count = hand.iter().filter(|&&t| t / 4 == tile / 4).count();

                if count >= 2 {
                    // Pon
                    let consume: Vec<u8> = hand
                        .iter()
                        .cloned()
                        .filter(|&t| t / 4 == tile / 4)
                        .take(2)
                        .collect();
                    self.current_claims
                        .entry(pid)
                        .or_default()
                        .push(Action::new(ActionType::Pon, Some(tile), consume));
                }
                if count >= 3 {
                    // Daiminkan
                    let consume: Vec<u8> = hand
                        .iter()
                        .cloned()
                        .filter(|&t| t / 4 == tile / 4)
                        .take(3)
                        .collect();
                    self.current_claims
                        .entry(pid)
                        .or_default()
                        .push(Action::new(ActionType::Daiminkan, Some(tile), consume));
                }
            }

            // 3. Chi (Only next player)
            let next_pid = (discarded_pid + 1) % 4;
            if !self.riichi_declared[next_pid as usize] && tile < 108 {
                let hand = &self.hands[next_pid as usize];
                let t_type = tile / 4;
                let suit = t_type / 9;
                let num = t_type % 9;

                let has = |n: u8| -> bool {
                    if n >= 9 {
                        return false;
                    }
                    let target = suit * 9 + n;
                    hand.iter().any(|&t| t / 4 == target)
                };

                // Helper to find first tile of type
                let find =
                    |n: u8| -> u8 { *hand.iter().find(|&&t| t / 4 == suit * 9 + n).unwrap() };

                // Left: T-2, T-1
                if num >= 2 && has(num - 2) && has(num - 1) {
                    let t1 = find(num - 2);
                    let t2 = find(num - 1);
                    self.current_claims
                        .entry(next_pid)
                        .or_default()
                        .push(Action::new(ActionType::Chi, Some(tile), vec![t1, t2]));
                }
                // Middle: T-1, T+1
                if (1..=7).contains(&num) && has(num - 1) && has(num + 1) {
                    let t1 = find(num - 1);
                    let t2 = find(num + 1);
                    self.current_claims
                        .entry(next_pid)
                        .or_default()
                        .push(Action::new(ActionType::Chi, Some(tile), vec![t1, t2]));
                }
                // Right: T+1, T+2
                if num <= 6 && has(num + 1) && has(num + 2) {
                    let t1 = find(num + 1);
                    let t2 = find(num + 2);
                    self.current_claims
                        .entry(next_pid)
                        .or_default()
                        .push(Action::new(ActionType::Chi, Some(tile), vec![t1, t2]));
                }
            }
        }
    }

    fn _check_ron(&self, pid: u8, tile: u8, _discarded_pid: u8) -> bool {
        let hand = &self.hands[pid as usize];
        let melds = &self.melds[pid as usize];

        let cond = Conditions {
            tsumo: false,
            riichi: self.riichi_declared[pid as usize],
            double_riichi: self.double_riichi_declared[pid as usize],
            ippatsu: false, // TODO
            player_wind: Wind::from((pid + 4 - self.oya) % 4),
            round_wind: Wind::East,
            chankan: false,
            haitei: false,
            houtei: self.wall.len() <= 14,
            rinshan: false,
            tsumo_first_turn: false,
            kyoutaku: self.riichi_sticks,
            tsumi: self.honba as u32,
        };

        let calc = crate::agari_calculator::AgariCalculator::new(hand.clone(), melds.clone());
        let agari = calc.calc(tile, self.dora_indicators.clone(), vec![], Some(cond));

        agari.agari && (agari.yakuman || agari.han >= 1)
    }
}
impl RiichiEnv {
    fn _initialize_round(
        &mut self,
        oya: u8,
        bakaze: u8,
        honba: u8,
        kyotaku: u32,
        wall: Option<Vec<u8>>,
        scores: Option<[i32; 4]>,
    ) {
        self.oya = oya;
        self.honba = honba;
        self.round_wind = bakaze;
        self.hands = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        self.melds = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        self.discards = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
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

        if let Some(s) = scores {
            self.scores = s;
        }

        if let Some(w) = wall {
            let mut w_rev = w;
            w_rev.reverse();
            self.wall = w_rev;
        } else {
            let mut w: Vec<u8> = (0..136).collect();
            let mut rng = if let Some(s) = self.seed {
                StdRng::seed_from_u64(s)
            } else {
                StdRng::from_entropy()
            };
            w.shuffle(&mut rng);
            self.wall = w;
        }

        self.dora_indicators = vec![self.wall[4]];

        for i in 0..4 {
            self.hands[i] = Vec::new();
        }

        // 4-4-4-1 pattern
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

        self.current_player = self.oya;
        self.phase = Phase::WaitAct;
        self.active_players = vec![];
        self.needs_tsumo = true;
        self.drawn_tile = None;

        let tehais: Vec<Vec<String>> = self
            .hands
            .iter()
            .map(|h| h.iter().map(|&t| tid_to_mjai(t)).collect())
            .collect();

        let winds = ["E", "S", "W", "N"];
        let bakaze_str = winds[(bakaze as usize) % 4];

        let start_kyoku = serde_json::json!({
            "type": "start_kyoku",
            "bakaze": bakaze_str,
            "kyoku": oya + 1,
            "honba": honba,
            "kyotaku": kyotaku,
            "oya": oya,
            "dora_marker": tid_to_mjai(self.dora_indicators[0]),
            "tehais": tehais,
            "scores": self.scores
        });
        self.mjai_log.push(start_kyoku.to_string());
    }

    fn _get_obs(&self, pid: u8) -> Observation {
        Observation {
            player_id: pid,
            hand: self.hands[pid as usize].clone(),
            events_json: self.mjai_log.clone(),
            prev_events_size: self.player_event_counts[pid as usize],
            legal_actions: self._get_legal_actions(pid),
        }
    }

    fn _calculate_deltas(
        &mut self,
        agari: &Agari,
        winner: u8,
        is_tsumo: bool,
        loser: Option<u8>,
        include_bonus: bool,
    ) -> [i32; 4] {
        let mut deltas = [0; 4];
        let h_val = if include_bonus { self.honba as i32 } else { 0 };
        let pao_pid: Option<u8> = None; // TODO: Implement Pao if needed

        if is_tsumo {
            let total = if winner == self.oya {
                (agari.tsumo_agari_ko * 3) as i32 + h_val * 300
            } else {
                (agari.tsumo_agari_oya + agari.tsumo_agari_ko * 2) as i32 + h_val * 300
            };

            if let Some(p) = pao_pid {
                deltas[winner as usize] = total;
                deltas[p as usize] = -total;
            } else {
                let h = h_val * 100;
                if winner == self.oya {
                    let p = agari.tsumo_agari_ko as i32;
                    for i in 0..4 {
                        deltas[i as usize] = if i == winner { (p + h) * 3 } else { -(p + h) };
                    }
                } else {
                    let po = agari.tsumo_agari_oya as i32;
                    let pk = agari.tsumo_agari_ko as i32;
                    for i in 0..4 {
                        if i == winner {
                            deltas[i as usize] = po + pk * 2 + h * 3;
                        } else if i == self.oya {
                            deltas[i as usize] = -(po + h);
                        } else {
                            deltas[i as usize] = -(pk + h);
                        }
                    }
                }
            }
        } else {
            // Ron
            let s_base = agari.ron_agari as i32;
            let s_total = s_base + h_val * 300;
            if let Some(p) = pao_pid {
                if loser != Some(p) {
                    // Split
                    let half = s_base / 2;
                    if let Some(l) = loser {
                        deltas[l as usize] = -(half);
                        deltas[p as usize] = -(half) - h_val * 300;
                        deltas[winner as usize] = s_total;
                    }
                } else {
                    // Loser is p (unlikely in Ron? Pao is third person responsibility)
                    if let Some(l) = loser {
                        deltas[l as usize] = -s_total;
                        deltas[winner as usize] = s_total;
                    }
                }
            } else if let Some(l) = loser {
                deltas[l as usize] = -s_total;
                deltas[winner as usize] = s_total;
            }
        }

        if include_bonus && self.riichi_sticks > 0 {
            deltas[winner as usize] += (self.riichi_sticks * 1000) as i32;
            self.riichi_sticks = 0;
        }

        for i in 0..4 {
            self.scores[i as usize] += deltas[i as usize];
            self.score_deltas[i as usize] += deltas[i as usize];
        }

        deltas
    }

    fn _get_legal_actions(&self, pid: u8) -> Vec<Action> {
        let mut actions = Vec::new();
        let hand = &self.hands[pid as usize];

        let mut h14 = hand.clone();
        if let Some(t) = self.drawn_tile {
            if pid == self.current_player {
                h14.push(t);
            }
        }
        h14.sort();

        if self.phase == Phase::WaitAct {
            if pid != self.current_player || self.needs_tsumo {
                return Vec::new(); // Wait for tsumo or not your turn
            }

            // If in Riichi
            if self.riichi_declared[pid as usize] {
                // Can only Tsumo or Ankan (with restrictions) or Discard drawn tile

                // Tsumo Check
                if let Some(dt) = self.drawn_tile {
                    if self._check_tsumo(pid, dt) {
                        actions.push(Action::new(ActionType::Tsumo, None, vec![]));
                    }

                    // Ankan Check logic (complex restrictions during Riichi)
                    // Simple check: if 4 tiles exist, and changing them doesn't change wait?
                    // For now, if Python logic is simple, mimic it.
                    // Python: `_riichienv.check_riichi_candidates(h14)`? No that's for declarations.
                    // Python: `sum(1 for t in h14 if t // 4 == t_type) == 4`
                    // Rust AgariCalculator doesn't check Ankan validity during Riichi explicitly?
                    // Only logic: `t_type = self.drawn_tile // 4`.
                    // If 4 matches, allow Ankan.
                    // Note: Riichi Ankan rule is stricter, but let's stick to base logic.
                    let t_type = dt / 4;
                    let matches: Vec<u8> =
                        h14.iter().cloned().filter(|&t| t / 4 == t_type).collect();
                    if matches.len() == 4 {
                        actions.push(Action::new(ActionType::Ankan, Some(dt), matches));
                    }
                }

                if let Some(dt) = self.drawn_tile {
                    actions.push(Action::new(ActionType::Discard, Some(dt), vec![]));
                }
                return actions;
            }

            // Normal Turn
            // 1. Kyushukyuhai
            // ...

            // 2. Discards
            for &t in &h14 {
                actions.push(Action::new(ActionType::Discard, Some(t), vec![]));
                // Optimize: only unique tiles? Python does all.
            }

            // 3. Tsumo
            if let Some(dt) = self.drawn_tile {
                if self._check_tsumo(pid, dt) {
                    actions.push(Action::new(ActionType::Tsumo, None, vec![]));
                }
            }

            // 4. Riichi
            // menzen, socre >= 1000, wall >= 4
            let is_menzen = self.melds[pid as usize].iter().all(|m| !m.opened);
            if is_menzen && self.scores[pid as usize] >= 1000 && self.wall.len() >= 4 {
                // Check if any discard leads to Tenpai
                // Slow check?
                // For each unique discard, check if tenpai.
                let mut can_riichi = false;
                // De-dup tiles to check
                let mut checked_types = std::collections::HashSet::new();
                for &t in &h14 {
                    let tt = t / 4;
                    if checked_types.contains(&tt) {
                        continue;
                    }
                    checked_types.insert(tt);

                    // Try removing t
                    // AgariCalculator.is_tenpai
                    // Need to implement efficient is_tenpai or use crate.
                    // Instantiate AgariCalculator with h14 minus t.
                    let mut temp_hand = h14.clone();
                    if let Some(pos) = temp_hand.iter().position(|&x| x == t) {
                        temp_hand.remove(pos);
                    }
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
                    actions.push(Action::new(ActionType::Riichi, None, vec![]));
                }
            }

            // 5. Kan (Ankan / Kakan)
            if self.wall.len() > 14 {
                // Kakan needs replacement tile? No, triggers rinshan.
                // Kakan
                for m in &self.melds[pid as usize] {
                    if m.meld_type == MeldType::Peng {
                        // Check if we have the 4th tile
                        let target_type = m.tiles[0] / 4;
                        if let Some(&kakan_tile) = h14.iter().find(|&&t| t / 4 == target_type) {
                            actions.push(Action::new(
                                ActionType::Kakan,
                                Some(kakan_tile),
                                m.tiles.clone(),
                            ));
                        }
                    }
                }
                // Ankan
                let mut counts = HashMap::new();
                for &t in &h14 {
                    *counts.entry(t / 4).or_insert(0) += 1;
                }
                for (&tt, &count) in &counts {
                    if count == 4 {
                        let tiles: Vec<u8> = h14.iter().cloned().filter(|&t| t / 4 == tt).collect();
                        actions.push(Action::new(ActionType::Ankan, Some(tiles[0]), tiles));
                    }
                }
            }
        } else {
            // WaitResponse
            actions.push(Action::new(ActionType::Pass, None, vec![]));
            if let Some(claims) = self.current_claims.get(&pid) {
                for c in claims {
                    actions.push(c.clone());
                }
            }
        }

        actions
    }

    fn _execute_claim(&mut self, pid: u8, action: Action, from_pid: Option<u8>) {
        use serde_json::Value; // Added use statement for Value
        let hand = &mut self.hands[pid as usize];

        let mut m_type = MeldType::Peng;
        let mut opened = true;
        let consumed = action.consume_tiles.clone();

        match action.action_type {
            ActionType::Chi => {
                m_type = MeldType::Chi;
            }
            ActionType::Pon => {
                m_type = MeldType::Peng;
            }
            ActionType::Daiminkan => {
                m_type = MeldType::Gang;
            }
            ActionType::Ankan => {
                m_type = MeldType::Angang;
                opened = false;
            }
            ActionType::Kakan => {
                // m_type = MeldType::Addgang; // Redundant as it returns early or doesn't use it
                // Special handling: Action tile removed from hand
                if let Some(pos) = hand.iter().position(|&x| x == action.tile.unwrap()) {
                    hand.remove(pos);
                }
                // Find existing Pon logic
                if let Some(pos) = self.melds[pid as usize].iter().position(|m| {
                    m.meld_type == MeldType::Peng && m.tiles[0] / 4 == action.tile.unwrap() / 4
                }) {
                    let mut old_meld = self.melds[pid as usize].remove(pos);
                    old_meld.tiles.push(action.tile.unwrap());
                    old_meld.tiles.sort();
                    let new_meld = Meld::new(MeldType::Addgang, old_meld.tiles.clone(), true);
                    self.melds[pid as usize].push(new_meld);

                    // Log Kakan
                    let mut ev = serde_json::Map::new();
                    ev.insert("type".to_string(), Value::String("kakan".to_string()));
                    ev.insert("actor".to_string(), Value::Number(pid.into()));
                    ev.insert(
                        "pai".to_string(),
                        Value::String(tid_to_mjai(action.tile.unwrap())),
                    );
                    let old_cons: Vec<String> = old_meld
                        .tiles
                        .iter()
                        .filter(|&&t| t != action.tile.unwrap())
                        .map(|&t| tid_to_mjai(t))
                        .collect();
                    ev.insert(
                        "consumed".to_string(),
                        serde_json::to_value(old_cons).unwrap(),
                    );
                    self.mjai_log.push(Value::Object(ev).to_string());

                    self.pending_kan_dora_count += 1;
                    return;
                }
                // Fallback (Ankan logic if kakan fails?) - Python logic handles this.
                // Assuming valid Kakan implies existing Pon.
                return;
            }
            _ => {}
        }

        // Remove consume tiles from hand
        if action.action_type != ActionType::Kakan {
            for &c in &consumed {
                if let Some(pos) = hand.iter().position(|&x| x == c) {
                    hand.remove(pos);
                }
            }
            // Add tile to meld
            let mut tiles = consumed.clone();
            if let Some(t) = action.tile {
                tiles.push(t);
            }
            tiles.sort();

            let meld = Meld::new(m_type, tiles.clone(), opened);
            self.melds[pid as usize].push(meld);

            // Log Claim
            let type_str = match action.action_type {
                ActionType::Chi => "chi",
                ActionType::Pon => "pon",
                ActionType::Daiminkan => "daiminkan",
                ActionType::Ankan => "ankan",
                _ => "unknown",
            };
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String(type_str.to_string()));
            ev.insert("actor".to_string(), Value::Number(pid.into()));
            ev.insert(
                "target".to_string(),
                Value::Number(from_pid.unwrap_or(pid).into()),
            );
            if let Some(t) = action.tile {
                ev.insert("pai".to_string(), Value::String(tid_to_mjai(t)));
            }
            let cons_str: Vec<String> = consumed.iter().map(|&t| tid_to_mjai(t)).collect();
            ev.insert(
                "consumed".to_string(),
                serde_json::to_value(cons_str).unwrap(),
            );
            self.mjai_log.push(Value::Object(ev).to_string());

            if action.action_type == ActionType::Daiminkan
                || action.action_type == ActionType::Ankan
            {
                self.pending_kan_dora_count += 1;
            }

            // Hand sorting? Usually done.
            self.hands[pid as usize].sort();
        }
    }

    fn _check_tsumo(&self, pid: u8, tile: u8) -> bool {
        let hand = &self.hands[pid as usize];
        let melds = &self.melds[pid as usize];

        // Is first turn?
        let is_first = self.is_first_turn && self.discards[pid as usize].is_empty();

        let cond = Conditions {
            tsumo: true,
            riichi: self.riichi_declared[pid as usize],
            double_riichi: self.double_riichi_declared[pid as usize],
            ippatsu: false, // TODO: track ippatsu
            player_wind: Wind::from((pid + 4 - self.oya) % 4),
            round_wind: Wind::from(self.round_wind),
            chankan: false,
            haitei: self.wall.len() <= 14,
            houtei: false,
            rinshan: self.is_rinshan_flag,
            tsumo_first_turn: is_first,
            kyoutaku: self.riichi_sticks,
            tsumi: self.honba as u32,
        };

        // Need to construct AgariCalculator
        // ...
        // It takes Vec<u8> of hand (13 tiles).
        // self.hands has 13 tiles? Yes if we haven't added drawn tile yet.
        // Wait, `_get_legal_actions` adds drawn tile to local `h14`.
        // `AgariCalculator::new` takes `tiles_136` (hand).
        // `calc` takes `win_tile`.
        // So we pass `self.hands[pid]` (13 tiles) to `new`.
        let calc = crate::agari_calculator::AgariCalculator::new(hand.clone(), melds.clone());
        let agari = calc.calc(tile, self.dora_indicators.clone(), vec![], Some(cond));

        agari.agari && (agari.yakuman || agari.han >= 1) // Simple check (ignore yaku 31-33 exclusion for now)
    }
}

fn is_terminal_tile(tile: u8) -> bool {
    let t = tile / 4;
    t == 0 || t == 8 || t == 9 || t == 17 || t == 18 || t >= 26
}
