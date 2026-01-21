#![allow(clippy::useless_conversion)]
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyListMethods};
use pyo3::{pyclass, pymethods, Bound, IntoPyObject, Py, PyAny, PyErr, PyResult, Python};
// IntoPy might be needed for .into_py() calls if I revert?
// I used .to_object() which needs ToPyObject.
use rand::prelude::*;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

use crate::parser::tid_to_mjai;
use crate::types::{Agari, Conditions, Meld, MeldType, Wind};
use crate::yaku;
use sha2::Digest;

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

// --- Enums ---

#[pyclass(module = "riichienv._riichienv", eq, eq_int)]
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

#[pyclass(module = "riichienv._riichienv", eq, eq_int)]
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

#[pyclass(module = "riichienv._riichienv")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Action {
    #[pyo3(get, set)]
    pub action_type: ActionType,
    #[pyo3(get, set)]
    pub tile: Option<u8>,
    pub consume_tiles: Vec<u8>,
}

#[pymethods]
impl Action {
    #[new]
    #[pyo3(signature = (r#type=ActionType::Pass, tile=None, consume_tiles=vec![]))]
    pub fn new(r#type: ActionType, tile: Option<u8>, consume_tiles: Vec<u8>) -> Self {
        let mut sorted_consume = consume_tiles;
        sorted_consume.sort();
        Self {
            action_type: r#type,
            tile,
            consume_tiles: sorted_consume,
        }
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("type", self.action_type as i32)?;
        dict.set_item("tile", self.tile)?;

        let cons: Vec<u32> = self.consume_tiles.iter().map(|&x| x as u32).collect();
        dict.set_item("consume_tiles", cons)?;
        Ok(dict.unbind().into())
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

    fn __repr__(&self) -> String {
        format!(
            "Action(action_type={:?}, tile={:?}, consume_tiles={:?})",
            self.action_type, self.tile, self.consume_tiles
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[getter]
    fn get_consume_tiles(&self) -> Vec<u32> {
        self.consume_tiles.iter().map(|&x| x as u32).collect()
    }

    #[setter]
    fn set_consume_tiles(&mut self, value: Vec<u8>) {
        self.consume_tiles = value;
    }
}

#[pyclass(module = "riichienv._riichienv")]
#[derive(Debug, Clone)]
pub struct Observation {
    #[pyo3(get)]
    pub player_id: u8,
    pub hand: Vec<u8>,
    pub events_json: Vec<String>,
    #[pyo3(get)]
    pub prev_events_size: usize,
    pub legal_actions: Vec<Action>,
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
    pub fn hand(&self) -> Vec<u32> {
        self.hand.iter().map(|&x| x as u32).collect()
    }

    #[getter]
    pub fn events(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let loads = py.import("json")?.getattr("loads")?;
        let list = pyo3::types::PyList::empty(py);
        for s in &self.events_json {
            list.append(loads.call1((s,))?)?;
        }
        Ok(list.unbind().into())
    }

    pub fn new_events(&self, py: Python) -> PyResult<Py<PyAny>> {
        let list = pyo3::types::PyList::empty(py);
        for s in &self.events_json[self.prev_events_size..] {
            list.append(s)?;
        }
        Ok(list.unbind().into())
    }

    pub fn select_action_from_mjai(
        &self,
        _py: Python,
        mjai_resp: Bound<PyDict>,
    ) -> PyResult<Option<Action>> {
        let r_type: String = mjai_resp
            .get_item("type")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("missing type"))?
            .extract()?;
        let r_pai: Option<String> = mjai_resp.get_item("pai")?.and_then(|v| v.extract().ok());
        let r_consumed: Option<Vec<String>> = mjai_resp
            .get_item("consumed")?
            .and_then(|v| v.extract().ok());

        for act in &self.legal_actions {
            let m_type = match act.action_type {
                ActionType::Discard => "dahai",
                ActionType::Chi => "chi",
                ActionType::Pon => "pon",
                ActionType::Daiminkan => "daiminkan",
                ActionType::Ankan => "ankan",
                ActionType::Kakan => "kakan",
                ActionType::Riichi => "reach",
                ActionType::Tsumo | ActionType::Ron => "hora",
                ActionType::Pass => "none",
                ActionType::KyushuKyuhai => "ryukyoku",
            };

            if m_type != r_type {
                continue;
            }

            // Tile comparison
            if let Some(ref rp) = r_pai {
                if let Some(at) = act.tile {
                    if tid_to_mjai(at) != *rp {
                        continue;
                    }
                } else if act.action_type != ActionType::Riichi
                    && act.action_type != ActionType::Tsumo
                    && act.action_type != ActionType::Ron
                    && act.action_type != ActionType::Ankan
                    && act.action_type != ActionType::Kakan
                {
                    continue;
                }
            }

            // Consumed comparison
            if let Some(ref rc) = r_consumed {
                let mut ac: Vec<String> =
                    act.consume_tiles.iter().map(|&t| tid_to_mjai(t)).collect();
                ac.sort();
                let mut rc_sorted = rc.clone();
                rc_sorted.sort();
                if ac != rc_sorted {
                    continue;
                }
            } else if !act.consume_tiles.is_empty() {
                continue;
            }

            return Ok(Some(act.clone()));
        }

        Ok(None)
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("player_id", self.player_id)?;
        let hand: Vec<u32> = self.hand.iter().map(|&x| x as u32).collect();
        dict.set_item("hand", hand)?;
        dict.set_item("events", self.events(py)?)?;
        dict.set_item("prev_events_size", self.prev_events_size)?;

        let actions = pyo3::types::PyList::empty(py);
        for a in &self.legal_actions {
            actions.append(a.to_dict(py)?)?;
        }
        dict.set_item("legal_actions", actions)?;

        Ok(dict.unbind().into())
    }

    pub fn legal_actions(&self) -> Vec<Action> {
        self.legal_actions.clone()
    }
}

#[pyclass(module = "riichienv._riichienv")]
#[derive(Debug, Clone)]
pub struct RiichiEnv {
    // Game State
    #[pyo3(get, set)]
    pub wall: Vec<u8>,
    #[pyo3(get, set)]
    pub hands: [Vec<u8>; 4],
    #[pyo3(get, set)]
    pub melds: [Vec<Meld>; 4],
    #[pyo3(get, set)]
    pub discards: [Vec<u8>; 4],
    #[pyo3(get, set)]
    pub discard_from_hand: [Vec<bool>; 4],
    #[pyo3(get, set)]
    pub discard_is_riichi: [Vec<bool>; 4],
    #[pyo3(get, set)]
    pub riichi_declaration_index: [Option<usize>; 4],
    #[pyo3(get, set)]
    pub current_player: u8,
    #[pyo3(get, set)]
    pub turn_count: u32,
    #[pyo3(get, set)]
    pub is_done: bool,
    #[pyo3(get, set)]
    pub needs_tsumo: bool,
    #[pyo3(get, set)]
    pub needs_initialize_next_round: bool,
    #[pyo3(get, set)]
    pub pending_oya_won: bool,
    #[pyo3(get, set)]
    pub pending_is_draw: bool,
    #[pyo3(get)]
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
    #[pyo3(get)]
    pub phase: Phase,
    pub active_players: Vec<u8>,
    pub last_discard: Option<(u8, u8)>,
    #[pyo3(get, set)]
    pub current_claims: HashMap<u8, Vec<Action>>,

    #[pyo3(get, set)]
    pub pending_kan: Option<(u8, Action)>,

    #[pyo3(get)]
    pub oya: u8,
    #[pyo3(get)]
    pub honba: u8, // Added
    #[pyo3(get)]
    pub kyoku_idx: u8,
    #[pyo3(get)]
    pub round_wind: u8,
    // ...
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
    #[pyo3(get, set)]
    pub ippatsu_cycle: [bool; 4],

    #[pyo3(get)]
    pub wall_digest: String,
    #[pyo3(get)]
    pub salt: String,
    #[pyo3(get)]
    pub agari_results: HashMap<u8, Agari>,
    #[pyo3(get)]
    pub last_agari_results: HashMap<u8, Agari>,
    #[pyo3(get, set)]
    pub round_end_scores: Option<[i32; 4]>,
    pub forbidden_discards: [Vec<u8>; 4],

    #[pyo3(get)]
    pub mjai_log: Vec<String>,
    #[pyo3(get)]
    pub mjai_log_per_player: [Vec<String>; 4],
    #[pyo3(get)]
    pub player_event_counts: [usize; 4],

    // Config
    #[pyo3(get)]
    pub game_mode: u8,
    // If true, disables the generation of MJAI-compatible event logs.
    // Enabling this can improve performance for RL training where visualizer data is not needed.
    #[pyo3(get, set)]
    pub skip_mjai_logging: bool,
    #[pyo3(get)]
    pub seed: Option<u64>,
    pub(crate) hand_index: u64,
    #[pyo3(get)]
    pub rule: crate::rule::GameRule,
    #[pyo3(get)]
    pub pao: [HashMap<u8, u8>; 4],
}

impl RiichiEnv {
    pub(crate) fn _trigger_ryukyoku(&mut self, reason: &str) {
        self._accept_riichi(); // Ensure riichi sticks are collected if pending
        let mut tenpai = [false; 4];
        let mut final_reason = reason.to_string();
        let mut nagashi_winners = Vec::new();

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
            for (i, &eligible) in self.nagashi_eligible.iter().enumerate() {
                if eligible {
                    nagashi_winners.push(i as u8);
                }
            }

            if !nagashi_winners.is_empty() {
                final_reason = "nagashimangan".to_string();
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
        ev.insert("reason".to_string(), Value::String(final_reason.clone()));
        self._push_mjai_event(Value::Object(ev));

        let mut is_renchan = false;
        if final_reason == "exhaustive_draw" {
            is_renchan = tenpai[self.oya as usize];
        } else if final_reason == "nagashimangan" {
            is_renchan = nagashi_winners.contains(&self.oya);
        } else if ["kyushu_kyuhai", "suurechi", "suukansansen", "sufuurenta"]
            .contains(&final_reason.as_str())
        {
            is_renchan = true;
        }

        self._end_kyoku_ryukyoku(is_renchan, true);
    }

    pub(crate) fn _trigger_error_penalty(&mut self, offender: u8, reason: String) {
        // Chombo Penalty Logic
        // Offender pays Mangan amount.
        // If Oya: 4000 all -> 12000 total.
        // If Ko: 4000 to Oya, 2000 to others -> 8000 total.

        let mut deltas = [0; 4];
        let is_offender_oya = offender == self.oya;

        if is_offender_oya {
            for (i, d) in deltas.iter_mut().enumerate() {
                if i as u8 == offender {
                    *d = -12000;
                } else {
                    *d = 4000;
                }
            }
        } else {
            for (i, d) in deltas.iter_mut().enumerate() {
                if i as u8 == offender {
                    *d = -8000;
                } else if i as u8 == self.oya {
                    *d = 4000;
                } else {
                    *d = 2000;
                }
            }
        }

        // Update scores
        for (i, d) in deltas.iter().enumerate() {
            self.scores[i] += d;
            self.score_deltas[i] += d;
        }

        // Log Ryukyoku with Error reason
        let mut ev = serde_json::Map::new();
        ev.insert("type".to_string(), Value::String("ryukyoku".to_string()));
        let mut full_reason = "Error: ".to_string();
        full_reason.push_str(&reason);
        ev.insert("reason".to_string(), Value::String(full_reason));
        ev.insert("deltas".to_string(), serde_json::to_value(deltas).unwrap());
        self._push_mjai_event(Value::Object(ev));

        // End Kyoku as Ryukyoku (Renchan + Honba+1)
        // is_renchan=true (Keep Oya), is_draw=true (It is a draw type).
        self._end_kyoku_ryukyoku(true, true);
    }

    fn _is_game_over(&self) -> bool {
        if self.scores.iter().any(|&s| s < 0) {
            return true;
        }
        let max_score = self.scores.iter().cloned().max().unwrap_or(0);

        if self.game_mode == 1 || self.game_mode == 4 {
            // Tonpu
            if self.round_wind >= 1 {
                // If South (1), check if sudden death conditions met (score < 30000).
                // If West (2), it's over (limit).
                if self.round_wind > 1 || max_score >= 30000 {
                    return true;
                }
            }
        } else if self.game_mode == 2 || self.game_mode == 5 {
            // Hanchan
            if self.round_wind >= 2 {
                // If West (2), check sudden death.
                // If North (3), it's over (limit).
                if self.round_wind > 2 || max_score >= 30000 {
                    return true;
                }
            }
        } else if self.game_mode == 0 || self.game_mode == 3 {
            // Ikkyoku (One Round)
            // For Ikkyoku, any kyoku end = game over.
            return true;
        }

        false
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
        let mut sticks_awarded = false;
        for &w in &winners {
            if let Some(agari) = self.agari_results.get(&w).cloned() {
                let d = self._calculate_deltas(&agari, w, is_tsumo, loser, true, !sticks_awarded);
                sticks_awarded = true;
                for i in 0..4 {
                    total_deltas[i] += d[i];
                }
            }
        }

        for (i, delta) in total_deltas.iter().enumerate() {
            self.scores[i] += delta;
            self.score_deltas[i] += delta;
        }
        self.riichi_sticks = 0;

        self.last_agari_results = self.agari_results.clone();
        self.round_end_scores = Some(self.scores);
        if self._is_game_over() {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("end_game".to_string()));
            self._push_mjai_event(Value::Object(ev));
            self.is_done = true;
        } else {
            self.needs_initialize_next_round = true;
            self.pending_oya_won = winners.contains(&self.oya);
            self.pending_is_draw = false;
        }
        self.phase = Phase::WaitAct; // Reset phase to prevent re-triggering
        self.active_players = vec![];
    }

    pub(crate) fn _end_kyoku_ryukyoku(&mut self, is_renchan: bool, is_draw: bool) {
        self.round_end_scores = Some(self.scores); // Set round end scores for verification
        let mut ev = serde_json::Map::new();
        ev.insert("type".to_string(), Value::String("end_kyoku".to_string()));
        self._push_mjai_event(Value::Object(ev));

        if self._is_game_over() {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("end_game".to_string()));
            self._push_mjai_event(Value::Object(ev));
            self.is_done = true;
        } else {
            self.needs_initialize_next_round = true;
            self.pending_oya_won = is_renchan;
            self.pending_is_draw = is_draw;
        }
        self.phase = Phase::WaitAct;
        self.active_players = vec![];
    }

    pub(crate) fn _initialize_next_round(&mut self, oya_won: bool, is_draw: bool) {
        if self.is_done {
            return;
        }

        let mut next_honba = self.honba;
        let mut next_oya = self.oya;
        let mut next_round_wind = self.round_wind;

        if oya_won {
            // Renchan (Oya Win or Oya Tenpai in Draw)
            next_honba = next_honba.saturating_add(1);
        } else if is_draw {
            // Ryukyoku, Oya Nontenpai -> Rotate but keep sticks (honba + 1)
            next_honba = next_honba.saturating_add(1);
            next_oya = (next_oya + 1) % 4;
            if next_oya == 0 {
                next_round_wind += 1;
            }
        } else {
            // Ko Win -> Rotate, clear honba
            next_honba = 0;
            next_oya = (next_oya + 1) % 4;
            if next_oya == 0 {
                next_round_wind += 1;
            }
        }

        // Check game end conditions (e.g. Hanchan ends after South 4)
        // Check game end conditions
        // Tonpu: Ends if we move to South (Round Wind 1) or later?
        // Actually:
        // Tonpu (1, 4): Ends when round_wind becomes 1 (South) -> "East round finished"
        // Hanchan (2, 5): Ends when round_wind becomes 2 (West) -> "South round finished"
        // Ikkyoku (0, 3): Usually just 1 hand, but assuming standard logic:
        // If it was just one kyoku, we might rely on the caller or external config?
        // But let's assume standard behavior:

        match self.game_mode {
            1 | 4 => {
                // Tonpusen (Yon, San)
                let max_score = self.scores.iter().cloned().max().unwrap_or(0);
                if next_round_wind >= 1 && (max_score >= 30000 || next_round_wind > 1) {
                    self.is_done = true;
                    let mut ev = serde_json::Map::new();
                    ev.insert("type".to_string(), Value::String("end_game".to_string()));
                    self._push_mjai_event(Value::Object(ev));
                    return;
                }
            }
            2 | 5 => {
                // Hanchan (Yon, San)
                let max_score = self.scores.iter().cloned().max().unwrap_or(0);
                if next_round_wind >= 2 && (max_score >= 30000 || next_round_wind > 2) {
                    self.is_done = true;
                    let mut ev = serde_json::Map::new();
                    ev.insert("type".to_string(), Value::String("end_game".to_string()));
                    self._push_mjai_event(Value::Object(ev));
                    return;
                }
            }
            0 | 3 => {
                // Ikkyoku
                // End after 1 hand regardless?
                // Or allow renchan? Most Ikkyoku modes end after 1 hand.
                // Let's assume end if we transitioned (next_honba == 0 and different oya/wind)
                // Simplest: Always end after 1 hand for now to be safe.
                self.is_done = true;
                let mut ev = serde_json::Map::new();
                ev.insert("type".to_string(), Value::String("end_game".to_string()));
                self._push_mjai_event(Value::Object(ev));
                return;
            }
            _ => {
                // Unknown, maybe default to Tonpu limit?
                if next_round_wind >= 1 {
                    self.is_done = true;
                    // Emit end_game for safety?
                    let mut ev = serde_json::Map::new();
                    ev.insert("type".to_string(), Value::String("end_game".to_string()));
                    self._push_mjai_event(Value::Object(ev));
                    return;
                }
            }
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
            self._push_mjai_event(Value::Object(ev));

            self.riichi_pending_acceptance = None;
        }
    }
}

fn _tid_to_mjai_hand(hand: &[u8]) -> Vec<String> {
    hand.iter().map(|&t| tid_to_mjai(t)).collect()
}

#[pymethods]
impl RiichiEnv {
    #[new]
    #[pyo3(signature = (game_mode=None, skip_mjai_logging=false, seed=None, round_wind=None, rule=None))]
    pub fn new(
        game_mode: Option<Bound<'_, PyAny>>,
        skip_mjai_logging: bool,
        seed: Option<u64>,
        round_wind: Option<u8>,
        rule: Option<crate::rule::GameRule>,
    ) -> PyResult<Self> {
        let gt = if let Some(val) = game_mode {
            if let Ok(s) = val.extract::<String>() {
                match s.as_str() {
                    "4p-red-single" => 0,
                    "4p-red-east" => 1,
                    "4p-red-half" => 2,
                    "3p-red-single" => 3,
                    "3p-red-east" => 4,
                    "3p-red-half" => 5,
                    _ => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Unsupported game_mode: {}",
                            s
                        )))
                    }
                }
            } else if let Ok(i) = val.extract::<u8>() {
                i
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "game_mode must be str or int",
                ));
            }
        } else {
            0 // Default to 4p-red-single
        };

        let mut env = RiichiEnv {
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
            round_wind: round_wind.unwrap_or(0),
            ippatsu_cycle: [false; 4],
            game_mode: gt,
            skip_mjai_logging,
            seed,
            hand_index: 0,
            forbidden_discards: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            rule: rule.unwrap_or_default(),
            pao: [
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
            ],
        };
        Python::attach(|py| env.reset(py, None, None, round_wind, None, None, None, seed))?;
        Ok(env)
    }

    #[getter]
    fn get_wall(&self) -> Vec<u32> {
        self.wall.iter().map(|&x| x as u32).collect()
    }

    #[setter]
    fn set_wall(&mut self, wall: Vec<u32>) {
        self.wall = wall.iter().map(|&x| x as u8).collect();
    }

    #[getter]
    fn get_hands(&self) -> [Vec<u32>; 4] {
        [
            self.hands[0].iter().map(|&x| x as u32).collect(),
            self.hands[1].iter().map(|&x| x as u32).collect(),
            self.hands[2].iter().map(|&x| x as u32).collect(),
            self.hands[3].iter().map(|&x| x as u32).collect(),
        ]
    }

    #[getter]
    fn get_discards(&self) -> [Vec<u32>; 4] {
        [
            self.discards[0].iter().map(|&x| x as u32).collect(),
            self.discards[1].iter().map(|&x| x as u32).collect(),
            self.discards[2].iter().map(|&x| x as u32).collect(),
            self.discards[3].iter().map(|&x| x as u32).collect(),
        ]
    }

    #[getter]
    fn get_discard_is_riichi(&self) -> [Vec<bool>; 4] {
        self.discard_is_riichi.clone()
    }

    #[getter]
    fn get_riichi_declaration_index(&self) -> [Option<usize>; 4] {
        self.riichi_declaration_index
    }

    #[setter]
    fn set_discards(&mut self, discards: [Vec<u32>; 4]) {
        for (i, d) in discards.iter().enumerate() {
            self.discards[i] = d.iter().map(|&x| x as u8).collect();
        }
    }

    #[getter]
    fn get_active_players(&self) -> Vec<u32> {
        self.active_players.iter().map(|&x| x as u32).collect()
    }

    #[setter]
    fn set_active_players(&mut self, active_players: Vec<u32>) {
        self.active_players = active_players.iter().map(|&x| x as u8).collect();
    }

    #[getter]
    fn get_dora_indicators(&self) -> Vec<u32> {
        self.dora_indicators.iter().map(|&x| x as u32).collect()
    }

    #[setter]
    fn set_dora_indicators(&mut self, dora_indicators: Vec<u32>) {
        self.dora_indicators = dora_indicators.iter().map(|&x| x as u8).collect();
    }

    #[getter]
    fn get_forbidden_discards(&self) -> [Vec<u32>; 4] {
        [
            self.forbidden_discards[0]
                .iter()
                .map(|&x| x as u32)
                .collect(),
            self.forbidden_discards[1]
                .iter()
                .map(|&x| x as u32)
                .collect(),
            self.forbidden_discards[2]
                .iter()
                .map(|&x| x as u32)
                .collect(),
            self.forbidden_discards[3]
                .iter()
                .map(|&x| x as u32)
                .collect(),
        ]
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
    ) -> PyResult<Py<PyAny>> {
        // ... existing reset impl ...
        if let Some(s) = seed {
            self.seed = Some(s);
        }
        self.hand_index = 0;

        // Reset MJAI log for new game/episode
        self.mjai_log.clear();
        for log in self.mjai_log_per_player.iter_mut() {
            log.clear();
        }
        self.player_event_counts = [0; 4];

        let initial_scores = if let Some(sc) = scores {
            let mut s = [0; 4];
            s.copy_from_slice(&sc[..4]);
            s
        } else {
            [25000; 4]
        };

        if self.mjai_log.is_empty() && !self.skip_mjai_logging {
            let mut start_game = serde_json::Map::new();
            start_game.insert("type".to_string(), Value::String("start_game".to_string()));
            start_game.insert("id".to_string(), Value::Number(0.into()));
            // names skipped for brevity
            self._push_mjai_event(Value::Object(start_game));
        }

        self.agari_results = HashMap::new();
        self.last_agari_results = HashMap::new();
        self.round_end_scores = None;
        self.pao = [
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        ];

        self._initialize_round(
            oya.unwrap_or(self.oya),
            bakaze.unwrap_or(self.round_wind),
            honba.unwrap_or(self.honba),
            kyotaku.unwrap_or(self.riichi_sticks),
            wall,
            Some(initial_scores),
        );

        self.step(py, HashMap::new())
    }

    #[pyo3(signature = (players=None))]
    fn get_obs_py<'py>(
        &mut self,
        py: Python<'py>,
        players: Option<Vec<u8>>,
    ) -> PyResult<Py<PyAny>> {
        Ok(self
            .get_observations(players)
            .into_pyobject(py)?
            .unbind()
            .into())
    }

    pub fn set_scores(&mut self, scores: Vec<i32>) -> PyResult<()> {
        if scores.len() != 4 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Scores must be length 4",
            ));
        }
        self.scores.copy_from_slice(&scores);
        Ok(())
    }

    #[pyo3(signature = (oya=None, honba=None, kyoku_idx=None, round_wind=None))]
    pub fn set_state(
        &mut self,
        oya: Option<u8>,
        honba: Option<u8>,
        kyoku_idx: Option<u8>,
        round_wind: Option<u8>,
    ) -> PyResult<()> {
        if let Some(v) = oya {
            self.oya = v;
            self.kyoku_idx = v;
        }
        if let Some(v) = honba {
            self.honba = v;
        }
        if let Some(v) = kyoku_idx {
            self.kyoku_idx = v;
            self.oya = v;
        }
        if let Some(v) = round_wind {
            self.round_wind = v;
        }
        Ok(())
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

    pub fn ranks(&self) -> [u32; 4] {
        let mut indices: Vec<usize> = (0..4).collect();
        indices.sort_by(|&a, &b| {
            if self.scores[a] != self.scores[b] {
                self.scores[b].cmp(&self.scores[a])
            } else {
                a.cmp(&b)
            }
        });
        let mut ranks = [0; 4];
        for (rank, &idx) in indices.iter().enumerate() {
            ranks[idx] = (rank + 1) as u32;
        }
        ranks
    }

    #[pyo3(signature = (preset_rule=None))]
    pub fn points(&self, preset_rule: Option<&str>) -> PyResult<[i32; 4]> {
        let rule_str = preset_rule.unwrap_or("basic");
        let (soten_weight, soten_base, jun_weight) = match rule_str {
            "basic" => (1, 25000, [50, 10, -10, -50]),
            "ouza-tyoujyo" => (0, 25000, [100, 40, -40, -100]),
            "ouza-normal" => (0, 25000, [50, 20, -20, -50]),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown preset rule: {}",
                    rule_str
                )))
            }
        };

        let ranks = self.ranks();
        let mut points = [0; 4];
        for i in 0..4 {
            points[i] = ((self.scores[i] - soten_base) as f64 / 1000.0 * soten_weight as f64
                + jun_weight[ranks[i] as usize - 1] as f64) as i32;
        }
        Ok(points)
    }

    pub fn _get_waits(&self, pid: u8) -> HashSet<u8> {
        let mut waits = HashSet::new();
        let hand = &self.hands[pid as usize];
        let melds = &self.melds[pid as usize];

        let calc = crate::agari_calculator::AgariCalculator::new(hand.clone(), melds.clone());
        if !calc.is_tenpai() {
            return waits;
        }

        for t_type in 0..34 {
            let win_tile = t_type * 4;
            let cond = Conditions {
                tsumo: false,
                riichi: self.riichi_declared[pid as usize],
                double_riichi: self.double_riichi_declared[pid as usize],
                player_wind: crate::types::Wind::from((pid + 4 - self.oya) % 4),
                round_wind: crate::types::Wind::from(self.round_wind),
                ..Default::default()
            };

            let res = calc.calc(
                win_tile,
                self.dora_indicators.clone(),
                vec![], // No Ura indicators for waits check
                Some(cond),
            );
            if res.agari {
                waits.insert(t_type);
            }
        }
        waits
    }

    pub fn _is_furiten(&self, pid: u8) -> bool {
        if self.missed_agari_riichi[pid as usize] || self.missed_agari_doujun[pid as usize] {
            return true;
        }

        let waits = self._get_waits(pid);
        if waits.is_empty() {
            return false;
        }

        for &t in &self.discards[pid as usize] {
            if waits.contains(&(t / 4)) {
                return true;
            }
        }
        false
    }

    fn _check_midway_draws(&mut self) -> bool {
        // 1. Sufuurenta
        {
            let mut discards = Vec::new();
            for p in 0..4 {
                if self.discards[p].len() != 1 {
                    break;
                }
                discards.push(self.discards[p][0]);
            }
            if discards.len() == 4 {
                let t = discards[0] / 4;
                if (27..=30).contains(&t)
                    && discards.iter().all(|&d| d / 4 == t)
                    && self.melds.iter().all(|m| m.is_empty())
                {
                    self._trigger_ryukyoku("sufuurenta");
                    return true;
                }
            }
        }

        // 2. Suukansansen
        {
            let mut tk = 0;
            let mut kp = HashSet::new();
            for (p, ms) in self.melds.iter().enumerate() {
                for m in ms {
                    if matches!(
                        m.meld_type,
                        MeldType::Gang | MeldType::Angang | MeldType::Addgang
                    ) {
                        tk += 1;
                        kp.insert(p);
                    }
                }
            }
            if tk >= 4 && kp.len() > 1 {
                self._trigger_ryukyoku("suukansansen");
                return true;
            }
        }

        // 3. Suurechi
        if self.riichi_declared.iter().all(|&r| r) {
            self._trigger_ryukyoku("suurechi");
            return true;
        }

        false
    }

    #[getter]
    pub fn mjai_log(&self, py: Python) -> PyResult<Py<PyAny>> {
        let json = py.import("json")?;
        let loads = json.getattr("loads")?;
        let list = pyo3::types::PyList::empty(py);
        for s in &self.mjai_log {
            list.append(loads.call1((s,))?)?;
        }
        Ok(list.unbind().into())
    }

    #[setter]
    pub fn set_phase(&mut self, val: Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(i) = val.extract::<i32>() {
            self.phase = match i {
                0 => Phase::WaitAct,
                1 => Phase::WaitResponse,
                _ => Phase::WaitAct,
            };
        } else if let Ok(p) = val.extract::<Phase>() {
            self.phase = p;
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Expected int or Phase",
            ));
        }
        Ok(())
    }

    #[getter]
    pub fn current_player(&self) -> u8 {
        self.current_player
    }

    #[setter]
    pub fn set_current_player(&mut self, val: u8) {
        self.current_player = val % 4;
        self.missed_agari_doujun[self.current_player as usize] = false;
    }

    #[getter]
    pub fn last_discard(&self) -> Option<(u8, u8)> {
        self.last_discard
    }

    #[setter]
    pub fn set_last_discard(&mut self, val: Option<(u8, u8)>) {
        self.last_discard = val;
    }

    pub fn _get_legal_actions(&self, pid: u8) -> Vec<Action> {
        // Internal method reused for Python testing
        // Requires importing _get_legal_actions logic (which is private but available to struct)
        self._get_legal_actions_internal(pid)
    }

    pub fn done(&self) -> bool {
        self.is_done
    }

    pub fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: HashMap<u8, Action>,
    ) -> PyResult<Py<PyAny>> {
        if self.is_done {
            return self.get_obs_py(py, None);
        }
        // --- Added: Validation for illegal actions ---
        if !self.is_done {
            let mut illegal_actor: Option<u8> = None;
            let mut sorted_keys: Vec<u8> = actions.keys().cloned().collect();
            sorted_keys.sort();

            for pid in sorted_keys {
                // Determine if this player interaction is expected
                let is_expected = if self.phase == Phase::WaitAct {
                    pid == self.current_player && !self.needs_tsumo
                } else {
                    // WaitResponse
                    self.active_players.contains(&pid)
                };

                // If the player is not expected to act in this phase, treat it as an illegal action.
                if !is_expected {
                    illegal_actor = Some(pid);
                    break;
                }

                let legals = self._get_legal_actions_internal(pid);
                let action = &actions[&pid];

                // Check strictly using PartialEq (requires sorted consume_tiles)
                // Relaxed Check: If action.consume_tiles is empty, match based on type and tile only.
                let is_legal = if action.consume_tiles.is_empty() {
                    legals
                        .iter()
                        .any(|l| l.action_type == action.action_type && l.tile == action.tile)
                } else if legals.contains(action)
                    || (action.action_type == ActionType::Ankan
                        && legals.iter().any(|l| {
                            l.action_type == ActionType::Ankan
                                && l.consume_tiles == action.consume_tiles
                        }))
                {
                    true
                } else {
                    // Relaxed Check: Match structural validity (equivalent tiles) AND actual hand possession
                    let matching_legal = legals.iter().find(|l| {
                        l.action_type == action.action_type
                            && l.tile == action.tile
                            && l.consume_tiles.len() == action.consume_tiles.len()
                            && l.consume_tiles.iter().zip(action.consume_tiles.iter()).all(
                                |(&lt, &at)| {
                                    // Equivalent if same ID OR (Same type and Both are Non-Red)
                                    // Red tiles: 16, 52, 88.
                                    // If one is red, they must be identical (strict match).
                                    // If both are non-red, they match if same type (t/4).
                                    let is_red = |t: u8| t == 16 || t == 52 || t == 88;
                                    if lt == at {
                                        true
                                    } else {
                                        !is_red(lt) && !is_red(at) && lt / 4 == at / 4
                                    }
                                },
                            )
                    });

                    if let Some(_match) = matching_legal {
                        // Structurally valid. Now check if player actually HAS these tiles.
                        let hand = &self.hands[pid as usize];
                        let mut temp_hand = hand.clone();
                        let mut has_all = true;
                        for &c in &action.consume_tiles {
                            if let Some(pos) = temp_hand.iter().position(|&h| h == c) {
                                temp_hand.remove(pos);
                            } else {
                                has_all = false;
                                break;
                            }
                        }
                        has_all
                    } else {
                        false
                    }
                };

                if !is_legal {
                    illegal_actor = Some(pid);
                    break;
                }
            }

            if let Some(offender) = illegal_actor {
                self._trigger_error_penalty(offender, "Illegal Action".to_string());
                return self.get_obs_py(py, Some(self.active_players.clone()));
            }
        }
        // ---------------------------------------------

        while !self.is_done {
            if self.needs_initialize_next_round {
                self._initialize_next_round(self.pending_oya_won, self.pending_is_draw);
                if self.is_done {
                    // Game ended during initialization (e.g. Sudden Death)
                    return self.get_obs_py(py, Some(self.active_players.clone()));
                }
            }
            if self.needs_tsumo {
                // Midway draws logic
                if self._check_midway_draws() {
                    return self.get_obs_py(py, Some(self.active_players.clone()));
                }
                // Exhaustive draw check
                if self.wall.len() <= 14 {
                    self._trigger_ryukyoku("exhaustive_draw");
                    return self.get_obs_py(py, Some(self.active_players.clone()));
                }

                if self.is_rinshan_flag {
                    if !self.wall.is_empty() {
                        let t = self.wall.remove(0);
                        self.drawn_tile = Some(t);
                        self.hands[self.current_player as usize].push(t);
                        // self.is_rinshan_flag = false; // Keep flag for Agari/Test check
                        self.rinshan_draw_count += 1;
                    }
                } else if !self.wall.is_empty() {
                    let t = self.wall.pop().unwrap();
                    self.drawn_tile = Some(t);
                    self.hands[self.current_player as usize].push(t);
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
                    self._push_mjai_event(Value::Object(ev));
                } else {
                    // Should have triggered ryukyoku?
                }

                self.needs_tsumo = false;
                self.missed_agari_doujun[self.current_player as usize] = false;
                self.active_players = vec![self.current_player];

                if !self.skip_mjai_logging || !self.riichi_declared[self.current_player as usize] {
                    return self.get_obs_py(py, Some(self.active_players.clone()));
                }
                continue;
            }

            // Phase handling
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
                            ippatsu: self.ippatsu_cycle[winner as usize],
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
                        // self.hands already contains the drawn tile (14 tiles).
                        // AgariCalculator::calc(tile) usually adds the win_tile to the hand check.
                        // So we should pass 13 tiles to AgariCalculator::new.
                        if let Some(idx) = hand_for_calc.iter().rposition(|&t| t == tile) {
                            hand_for_calc.remove(idx);
                        }
                        let calc = crate::agari_calculator::AgariCalculator::new(
                            hand_for_calc,
                            self.melds[winner as usize].clone(),
                        );
                        let ura = if self.riichi_declared[winner as usize] {
                            self._get_ura_markers_raw()
                        } else {
                            vec![]
                        };
                        let agari = calc.calc(tile, self.dora_indicators.clone(), ura, Some(cond));

                        let deltas =
                            self._calculate_deltas(&agari, winner, true, Some(winner), true, true);

                        // Log Hora
                        let mut ev = serde_json::Map::new();
                        ev.insert("type".to_string(), Value::String("hora".to_string()));
                        ev.insert("actor".to_string(), Value::Number(winner.into()));
                        ev.insert("target".to_string(), Value::Number(winner.into()));
                        ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
                        ev.insert("tsumo".to_string(), Value::Bool(true));
                        ev.insert("deltas".to_string(), serde_json::to_value(deltas).unwrap());
                        let uras = if self.riichi_declared[winner as usize] {
                            serde_json::to_value(self._get_ura_markers()).unwrap()
                        } else {
                            Value::Array(vec![])
                        };
                        ev.insert("ura_markers".to_string(), uras);
                        self._push_mjai_event(Value::Object(ev));

                        let mut agaris = HashMap::new();
                        agaris.insert(winner, agari);
                        self._end_kyoku_win(vec![winner], true, Some(winner), agaris);
                        return self.get_obs_py(py, Some(self.active_players.clone()));
                    }

                    if act.action_type == ActionType::Kakan {
                        let pid = self.current_player;
                        let tile = act.tile.unwrap();
                        let p_usize = pid as usize;
                        // Validate: Tile in hand + Pon in melds
                        let mut has_tile_idx = None;
                        if let Some(idx) = self.hands[p_usize].iter().position(|&t| t == tile) {
                            has_tile_idx = Some(idx);
                        } else if let Some(dt) = self.drawn_tile {
                            if dt == tile {
                                has_tile_idx = self.hands[p_usize].len().checked_sub(1);
                                // Drawn tile is last?
                                // Actually we should look up by value in hands, as we pushed it.
                                // The iter position found it if it's there.
                            }
                        }

                        let meld_idx = self.melds[p_usize].iter().position(|m| {
                            m.meld_type == MeldType::Peng && m.tiles[0] / 4 == tile / 4
                        });

                        if let (Some(h_idx), Some(m_idx)) = (has_tile_idx, meld_idx) {
                            // Check Chankan
                            let mut chankan_ronners = Vec::new();
                            for i in 0..4 {
                                if i == pid {
                                    continue;
                                }
                                // Check missed agari (Temporary Furiten)
                                if self.missed_agari_doujun[i as usize]
                                    || self.missed_agari_riichi[i as usize]
                                {
                                    continue;
                                }
                                // Enhanced Check (Furiten)
                                let hand = &self.hands[i as usize];
                                let melds = &self.melds[i as usize];
                                let calc = crate::agari_calculator::AgariCalculator::new(
                                    hand.clone(),
                                    melds.clone(),
                                );
                                if calc
                                    .get_waits_u8()
                                    .iter()
                                    .any(|&w| self.discards[i as usize].iter().any(|&d| d / 4 == w))
                                {
                                    continue;
                                }

                                if self._check_ron(i, tile, pid, true) {
                                    chankan_ronners.push(i);
                                }
                            }

                            if !chankan_ronners.is_empty() {
                                // Log Kakan Event IMMEDIATELY so Ronners can see it
                                let mut ev = serde_json::Map::new();
                                ev.insert("type".to_string(), Value::String("kakan".to_string()));
                                ev.insert("actor".to_string(), Value::Number(pid.into()));
                                ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
                                // Consumed: Original Pon tiles (from meld or act?)
                                // act.consume_tiles usually has 3 tiles for Kakan?
                                // Actually Kakan consumes 1 tile from hand + 2 from Pon?
                                // Standard MJAI Kakan 'consumed' field lists the 3 tiles of the Pon?
                                // Or the 3 tiles involved?
                                // Looking at Block D (Line 1255): It serves `m.tiles` (taken 3).
                                // `m` is the meld.
                                let m_curr = &self.melds[p_usize][m_idx];
                                let consumed_mjai: Vec<Value> = m_curr
                                    .tiles
                                    .iter()
                                    .take(3)
                                    .map(|&t| Value::String(tid_to_mjai(t)))
                                    .collect();
                                ev.insert("consumed".to_string(), Value::Array(consumed_mjai));
                                self._push_mjai_event(Value::Object(ev));

                                self.pending_kan = Some((pid, act.clone()));
                                self.phase = Phase::WaitResponse;
                                self.active_players = chankan_ronners;
                                self.active_players.sort();
                                self.needs_tsumo = false;
                                return self.get_obs_py(py, Some(self.active_players.clone()));
                            }

                            // Execute Kakan
                            let t = self.hands[p_usize].remove(h_idx);
                            let mut m = self.melds[p_usize][m_idx].clone();
                            m.meld_type = MeldType::Addgang;
                            m.tiles.push(t);
                            m.tiles.sort();
                            self.hands[p_usize].sort();
                            self.drawn_tile = None;

                            self.is_rinshan_flag = true;
                            self.needs_tsumo = true;
                            self.active_players = vec![];
                            // Log Kakan
                            let mut ev = serde_json::Map::new();
                            ev.insert("type".to_string(), Value::String("kakan".to_string()));
                            ev.insert("actor".to_string(), Value::Number(pid.into()));
                            ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
                            // FIX: Only take 3 tiles for consumed
                            let consumed_mjai: Vec<Value> = m
                                .tiles
                                .iter()
                                .take(3)
                                .map(|&t| Value::String(tid_to_mjai(t)))
                                .collect();
                            ev.insert("consumed".to_string(), Value::Array(consumed_mjai));
                            self._push_mjai_event(Value::Object(ev));

                            self.melds[p_usize][m_idx] = m;

                            self._reveal_kan_dora();
                            self._check_midway_draws();

                            if !self.skip_mjai_logging {
                                // return self.get_obs_py(py, Some(vec![pid]));
                            }
                            continue;
                        }
                    }

                    if act.action_type == ActionType::Ankan {
                        let pid = self.current_player;
                        let tile = act.tile.unwrap();
                        let p_usize = pid as usize;
                        // Validate: 4 tiles
                        let t_type = tile / 4;
                        let indices: Vec<usize> = self.hands[p_usize]
                            .iter()
                            .enumerate()
                            .filter(|(_, &t)| t / 4 == t_type)
                            .map(|(i, _)| i)
                            .collect();

                        if indices.len() >= 4 {
                            // Check Chankan (Kokushi)
                            let mut chankan_ronners = Vec::new();
                            // If rule allows Ron on Ankan for Kokushi check
                            // Standard: Tenhou (False), MJSoul (True)
                            let check_chankan = self.rule.allows_ron_on_ankan_for_kokushi_musou;

                            if check_chankan {
                                for i in 0..4 {
                                    if i == pid {
                                        continue;
                                    }
                                    // Check missed agari (Temporary Furiten)
                                    if self.missed_agari_doujun[i as usize]
                                        || self.missed_agari_riichi[i as usize]
                                    {
                                        continue;
                                    }
                                    // Kokushi Check (Ankan Chankan restriction)
                                    let melds = &self.melds[i as usize];
                                    if !melds.is_empty() {
                                        continue;
                                    }

                                    let hand = &self.hands[i as usize];
                                    let calc = crate::agari_calculator::AgariCalculator::new(
                                        hand.clone(),
                                        melds.clone(),
                                    );

                                    // Furiten Check
                                    if calc.get_waits_u8().iter().any(|&w| {
                                        self.discards[i as usize].iter().any(|&d| d / 4 == w)
                                    }) {
                                        continue;
                                    }

                                    // Agari check with Kokushi constraint
                                    let cond = Conditions {
                                        chankan: true,
                                        player_wind: Wind::from((i + 4 - self.oya) % 4),
                                        round_wind: Wind::from(self.round_wind),
                                        ..Default::default()
                                    };
                                    let agari = calc.calc(
                                        tile,
                                        self.dora_indicators.clone(),
                                        vec![],
                                        Some(cond),
                                    );

                                    if agari.agari
                                        && (agari.yaku.contains(&yaku::ID_KOKUSHI)
                                            || agari.yaku.contains(&yaku::ID_KOKUSHI_13))
                                    {
                                        chankan_ronners.push(i);
                                    }
                                }
                            }

                            if !chankan_ronners.is_empty() {
                                // Log Ankan Event IMMEDIATELY
                                let mut consumed: Vec<u8> =
                                    indices.iter().map(|&i| self.hands[p_usize][i]).collect();
                                consumed.sort();

                                let mut ev = serde_json::Map::new();
                                ev.insert("type".to_string(), Value::String("ankan".to_string()));
                                ev.insert("actor".to_string(), Value::Number(pid.into()));
                                ev.insert(
                                    "consumed".to_string(),
                                    Value::Array(
                                        consumed
                                            .iter()
                                            .map(|&t| Value::String(tid_to_mjai(t)))
                                            .collect(),
                                    ),
                                );
                                self._push_mjai_event(Value::Object(ev));

                                self.pending_kan = Some((pid, act.clone()));
                                self.phase = Phase::WaitResponse;
                                self.active_players = chankan_ronners;
                                self.active_players.sort();
                                self.needs_tsumo = false;
                                return self.get_obs_py(py, Some(self.active_players.clone()));
                            }

                            // Execute Ankan
                            // Remove 4 tiles (reverse index to avoid shift)
                            let mut consumed = Vec::new();
                            for &idx in indices.iter().rev() {
                                consumed.push(self.hands[p_usize].remove(idx));
                            }
                            consumed.sort(); // Should be 4 sorted tiles

                            let m = Meld {
                                meld_type: MeldType::Angang,
                                tiles: consumed.clone(),
                                opened: false, // Ankan is closed
                                from_who: -1,
                            };
                            self.melds[p_usize].push(m);
                            self.hands[p_usize].sort();
                            self.drawn_tile = None;

                            self.is_rinshan_flag = true;
                            self.needs_tsumo = true;
                            self.active_players = vec![];

                            // Log Ankan
                            let mut ev = serde_json::Map::new();
                            ev.insert("type".to_string(), Value::String("ankan".to_string()));
                            ev.insert("actor".to_string(), Value::Number(pid.into()));
                            ev.insert(
                                "consumed".to_string(),
                                Value::Array(
                                    consumed
                                        .iter()
                                        .map(|&t| Value::String(tid_to_mjai(t)))
                                        .collect(),
                                ),
                            );
                            self._push_mjai_event(Value::Object(ev));

                            self._reveal_kan_dora();
                            self._check_midway_draws();

                            if !self.skip_mjai_logging {
                                // return self.get_obs_py(py, Some(vec![pid]));
                            }
                            continue;
                        }
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
                        self._push_mjai_event(Value::Object(ev));
                        return self.get_obs_py(py, Some(vec![self.current_player]));
                    }

                    if act.action_type == ActionType::KyushuKyuhai {
                        self._trigger_ryukyoku("kyushu_kyuhai");
                        continue;
                    }
                }
            } else if self.phase == Phase::WaitResponse {
                // Check if all active players have responded
                if !self.active_players.iter().all(|p| actions.contains_key(p)) {
                    return self.get_obs_py(py, Some(self.active_players.clone()));
                }

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

                    let tile = if let Some(ld) = self.last_discard {
                        ld.1
                    } else if let Some((_, ref pact)) = self.pending_kan {
                        pact.tile.unwrap()
                    } else {
                        0 // Should trigger panic or error?
                    };
                    let mut agaris = HashMap::new();
                    for (idx, &winner) in sorted_ronners.iter().enumerate() {
                        let cond = Conditions {
                            tsumo: false,
                            riichi: self.riichi_declared[winner as usize],
                            double_riichi: self.double_riichi_declared[winner as usize],
                            ippatsu: self.ippatsu_cycle[winner as usize],
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
                        let ura = if self.riichi_declared[winner as usize] {
                            self._get_ura_markers_raw()
                        } else {
                            vec![]
                        };
                        let agari = calc.calc(tile, self.dora_indicators.clone(), ura, Some(cond));

                        // Calculate Deltas
                        let deltas = self._calculate_deltas(
                            &agari,
                            winner,
                            false,
                            Some(discarder),
                            idx == 0,
                            idx == 0,
                        );

                        // Log Hora
                        let mut ev = serde_json::Map::new();
                        ev.insert("type".to_string(), Value::String("hora".to_string()));
                        ev.insert("actor".to_string(), Value::Number(winner.into()));
                        ev.insert("target".to_string(), Value::Number(discarder.into()));
                        ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
                        ev.insert("deltas".to_string(), serde_json::to_value(deltas).unwrap());

                        let uras = if self.riichi_declared[winner as usize] {
                            serde_json::to_value(self._get_ura_markers()).unwrap()
                        } else {
                            Value::Array(vec![])
                        };
                        ev.insert("ura_markers".to_string(), uras);
                        self._push_mjai_event(Value::Object(ev));

                        agaris.insert(winner, agari);
                    }

                    self._end_kyoku_win(sorted_ronners, false, Some(discarder), agaris);
                    return self.get_obs_py(py, Some(self.active_players.clone()));
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
                    let action = valid_actions[&claimer].clone();
                    let tile = self.last_discard.unwrap().1;

                    self._execute_claim(claimer, action.clone(), Some(self.current_player))?;

                    if let Some((_, pk)) = self.pending_kan.clone() {
                        if pk.action_type == ActionType::Kakan
                            || pk.action_type == ActionType::Daiminkan
                        {
                            // Resuming from Kan interruption (all passed)
                            self.is_rinshan_flag = true;
                            self.riichi_stage = [false; 4]; // Ippatsu eligible reset
                            self.needs_tsumo = true;
                            self.phase = Phase::WaitAct;
                            self.active_players = vec![];
                            self.pending_kan = None;
                            self._reveal_kan_dora();
                            self._check_midway_draws();
                            if !self.skip_mjai_logging {
                                return self.get_obs_py(py, Some(vec![]));
                            }
                            continue; // Proceed to draw
                        }
                    }

                    if action.action_type == ActionType::Daiminkan {
                        // Pause for Chankan
                        self.phase = Phase::WaitResponse;
                        let mut chankan_ronners = Vec::new();
                        for pid in 0..4 {
                            if pid == claimer {
                                continue;
                            }
                            if self.is_done {
                                continue;
                            }
                            if self.missed_agari_doujun[pid as usize]
                                || self.missed_agari_riichi[pid as usize]
                            {
                                continue;
                            }
                            let is_furiten = self.discards[pid as usize].contains(&tile);
                            if is_furiten {
                                continue;
                            }
                            if self._check_ron(pid, tile, claimer, false) {
                                chankan_ronners.push(pid);
                            }
                        }

                        if !chankan_ronners.is_empty() {
                            self.active_players = chankan_ronners.clone();
                            self.active_players.sort();
                            self.pending_kan = Some((claimer, action)); // Store action to resume later
                            self.current_claims = HashMap::new();
                            for &pid in &self.active_players {
                                self.current_claims.insert(
                                    pid,
                                    vec![Action::new(ActionType::Ron, Some(tile), vec![])],
                                );
                            }
                            if !self.skip_mjai_logging {
                                // return self.get_obs_py(py, Some(self.active_players.clone()));
                            }
                        } else {
                            // No Chankan, proceed with Daiminkan
                            self.current_player = claimer;
                            self.phase = Phase::WaitAct;
                            self.is_rinshan_flag = true;
                            self.needs_tsumo = true;
                            self.active_players = vec![];
                            self._reveal_kan_dora();
                            self._check_midway_draws();
                            continue;
                        }
                    } else {
                        // Pon
                        self.current_player = claimer;
                        self.missed_agari_doujun[claimer as usize] = false;
                        self.phase = Phase::WaitAct;
                        self.is_rinshan_flag = false;
                        self.drawn_tile = None;
                        self.needs_tsumo = false;
                        self.active_players = vec![claimer];
                    }

                    if !self.skip_mjai_logging {
                        // return self.get_obs_py(py, Some(self.active_players.clone()));
                    }
                    continue;
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
                    )?;
                    self.current_player = claimer;
                    self.missed_agari_doujun[claimer as usize] = false;
                    self.phase = Phase::WaitAct;
                    self.active_players = vec![claimer];
                    self.is_rinshan_flag = false;
                    self.drawn_tile = None;
                    self.needs_tsumo = false;

                    return self.get_obs_py(py, Some(self.active_players.clone()));
                }

                if self.pending_kan.is_some() {
                    let (pid, act) = self.pending_kan.take().unwrap();
                    let p_usize = pid as usize;

                    if act.action_type == ActionType::Kakan {
                        let tile = act.tile.unwrap();
                        // We need to find h_idx and m_idx again? Or store them?
                        // Recalculate to be safe.
                        let mut has_tile_idx = None;
                        if let Some(idx) = self.hands[p_usize].iter().position(|&t| t == tile) {
                            has_tile_idx = Some(idx);
                        } else if let Some(dt) = self.drawn_tile {
                            if dt == tile {
                                has_tile_idx = self.hands[p_usize].len().checked_sub(1);
                            }
                        }

                        let meld_idx = self.melds[p_usize].iter().position(|m| {
                            m.meld_type == MeldType::Peng && m.tiles[0] / 4 == tile / 4
                        });

                        if let (Some(h_idx), Some(m_idx)) = (has_tile_idx, meld_idx) {
                            // Execute Kakan
                            let t = self.hands[p_usize].remove(h_idx);

                            // Capture original Pon tiles for logging
                            let _consumed_tiles = self.melds[p_usize][m_idx].tiles.clone();

                            let mut m = self.melds[p_usize][m_idx].clone();
                            m.meld_type = MeldType::Addgang;
                            m.tiles.push(t);
                            m.tiles.sort();
                            self.melds[p_usize][m_idx] = m;
                            self.hands[p_usize].sort();
                            self.drawn_tile = None;

                            self.is_rinshan_flag = true;
                            self.needs_tsumo = true;
                            self.active_players = vec![];

                            // Log Kakan - ALREADY LOGGED IN CHANKAN CHECK
                            // let mut ev = serde_json::Map::new();
                            // ev.insert("type".to_string(), Value::String("kakan".to_string()));
                            // ev.insert("actor".to_string(), Value::Number(pid.into()));
                            // ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
                            // ev.insert(
                            //     "consumed".to_string(),
                            //     Value::Array(
                            //         act.consume_tiles
                            //             .iter()
                            //             .take(3)
                            //             .map(|&t| Value::String(tid_to_mjai(t)))
                            //             .collect(),
                            //     ),
                            // );
                            // self._push_mjai_event(Value::Object(ev));

                            self._reveal_kan_dora();
                            self._check_midway_draws();

                            self.current_player = pid; // Remain current player
                            self.phase = Phase::WaitAct;
                            if !self.skip_mjai_logging {
                                // return self.get_obs_py(py, Some(vec![pid]));
                            }
                            continue;
                        }
                    } else if act.action_type == ActionType::Ankan {
                        let tile = act.tile.unwrap();
                        let t_type = tile / 4;
                        let indices: Vec<usize> = self.hands[p_usize]
                            .iter()
                            .enumerate()
                            .filter(|(_, &t)| t / 4 == t_type)
                            .map(|(i, _)| i)
                            .collect();

                        if indices.len() >= 4 {
                            // Execute Ankan
                            let mut consumed = Vec::new();
                            for &idx in indices.iter().rev() {
                                consumed.push(self.hands[p_usize].remove(idx));
                            }
                            consumed.sort();

                            let m = Meld {
                                meld_type: MeldType::Angang,
                                tiles: consumed.clone(),
                                opened: false,
                                from_who: -1,
                            };
                            self.melds[p_usize].push(m);
                            self.hands[p_usize].sort();
                            self.drawn_tile = None;

                            self.is_rinshan_flag = true;
                            self.needs_tsumo = true;
                            self.active_players = vec![];

                            // Log Ankan - ALREADY LOGGED IN CHANKAN CHECK
                            // let mut ev = serde_json::Map::new();
                            // ev.insert("type".to_string(), Value::String("ankan".to_string()));
                            // ev.insert("actor".to_string(), Value::Number(pid.into()));
                            // ev.insert(
                            //     "consumed".to_string(),
                            //     Value::Array(
                            //         consumed
                            //             .iter()
                            //             .map(|&t| Value::String(tid_to_mjai(t)))
                            //             .collect(),
                            //     ),
                            // );
                            // self._push_mjai_event(Value::Object(ev));

                            self._reveal_kan_dora();
                            self._check_midway_draws();

                            self.current_player = pid;
                            self.phase = Phase::WaitAct;

                            if !self.skip_mjai_logging {
                                // return self.get_obs_py(py, Some(vec![pid]));
                            }
                            continue;
                        }
                    }
                }

                // 6. All Passed -> Next Player
                let discarder = self.current_player;

                if self.riichi_stage[discarder as usize] {
                    self.riichi_stage[discarder as usize] = false;
                    self.riichi_declared[discarder as usize] = true;
                    self.scores[discarder as usize] -= 1000;
                    self.score_deltas[discarder as usize] -= 1000;
                    self.riichi_sticks += 1;
                    self.ippatsu_cycle[discarder as usize] = true;

                    // Check for Double Riichi
                    if self.is_first_turn
                        && self.discards[discarder as usize].len() == 1
                        && self.melds.iter().all(|m| m.is_empty())
                    {
                        self.double_riichi_declared[discarder as usize] = true;
                    }

                    let mut ev = serde_json::Map::new();
                    ev.insert(
                        "type".to_string(),
                        Value::String("reach_accepted".to_string()),
                    );
                    ev.insert("actor".to_string(), Value::Number(discarder.into()));
                    self._push_mjai_event(Value::Object(ev));
                }

                self.current_player = (discarder + 1) % 4;
                self.phase = Phase::WaitAct;
                self.needs_tsumo = true;
                self.active_players = vec![];
                self.current_claims.clear(); // Clear stale claims
                continue; // Loop back for tsumo
            }

            // Fallback to break loop
            break;
        }
        self.get_obs_py(py, Some(self.active_players.clone()))
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

    pub fn _get_ura_markers_raw(&self) -> Vec<u8> {
        let mut uras = Vec::new();
        for i in 0..self.dora_indicators.len() {
            let target_idx = (5 + 2 * i) as isize - self.rinshan_draw_count as isize;
            if target_idx >= 0 && (target_idx as usize) < self.wall.len() {
                uras.push(self.wall[target_idx as usize]);
            }
        }
        uras
    }

    pub fn _get_ura_markers_u8(&self) -> Vec<u32> {
        self._get_ura_markers_raw()
            .iter()
            .map(|&x| x as u32)
            .collect()
    }
}

impl RiichiEnv {
    fn _perform_discard(&mut self, pid: u8, tile: u8, is_tsumogiri: bool) {
        self.is_rinshan_flag = false; // Clear Rinshan flag on discard
        self.ippatsu_cycle[pid as usize] = false; // Discard ends your Ippatsu chance
        self.forbidden_discards[pid as usize].clear();
        let tsumogiri = is_tsumogiri;
        // Simplified Discard Logic (14 tiles hand)
        if let Some(pos) = self.hands[pid as usize].iter().position(|&t| t == tile) {
            self.hands[pid as usize].remove(pos);
            self.hands[pid as usize].sort();
        } else {
            // Should not happen if valid action
            // But if tsumogiri and tile was drawn_tile (which is in hand), it should find it.
        }

        self.discards[pid as usize].push(tile);
        self.discard_from_hand[pid as usize].push(!tsumogiri);
        let is_riichi_decl = self.riichi_stage[pid as usize];
        self.discard_is_riichi[pid as usize].push(is_riichi_decl);
        if is_riichi_decl {
            self.riichi_declaration_index[pid as usize] =
                Some(self.discards[pid as usize].len() - 1);
        }

        self.drawn_tile = None;
        self.last_discard = Some((pid, tile));

        let mut ev = serde_json::Map::new();
        ev.insert("type".to_string(), Value::String("dahai".to_string()));
        ev.insert("actor".to_string(), Value::Number(pid.into()));
        ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
        ev.insert("tsumogiri".to_string(), Value::Bool(tsumogiri));
        self._push_mjai_event(Value::Object(ev));

        self.missed_agari_doujun[pid as usize] = false; // Discard ends temporary furiten
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
                self.score_deltas[pid as usize] -= 1000;
                self.riichi_sticks += 1;
                self.ippatsu_cycle[pid as usize] = true; // Start Ippatsu cycle

                // Check for Double Riichi
                if self.is_first_turn
                    && self.discards[pid as usize].len() == 1
                    && self.melds.iter().all(|m| m.is_empty())
                {
                    self.double_riichi_declared[pid as usize] = true;
                }

                let mut ev = serde_json::Map::new();
                ev.insert(
                    "type".to_string(),
                    Value::String("reach_accepted".to_string()),
                );
                ev.insert("actor".to_string(), Value::Number(pid.into()));
                self._push_mjai_event(Value::Object(ev));
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
            if self.riichi_declared[pid as usize]
                || self.double_riichi_declared[pid as usize]
                || self.riichi_stage[pid as usize]
            {
                // Can still Ron in Riichi
            }
            if self.is_done {
                continue;
            }

            if self.missed_agari_doujun[pid as usize] || self.missed_agari_riichi[pid as usize] {
                continue;
            }

            // Enhanced Furiten Check (Genbutsu + General Furiten)
            // 1. Simple check: Is the tile ITSELF in discards? (Genbutsu)
            if self.discards[pid as usize].contains(&tile) {
                continue;
            }

            // 2. Exact Furiten Check: Is ANY wait tile in discards?
            let hand = &self.hands[pid as usize];
            let melds = &self.melds[pid as usize];
            let calc = crate::agari_calculator::AgariCalculator::new(hand.clone(), melds.clone());
            let waits = calc.get_waits_u8();
            if waits.is_empty() {
                continue; // Not Tenpai -> Cannot Ron logic
            }

            let is_strictly_furiten = waits
                .iter()
                .any(|&w| self.discards[pid as usize].iter().any(|&d| d / 4 == w));

            if is_strictly_furiten {
                continue;
            }

            if self._check_ron(pid, tile, discarded_pid, false) {
                self.current_claims
                    .entry(pid)
                    .or_default()
                    .push(Action::new(ActionType::Ron, Some(tile), vec![]));
            } else if waits.contains(&(tile / 4)) {
                self.missed_agari_doujun[pid as usize] = true;
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
                if self.riichi_declared[pid as usize]
                    || self.double_riichi_declared[pid as usize]
                    || self.riichi_stage[pid as usize]
                {
                    continue;
                }

                let hand = &self.hands[pid as usize];
                // println!("PID {} checking claims for tile {}. Hand: {:?}", pid, tile, hand);
                let count = hand.iter().filter(|&&t| t / 4 == tile / 4).count();
                // println!("Count: {}", count);

                if count >= 2 {
                    // Pon
                    // Because pon is Any 3 combination
                    // Possible counts of red: 0 or 1 (since total is 4 and only 1 is red).
                    // Pon can have:
                    // 1. Two black tiles
                    // 2. One red and one black
                    let reds: Vec<u8> = hand
                        .iter()
                        .cloned()
                        .filter(|&t| (t == 16 || t == 52 || t == 88) && t / 4 == tile / 4)
                        .collect();
                    let blacks: Vec<u8> = hand
                        .iter()
                        .cloned()
                        .filter(|&t| (t != 16 && t != 52 && t != 88) && t / 4 == tile / 4)
                        .collect();

                    // Option 1: Two blacks
                    if blacks.len() >= 2 {
                        self.current_claims
                            .entry(pid)
                            .or_default()
                            .push(Action::new(
                                ActionType::Pon,
                                Some(tile),
                                vec![blacks[0], blacks[1]],
                            ));
                    }
                    // Option 2: One red, one black
                    if !reds.is_empty() && !blacks.is_empty() {
                        self.current_claims
                            .entry(pid)
                            .or_default()
                            .push(Action::new(
                                ActionType::Pon,
                                Some(tile),
                                vec![reds[0], blacks[0]],
                            ));
                    }
                }
                if count >= 3 {
                    // Daiminkan
                    // Option 1: Three blacks
                    // Option 2: One red, two blacks
                    let reds: Vec<u8> = hand
                        .iter()
                        .cloned()
                        .filter(|&t| (t == 16 || t == 52 || t == 88) && t / 4 == tile / 4)
                        .collect();
                    let blacks: Vec<u8> = hand
                        .iter()
                        .cloned()
                        .filter(|&t| (t != 16 && t != 52 && t != 88) && t / 4 == tile / 4)
                        .collect();

                    if blacks.len() >= 3 {
                        self.current_claims
                            .entry(pid)
                            .or_default()
                            .push(Action::new(
                                ActionType::Daiminkan,
                                Some(tile),
                                vec![blacks[0], blacks[1], blacks[2]],
                            ));
                    }
                    if !reds.is_empty() && blacks.len() >= 2 {
                        self.current_claims
                            .entry(pid)
                            .or_default()
                            .push(Action::new(
                                ActionType::Daiminkan,
                                Some(tile),
                                vec![reds[0], blacks[0], blacks[1]],
                            ));
                    }
                }
            }

            // 3. Chi (Only next player)
            let next_pid = (discarded_pid + 1) % 4;
            // println!("DEBUG RUST: next_pid={} riichi_declared={}", next_pid, self.riichi_declared[next_pid as usize]);
            if !self.riichi_declared[next_pid as usize]
                && !self.double_riichi_declared[next_pid as usize]
                && !self.riichi_stage[next_pid as usize]
                && tile < 108
            {
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

                // Left: T-2, T-1
                if num >= 2 && has(num - 2) && has(num - 1) {
                    let ts1: Vec<u8> = hand
                        .iter()
                        .cloned()
                        .filter(|&t| t / 4 == suit * 9 + num - 2)
                        .collect();
                    let ts2: Vec<u8> = hand
                        .iter()
                        .cloned()
                        .filter(|&t| t / 4 == suit * 9 + num - 1)
                        .collect();
                    let mut distinct1 = Vec::new();
                    let mut seen1 = HashSet::new();
                    for t in ts1 {
                        let is_red = t == 16 || t == 52 || t == 88;
                        if seen1.insert(is_red) {
                            distinct1.push(t);
                        }
                    }
                    let mut distinct2 = Vec::new();
                    let mut seen2 = HashSet::new();
                    for t in ts2 {
                        let is_red = t == 16 || t == 52 || t == 88;
                        if seen2.insert(is_red) {
                            distinct2.push(t);
                        }
                    }

                    for &t1 in &distinct1 {
                        for &t2 in &distinct2 {
                            let consumed = vec![t1, t2];
                            if self._is_kuikae_valid(
                                &self.hands[next_pid as usize],
                                tile,
                                &consumed,
                            ) {
                                self.current_claims
                                    .entry(next_pid)
                                    .or_default()
                                    .push(Action::new(ActionType::Chi, Some(tile), consumed));
                            }
                        }
                    }
                }
                // Middle: T-1, T+1
                if (1..=7).contains(&num) && has(num - 1) && has(num + 1) {
                    let ts1: Vec<u8> = hand
                        .iter()
                        .cloned()
                        .filter(|&t| t / 4 == suit * 9 + num - 1)
                        .collect();
                    let ts2: Vec<u8> = hand
                        .iter()
                        .cloned()
                        .filter(|&t| t / 4 == suit * 9 + num + 1)
                        .collect();
                    let mut distinct1 = Vec::new();
                    let mut seen1 = HashSet::new();
                    for t in ts1 {
                        let is_red = t == 16 || t == 52 || t == 88;
                        if seen1.insert(is_red) {
                            distinct1.push(t);
                        }
                    }
                    let mut distinct2 = Vec::new();
                    let mut seen2 = HashSet::new();
                    for t in ts2 {
                        let is_red = t == 16 || t == 52 || t == 88;
                        if seen2.insert(is_red) {
                            distinct2.push(t);
                        }
                    }

                    for &t1 in &distinct1 {
                        for &t2 in &distinct2 {
                            let consumed = vec![t1, t2];
                            if self._is_kuikae_valid(
                                &self.hands[next_pid as usize],
                                tile,
                                &consumed,
                            ) {
                                self.current_claims
                                    .entry(next_pid)
                                    .or_default()
                                    .push(Action::new(ActionType::Chi, Some(tile), consumed));
                            }
                        }
                    }
                }
                // Right: T+1, T+2
                if num <= 6 && has(num + 1) && has(num + 2) {
                    let ts1: Vec<u8> = hand
                        .iter()
                        .cloned()
                        .filter(|&t| t / 4 == suit * 9 + num + 1)
                        .collect();
                    let ts2: Vec<u8> = hand
                        .iter()
                        .cloned()
                        .filter(|&t| t / 4 == suit * 9 + num + 2)
                        .collect();
                    let mut distinct1 = Vec::new();
                    let mut seen1 = HashSet::new();
                    for t in ts1 {
                        let is_red = t == 16 || t == 52 || t == 88;
                        if seen1.insert(is_red) {
                            distinct1.push(t);
                        }
                    }
                    let mut distinct2 = Vec::new();
                    let mut seen2 = HashSet::new();
                    for t in ts2 {
                        let is_red = t == 16 || t == 52 || t == 88;
                        if seen2.insert(is_red) {
                            distinct2.push(t);
                        }
                    }

                    for &t1 in &distinct1 {
                        for &t2 in &distinct2 {
                            let consumed = vec![t1, t2];
                            if self._is_kuikae_valid(
                                &self.hands[next_pid as usize],
                                tile,
                                &consumed,
                            ) {
                                self.current_claims
                                    .entry(next_pid)
                                    .or_default()
                                    .push(Action::new(ActionType::Chi, Some(tile), consumed));
                            }
                        }
                    }
                }
            }
        }
    }

    fn _check_ron(&self, pid: u8, tile: u8, _discarded_pid: u8, is_chankan: bool) -> bool {
        let hand = &self.hands[pid as usize];
        let melds = &self.melds[pid as usize];

        let cond = Conditions {
            tsumo: false,
            riichi: self.riichi_declared[pid as usize],
            double_riichi: self.double_riichi_declared[pid as usize],
            ippatsu: self.ippatsu_cycle[pid as usize],
            player_wind: Wind::from((pid + 4 - self.oya) % 4),
            round_wind: Wind::from(self.round_wind),
            chankan: is_chankan,
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
        self.kyoku_idx = oya; // Update kyoku_idx to match oya
        self.honba = honba;
        self.riichi_sticks = kyotaku; // Initialize sticks
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
            // println!(
            //     "DEBUG RUST: _initialize_round wall provided. First 34 tiles BEFORE reversal: {:?}",
            //     &w[0..34]
            // );
            w.reverse();
            // println!(
            //     "DEBUG RUST: _initialize_round wall reversed. First 34 tiles (drawing order): {:?}",
            //     &w[0..34]
            // );
            self.wall = w;
        } else {
            let mut w: Vec<u8> = (0..136).collect();
            let mut rng = if let Some(episode_seed) = self.seed {
                let hand_seed = splitmix64(episode_seed ^ self.hand_index);
                self.hand_index = self.hand_index.wrapping_add(1);
                StdRng::seed_from_u64(hand_seed)
            } else {
                self.hand_index = self.hand_index.wrapping_add(1);
                StdRng::from_entropy()
            };
            w.shuffle(&mut rng);
            self.wall = w;
        }

        // println!(
        //     "DEBUG RUST: Clearing Melds in _initialize_round. Melds len: {}",
        //     self.melds[0].len()
        // );
        self.dora_indicators = vec![self.wall[5]];

        // Generate Salt
        if self.salt.is_empty() {
            let mut rng = if let Some(s) = self.seed {
                StdRng::seed_from_u64(s)
            } else {
                StdRng::from_entropy()
            };
            // 16 chars random hex
            let chars: Vec<u8> = (0..8).map(|_| rng.gen()).collect();
            self.salt = hex::encode(chars);
        }

        // Calculate Wall Digest
        // Format: salt + wall_str
        // wall_str: join ints comma separated?
        // Python: hashlib.sha256((self.salt + ",".join(map(str, self.wall))).encode()).hexdigest()
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
            // println!(
            //     "DEBUG RUST: Initial hand player {} : {:?}",
            //     pid, self.hands[pid]
            // );
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
        self._push_mjai_event(start_kyoku);
    }

    fn _filter_mjai_event(&self, pid: u8, v: &Value) -> Value {
        let mut v = v.clone();
        if let Some(obj) = v.as_object_mut() {
            if let Some(type_val) = obj.get("type").and_then(|t| t.as_str()) {
                match type_val {
                    "start_kyoku" => {
                        if let Some(tehais) = obj.get_mut("tehais").and_then(|t| t.as_array_mut()) {
                            for (i, hand_val) in tehais.iter_mut().enumerate() {
                                if i != pid as usize {
                                    if let Some(hand_arr) = hand_val.as_array() {
                                        let len = hand_arr.len();
                                        *hand_val = serde_json::Value::Array(vec![
                                            serde_json::Value::String("?".to_string());
                                            len
                                        ]);
                                    }
                                }
                            }
                        }
                    }
                    "tsumo" => {
                        if let Some(actor) = obj.get("actor").and_then(|a| a.as_u64()) {
                            if actor != pid as u64 && obj.contains_key("pai") {
                                obj.insert(
                                    "pai".to_string(),
                                    serde_json::Value::String("?".to_string()),
                                );
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        v
    }

    pub(crate) fn _push_mjai_event(&mut self, ev: Value) {
        if self.skip_mjai_logging {
            return;
        }
        let s = ev.to_string();
        self.mjai_log.push(s.clone());

        let type_str = ev.get("type").and_then(|t| t.as_str());
        let needs_filter = matches!(type_str, Some("start_kyoku") | Some("tsumo"));

        if needs_filter {
            for p in 0..4 {
                let filtered = self._filter_mjai_event(p as u8, &ev);
                self.mjai_log_per_player[p].push(filtered.to_string());
            }
        } else {
            for p in 0..4 {
                self.mjai_log_per_player[p].push(s.clone());
            }
        }
    }

    fn _get_obs(&self, pid: u8) -> Observation {
        Observation {
            player_id: pid,
            hand: self.hands[pid as usize].clone(),
            events_json: self.mjai_log_per_player[pid as usize].clone(),
            prev_events_size: self.player_event_counts[pid as usize],
            legal_actions: self._get_legal_actions_internal(pid),
        }
    }

    fn _calculate_deltas(
        &self,
        agari: &Agari,
        winner: u8,
        is_tsumo: bool,
        loser: Option<u8>,
        include_bonus: bool,
        include_riichi_sticks: bool,
    ) -> [i32; 4] {
        let mut deltas = [0; 4];
        let h_val = if include_bonus { self.honba as i32 } else { 0 };
        let mut pao_pid: Option<u8> = None;
        if agari.yaku.contains(&crate::yaku::ID_DAISANGEN) {
            if let Some(&p) = self.pao[winner as usize].get(&(crate::yaku::ID_DAISANGEN as u8)) {
                pao_pid = Some(p);
            }
        } else if agari.yaku.contains(&crate::yaku::ID_DAISUUSHI) {
            if let Some(&p) = self.pao[winner as usize].get(&(crate::yaku::ID_DAISUUSHI as u8)) {
                pao_pid = Some(p);
            }
        }

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

        if include_bonus && include_riichi_sticks && self.riichi_sticks > 0 {
            deltas[winner as usize] += (self.riichi_sticks * 1000) as i32;
        }

        deltas
    }

    // Helper to check valid discards under Kuikae rules
    fn _is_kuikae_valid(&self, hand: &[u8], tile: u8, consumed: &[u8]) -> bool {
        let mut sim_hand = hand.to_vec();
        for &c in consumed {
            if let Some(pos) = sim_hand.iter().position(|&t| t == c) {
                sim_hand.swap_remove(pos);
            }
        }

        let called_kv = tile / 4;
        let mut forbidden_kvs = Vec::new();
        forbidden_kvs.push(called_kv); // Rule 1: Cannot discard same tile

        // Rule 2: Suji-gui (Eating-Swap)
        // Check if called + consumed form a sequence
        let mut full_set_kvs = vec![called_kv];
        for &c in consumed {
            full_set_kvs.push(c / 4);
        }
        full_set_kvs.sort();

        if full_set_kvs.len() == 3
            && full_set_kvs[2] == full_set_kvs[1] + 1
            && full_set_kvs[1] == full_set_kvs[0] + 1
        {
            let min = full_set_kvs[0];
            let max = full_set_kvs[2];
            let num_min = min % 9;

            if called_kv == min && num_min <= 5 {
                forbidden_kvs.push(max + 1);
            } else if called_kv == max && num_min >= 1 {
                forbidden_kvs.push(min - 1);
            }
        }

        // Check if any legal discard remains
        sim_hand.iter().any(|&t| !forbidden_kvs.contains(&(t / 4)))
    }

    fn _get_legal_actions_internal(&self, pid: u8) -> Vec<Action> {
        let mut actions = Vec::new();
        let hand = &self.hands[pid as usize];

        let mut h14 = hand.clone();
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
                    // Rule: Ankan is allowed ONLY IF it does not change the waits.
                    let t_type = dt / 4;
                    let matches: Vec<u8> =
                        h14.iter().cloned().filter(|&t| t / 4 == t_type).collect();
                    if matches.len() == 4 {
                        use crate::agari_calculator::AgariCalculator;
                        let mut hand13 = hand.clone();
                        if let Some(pos) = hand13.iter().position(|&x| x == dt) {
                            hand13.remove(pos);
                        }
                        let mut old_waits =
                            AgariCalculator::new(hand13, self.melds[pid as usize].clone())
                                .get_waits_u8();
                        old_waits.sort();

                        // Simulate ankan
                        let mut next_melds = self.melds[pid as usize].clone();
                        next_melds.push(Meld::new(MeldType::Angang, matches.clone(), false, -1));
                        let mut next_hand = hand.clone();
                        for &m in &matches {
                            if let Some(pos) = next_hand.iter().position(|&x| x == m) {
                                next_hand.remove(pos);
                            }
                        }
                        let mut new_waits =
                            AgariCalculator::new(next_hand, next_melds).get_waits_u8();
                        new_waits.sort();

                        if !old_waits.is_empty() && old_waits == new_waits {
                            actions.push(Action::new(ActionType::Ankan, Some(dt), matches));
                        }
                    }
                }

                if let Some(dt) = self.drawn_tile {
                    actions.push(Action::new(ActionType::Discard, Some(dt), vec![]));
                }
                return actions;
            }

            // Normal Turn
            if self.riichi_stage[pid as usize] {
                // Must discard after declaring Riichi
                // Filter discards that maintain Tenpai.
                let mut valid_discard_types = std::collections::HashSet::new();
                let mut checked_types = std::collections::HashSet::new();

                let mut calc = crate::agari_calculator::AgariCalculator::new(
                    h14.clone(),
                    self.melds[pid as usize].clone(),
                );

                for &t in &h14 {
                    let tt = t / 4;
                    if checked_types.contains(&tt) {
                        continue;
                    }
                    checked_types.insert(tt);

                    // Optimization: Reuse calculator by temporarily removing the tile
                    calc.hand.remove(tt);
                    if calc.is_tenpai() {
                        valid_discard_types.insert(tt);
                    }
                    calc.hand.add(tt);
                }

                for &t in &h14 {
                    if valid_discard_types.contains(&(t / 4)) {
                        actions.push(Action::new(ActionType::Discard, Some(t), vec![]));
                    }
                }
                return actions;
            }

            // 1. Kyushukyuhai
            // Condition: First turn (no discards by self) and no melds on board (uninterrupted)
            let is_first_turn_personal = self.discards[pid as usize].is_empty();
            let no_melds = self.melds.iter().all(|pm| pm.is_empty());

            if is_first_turn_personal && no_melds {
                let mut yaochuu_count = 0;
                let mut seen = [false; 34];
                for &t in &h14 {
                    let tt = t / 4;
                    let is_yaochuu = tt == 0
                        || tt == 8
                        || tt == 9
                        || tt == 17
                        || tt == 18
                        || tt == 26
                        || tt >= 27;
                    if is_yaochuu && !seen[tt as usize] {
                        seen[tt as usize] = true;
                        yaochuu_count += 1;
                    }
                }
                if yaochuu_count >= 9 {
                    actions.push(Action::new(ActionType::KyushuKyuhai, None, vec![]));
                }
            }

            // 2. Discards
            for &t in &h14 {
                if self.forbidden_discards[pid as usize].contains(&(t / 4)) {
                    continue;
                }
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
            // menzen, socre >= 1000, wall >= 18
            let is_menzen = self.melds[pid as usize].iter().all(|m| !m.opened);
            if is_menzen && self.scores[pid as usize] >= 1000 && self.wall.len() >= 18 {
                let mut can_riichi = false;
                // De-dup tiles to check
                let mut checked_types = std::collections::HashSet::new();
                for &t in &h14 {
                    let tt = t / 4;
                    if checked_types.contains(&tt) {
                        continue;
                    }
                    checked_types.insert(tt);

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
                            let act =
                                Action::new(ActionType::Kakan, Some(kakan_tile), m.tiles.clone());
                            // Ensure we only send 3 tiles?
                            // m.tiles should be 3.
                            actions.push(act);
                        }
                    }
                }
                // Ankan
                let mut counts = HashMap::new();
                for &t in &h14 {
                    *counts.entry(t / 4).or_insert(0) += 1;
                }
                let mut keys: Vec<u8> = counts.keys().cloned().collect();
                keys.sort(); // Deterministic order
                for tt in keys {
                    if counts[&tt] == 4 {
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
            } else if self.pending_kan.is_some() {
                // Chankan opportunity
                // If this player is active (WaitResponse), and pending_kan exists,
                // it implies they can Ron (Chankan).
                // We don't have the tile in 'current_claims', but we know the target tile is in pending_kan.
                let (_kan_actor, ref kan_act) = self.pending_kan.as_ref().unwrap();
                let tile = kan_act.tile.unwrap();

                // We must verify Ron validity?
                // step() logic filtered `active_players` using `_check_ron`.
                // So if we are active, we can Ron.
                actions.push(Action::new(ActionType::Ron, Some(tile), vec![]));
            }
        }

        actions
    }

    fn _execute_claim(&mut self, pid: u8, action: Action, from_pid: Option<u8>) -> PyResult<()> {
        // if !self.skip_mjai_logging {
        //     println!(
        //         "DEBUG RUST: _execute_claim START pid={} action={:?} current_melds_len={}",
        //         pid,
        //         action.action_type,
        //         self.melds[pid as usize].len()
        //     );
        // }
        use serde_json::Value; // Added use statement for Value
                               // If the discarded tile was part of a Riichi declaration, accept it
        if let Some(f_pid) = from_pid {
            if self.riichi_stage[f_pid as usize] {
                self.riichi_stage[f_pid as usize] = false;
                self.riichi_declared[f_pid as usize] = true;
                self.scores[f_pid as usize] -= 1000;
                self.score_deltas[f_pid as usize] -= 1000;
                self.riichi_sticks += 1;

                let mut ev = serde_json::Map::new();
                ev.insert(
                    "type".to_string(),
                    Value::String("reach_accepted".to_string()),
                );
                ev.insert("actor".to_string(), Value::Number(f_pid.into()));
                self._push_mjai_event(Value::Object(ev));
            }
        }
        self.ippatsu_cycle = [false; 4]; // Any claim breaks Ippatsu for everyone
        let hand = &mut self.hands[pid as usize];

        let mut m_type = MeldType::Peng;
        let mut opened = true;
        let consumed = action.consume_tiles.clone();

        match action.action_type {
            ActionType::Chi => {
                m_type = MeldType::Chi;
                if let Some(called) = action.tile {
                    let called_kv = called / 4;
                    // Forbid the same tile
                    self.forbidden_discards[pid as usize].push(called_kv);

                    let mut full_set_kvs: Vec<u8> = vec![called_kv];
                    for &c in &action.consume_tiles {
                        full_set_kvs.push(c / 4);
                    }
                    full_set_kvs.sort();
                    if full_set_kvs.len() == 3 {
                        let min = full_set_kvs[0];
                        let max = full_set_kvs[2];
                        if called_kv == min && min % 9 <= 5 {
                            self.forbidden_discards[pid as usize].push(max + 1);
                        } else if called_kv == max && min % 9 >= 1 {
                            self.forbidden_discards[pid as usize].push(min - 1);
                        }
                    }
                }
            }
            ActionType::Pon => {
                m_type = MeldType::Peng;
                if let Some(called) = action.tile {
                    self.forbidden_discards[pid as usize].push(called / 4);
                }
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
                    let new_meld = Meld::new(
                        MeldType::Addgang,
                        old_meld.tiles.clone(),
                        true,
                        old_meld.from_who,
                    );
                    self.melds[pid as usize].push(new_meld);

                    // Log Kakan
                    let mut ev = serde_json::Map::new();
                    ev.insert("type".to_string(), Value::String("kakan".to_string()));
                    ev.insert("actor".to_string(), Value::Number(pid.into()));
                    ev.insert(
                        "pai".to_string(),
                        Value::String(tid_to_mjai(action.tile.unwrap())),
                    );
                    let _kakan_tile = action.tile.unwrap();
                    let old_cons: Vec<String> = action
                        .consume_tiles
                        .iter()
                        .take(3)
                        .map(|&t| {
                            // Check Red 5s
                            if t == 16 {
                                return "0m".to_string();
                            }
                            if t == 52 {
                                return "0p".to_string();
                            }
                            if t == 88 {
                                return "0s".to_string();
                            }
                            tid_to_mjai(t)
                        })
                        .collect();
                    ev.insert(
                        "consumed".to_string(),
                        serde_json::to_value(old_cons).unwrap(),
                    );
                    self._push_mjai_event(Value::Object(ev));

                    self.pending_kan_dora_count += 1;
                    return Ok(());
                }
                // Fallback (Ankan logic if kakan fails?) - Python logic handles this.
                // Assuming valid Kakan implies existing Pon.
                return Ok(());
            }
            _ => {}
        }

        // Remove consume tiles from hand
        if action.action_type != ActionType::Kakan {
            // Validation: Structural check
            let mut all_tiles = consumed.clone();
            if let Some(t) = action.tile {
                all_tiles.push(t);
            }
            all_tiles.sort();

            match action.action_type {
                ActionType::Chi => {
                    // 3 tiles, same suit, sequence
                    if all_tiles.len() != 3 {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "CHI must have 3 tiles",
                        ));
                    }
                    let t1 = all_tiles[0] / 4;
                    let t2 = all_tiles[1] / 4;
                    let t3 = all_tiles[2] / 4;
                    if t1 >= 27 || t2 >= 27 || t3 >= 27 {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "CHI cannot include honor tiles",
                        ));
                    }
                    if (t1 / 9 != t2 / 9) || (t2 / 9 != t3 / 9) {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "CHI tiles must be in the same suit",
                        ));
                    }
                    if t1 + 1 != t2 || t2 + 1 != t3 {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "CHI tiles must be a sequence",
                        ));
                    }
                }
                ActionType::Pon => {
                    if all_tiles.len() != 3 {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "PON must have 3 tiles",
                        ));
                    }
                    let t1 = all_tiles[0] / 4;
                    let t2 = all_tiles[1] / 4;
                    let t3 = all_tiles[2] / 4;
                    if t1 != t2 || t2 != t3 {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "PON tiles must be of the same type",
                        ));
                    }
                }
                ActionType::Daiminkan | ActionType::Ankan => {
                    if all_tiles.len() != 4 {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "KAN must have 4 tiles",
                        ));
                    }
                    let t1 = all_tiles[0] / 4;
                    let t2 = all_tiles[1] / 4;
                    let t3 = all_tiles[2] / 4;
                    let t4 = all_tiles[3] / 4;
                    if t1 != t2 || t2 != t3 || t3 != t4 {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "KAN tiles must be of the same type",
                        ));
                    }
                }
                _ => {}
            }

            for &c in &consumed {
                if let Some(pos) = hand.iter().position(|&x| x == c) {
                    hand.remove(pos);
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Tile {} not in hand",
                        c
                    )));
                }
            }
            // Add tile to meld
            let mut tiles = consumed.clone();
            if let Some(t) = action.tile {
                tiles.push(t);
            }
            tiles.sort();

            let from_who = from_pid.map(|p| p as i8).unwrap_or(-1);
            let meld = Meld::new(m_type, tiles.clone(), opened, from_who);
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
                if action.action_type != ActionType::Ankan {
                    ev.insert("pai".to_string(), Value::String(tid_to_mjai(t)));
                }
            }
            let cons_str: Vec<String> = consumed.iter().map(|&t| tid_to_mjai(t)).collect();
            ev.insert(
                "consumed".to_string(),
                serde_json::to_value(cons_str).unwrap(),
            );
            self._push_mjai_event(Value::Object(ev));

            if action.action_type == ActionType::Daiminkan
                || action.action_type == ActionType::Ankan
            {
                self.pending_kan_dora_count += 1;
            }

            self.hands[pid as usize].sort();
            self._check_pao_conditions(pid);
        }
        Ok(())
    }

    // Helper to update Pao state
    fn _check_pao_conditions(&mut self, pid: u8) {
        const DRAGON_TILE_TYPES: &[u8] = &[31, 32, 33]; // Haku, Hatsu, Chun
        const WIND_TILE_TYPES: &[u8] = &[27, 28, 29, 30]; // East, South, West, North

        let melds = &self.melds[pid as usize];

        let mut d_melds = Vec::new();
        let mut w_melds = Vec::new();

        for m in melds {
            let t = m.tiles[0] / 4;
            if DRAGON_TILE_TYPES.contains(&t) {
                d_melds.push(m);
            } else if WIND_TILE_TYPES.contains(&t) {
                w_melds.push(m);
            }
        }

        if d_melds.len() == 3 {
            let last = d_melds.last().unwrap();
            if last.from_who != -1 && last.from_who != pid as i8 {
                self.pao[pid as usize].insert(crate::yaku::ID_DAISANGEN as u8, last.from_who as u8);
            }
        }

        if w_melds.len() == 4 {
            let last = w_melds.last().unwrap();
            if last.from_who != -1 && last.from_who != pid as i8 {
                self.pao[pid as usize].insert(crate::yaku::ID_DAISUUSHI as u8, last.from_who as u8);
            }
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
            ippatsu: self.ippatsu_cycle[pid as usize],
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

        // `AgariCalculator::new` takes `tiles_136` (hand).
        // `calc` takes `win_tile`.
        let calc = crate::agari_calculator::AgariCalculator::new(hand.clone(), melds.clone());
        let agari = calc.calc(tile, self.dora_indicators.clone(), vec![], Some(cond));

        agari.agari && (agari.yakuman || agari.han >= 1) // Simple check (ignore yaku 31-33 exclusion for now)
    }
}

fn is_terminal_tile(tile: u8) -> bool {
    let t = tile / 4;
    t == 0 || t == 8 || t == 9 || t == 17 || t == 18 || t >= 26
}

#[allow(dead_code)]
fn serde_json_to_pyobject(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => Ok(b.into_pyobject(py)?.to_owned().unbind().into()),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.to_owned().unbind().into())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.to_owned().unbind().into())
            } else {
                Ok(n.to_string().into_pyobject(py)?.to_owned().unbind().into())
            }
        }
        Value::String(s) => Ok(s.into_pyobject(py)?.to_owned().unbind().into()),
        Value::Array(arr) => {
            let list = pyo3::types::PyList::empty(py);
            for v in arr {
                list.append(serde_json_to_pyobject(py, v)?)?;
            }
            Ok(list.unbind().into())
        }
        Value::Object(obj) => {
            let dict = pyo3::types::PyDict::new(py);
            for (k, v) in obj {
                dict.set_item(k, serde_json_to_pyobject(py, v)?)?;
            }
            Ok(dict.unbind().into())
        }
    }
}
