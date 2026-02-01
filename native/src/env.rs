use pyo3::prelude::*;
use pyo3::IntoPyObject;
use std::collections::HashMap;

use crate::action::{Action, Phase};
use crate::observation::Observation;
use crate::replay::MjaiEvent;
use crate::rule::GameRule;
use crate::state::legal_actions::GameStateLegalActions; // Import trait
use crate::state::GameState;
use crate::types::{Agari, Meld};

#[pyclass(module = "riichienv._riichienv")]
#[derive(Debug, Clone)]
pub struct RiichiEnv {
    #[pyo3(get)]
    pub state: GameState,
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
        rule: Option<GameRule>,
    ) -> PyResult<Self> {
        let gt = if let Some(val) = game_mode {
            if let Ok(s) = val.extract::<String>() {
                match s.as_str() {
                    "4p-red-single" => 0,
                    "4p-red-east" => 1,
                    "4p-red-half" => 2,
                    _ => 0,
                }
            } else {
                val.extract::<u8>().unwrap_or_default()
            }
        } else {
            0
        };

        Ok(RiichiEnv {
            state: GameState::new(
                gt,
                skip_mjai_logging,
                seed,
                round_wind.unwrap_or(0),
                rule.unwrap_or_default(),
            ),
        })
    }

    // --- Delegation Getters/Setters ---

    #[getter]
    pub fn get_wall(&self) -> Vec<u32> {
        self.state.wall.tiles.iter().map(|&x| x as u32).collect()
    }
    #[setter]
    pub fn set_wall(&mut self, v: Vec<u32>) {
        self.state.wall.tiles = v.iter().map(|&x| x as u8).collect();
    }

    #[getter]
    pub fn get_hands(&self) -> Vec<Vec<u32>> {
        self.state
            .players
            .iter()
            .map(|p| p.hand.iter().map(|&x| x as u32).collect())
            .collect()
    }
    #[setter]
    pub fn set_hands(&mut self, v: Vec<Vec<u32>>) {
        if v.len() == 4 {
            for (i, h) in v.into_iter().enumerate() {
                self.state.players[i].hand = h.iter().map(|&x| x as u8).collect();
            }
        }
    }

    #[getter]
    pub fn get_melds(&self) -> Vec<Vec<Meld>> {
        self.state.players.iter().map(|p| p.melds.clone()).collect()
    }
    #[setter]
    pub fn set_melds(&mut self, v: Vec<Vec<Meld>>) {
        if v.len() == 4 {
            for (i, m) in v.into_iter().enumerate() {
                self.state.players[i].melds = m;
            }
        }
    }

    #[getter]
    pub fn get_discards(&self) -> Vec<Vec<u32>> {
        self.state
            .players
            .iter()
            .map(|p| p.discards.iter().map(|&x| x as u32).collect())
            .collect()
    }
    #[setter]
    pub fn set_discards(&mut self, v: Vec<Vec<u32>>) {
        if v.len() == 4 {
            for (i, d) in v.into_iter().enumerate() {
                self.state.players[i].discards = d.iter().map(|&x| x as u8).collect();
            }
        }
    }

    #[getter]
    pub fn get_discard_from_hand(&self) -> Vec<Vec<bool>> {
        self.state
            .players
            .iter()
            .map(|p| p.discard_from_hand.clone())
            .collect()
    }
    #[setter]
    pub fn set_discard_from_hand(&mut self, v: Vec<Vec<bool>>) {
        if v.len() == 4 {
            for (i, d) in v.into_iter().enumerate() {
                self.state.players[i].discard_from_hand = d;
            }
        }
    }

    #[getter]
    pub fn get_discard_is_riichi(&self) -> Vec<Vec<bool>> {
        self.state
            .players
            .iter()
            .map(|p| p.discard_is_riichi.clone())
            .collect()
    }
    #[setter]
    pub fn set_discard_is_riichi(&mut self, v: Vec<Vec<bool>>) {
        if v.len() == 4 {
            for (i, d) in v.into_iter().enumerate() {
                self.state.players[i].discard_is_riichi = d;
            }
        }
    }

    #[getter]
    pub fn get_dora_indicators(&self) -> Vec<u32> {
        self.state
            .wall
            .dora_indicators
            .iter()
            .map(|&x| x as u32)
            .collect()
    }
    #[setter]
    pub fn set_dora_indicators(&mut self, v: Vec<u32>) {
        self.state.wall.dora_indicators = v.iter().map(|&x| x as u8).collect();
    }

    #[getter]
    pub fn get_rinshan_draw_count(&self) -> u8 {
        self.state.wall.rinshan_draw_count
    }
    #[setter]
    pub fn set_rinshan_draw_count(&mut self, v: u8) {
        self.state.wall.rinshan_draw_count = v;
    }

    #[getter]
    pub fn get_pending_kan_dora_count(&self) -> u8 {
        self.state.wall.pending_kan_dora_count
    }
    #[setter]
    pub fn set_pending_kan_dora_count(&mut self, v: u8) {
        self.state.wall.pending_kan_dora_count = v;
    }

    #[getter]
    pub fn get_is_rinshan_flag(&self) -> bool {
        self.state.is_rinshan_flag
    }
    #[setter]
    pub fn set_is_rinshan_flag(&mut self, v: bool) {
        self.state.is_rinshan_flag = v;
    }

    #[getter]
    pub fn get_riichi_declaration_index(&self) -> Vec<Option<usize>> {
        self.state
            .players
            .iter()
            .map(|p| p.riichi_declaration_index)
            .collect()
    }
    #[setter]
    pub fn set_riichi_declaration_index(&mut self, v: Vec<Option<usize>>) {
        if v.len() == 4 {
            for (i, d) in v.into_iter().enumerate() {
                self.state.players[i].riichi_declaration_index = d;
            }
        }
    }

    #[getter]
    pub fn get_current_player(&self) -> u8 {
        self.state.current_player
    }
    #[setter]
    pub fn set_current_player(&mut self, v: u8) {
        self.state.current_player = v;
    }

    #[getter]
    pub fn get_game_mode(&self) -> u8 {
        self.state.game_mode
    }

    #[getter]
    pub fn get_turn_count(&self) -> u32 {
        self.state.turn_count
    }
    #[setter]
    pub fn set_turn_count(&mut self, v: u32) {
        self.state.turn_count = v;
    }

    #[getter]
    pub fn get_kyoku_idx(&self) -> u8 {
        self.state.kyoku_idx
    }

    #[pyo3(name = "done")]
    pub fn done_method(&self) -> bool {
        self.state.is_done
    }

    #[getter]
    pub fn get_is_done(&self) -> bool {
        self.state.is_done
    }
    #[setter]
    pub fn set_is_done(&mut self, v: bool) {
        self.state.is_done = v;
    }

    #[getter]
    pub fn get_needs_tsumo(&self) -> bool {
        self.state.needs_tsumo
    }
    #[setter]
    pub fn set_needs_tsumo(&mut self, v: bool) {
        self.state.needs_tsumo = v;
    }

    #[getter]
    pub fn get_needs_initialize_next_round(&self) -> bool {
        self.state.needs_initialize_next_round
    }
    #[setter]
    pub fn set_needs_initialize_next_round(&mut self, v: bool) {
        self.state.needs_initialize_next_round = v;
    }

    #[pyo3(name = "scores")]
    pub fn scores_method(&self) -> Vec<i32> {
        self.state.players.iter().map(|p| p.score).collect()
    }
    #[pyo3(name = "set_scores")]
    pub fn set_scores_method(&mut self, v: Vec<i32>) {
        if v.len() == 4 {
            for (i, &s) in v.iter().enumerate() {
                self.state.players[i].score = s;
            }
        }
    }
    #[getter]
    pub fn get_scores(&self) -> Vec<i32> {
        self.state.players.iter().map(|p| p.score).collect()
    }
    #[setter]
    pub fn set_scores(&mut self, v: Vec<i32>) {
        if v.len() == 4 {
            for (i, &s) in v.iter().enumerate() {
                self.state.players[i].score = s;
            }
        }
    }

    #[getter]
    pub fn get_riichi_sticks(&self) -> u32 {
        self.state.riichi_sticks
    }
    #[setter]
    pub fn set_riichi_sticks(&mut self, v: u32) {
        self.state.riichi_sticks = v;
    }

    #[getter]
    pub fn get_riichi_declared(&self) -> Vec<bool> {
        self.state
            .players
            .iter()
            .map(|p| p.riichi_declared)
            .collect()
    }
    #[setter]
    pub fn set_riichi_declared(&mut self, v: Vec<bool>) {
        if v.len() == 4 {
            for (i, &val) in v.iter().enumerate() {
                self.state.players[i].riichi_declared = val;
            }
        }
    }

    #[getter]
    pub fn get_riichi_stage(&self) -> Vec<bool> {
        self.state.players.iter().map(|p| p.riichi_stage).collect()
    }
    #[setter]
    pub fn set_riichi_stage(&mut self, v: Vec<bool>) {
        if v.len() == 4 {
            for (i, &val) in v.iter().enumerate() {
                self.state.players[i].riichi_stage = val;
            }
        }
    }

    #[getter]
    pub fn get_phase(&self) -> Phase {
        self.state.phase
    }
    #[setter]
    pub fn set_phase(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(p) = v.extract::<Phase>() {
            self.state.phase = p;
        } else if let Ok(i) = v.extract::<i32>() {
            self.state.phase = match i {
                0 => Phase::WaitAct,
                1 => Phase::WaitResponse,
                _ => Phase::WaitAct,
            };
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Expected Phase or int",
            ));
        }
        Ok(())
    }

    #[getter]
    pub fn get_active_players(&self) -> Vec<u32> {
        self.state
            .active_players
            .iter()
            .map(|&x| x as u32)
            .collect()
    }
    #[setter]
    pub fn set_active_players(&mut self, v: Vec<u32>) {
        self.state.active_players = v.iter().map(|&x| x as u8).collect();
    }

    #[getter]
    pub fn get_oya(&self) -> u8 {
        self.state.oya
    }
    #[setter]
    pub fn set_oya(&mut self, v: u8) {
        self.state.oya = v;
    }

    #[getter]
    pub fn get_honba(&self) -> u8 {
        self.state.honba
    }
    #[setter]
    pub fn set_honba(&mut self, v: u8) {
        self.state.honba = v;
    }

    #[getter]
    pub fn is_first_turn(&self) -> bool {
        self.state.is_first_turn
    }
    #[setter]
    pub fn set_is_first_turn(&mut self, v: bool) {
        self.state.is_first_turn = v;
    }

    #[getter]
    pub fn get_drawn_tile(&self) -> Option<u8> {
        self.state.drawn_tile
    }
    #[setter]
    pub fn set_drawn_tile(&mut self, v: Option<u8>) {
        self.state.drawn_tile = v;
    }

    #[getter]
    pub fn current_claims(&self) -> HashMap<u8, Vec<Action>> {
        self.state.current_claims.clone()
    }
    #[setter]
    pub fn set_current_claims(&mut self, v: HashMap<u8, Vec<Action>>) {
        self.state.current_claims = v;
    }

    #[getter]
    pub fn get_last_discard(&self) -> Option<(u32, u32)> {
        self.state.last_discard.map(|(s, t)| (s as u32, t as u32))
    }
    #[setter]
    pub fn set_last_discard(&mut self, v: Option<(u32, u32)>) {
        if let Some((pid, tile)) = v {
            self.state.last_discard = Some((pid as u8, tile as u8));
        } else {
            self.state.last_discard = None;
        }
    }

    #[getter]
    pub fn get_pao(&self) -> Vec<HashMap<u8, u8>> {
        self.state.players.iter().map(|p| p.pao.clone()).collect()
    }
    #[setter]
    pub fn set_pao(&mut self, v: Vec<HashMap<u8, u8>>) {
        if v.len() == 4 {
            for (i, p) in v.into_iter().enumerate() {
                self.state.players[i].pao = p;
            }
        }
    }

    #[getter]
    pub fn get_missed_agari_doujun(&self) -> Vec<bool> {
        self.state
            .players
            .iter()
            .map(|p| p.missed_agari_doujun)
            .collect()
    }
    #[setter]
    pub fn set_missed_agari_doujun(&mut self, v: Vec<bool>) {
        if v.len() == 4 {
            for (i, &val) in v.iter().enumerate() {
                self.state.players[i].missed_agari_doujun = val;
            }
        }
    }

    #[getter]
    pub fn get_agari_results(&self) -> HashMap<u8, Agari> {
        self.state.agari_results.clone()
    }

    #[getter]
    pub fn get_score_deltas(&self) -> Vec<i32> {
        self.state.players.iter().map(|p| p.score_delta).collect()
    }

    #[getter]
    pub fn get_round_wind(&self) -> u8 {
        self.state.round_wind
    }
    #[setter]
    pub fn set_round_wind(&mut self, v: u8) {
        self.state.round_wind = v;
    }

    pub fn _reveal_kan_dora(&mut self) {
        self.state._reveal_kan_dora();
    }

    pub fn _get_ura_markers(&self) -> Vec<String> {
        self.state._get_ura_markers()
    }

    #[getter(_custom_round_wind)]
    pub fn get_custom_round_wind(&self) -> u8 {
        self.state.round_wind
    }

    // --- Methods ---

    #[pyo3(signature = (oya=None, honba=None, riichi_sticks=None, scores=None, round_wind=None))]
    pub fn set_state(
        &mut self,
        oya: Option<u8>,
        honba: Option<u8>,
        riichi_sticks: Option<u32>,
        scores: Option<Vec<i32>>,
        round_wind: Option<u8>,
    ) {
        if let Some(o) = oya {
            self.state.oya = o;
            self.state.kyoku_idx = o;
        }
        if let Some(h) = honba {
            self.state.honba = h;
        }
        if let Some(r) = riichi_sticks {
            self.state.riichi_sticks = r;
        }
        if let Some(sc) = scores {
            if sc.len() == 4 {
                for (i, &s) in sc.iter().enumerate() {
                    self.state.players[i].score = s;
                }
            }
        }
        if let Some(rw) = round_wind {
            self.state.round_wind = rw;
        }
    }

    pub fn ranks(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..4).collect();
        indices.sort_by(|&a, &b| {
            let score_a = self.state.players[a].score;
            let score_b = self.state.players[b].score;
            if score_a != score_b {
                score_b.cmp(&score_a)
            } else {
                a.cmp(&b)
            }
        });
        let mut result = vec![0; 4];
        for (rank, &pid) in indices.iter().enumerate() {
            result[pid] = rank + 1;
        }
        result
    }

    pub fn points(&self, rule_name: &str) -> PyResult<Vec<f64>> {
        let (soten_weight, soten_base, jun_weight) = match rule_name {
            "basic" => (1.0, 25000.0, vec![50.0, 10.0, -10.0, -50.0]),
            "ouza-tyoujyo" => (0.0, 25000.0, vec![100.0, 40.0, -40.0, -100.0]),
            "ouza-normal" => (0.0, 25000.0, vec![50.0, 20.0, -20.0, -50.0]),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown preset rule: {}",
                    rule_name
                )))
            }
        };

        let ranks = self.ranks();
        let mut points = vec![0.0; 4];
        for i in 0..4 {
            let score = self.state.players[i].score as f64;
            let rank = ranks[i];
            let uma = jun_weight[rank - 1];
            points[i] = (score - soten_base) / 1000.0 * soten_weight + uma;
        }
        points.into_iter().map(Ok).collect()
    }

    #[getter]
    pub fn mjai_log(&self, py: Python) -> PyResult<Py<PyAny>> {
        let json = py.import("json")?;
        let loads = json.getattr("loads")?;
        let list = pyo3::types::PyList::empty(py);
        for s in &self.state.mjai_log {
            list.append(loads.call1((s,))?)?;
        }
        Ok(list.unbind().into())
    }

    #[pyo3(signature = (players=None))]
    pub fn get_observations(&mut self, players: Option<Vec<u8>>) -> HashMap<u8, Observation> {
        let targets = players.unwrap_or_else(|| (0..4).collect());
        let mut map = HashMap::new();
        for p in targets {
            map.insert(p, self.state.get_observation(p));
        }
        map
    }

    pub fn get_observation(&mut self, player_id: u8) -> Observation {
        self.state.get_observation(player_id)
    }

    #[pyo3(signature = (players=None))]
    fn get_obs_py<'py>(
        &mut self,
        py: Python<'py>,
        players: Option<Vec<u8>>,
    ) -> PyResult<Py<PyAny>> {
        // println!("TRACE: get_obs_py(players={:?})", players);
        let obs_map = self.get_observations(players);
        obs_map.into_pyobject(py).map(|o| o.unbind().into())
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
        if let Some(s) = seed {
            self.state.seed = Some(s);
        }

        let initial_scores = if let Some(sc) = scores {
            let mut s = [25000; 4];
            for (i, &score) in sc.iter().enumerate().take(4) {
                s[i] = score;
            }
            Some(s)
        } else {
            None
        };

        self.state.reset();
        self.state._initialize_round(
            oya.unwrap_or(self.state.oya),
            bakaze.unwrap_or(self.state.round_wind),
            honba.unwrap_or(self.state.honba),
            kyotaku.unwrap_or(self.state.riichi_sticks),
            wall,
            initial_scores,
        );

        self.get_obs_py(py, Some(self.state.active_players.clone()))
    }

    pub fn _get_legal_actions(&mut self, pid: u8) -> Vec<Action> {
        self.state._get_legal_actions_internal(pid)
    }

    #[pyo3(signature = (actions))]
    pub fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: HashMap<u8, Action>,
    ) -> PyResult<Py<PyAny>> {
        self.state.step(&actions);
        if self.state.last_error.is_some() {
            let dict = pyo3::types::PyDict::new(py);
            return Ok(dict.unbind().into());
        }
        self.get_obs_py(py, Some(self.state.active_players.clone()))
    }

    pub fn apply_mjai_event(&mut self, py: Python, event: Py<PyAny>) -> PyResult<()> {
        // Use python json to dump to string, then parse in rust
        let json = py.import("json")?;
        let s: String = json.call_method1("dumps", (event,))?.extract()?;
        let ev: MjaiEvent = serde_json::from_str(&s).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("JSON Parse Error: {}", e))
        })?;
        self.state.apply_mjai_event(ev);
        Ok(())
    }
}
