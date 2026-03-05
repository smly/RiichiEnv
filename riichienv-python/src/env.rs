use pyo3::IntoPyObject;
use pyo3::prelude::*;
use std::collections::HashMap;

use riichienv_core::action::{Action, Action3P, ActionEncoder, Phase};

/// Wrapper that accepts both `Action` and `Action3P` from Python.
pub struct AnyAction(Action);

impl<'a, 'py> FromPyObject<'a, 'py> for AnyAction {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(a) = obj.extract::<Action>() {
            Ok(AnyAction(a))
        } else if let Ok(a3p) = obj.extract::<Action3P>() {
            Ok(AnyAction(Action::from(a3p)))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "expected Action or Action3P",
            ))
        }
    }
}
use riichienv_core::game_variant::GameStateVariant;
use riichienv_core::replay::MjaiEvent;
use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;
use riichienv_core::state::legal_actions::GameStateLegalActions;
use riichienv_core::state_3p::legal_actions::GameState3PLegalActions;
use riichienv_core::types::{Meld, WinResult};

/// Dispatch macro for immutable access to the inner variant.
/// Both GameState and GameState3P share identical field names.
macro_rules! with_variant {
    ($self:expr, |$s:ident| $body:expr) => {
        match &$self.variant {
            GameStateVariant::FourPlayer($s) => $body,
            GameStateVariant::ThreePlayer($s) => $body,
        }
    };
}

/// Dispatch macro for mutable access to the inner variant.
macro_rules! with_variant_mut {
    ($self:expr, |$s:ident| $body:expr) => {
        match &mut $self.variant {
            GameStateVariant::FourPlayer($s) => $body,
            GameStateVariant::ThreePlayer($s) => $body,
        }
    };
}

#[pyclass(module = "riichienv._riichienv", from_py_object)]
#[derive(Debug, Clone)]
pub struct RiichiEnv {
    pub variant: GameStateVariant,
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
                    "3p-red-single" => 3,
                    "3p-red-east" => 4,
                    "3p-red-half" => 5,
                    _ => 0,
                }
            } else {
                val.extract::<u8>().unwrap_or_default()
            }
        } else {
            0
        };

        Ok(RiichiEnv {
            variant: GameStateVariant::new(
                gt,
                skip_mjai_logging,
                seed,
                round_wind.unwrap_or(0),
                rule.unwrap_or_default(),
            ),
        })
    }

    // --- Backward-compatible state getter (4P only) ---

    #[getter]
    pub fn get_state(&self) -> PyResult<GameState> {
        match &self.variant {
            GameStateVariant::FourPlayer(s) => Ok(*s.clone()),
            GameStateVariant::ThreePlayer(_) => Err(pyo3::exceptions::PyAttributeError::new_err(
                "state property is not available for 3P games. Use individual getters instead.",
            )),
        }
    }

    // --- Delegation Getters/Setters ---

    #[getter]
    pub fn get_wall(&self) -> Vec<u32> {
        with_variant!(self, |s| s.wall.tiles.iter().map(|&x| x as u32).collect())
    }
    #[setter]
    pub fn set_wall(&mut self, v: Vec<u32>) {
        with_variant_mut!(self, |s| s.wall.tiles =
            v.iter().map(|&x| x as u8).collect());
    }

    #[getter]
    pub fn get_hands(&self) -> Vec<Vec<u32>> {
        with_variant!(self, |s| s
            .players
            .iter()
            .map(|p| p.hand.iter().map(|&x| x as u32).collect())
            .collect())
    }
    #[setter]
    pub fn set_hands(&mut self, v: Vec<Vec<u32>>) {
        let np = self.variant.num_players() as usize;
        if v.len() == np {
            with_variant_mut!(self, |s| {
                for (i, h) in v.into_iter().enumerate() {
                    s.players[i].hand = h.iter().map(|&x| x as u8).collect();
                }
            });
        }
    }

    #[getter]
    pub fn get_melds(&self) -> Vec<Vec<Meld>> {
        with_variant!(self, |s| s
            .players
            .iter()
            .map(|p| p.melds.clone())
            .collect())
    }
    #[setter]
    pub fn set_melds(&mut self, v: Vec<Vec<Meld>>) {
        let np = self.variant.num_players() as usize;
        if v.len() == np {
            with_variant_mut!(self, |s| {
                for (i, m) in v.into_iter().enumerate() {
                    s.players[i].melds = m;
                }
            });
        }
    }

    #[getter]
    pub fn get_discards(&self) -> Vec<Vec<u32>> {
        with_variant!(self, |s| s
            .players
            .iter()
            .map(|p| p.discards.iter().map(|&x| x as u32).collect())
            .collect())
    }
    #[setter]
    pub fn set_discards(&mut self, v: Vec<Vec<u32>>) {
        let np = self.variant.num_players() as usize;
        if v.len() == np {
            with_variant_mut!(self, |s| {
                for (i, d) in v.into_iter().enumerate() {
                    s.players[i].discards = d.iter().map(|&x| x as u8).collect();
                }
            });
        }
    }

    #[getter]
    pub fn get_discard_from_hand(&self) -> Vec<Vec<bool>> {
        with_variant!(self, |s| s
            .players
            .iter()
            .map(|p| p.discard_from_hand.clone())
            .collect())
    }
    #[setter]
    pub fn set_discard_from_hand(&mut self, v: Vec<Vec<bool>>) {
        let np = self.variant.num_players() as usize;
        if v.len() == np {
            with_variant_mut!(self, |s| {
                for (i, d) in v.into_iter().enumerate() {
                    s.players[i].discard_from_hand = d;
                }
            });
        }
    }

    #[getter]
    pub fn get_discard_is_riichi(&self) -> Vec<Vec<bool>> {
        with_variant!(self, |s| s
            .players
            .iter()
            .map(|p| p.discard_is_riichi.clone())
            .collect())
    }
    #[setter]
    pub fn set_discard_is_riichi(&mut self, v: Vec<Vec<bool>>) {
        let np = self.variant.num_players() as usize;
        if v.len() == np {
            with_variant_mut!(self, |s| {
                for (i, d) in v.into_iter().enumerate() {
                    s.players[i].discard_is_riichi = d;
                }
            });
        }
    }

    #[getter]
    pub fn get_dora_indicators(&self) -> Vec<u32> {
        with_variant!(self, |s| s
            .wall
            .dora_indicators
            .iter()
            .map(|&x| x as u32)
            .collect())
    }
    #[setter]
    pub fn set_dora_indicators(&mut self, v: Vec<u32>) {
        with_variant_mut!(self, |s| s.wall.dora_indicators =
            v.iter().map(|&x| x as u8).collect());
    }

    #[getter]
    pub fn get_rinshan_draw_count(&self) -> u8 {
        with_variant!(self, |s| s.wall.rinshan_draw_count)
    }
    #[setter]
    pub fn set_rinshan_draw_count(&mut self, v: u8) {
        with_variant_mut!(self, |s| s.wall.rinshan_draw_count = v);
    }

    #[getter]
    pub fn get_pending_kan_dora_count(&self) -> u8 {
        with_variant!(self, |s| s.wall.pending_kan_dora_count)
    }
    #[setter]
    pub fn set_pending_kan_dora_count(&mut self, v: u8) {
        with_variant_mut!(self, |s| s.wall.pending_kan_dora_count = v);
    }

    #[getter]
    pub fn get_is_rinshan_flag(&self) -> bool {
        with_variant!(self, |s| s.is_rinshan_flag)
    }
    #[setter]
    pub fn set_is_rinshan_flag(&mut self, v: bool) {
        with_variant_mut!(self, |s| s.is_rinshan_flag = v);
    }

    #[getter]
    pub fn get_riichi_declaration_index(&self) -> Vec<Option<usize>> {
        with_variant!(self, |s| s
            .players
            .iter()
            .map(|p| p.riichi_declaration_index)
            .collect())
    }
    #[setter]
    pub fn set_riichi_declaration_index(&mut self, v: Vec<Option<usize>>) {
        let np = self.variant.num_players() as usize;
        if v.len() == np {
            with_variant_mut!(self, |s| {
                for (i, d) in v.into_iter().enumerate() {
                    s.players[i].riichi_declaration_index = d;
                }
            });
        }
    }

    #[getter]
    pub fn get_current_player(&self) -> u8 {
        with_variant!(self, |s| s.current_player)
    }
    #[setter]
    pub fn set_current_player(&mut self, v: u8) {
        with_variant_mut!(self, |s| s.current_player = v);
    }

    #[getter]
    pub fn get_game_mode(&self) -> u8 {
        with_variant!(self, |s| s.game_mode)
    }

    #[getter]
    pub fn get_num_players(&self) -> u8 {
        self.variant.num_players()
    }

    #[getter]
    pub fn get_action_space_size(&self) -> usize {
        ActionEncoder::from_num_players(self.variant.num_players()).action_space_size()
    }

    #[getter]
    pub fn get_kita_tiles(&self) -> Vec<Vec<u8>> {
        with_variant!(self, |s| s
            .players
            .iter()
            .map(|p| p.kita_tiles.clone())
            .collect())
    }

    #[getter]
    pub fn get_turn_count(&self) -> u32 {
        with_variant!(self, |s| s.turn_count)
    }
    #[setter]
    pub fn set_turn_count(&mut self, v: u32) {
        with_variant_mut!(self, |s| s.turn_count = v);
    }

    #[getter]
    pub fn get_kyoku_idx(&self) -> u8 {
        with_variant!(self, |s| s.kyoku_idx)
    }

    #[pyo3(name = "done")]
    pub fn done_method(&self) -> bool {
        with_variant!(self, |s| s.is_done)
    }

    /// Return a deep copy of this environment (full game state clone).
    #[pyo3(name = "clone")]
    pub fn py_clone(&self) -> Self {
        Self {
            variant: self.variant.clone(),
        }
    }

    pub fn __copy__(&self) -> Self {
        self.py_clone()
    }

    pub fn __deepcopy__(&self, _memo: &pyo3::Bound<'_, pyo3::types::PyAny>) -> Self {
        self.py_clone()
    }

    #[getter]
    pub fn get_is_done(&self) -> bool {
        with_variant!(self, |s| s.is_done)
    }
    #[setter]
    pub fn set_is_done(&mut self, v: bool) {
        with_variant_mut!(self, |s| s.is_done = v);
    }

    #[getter]
    pub fn get_needs_tsumo(&self) -> bool {
        with_variant!(self, |s| s.needs_tsumo)
    }
    #[setter]
    pub fn set_needs_tsumo(&mut self, v: bool) {
        with_variant_mut!(self, |s| s.needs_tsumo = v);
    }

    #[getter]
    pub fn get_needs_initialize_next_round(&self) -> bool {
        with_variant!(self, |s| s.needs_initialize_next_round)
    }
    #[setter]
    pub fn set_needs_initialize_next_round(&mut self, v: bool) {
        with_variant_mut!(self, |s| s.needs_initialize_next_round = v);
    }

    #[pyo3(name = "scores")]
    pub fn scores_method(&self) -> Vec<i32> {
        with_variant!(self, |s| s.players.iter().map(|p| p.score).collect())
    }
    #[pyo3(name = "set_scores")]
    pub fn set_scores_method(&mut self, v: Vec<i32>) {
        let np = self.variant.num_players() as usize;
        if v.len() == np {
            with_variant_mut!(self, |s| {
                for (i, &sc) in v.iter().enumerate() {
                    s.players[i].score = sc;
                }
            });
        }
    }
    #[getter]
    pub fn get_scores(&self) -> Vec<i32> {
        with_variant!(self, |s| s.players.iter().map(|p| p.score).collect())
    }
    #[setter]
    pub fn set_scores(&mut self, v: Vec<i32>) {
        let np = self.variant.num_players() as usize;
        if v.len() == np {
            with_variant_mut!(self, |s| {
                for (i, &sc) in v.iter().enumerate() {
                    s.players[i].score = sc;
                }
            });
        }
    }

    #[getter]
    pub fn get_riichi_sticks(&self) -> u32 {
        with_variant!(self, |s| s.riichi_sticks)
    }
    #[setter]
    pub fn set_riichi_sticks(&mut self, v: u32) {
        with_variant_mut!(self, |s| s.riichi_sticks = v);
    }

    #[getter]
    pub fn get_riichi_declared(&self) -> Vec<bool> {
        with_variant!(self, |s| s
            .players
            .iter()
            .map(|p| p.riichi_declared)
            .collect())
    }
    #[setter]
    pub fn set_riichi_declared(&mut self, v: Vec<bool>) {
        let np = self.variant.num_players() as usize;
        if v.len() == np {
            with_variant_mut!(self, |s| {
                for (i, &val) in v.iter().enumerate() {
                    s.players[i].riichi_declared = val;
                }
            });
        }
    }

    #[getter]
    pub fn get_riichi_stage(&self) -> Vec<bool> {
        with_variant!(self, |s| s.players.iter().map(|p| p.riichi_stage).collect())
    }
    #[setter]
    pub fn set_riichi_stage(&mut self, v: Vec<bool>) {
        let np = self.variant.num_players() as usize;
        if v.len() == np {
            with_variant_mut!(self, |s| {
                for (i, &val) in v.iter().enumerate() {
                    s.players[i].riichi_stage = val;
                }
            });
        }
    }

    #[getter]
    pub fn get_phase(&self) -> Phase {
        with_variant!(self, |s| s.phase)
    }
    #[setter]
    pub fn set_phase(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        let phase = if let Ok(p) = v.extract::<Phase>() {
            p
        } else if let Ok(i) = v.extract::<i32>() {
            match i {
                0 => Phase::WaitAct,
                1 => Phase::WaitResponse,
                _ => Phase::WaitAct,
            }
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Expected Phase or int",
            ));
        };
        with_variant_mut!(self, |s| s.phase = phase);
        Ok(())
    }

    #[getter]
    pub fn get_active_players(&self) -> Vec<u32> {
        with_variant!(self, |s| s
            .active_players
            .iter()
            .map(|&x| x as u32)
            .collect())
    }
    #[setter]
    pub fn set_active_players(&mut self, v: Vec<u32>) {
        with_variant_mut!(self, |s| s.active_players =
            v.iter().map(|&x| x as u8).collect());
    }

    #[getter]
    pub fn get_oya(&self) -> u8 {
        with_variant!(self, |s| s.oya)
    }
    #[setter]
    pub fn set_oya(&mut self, v: u8) {
        with_variant_mut!(self, |s| s.oya = v);
    }

    #[getter]
    pub fn get_honba(&self) -> u8 {
        with_variant!(self, |s| s.honba)
    }
    #[setter]
    pub fn set_honba(&mut self, v: u8) {
        with_variant_mut!(self, |s| s.honba = v);
    }

    #[getter]
    pub fn is_first_turn(&self) -> bool {
        with_variant!(self, |s| s.is_first_turn)
    }
    #[setter]
    pub fn set_is_first_turn(&mut self, v: bool) {
        with_variant_mut!(self, |s| s.is_first_turn = v);
    }

    #[getter]
    pub fn get_drawn_tile(&self) -> Option<u8> {
        with_variant!(self, |s| s.drawn_tile)
    }
    #[setter]
    pub fn set_drawn_tile(&mut self, v: Option<u8>) {
        with_variant_mut!(self, |s| s.drawn_tile = v);
    }

    #[getter]
    pub fn current_claims(&self) -> HashMap<u8, Vec<Action>> {
        with_variant!(self, |s| s.current_claims.clone())
    }
    #[setter]
    pub fn set_current_claims(&mut self, v: HashMap<u8, Vec<Action>>) {
        with_variant_mut!(self, |s| s.current_claims = v);
    }

    #[getter]
    pub fn get_last_discard(&self) -> Option<(u32, u32)> {
        with_variant!(self, |s| s.last_discard.map(|(a, b)| (a as u32, b as u32)))
    }
    #[setter]
    pub fn set_last_discard(&mut self, v: Option<(u32, u32)>) {
        let ld = v.map(|(pid, tile)| (pid as u8, tile as u8));
        with_variant_mut!(self, |s| s.last_discard = ld);
    }

    #[getter]
    pub fn get_pao(&self) -> Vec<HashMap<u8, u8>> {
        with_variant!(self, |s| s.players.iter().map(|p| p.pao.clone()).collect())
    }
    #[setter]
    pub fn set_pao(&mut self, v: Vec<HashMap<u8, u8>>) {
        let np = self.variant.num_players() as usize;
        if v.len() == np {
            with_variant_mut!(self, |s| {
                for (i, p) in v.into_iter().enumerate() {
                    s.players[i].pao = p;
                }
            });
        }
    }

    #[getter]
    pub fn get_missed_agari_doujun(&self) -> Vec<bool> {
        with_variant!(self, |s| s
            .players
            .iter()
            .map(|p| p.missed_agari_doujun)
            .collect())
    }
    #[setter]
    pub fn set_missed_agari_doujun(&mut self, v: Vec<bool>) {
        let np = self.variant.num_players() as usize;
        if v.len() == np {
            with_variant_mut!(self, |s| {
                for (i, &val) in v.iter().enumerate() {
                    s.players[i].missed_agari_doujun = val;
                }
            });
        }
    }

    #[getter]
    pub fn get_win_results(&self) -> HashMap<u8, WinResult> {
        with_variant!(self, |s| s.win_results.clone())
    }

    #[getter]
    pub fn get_score_deltas(&self) -> Vec<i32> {
        with_variant!(self, |s| s.players.iter().map(|p| p.score_delta).collect())
    }

    #[getter]
    pub fn get_round_wind(&self) -> u8 {
        with_variant!(self, |s| s.round_wind)
    }
    #[setter]
    pub fn set_round_wind(&mut self, v: u8) {
        with_variant_mut!(self, |s| s.round_wind = v);
    }

    pub fn _reveal_kan_dora(&mut self) {
        with_variant_mut!(self, |s| s._reveal_kan_dora());
    }

    pub fn _get_ura_markers(&self) -> Vec<String> {
        with_variant!(self, |s| s._get_ura_markers())
    }

    #[getter(_custom_round_wind)]
    pub fn get_custom_round_wind(&self) -> u8 {
        with_variant!(self, |s| s.round_wind)
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
        let np = self.variant.num_players() as usize;
        with_variant_mut!(self, |s| {
            if let Some(o) = oya {
                s.oya = o;
                s.kyoku_idx = o;
            }
            if let Some(h) = honba {
                s.honba = h;
            }
            if let Some(r) = riichi_sticks {
                s.riichi_sticks = r;
            }
            if let Some(ref sc) = scores
                && sc.len() == np
            {
                for (i, &val) in sc.iter().enumerate() {
                    s.players[i].score = val;
                }
            }
            if let Some(rw) = round_wind {
                s.round_wind = rw;
            }
        });
    }

    pub fn ranks(&self) -> Vec<usize> {
        let np = self.variant.num_players() as usize;
        let scores: Vec<i32> = with_variant!(self, |s| s.players.iter().map(|p| p.score).collect());
        let mut indices: Vec<usize> = (0..np).collect();
        indices.sort_by(|&a, &b| {
            if scores[a] != scores[b] {
                scores[b].cmp(&scores[a])
            } else {
                a.cmp(&b)
            }
        });
        let mut result = vec![0; np];
        for (rank, &pid) in indices.iter().enumerate() {
            result[pid] = rank + 1;
        }
        result
    }

    pub fn points(&self, rule_name: &str) -> PyResult<Vec<f64>> {
        let np = self.variant.num_players() as usize;
        let (soten_weight, soten_base, jun_weight) = if np == 3 {
            match rule_name {
                "basic" => (1.0, 35000.0, vec![40.0, 0.0, -40.0]),
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Unknown preset rule for 3P: {}",
                        rule_name
                    )));
                }
            }
        } else {
            match rule_name {
                "basic" => (1.0, 25000.0, vec![50.0, 10.0, -10.0, -50.0]),
                "ouza-tyoujyo" => (0.0, 25000.0, vec![100.0, 40.0, -40.0, -100.0]),
                "ouza-normal" => (0.0, 25000.0, vec![50.0, 20.0, -20.0, -50.0]),
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Unknown preset rule: {}",
                        rule_name
                    )));
                }
            }
        };

        let scores: Vec<i32> = with_variant!(self, |s| s.players.iter().map(|p| p.score).collect());
        let ranks = self.ranks();
        let mut points = vec![0.0; np];
        for i in 0..np {
            let score = scores[i] as f64;
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
        let log: &Vec<String> = with_variant!(self, |s| &s.mjai_log);
        for s in log {
            list.append(loads.call1((s,))?)?;
        }
        Ok(list.unbind().into())
    }

    #[pyo3(signature = (players=None))]
    pub fn get_observations<'py>(
        &mut self,
        py: Python<'py>,
        players: Option<Vec<u8>>,
    ) -> PyResult<Py<PyAny>> {
        let np = self.variant.num_players();
        let targets = players.unwrap_or_else(|| (0..np).collect());
        match &mut self.variant {
            GameStateVariant::FourPlayer(s) => {
                let mut map = HashMap::new();
                for p in targets {
                    map.insert(p, s.get_observation(p));
                }
                map.into_pyobject(py).map(|o| o.unbind().into())
            }
            GameStateVariant::ThreePlayer(s) => {
                let mut map = HashMap::new();
                for p in targets {
                    map.insert(p, s.get_observation(p));
                }
                map.into_pyobject(py).map(|o| o.unbind().into())
            }
        }
    }

    pub fn get_observation<'py>(&mut self, py: Python<'py>, player_id: u8) -> PyResult<Py<PyAny>> {
        match &mut self.variant {
            GameStateVariant::FourPlayer(s) => s
                .get_observation(player_id)
                .into_pyobject(py)
                .map(|o| o.unbind().into()),
            GameStateVariant::ThreePlayer(s) => s
                .get_observation(player_id)
                .into_pyobject(py)
                .map(|o| o.unbind().into()),
        }
    }

    fn get_obs_py<'py>(
        &mut self,
        py: Python<'py>,
        players: Option<Vec<u8>>,
    ) -> PyResult<Py<PyAny>> {
        self.get_observations(py, players)
    }

    #[pyo3(signature = (oya=None, wall=None, round_wind=None, scores=None, honba=None, kyotaku=None, seed=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn reset<'py>(
        &mut self,
        py: Python<'py>,
        oya: Option<u8>,
        wall: Option<Vec<u8>>,
        round_wind: Option<u8>,
        scores: Option<Vec<i32>>,
        honba: Option<u8>,
        kyotaku: Option<u32>,
        seed: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        // Read defaults before mutable borrow
        let default_oya = with_variant!(self, |s| s.oya);
        let default_round_wind = with_variant!(self, |s| s.round_wind);
        let default_honba = with_variant!(self, |s| s.honba);
        let default_riichi_sticks = with_variant!(self, |s| s.riichi_sticks);

        with_variant_mut!(self, |s| {
            if let Some(sd) = seed {
                s.seed = Some(sd);
            }
            s.reset();
            s._initialize_round(
                oya.unwrap_or(default_oya),
                round_wind.unwrap_or(default_round_wind),
                honba.unwrap_or(default_honba),
                kyotaku.unwrap_or(default_riichi_sticks),
                wall,
                scores,
            );
        });

        let active = with_variant!(self, |s| s.active_players.clone());
        self.get_obs_py(py, Some(active))
    }

    pub fn _get_legal_actions(&mut self, pid: u8) -> Vec<Action> {
        with_variant_mut!(self, |s| s._get_legal_actions_internal(pid))
    }

    #[pyo3(signature = (actions))]
    pub fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: HashMap<u8, AnyAction>,
    ) -> PyResult<Py<PyAny>> {
        let actions: HashMap<u8, Action> = actions.into_iter().map(|(k, v)| (k, v.0)).collect();
        with_variant_mut!(self, |s| s.step(&actions));
        let has_error = with_variant!(self, |s| s.last_error.is_some());
        if has_error {
            let dict = pyo3::types::PyDict::new(py);
            return Ok(dict.unbind().into());
        }
        let active = with_variant!(self, |s| s.active_players.clone());
        self.get_obs_py(py, Some(active))
    }

    pub fn apply_mjai_event(&mut self, py: Python, event: Py<PyAny>) -> PyResult<()> {
        let json = py.import("json")?;
        let s: String = json.call_method1("dumps", (event,))?.extract()?;
        let ev: MjaiEvent = serde_json::from_str(&s).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("JSON Parse Error: {}", e))
        })?;
        with_variant_mut!(self, |s| s.apply_mjai_event(ev));
        Ok(())
    }
}
