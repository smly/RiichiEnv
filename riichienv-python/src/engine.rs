use std::collections::HashMap;
use std::str::FromStr;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use riichienv_core::action::Action;
use riichienv_core::engine as core_engine;
use riichienv_core::engine::{
    BatchDecision, Decision, EngineConfig, EngineStepError, EventLogPolicy, GameMode, GameSnapshot,
    ObservationVariant, ResetOptions,
};
use riichienv_core::rule::GameRule;

use crate::env::AnyAction;
use crate::event_journal::EventJournal;

fn config(
    game_mode: &str,
    seed: Option<u64>,
    rule: Option<GameRule>,
    event_log: bool,
) -> PyResult<EngineConfig> {
    let mode = GameMode::from_str(game_mode).map_err(PyErr::from)?;
    Ok(EngineConfig {
        mode,
        rule: rule.unwrap_or_default(),
        seed,
        round_wind: 0,
        event_log: if event_log {
            EventLogPolicy::Full
        } else {
            EventLogPolicy::Off
        },
    })
}

fn observation_object(py: Python<'_>, observation: ObservationVariant) -> PyResult<Py<PyAny>> {
    match observation {
        ObservationVariant::FourPlayer(observation) => observation
            .into_pyobject(py)
            .map(|value| value.unbind().into()),
        ObservationVariant::ThreePlayer(observation) => observation
            .into_pyobject(py)
            .map(|value| value.unbind().into()),
    }
}

fn decisions_dict(py: Python<'_>, decisions: Vec<Decision>) -> PyResult<Py<PyAny>> {
    let result = PyDict::new(py);
    for decision in decisions {
        result.set_item(
            decision.player_id,
            observation_object(py, decision.observation)?,
        )?;
    }
    Ok(result.unbind().into())
}

fn batch_decisions_list(py: Python<'_>, decisions: Vec<BatchDecision>) -> PyResult<Py<PyAny>> {
    let result = PyList::empty(py);
    for decision in decisions {
        result.append((
            decision.env_index,
            decision.player_id,
            observation_object(py, decision.observation)?,
        ))?;
    }
    Ok(result.unbind().into())
}

fn snapshot_dict<'py>(py: Python<'py>, snapshot: &GameSnapshot) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("mode", snapshot.mode.as_str())?;
    result.set_item("num_players", snapshot.num_players)?;
    result.set_item("phase", format!("{:?}", snapshot.phase))?;
    result.set_item("current_player", snapshot.current_player)?;
    let active_players = snapshot
        .active_players
        .iter()
        .copied()
        .map(u32::from)
        .collect::<Vec<_>>();
    result.set_item("active_players", active_players)?;
    result.set_item("scores", &snapshot.scores)?;
    result.set_item("round_wind", snapshot.round_wind)?;
    result.set_item("kyoku", snapshot.kyoku)?;
    result.set_item("honba", snapshot.honba)?;
    result.set_item("riichi_sticks", snapshot.riichi_sticks)?;
    result.set_item("wall_tiles_remaining", snapshot.wall_tiles_remaining)?;
    result.set_item("done", snapshot.done)?;
    Ok(result)
}

fn error_dict<'py>(py: Python<'py>, error: &EngineStepError) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    match error {
        EngineStepError::IllegalAction { player_id, message } => {
            result.set_item("kind", "illegal_action")?;
            result.set_item("player_id", player_id)?;
            result.set_item("message", message)?;
        }
        EngineStepError::State { message } => {
            result.set_item("kind", "state")?;
            result.set_item("player_id", py.None())?;
            result.set_item("message", message)?;
        }
    }
    Ok(result)
}

/// Strict, variant-independent engine facade for new Python callers.
#[pyclass(
    name = "GameEngine",
    module = "riichienv._riichienv",
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub struct GameEngine {
    inner: core_engine::GameEngine,
}

#[pymethods]
impl GameEngine {
    #[new]
    #[pyo3(signature = (game_mode="4p-red-single", seed=None, rule=None, event_log=true))]
    fn new(
        py: Python<'_>,
        game_mode: &str,
        seed: Option<u64>,
        rule: Option<GameRule>,
        event_log: bool,
    ) -> PyResult<Self> {
        let config = config(game_mode, seed, rule, event_log)?;
        Ok(Self {
            inner: py.detach(move || core_engine::GameEngine::new(config))?,
        })
    }

    #[getter]
    fn mode(&self) -> &'static str {
        self.inner.mode().as_str()
    }

    #[getter]
    fn num_players(&self) -> u8 {
        self.inner.num_players()
    }

    #[getter]
    fn is_done(&self) -> bool {
        self.inner.is_done()
    }

    #[getter]
    fn mjai_log(&self, py: Python<'_>) -> Vec<String> {
        py.detach(|| self.inner.mjai_log().to_vec())
    }

    #[getter]
    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        Ok(snapshot_dict(py, &self.inner.snapshot())?.unbind().into())
    }

    #[getter]
    fn event_journal(&self, py: Python<'_>) -> PyResult<EventJournal> {
        py.detach(|| self.inner.event_journal())
            .map(EventJournal::from_core)
            .map_err(Into::into)
    }

    fn decisions(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let decisions = py.detach(|| self.inner.decisions());
        decisions_dict(py, decisions)
    }

    #[pyo3(signature = (oya=0, wall=None, round_wind=0, scores=None, honba=0, riichi_sticks=0, seed=None))]
    #[allow(clippy::too_many_arguments)]
    fn reset(
        &mut self,
        py: Python<'_>,
        oya: u8,
        wall: Option<Vec<u8>>,
        round_wind: u8,
        scores: Option<Vec<i32>>,
        honba: u8,
        riichi_sticks: u32,
        seed: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        let decisions = py.detach(|| {
            self.inner.reset(ResetOptions {
                oya,
                wall,
                round_wind,
                scores,
                honba,
                riichi_sticks,
                seed,
            })
        })?;
        decisions_dict(py, decisions)
    }

    fn step(&mut self, py: Python<'_>, actions: HashMap<u8, AnyAction>) -> PyResult<Py<PyAny>> {
        let actions: HashMap<u8, Action> = actions
            .into_iter()
            .map(|(player_id, action)| (player_id, action.0))
            .collect();
        let outcome = py.detach(|| self.inner.step(&actions));
        let result = PyDict::new(py);
        result.set_item("observations", decisions_dict(py, outcome.decisions)?)?;
        result.set_item("events", outcome.events)?;
        result.set_item("snapshot", snapshot_dict(py, &outcome.snapshot)?)?;
        match outcome.error {
            Some(error) => result.set_item("error", error_dict(py, &error)?)?,
            None => result.set_item("error", py.None())?,
        }
        Ok(result.unbind().into())
    }

    fn __repr__(&self) -> String {
        format!(
            "GameEngine(mode='{}', done={})",
            self.inner.mode().as_str(),
            self.inner.is_done()
        )
    }
}

/// Serial vector of independent engines for batched self-play orchestration.
#[pyclass(
    name = "BatchGameEngine",
    module = "riichienv._riichienv",
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub struct BatchGameEngine {
    inner: core_engine::BatchGameEngine,
}

#[pymethods]
impl BatchGameEngine {
    #[new]
    #[pyo3(signature = (count, game_mode="4p-red-single", seed=None, rule=None, event_log=false))]
    fn new(
        py: Python<'_>,
        count: usize,
        game_mode: &str,
        seed: Option<u64>,
        rule: Option<GameRule>,
        event_log: bool,
    ) -> PyResult<Self> {
        let config = config(game_mode, seed, rule, event_log)?;
        Ok(Self {
            inner: py
                .detach(move || core_engine::BatchGameEngine::try_homogeneous(config, count))?,
        })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn decisions(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let decisions = py.detach(|| self.inner.decisions());
        batch_decisions_list(py, decisions)
    }

    fn step(
        &mut self,
        py: Python<'_>,
        actions: Vec<HashMap<u8, AnyAction>>,
    ) -> PyResult<Py<PyAny>> {
        let actions = actions
            .into_iter()
            .map(|environment| {
                environment
                    .into_iter()
                    .map(|(player_id, action)| (player_id, action.0))
                    .collect::<HashMap<u8, Action>>()
            })
            .collect::<Vec<_>>();
        let outcome = py.detach(|| self.inner.step(&actions))?;
        let result = PyDict::new(py);
        result.set_item("decisions", batch_decisions_list(py, outcome.decisions)?)?;
        let snapshots = PyList::empty(py);
        for snapshot in &outcome.snapshots {
            snapshots.append(snapshot_dict(py, snapshot)?)?;
        }
        result.set_item("snapshots", snapshots)?;
        let errors = PyList::empty(py);
        for error in &outcome.errors {
            match error {
                Some(error) => errors.append(error_dict(py, error)?)?,
                None => errors.append(py.None())?,
            }
        }
        result.set_item("errors", errors)?;
        Ok(result.unbind().into())
    }

    fn __repr__(&self) -> String {
        format!("BatchGameEngine(len={})", self.inner.len())
    }
}
