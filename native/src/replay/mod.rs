/*
 * replay/mod.rs: Utilities for replaying games to verify the agari calculator.
 */
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyList, PyListMethods};
use std::sync::Arc;

use crate::action::Action as EnvAction;
use crate::agari_calculator::AgariCalculator;
use crate::types::{Agari, Conditions, Meld, MeldType};

pub mod mjai_replay;
pub mod mjsoul_replay;

pub use mjai_replay::MjaiEvent;
pub use mjai_replay::MjaiReplay;
pub use mjsoul_replay::MjSoulReplay;

#[derive(Clone, Debug)]
pub enum Action {
    DiscardTile {
        seat: usize,
        tile: u8,
        is_liqi: bool,
        is_wliqi: bool,
        doras: Option<Vec<u8>>,
    },
    DealTile {
        seat: usize,
        tile: u8,
        doras: Option<Vec<u8>>,
        left_tile_count: Option<u8>,
    },
    ChiPengGang {
        seat: usize,
        meld_type: MeldType,
        tiles: Vec<u8>,
        froms: Vec<usize>,
    },
    AnGangAddGang {
        seat: usize,
        meld_type: MeldType,
        tiles: Vec<u8>,
        tile_raw_id: u8,
        doras: Option<Vec<u8>>,
    },
    Dora {
        dora_marker: u8,
    },
    Hule {
        hules: Vec<HuleData>,
    },
    NoTile,
    LiuJu {
        lj_type: u8,
        seat: usize,
        tiles: Vec<u8>,
    },
    Other(String),
}

#[derive(Clone, Debug)]
pub struct HuleData {
    pub seat: usize,
    pub hu_tile: u8,
    pub zimo: bool,
    pub count: u32,
    pub fu: u32,
    pub fans: Vec<u32>,
    pub li_doras: Option<Vec<u8>>,
    pub yiman: bool,
    pub point_rong: u32,
    pub point_zimo_qin: u32,
    pub point_zimo_xian: u32,
}

#[pyclass(module = "riichienv._riichienv")]
pub struct KyokuStepIterator {
    state: crate::state::GameState,
    actions: Arc<[Action]>,
    idx: usize,
    pending_action: Option<(u8, EnvAction)>,
    filter_seat: Option<u8>,
    skip_single_action: bool,
}

#[pymethods]
impl KyokuStepIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let actions = slf.actions.clone();

        loop {
            if let Some((pid, action)) = slf.pending_action.take() {
                let obs =
                    slf.state
                        .get_observation_for_replay(pid, &action, &format!("{:?}", action))?;

                let current_log_action = &actions[slf.idx];
                slf.state.apply_log_action(current_log_action);
                slf.idx += 1;

                if let Some(target) = slf.filter_seat {
                    if pid == target {
                        // Filter forced actions if requested
                        if slf.skip_single_action && obs._legal_actions.len() <= 1 {
                            continue;
                        }

                        let py = slf.py();
                        return Ok(Some((obs, action).into_pyobject(py)?.unbind().into()));
                    }
                    continue;
                } else {
                    let py = slf.py();
                    return Ok(Some((pid, obs, action).into_pyobject(py)?.unbind().into()));
                }
            }

            if slf.idx >= actions.len() {
                return Ok(None);
            }

            let action = &actions[slf.idx];
            match action {
                Action::DealTile { .. }
                | Action::Dora { .. }
                | Action::NoTile
                | Action::LiuJu { .. } => {
                    slf.state.apply_log_action(action);
                    slf.idx += 1;
                }
                Action::Other(_) => {
                    slf.idx += 1;
                }
                Action::DiscardTile {
                    seat,
                    tile,
                    is_liqi,
                    ..
                } => {
                    let pid = *seat as u8;
                    let env_action =
                        EnvAction::new(crate::action::ActionType::Discard, Some(*tile), Vec::new());

                    if *is_liqi {
                        let riichi_action =
                            EnvAction::new(crate::action::ActionType::Riichi, None, Vec::new());

                        let obs = slf.state.get_observation_for_replay(
                            pid,
                            &riichi_action,
                            &format!("{:?}", action),
                        )?;

                        slf.pending_action = Some((pid, env_action));

                        if let Some(target) = slf.filter_seat {
                            if pid == target {
                                let py = slf.py();
                                return Ok(Some(
                                    (obs, riichi_action).into_pyobject(py)?.unbind().into(),
                                ));
                            }
                        } else {
                            let py = slf.py();
                            return Ok(Some(
                                (pid, obs, riichi_action).into_pyobject(py)?.unbind().into(),
                            ));
                        }
                    } else {
                        let obs = slf.state.get_observation_for_replay(
                            pid,
                            &env_action,
                            &format!("{:?}", action),
                        )?;

                        slf.state.apply_log_action(action);
                        slf.idx += 1;

                        if let Some(target) = slf.filter_seat {
                            if pid == target {
                                if slf.skip_single_action && obs._legal_actions.len() <= 1 {
                                    continue;
                                }

                                let py = slf.py();
                                return Ok(Some(
                                    (obs, env_action).into_pyobject(py)?.unbind().into(),
                                ));
                            }
                        } else {
                            let py = slf.py();
                            return Ok(Some(
                                (pid, obs, env_action).into_pyobject(py)?.unbind().into(),
                            ));
                        }
                    }
                }
                Action::ChiPengGang {
                    seat,
                    meld_type,
                    tiles,
                    ..
                } => {
                    let pid = *seat as u8;

                    let env_action_type = match meld_type {
                        MeldType::Chi => crate::action::ActionType::Chi,
                        MeldType::Peng => crate::action::ActionType::Pon,
                        MeldType::Gang => crate::action::ActionType::Daiminkan,
                        _ => crate::action::ActionType::Chi,
                    };

                    let t = tiles.first().copied();
                    let env_action = EnvAction::new(env_action_type, t, tiles.to_vec());

                    let obs = slf.state.get_observation_for_replay(
                        pid,
                        &env_action,
                        &format!("{:?}", action),
                    )?;

                    slf.state.apply_log_action(action);
                    slf.idx += 1;

                    if let Some(target) = slf.filter_seat {
                        if pid == target {
                            if slf.skip_single_action && obs._legal_actions.len() <= 1 {
                                continue;
                            }
                            let py = slf.py();
                            return Ok(Some((obs, env_action).into_pyobject(py)?.unbind().into()));
                        }
                    } else {
                        let py = slf.py();
                        return Ok(Some(
                            (pid, obs, env_action).into_pyobject(py)?.unbind().into(),
                        ));
                    }
                }
                Action::AnGangAddGang {
                    seat,
                    meld_type,
                    tiles,
                    ..
                } => {
                    let pid = *seat as u8;
                    let atype = match meld_type {
                        MeldType::Angang => crate::action::ActionType::Ankan,
                        MeldType::Addgang => crate::action::ActionType::Kakan,
                        _ => crate::action::ActionType::Ankan,
                    };

                    let env_action = match atype {
                        crate::action::ActionType::Ankan => {
                            let t34 = tiles[0] / 4;
                            let lowest = t34 * 4;
                            EnvAction::new(
                                atype,
                                Some(lowest),
                                vec![lowest, lowest + 1, lowest + 2, lowest + 3],
                            )
                        }
                        crate::action::ActionType::Kakan => {
                            let t34 = tiles[0] / 4;
                            let tile = tiles[0];
                            // Find the existing Peng meld to get its tiles
                            let mut consume = Vec::new();
                            for m in &slf.state.players[pid as usize].melds {
                                if m.meld_type == MeldType::Peng && m.tiles[0] / 4 == t34 {
                                    consume = m.tiles.clone();
                                    break;
                                }
                            }
                            EnvAction::new(atype, Some(tile), consume)
                        }
                        _ => {
                            let tile = tiles.first().copied();
                            EnvAction::new(atype, tile, tiles.to_vec())
                        }
                    };

                    let obs = slf.state.get_observation_for_replay(
                        pid,
                        &env_action,
                        &format!("{:?}", action),
                    )?;

                    slf.state.apply_log_action(action);
                    slf.idx += 1;

                    if let Some(target) = slf.filter_seat {
                        if pid == target {
                            if slf.skip_single_action && obs._legal_actions.len() <= 1 {
                                continue;
                            }
                            let py = slf.py();
                            return Ok(Some((obs, env_action).into_pyobject(py)?.unbind().into()));
                        }
                    } else {
                        let py = slf.py();
                        return Ok(Some(
                            (pid, obs, env_action).into_pyobject(py)?.unbind().into(),
                        ));
                    }
                }
                Action::Hule { hules } => {
                    let first = &hules[0];
                    let pid = first.seat as u8;

                    let atype = if first.zimo {
                        crate::action::ActionType::Tsumo
                    } else {
                        crate::action::ActionType::Ron
                    };
                    let tile = if first.zimo {
                        slf.state.drawn_tile
                    } else {
                        slf.state.last_discard.map(|(_, t)| t)
                    };
                    let env_action = EnvAction::new(atype, tile, Vec::new());

                    let obs = slf.state.get_observation_for_replay(
                        pid,
                        &env_action,
                        &format!("{:?}", action),
                    )?;

                    slf.state.apply_log_action(action);
                    slf.idx += 1;

                    if let Some(target) = slf.filter_seat {
                        if pid == target {
                            if slf.skip_single_action && obs._legal_actions.len() <= 1 {
                                continue;
                            }
                            let py = slf.py();
                            return Ok(Some((obs, env_action).into_pyobject(py)?.unbind().into()));
                        }
                    } else {
                        let py = slf.py();
                        return Ok(Some(
                            (pid, obs, env_action).into_pyobject(py)?.unbind().into(),
                        ));
                    }
                }
            }
        }
    }
}

#[pyclass(name = "Kyoku", module = "riichienv._riichienv")]
#[derive(Clone)]
pub struct LogKyoku {
    #[pyo3(get)]
    pub scores: Vec<i32>,
    #[pyo3(get)]
    pub doras: Vec<u8>,
    #[pyo3(get)]
    pub ura_doras: Vec<u8>,
    #[pyo3(get)]
    pub hands: Vec<Vec<u8>>,
    #[pyo3(get)]
    pub chang: u8,
    #[pyo3(get)]
    pub ju: u8,
    #[pyo3(get)]
    pub ben: u8,
    #[pyo3(get)]
    pub liqibang: u8,
    #[pyo3(get)]
    pub left_tile_count: u8,
    #[pyo3(get)]
    pub end_scores: Vec<i32>,
    #[pyo3(get)]
    pub wliqi: Vec<bool>,
    #[pyo3(get)]
    pub paishan: Option<String>,
    pub actions: Arc<[Action]>,
    #[pyo3(get)]
    pub rule: crate::rule::GameRule,
    #[pyo3(get)]
    pub game_end_scores: Option<Vec<i32>>,
}

#[pymethods]
impl LogKyoku {
    fn take_agari_contexts(&self) -> PyResult<AgariContextIterator> {
        Ok(AgariContextIterator::new(self.clone()))
    }

    #[pyo3(signature = (seat=None, rule=None, skip_single_action=None))]
    fn steps(
        &self,
        seat: Option<u8>,
        rule: Option<crate::rule::GameRule>,
        skip_single_action: Option<bool>,
    ) -> PyResult<KyokuStepIterator> {
        let rule = rule.unwrap_or(self.rule);
        let skip_single_action = skip_single_action.unwrap_or(true);
        let mut state = crate::state::GameState::new(0, false, None, 0, rule);

        // Initialize state from LogKyoku data
        let initial_scores: [i32; 4] = self.scores.clone().try_into().unwrap_or([25000; 4]);
        let doras = self.doras.clone();

        let mut oya_idx = (self.ju % 4) as usize;
        for (i, h) in self.hands.iter().enumerate() {
            if h.len() == 14 {
                oya_idx = i;
                break;
            }
        }
        let oya = oya_idx as u8;

        let bakaze = match self.chang {
            0 => crate::types::Wind::East,
            1 => crate::types::Wind::South,
            2 => crate::types::Wind::West,
            3 => crate::types::Wind::North,
            _ => crate::types::Wind::East,
        } as u8;

        let mut wall = None;
        if let Some(p_hex) = &self.paishan {
            if let Ok(w) = hex::decode(p_hex) {
                wall = Some(w);
            }
        }

        state._initialize_round(
            oya,
            bakaze,
            self.ben,
            self.liqibang as u32,
            wall.clone(),
            Some(initial_scores),
        );

        state._initialize_round(
            oya,
            bakaze,
            self.ben,
            self.liqibang as u32,
            wall.clone(),
            Some(initial_scores),
        );

        for (i, h) in self.hands.iter().enumerate() {
            state.players[i].hand = h.clone();
        }

        // If dealer starts with 14 tiles, set drawn_tile to allow immediate Tsumo/Discard
        if state.players[oya_idx].hand.len() == 14 {
            let mut dt = state.players[oya_idx].hand.last().copied();

            // Peek at the first action to see which tile was actually "drawn" (or acted upon)
            if let Some(first_action) = self.actions.first() {
                match first_action {
                    Action::Hule { hules } => {
                        if let Some(h) = hules.iter().find(|h| h.seat == oya_idx && h.zimo) {
                            dt = Some(h.hu_tile);
                        }
                    }
                    Action::DiscardTile { seat, tile, .. } => {
                        if *seat == oya_idx {
                            dt = Some(*tile);
                        }
                    }
                    Action::AnGangAddGang { seat, tiles, .. } => {
                        if *seat == oya_idx {
                            // For Ankan/Kakan, usually the first tile is the relevant one or part of the meld
                            dt = tiles.first().copied();
                        }
                    }
                    Action::ChiPengGang {
                        seat,
                        tiles,
                        meld_type,
                        ..
                    } => {
                        // Should not happen for Tsumo first turn, but if it does (e.g. late join?), handle it
                        if *seat == oya_idx && *meld_type == MeldType::Angang {
                            dt = tiles.first().copied();
                        }
                    }
                    _ => {}
                }
            }

            state.drawn_tile = dt;
            state.needs_tsumo = false;
        }

        for p in state.players.iter_mut() {
            p.hand.sort();
        }
        state.wall.dora_indicators = doras;

        // If wall was initialized from a full wall (136 tiles), pop all tiles starting hands consumed
        if state.wall.tiles.len() == 136 {
            let total_hand_tiles: usize = state.players.iter().map(|p| p.hand.len()).sum();
            for _ in 0..total_hand_tiles {
                if !state.wall.tiles.is_empty() {
                    state.wall.tiles.pop();
                }
            }
        }

        Ok(KyokuStepIterator {
            state,
            actions: self.actions.clone(),
            idx: 0,
            pending_action: None,
            filter_seat: seat,
            skip_single_action,
        })
    }

    fn events(&self, py: Python) -> PyResult<Py<PyAny>> {
        let events = PyList::empty(py);

        // Name: NewRound
        let nr_event = PyDict::new(py);
        nr_event.set_item("name", "NewRound")?;
        let nr_data = PyDict::new(py);

        nr_data.set_item("scores", self.scores.clone())?;

        if !self.doras.is_empty() {
            let d_list = PyList::new(py, self.doras.iter().map(|t| TileConverter::to_string(*t)))?;

            nr_data.set_item("doras", d_list)?;
        } else {
            nr_data.set_item("doras", PyList::empty(py))?;
        }

        if let Some(first) = self.doras.first() {
            nr_data.set_item("dora_marker", TileConverter::to_string(*first))?;
        }

        for i in 0..4 {
            let hand_list = PyList::new(
                py,
                self.hands[i].iter().map(|t| TileConverter::to_string(*t)),
            )?;

            nr_data.set_item(format!("tiles{}", i), hand_list)?;
        }

        nr_data.set_item("chang", self.chang)?;
        nr_data.set_item("ju", self.ju)?;
        nr_data.set_item("ben", self.ben)?;
        nr_data.set_item("liqibang", self.liqibang)?;
        nr_data.set_item("left_tile_count", self.left_tile_count)?;

        if !self.ura_doras.is_empty() {
            let ud_list = PyList::new(
                py,
                self.ura_doras.iter().map(|t| TileConverter::to_string(*t)),
            )?;

            nr_data.set_item("ura_doras", ud_list)?;
        }

        if let Some(paishan_str) = &self.paishan {
            nr_data.set_item("paishan", paishan_str)?;
        }

        nr_event.set_item("data", nr_data)?;
        events.append(nr_event)?;

        // Actions
        for action in self.actions.iter() {
            let a_event = PyDict::new(py);
            let a_data = PyDict::new(py);

            match action {
                Action::DiscardTile {
                    seat,
                    tile,
                    is_liqi,
                    is_wliqi,
                    doras,
                } => {
                    a_event.set_item("name", "DiscardTile")?;
                    a_data.set_item("seat", seat)?;
                    a_data.set_item("tile", TileConverter::to_string(*tile))?;
                    a_data.set_item("is_liqi", is_liqi)?;
                    a_data.set_item("is_wliqi", is_wliqi)?;
                    if let Some(d) = doras {
                        let d_list =
                            PyList::new(py, d.iter().map(|t| TileConverter::to_string(*t)))?;

                        a_data.set_item("doras", d_list)?;
                    }
                }
                Action::DealTile {
                    seat,
                    tile,
                    doras,
                    left_tile_count,
                } => {
                    a_event.set_item("name", "DealTile")?;
                    a_data.set_item("seat", seat)?;
                    a_data.set_item("tile", TileConverter::to_string(*tile))?;
                    if let Some(d) = doras {
                        let d_list =
                            PyList::new(py, d.iter().map(|t| TileConverter::to_string(*t)))?;
                        a_data.set_item("doras", d_list)?;
                    }
                    if let Some(ltc) = left_tile_count {
                        a_data.set_item("left_tile_count", ltc)?;
                    }
                }
                Action::ChiPengGang {
                    seat,
                    meld_type,
                    tiles,
                    froms,
                } => {
                    a_event.set_item("name", "ChiPengGang")?;
                    a_data.set_item("seat", seat)?;
                    let mt_int = match meld_type {
                        MeldType::Chi => 0,
                        MeldType::Peng => 1,
                        MeldType::Gang => 2,
                        MeldType::Angang => 3,
                        MeldType::Addgang => 2,
                    };
                    a_data.set_item("type", mt_int)?;
                    let t_list =
                        PyList::new(py, tiles.iter().map(|t| TileConverter::to_string(*t)))?;

                    a_data.set_item("tiles", t_list)?;
                    a_data.set_item("froms", froms.clone())?;
                }

                Action::AnGangAddGang {
                    seat,
                    meld_type,
                    tiles,
                    tile_raw_id: _,
                    doras,
                } => {
                    a_event.set_item("name", "AnGangAddGang")?;
                    a_data.set_item("seat", seat)?;
                    let mt_int = match meld_type {
                        MeldType::Angang => 3,
                        _ => 2,
                    };
                    a_data.set_item("type", mt_int)?;
                    if let Some(first) = tiles.first() {
                        a_data.set_item("tiles", TileConverter::to_string(*first))?;
                    }
                    if let Some(d) = doras {
                        let d_list =
                            PyList::new(py, d.iter().map(|t| TileConverter::to_string(*t)))?;
                        a_data.set_item("doras", d_list)?;
                    }
                }
                Action::Hule { hules } => {
                    a_event.set_item("name", "Hule")?;
                    let h_list = PyList::empty(py);
                    for h in hules {
                        let h_dict = PyDict::new(py);
                        let ht_str = TileConverter::to_string(h.hu_tile);
                        h_dict.set_item("seat", h.seat)?;
                        h_dict.set_item("hu_tile", ht_str)?;
                        h_dict.set_item("zimo", h.zimo)?;
                        h_dict.set_item("count", h.count)?;
                        h_dict.set_item("fu", h.fu)?;
                        let f_list = PyList::empty(py);
                        for f_id in &h.fans {
                            let f_dict = PyDict::new(py);
                            f_dict.set_item("id", f_id)?;
                            f_list.append(f_dict)?;
                        }
                        h_dict.set_item("fans", f_list)?;
                        h_dict.set_item("point_rong", h.point_rong)?;
                        h_dict.set_item("point_zimo_qin", h.point_zimo_qin)?;
                        h_dict.set_item("point_zimo_xian", h.point_zimo_xian)?;
                        h_dict.set_item("yiman", h.yiman)?;
                        if let Some(ld) = &h.li_doras {
                            let ld_list =
                                PyList::new(py, ld.iter().map(|t| TileConverter::to_string(*t)))?;
                            h_dict.set_item("li_doras", ld_list)?;
                        }
                        h_list.append(h_dict)?;
                    }
                    a_data.set_item("hules", h_list)?;
                }
                Action::Dora { dora_marker } => {
                    a_event.set_item("name", "Dora")?;
                    a_data.set_item("dora_marker", TileConverter::to_string(*dora_marker))?;
                }
                Action::NoTile => {
                    a_event.set_item("name", "NoTile")?;
                }
                Action::LiuJu {
                    lj_type,
                    seat,
                    tiles,
                } => {
                    a_event.set_item("name", "LiuJu")?;
                    a_data.set_item("type", lj_type)?;
                    a_data.set_item("seat", seat)?;
                    let t_strs: Vec<String> =
                        tiles.iter().map(|t| TileConverter::to_string(*t)).collect();
                    a_data.set_item("tiles", t_strs)?;
                }
                Action::Other(_) => {
                    continue;
                }
            }
            a_event.set_item("data", a_data)?;
            events.append(a_event)?;
        }

        Ok(events.into())
    }

    fn grp_features(&self, py: Python) -> PyResult<Py<PyAny>> {
        let features = PyDict::new(py);
        features.set_item("chang", self.chang)?;
        features.set_item("ju", self.ju)?;
        features.set_item("ben", self.ben)?;
        features.set_item("liqibang", self.liqibang)?;
        features.set_item("scores", self.scores.clone())?;
        features.set_item("end_scores", self.end_scores.clone())?;
        features.set_item("wliqi", self.wliqi.clone())?;

        // Calculate delta_scores
        let mut delta_scores = Vec::new();
        if self.scores.len() == self.end_scores.len() {
            for i in 0..self.scores.len() {
                delta_scores.push(self.end_scores[i] - self.scores[i]);
            }
        }
        features.set_item("delta_scores", delta_scores)?;

        Ok(features.into())
    }

    fn take_grp_features(&self, py: Python) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);

        // Basic Integers
        dict.set_item("chang", self.chang)?;
        dict.set_item("ju", self.ju)?;
        dict.set_item("ben", self.ben)?;
        dict.set_item("liqibang", self.liqibang)?;

        let initial_scores = &self.scores;
        // Default to initial scores if end scores not populated (e.g. last round in some logs, or incomplete data)
        let end_scores = if self.end_scores.is_empty() {
            initial_scores
        } else {
            &self.end_scores
        };

        // Helper to calc ranks: 0..3 (0 is top)
        fn get_ranks(scores: &[i32]) -> Vec<i32> {
            let mut indexed: Vec<(usize, i32)> = scores.iter().copied().enumerate().collect();
            // Sort descending by score. Then by seat index (lower seat index wins ties)
            indexed.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

            let mut ranks = vec![0; 4];
            for (rank, (seat, _)) in indexed.into_iter().enumerate() {
                ranks[seat] = rank as i32;
            }
            ranks
        }

        let initial_ranks = get_ranks(initial_scores);
        let end_ranks = get_ranks(end_scores);

        let delta_scores: Vec<i32> = initial_scores
            .iter()
            .zip(end_scores.iter())
            .map(|(s, e)| e - s)
            .collect();
        let delta_ranks: Vec<i32> = initial_ranks
            .iter()
            .zip(end_ranks.iter())
            .map(|(s, e)| e - s)
            .collect();

        dict.set_item("round_initial_scores", initial_scores.clone())?;
        dict.set_item("round_end_scores", end_scores.clone())?;
        dict.set_item("round_delta_scores", delta_scores)?;

        dict.set_item("round_initial_ranks", initial_ranks)?;
        dict.set_item("round_end_ranks", end_ranks.clone())?;
        dict.set_item("round_delta_ranks", delta_ranks)?;

        if let Some(final_scores) = &self.game_end_scores {
            let final_ranks = get_ranks(final_scores);
            dict.set_item("final_ranks", final_ranks)?;
        } else {
            dict.set_item("final_ranks", end_ranks)?;
        }

        for (i, hand) in self.hands.iter().enumerate() {
            let hand_u32: Vec<u32> = hand.iter().map(|&x| x as u32).collect();
            dict.set_item(format!("player{}_initial_hand_tids", i), hand_u32)?;
        }

        Ok(dict.into())
    }
}

#[pyclass(module = "riichienv._riichienv")]
pub struct AgariContextIterator {
    kyoku: LogKyoku,
    action_index: usize,
    pending_agari: Vec<AgariContext>,
    melds: Vec<Vec<Meld>>,
    current_hands: Vec<Vec<u8>>,
    liqi: Vec<bool>,
    wliqi: Vec<bool>,
    ippatsu: Vec<bool>,
    rinshan: Vec<bool>,
    is_first_turn: Vec<bool>,
    last_action_was_kakan: bool,
    kakan_tile: Option<u8>,
    current_doras: Vec<u8>,
    _current_liqibang: u8,
    current_left_tile_count: u8,
    wall: Vec<u8>,
    dora_count: u8,
    pending_minkan_doras: u8,
}

#[pymethods]
impl AgariContextIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<AgariContext> {
        slf.do_next()
    }
}

// Helper to parse paishan string
fn parse_paishan(s: &str) -> Vec<u8> {
    let mut wall = Vec::new();
    let mut chars = s.chars();
    while let (Some(n), Some(s_char)) = (chars.next(), chars.next()) {
        let mut t_str = String::with_capacity(2);
        t_str.push(n);
        t_str.push(s_char);
        wall.push(TileConverter::parse_tile_136(&t_str));
    }
    wall
}

impl AgariContextIterator {
    pub fn new(kyoku: LogKyoku) -> Self {
        let wall = if let Some(ref p) = kyoku.paishan {
            parse_paishan(p)
        } else {
            Vec::new()
        };

        AgariContextIterator {
            kyoku: kyoku.clone(),
            action_index: 0,
            pending_agari: Vec::new(),
            melds: vec![Vec::new(); 4],
            current_hands: kyoku.hands.clone(),
            liqi: vec![false; 4],
            wliqi: vec![false; 4],
            ippatsu: vec![false; 4],
            rinshan: vec![false; 4],
            is_first_turn: vec![true; 4],
            last_action_was_kakan: false,
            kakan_tile: None,
            current_doras: kyoku.doras.clone(),
            _current_liqibang: kyoku.liqibang,
            current_left_tile_count: kyoku.left_tile_count,
            wall,
            dora_count: 1, // Initial Dora is always 1
            pending_minkan_doras: 0,
        }
    }

    fn _recalc_doras(&mut self) {
        if self.wall.is_empty() {
            return;
        }
        let len = self.wall.len();
        // Base index for 1st Dora: len - 5
        // Base index for 1st Ura: len - 6
        // Indicators shift by -2 for each additional Dora
        self.current_doras.clear();
        for i in 0..self.dora_count {
            let offset = (i as usize) * 2;
            if len >= 5 + offset {
                let idx = len - 5 - offset;
                self.current_doras.push(self.wall[idx]);
            }
        }
    }

    fn _sync_doras_with_wall(&mut self) {
        if self.wall.is_empty() {
            return;
        }
        // If log has more doras, trust log and sync count
        if self.current_doras.len() > self.dora_count as usize {
            self.dora_count = self.current_doras.len() as u8;
            self.pending_minkan_doras = 0; // Log subsumes pending
        }
        // If count expects more doras, recalc from wall (Fixes Ankan bug)
        else if (self.dora_count as usize) > self.current_doras.len() {
            self._recalc_doras();
        }
    }

    fn _get_ura_indicators(&self) -> Vec<u8> {
        if self.wall.is_empty() {
            return Vec::new(); // Fallback or empty if no paishan
        }
        let mut uras = Vec::new();
        let len = self.wall.len();
        for i in 0..self.dora_count {
            let offset = (i as usize) * 2;
            if len >= 6 + offset {
                let idx = len - 6 - offset;
                uras.push(self.wall[idx]);
            }
        }
        uras
    }

    pub fn do_next(&mut self) -> Option<AgariContext> {
        if !self.pending_agari.is_empty() {
            return Some(self.pending_agari.remove(0));
        }

        while self.action_index < self.kyoku.actions.len() {
            let action = &self.kyoku.actions[self.action_index];
            self.action_index += 1;

            if !matches!(action, Action::Hule { .. }) {
                self.rinshan = vec![false; 4];
            }

            match action {
                Action::DiscardTile {
                    seat,
                    tile,
                    is_liqi,
                    is_wliqi,
                    doras,
                } => {
                    if self.last_action_was_kakan {
                        self.ippatsu = vec![false; 4];
                        self.is_first_turn = vec![false; 4];
                        self.last_action_was_kakan = false;
                        self.kakan_tile = None;
                    }

                    if *is_wliqi {
                        self.wliqi[*seat] = true;
                        self.ippatsu[*seat] = true;
                    }
                    if *is_liqi {
                        self.liqi[*seat] = true;
                        self.ippatsu[*seat] = true;
                    }
                    if !*is_liqi {
                        self.ippatsu[*seat] = false;
                    }
                    self.is_first_turn[*seat] = false;

                    TileConverter::match_and_remove_u8(&mut self.current_hands[*seat], *tile);

                    if let Some(d) = doras {
                        self.current_doras = d.clone();
                    }

                    // Discard reveals pending
                    if self.pending_minkan_doras > 0 {
                        self.dora_count += self.pending_minkan_doras;
                        self.pending_minkan_doras = 0;
                    }

                    self._sync_doras_with_wall();
                }
                Action::DealTile {
                    seat,
                    tile,
                    doras,
                    left_tile_count,
                } => {
                    if self.last_action_was_kakan {
                        self.ippatsu = vec![false; 4];
                        self.is_first_turn = vec![false; 4];
                        self.last_action_was_kakan = false;
                        self.kakan_tile = None;
                    }
                    self.current_hands[*seat].push(*tile);
                    if let Some(c) = left_tile_count {
                        self.current_left_tile_count = *c;
                    } else if self.current_left_tile_count > 0 {
                        self.current_left_tile_count -= 1;
                    }

                    if let Some(d) = doras {
                        self.current_doras = d.clone();
                        self.rinshan[*seat] = true;
                    }
                    self._sync_doras_with_wall();
                }
                Action::ChiPengGang {
                    seat,
                    meld_type,
                    tiles,
                    froms,
                } => {
                    self.rinshan = vec![false; 4];
                    self.ippatsu = vec![false; 4];
                    self.is_first_turn = vec![false; 4];
                    self.last_action_was_kakan = false;
                    self.kakan_tile = None;

                    for (i, t) in tiles.iter().enumerate() {
                        if i < froms.len() && froms[i] == *seat {
                            TileConverter::match_and_remove_u8(&mut self.current_hands[*seat], *t);
                        }
                    }
                    // Infer from_who from cpg_action if possible, or default to -1
                    let mut from_who = -1;
                    for &f in froms {
                        if f != *seat {
                            from_who = f as i8;
                            break;
                        }
                    }
                    self.melds[*seat].push(Meld {
                        meld_type: *meld_type,
                        tiles: tiles.clone(),
                        opened: true,
                        from_who,
                    });
                    if *meld_type == MeldType::Gang {
                        self.rinshan[*seat] = true;

                        // New Kan flushes pending
                        if self.pending_minkan_doras > 0 {
                            self.dora_count += self.pending_minkan_doras;
                            self.pending_minkan_doras = 0;
                        }
                        // Add this Kan to pending
                        self.pending_minkan_doras += 1;
                    }
                }
                Action::Dora { dora_marker } => {
                    if self.wall.is_empty() {
                        self.current_doras.push(*dora_marker);
                    } else {
                        // Dora action consumes pending if matches?
                        // Or just increments?
                        // Safer to increment and clear pending to avoid double count.
                        // But Action::Dora implies "One Dora".
                        // If we have 2 pending, and 1 Dora event...
                        // Usually events are specific.
                        self.dora_count += 1;
                        if self.pending_minkan_doras > 0 {
                            self.pending_minkan_doras -= 1;
                        }
                        self._sync_doras_with_wall();
                    }
                }
                Action::AnGangAddGang {
                    seat,
                    meld_type,
                    tiles,
                    tile_raw_id,
                    doras,
                } => {
                    self.rinshan = vec![false; 4];
                    if let Some(d) = doras {
                        self.current_doras = d.clone();
                    }

                    // New Kan flushes pending (for previous Minkan)
                    if self.pending_minkan_doras > 0 {
                        self.dora_count += self.pending_minkan_doras;
                        self.pending_minkan_doras = 0;
                    }

                    if *meld_type == MeldType::Angang {
                        self.ippatsu = vec![false; 4];
                        self.is_first_turn = vec![false; 4];
                        self.last_action_was_kakan = false;
                        self.kakan_tile = None;

                        let target_34 = *tile_raw_id;
                        for _ in 0..4 {
                            if let Some(pos) = self.current_hands[*seat]
                                .iter()
                                .position(|x| *x / 4 == target_34)
                            {
                                self.current_hands[*seat].remove(pos);
                            }
                        }

                        let mut m_tiles = vec![
                            target_34 * 4,
                            target_34 * 4 + 1,
                            target_34 * 4 + 2,
                            target_34 * 4 + 3,
                        ];
                        // Correct for red tiles
                        if target_34 == 4 {
                            m_tiles = vec![16, 17, 18, 19];
                        } else if target_34 == 13 {
                            m_tiles = vec![52, 53, 54, 55];
                        } else if target_34 == 22 {
                            m_tiles = vec![88, 89, 90, 91];
                        }

                        self.melds[*seat].push(Meld {
                            meld_type: *meld_type,
                            tiles: m_tiles,
                            opened: false,
                            from_who: -1,
                        });
                        self.rinshan[*seat] = true;

                        // Ankan: Immediate Reveal
                        if !self.wall.is_empty() {
                            self.dora_count += 1;
                        }
                    } else {
                        self.last_action_was_kakan = true;
                        self.kakan_tile = Some(tiles[0]);
                        self.rinshan[*seat] = true;
                        let mut upgraded = false;
                        for m in self.melds[*seat].iter_mut() {
                            if m.meld_type == MeldType::Peng && (m.tiles[0] / 4 == tiles[0] / 4) {
                                m.meld_type = MeldType::Addgang;
                                m.tiles.push(tiles[0]);
                                upgraded = true;
                                break;
                            }
                        }
                        if !upgraded {
                            self.melds[*seat].push(Meld {
                                meld_type: *meld_type,
                                tiles: tiles.clone(),
                                opened: true,
                                from_who: -1,
                            });
                        }
                        TileConverter::match_and_remove_u8(
                            &mut self.current_hands[*seat],
                            tiles[0],
                        );
                        // AddGang: Reveal Late (pending)
                        self.pending_minkan_doras += 1;
                    }
                    self._sync_doras_with_wall();
                }
                Action::Hule { hules } => {
                    for hule_data in hules {
                        let seat = hule_data.seat;
                        let win_tile = hule_data.hu_tile;

                        let is_zimo = hule_data.zimo;

                        let mut is_chankan = false;
                        if !is_zimo && self.last_action_was_kakan {
                            if let Some(k) = self.kakan_tile {
                                if k / 4 == win_tile / 4 {
                                    is_chankan = true;
                                }
                            }
                        }

                        let mut hand_136 = self.current_hands[seat].clone();
                        let melds_136 = self.melds[seat].clone();

                        let conditions = Conditions {
                            tsumo: is_zimo,
                            riichi: self.liqi[seat],
                            double_riichi: self.wliqi[seat],
                            ippatsu: self.ippatsu[seat],
                            haitei: (self.current_left_tile_count == 0)
                                && is_zimo
                                && !self.rinshan[seat],
                            houtei: (self.current_left_tile_count == 0)
                                && !is_zimo
                                && !self.rinshan[seat],
                            rinshan: self.rinshan[seat],
                            chankan: is_chankan,
                            tsumo_first_turn: self.is_first_turn[seat] && is_zimo,
                            player_wind: (((seat + 4 - self.kyoku.ju as usize) % 4) as u8).into(),
                            round_wind: self.kyoku.chang.into(),
                            kyoutaku: 0, // Not tracked in basic loop?
                            tsumi: 0,    // Not tracked
                        };

                        if !is_zimo {
                            hand_136.push(win_tile);
                        }

                        let dora_indicators = self.current_doras.clone();
                        // Use calculated Ura Indicators if wall exists
                        let ura_indicators = if self.liqi[seat] {
                            if let Some(ref li) = hule_data.li_doras {
                                li.clone()
                            } else if !self.wall.is_empty() {
                                self._get_ura_indicators()
                            } else {
                                self.kyoku.ura_doras.clone()
                            }
                        } else {
                            vec![]
                        };

                        let actual_result = {
                            let calc = AgariCalculator::new(hand_136.clone(), melds_136.clone());
                            calc.calc(
                                win_tile,
                                dora_indicators.clone(),
                                ura_indicators.clone(),
                                Some(conditions.clone()),
                            )
                        };

                        self.pending_agari.push(AgariContext {
                            seat: seat as u8,
                            tiles: hand_136,
                            melds: melds_136,
                            agari_tile: win_tile,
                            dora_indicators,
                            ura_indicators,
                            conditions,
                            expected_yaku: hule_data.fans.clone(),
                            expected_han: hule_data.count,
                            expected_fu: hule_data.fu,
                            actual: actual_result,
                        });
                    }
                    if !self.pending_agari.is_empty() {
                        return Some(self.pending_agari.remove(0));
                    }
                }
                _ => {}
            }
        }
        None
    }
}

#[pyclass(module = "riichienv._riichienv")]
pub struct AgariContext {
    pub seat: u8,
    pub tiles: Vec<u8>,
    pub melds: Vec<Meld>,
    pub agari_tile: u8,
    pub dora_indicators: Vec<u8>,
    pub ura_indicators: Vec<u8>,
    pub conditions: Conditions,
    pub expected_yaku: Vec<u32>,
    pub expected_han: u32,
    pub expected_fu: u32,
    pub actual: Agari,
}

#[pymethods]
impl AgariContext {
    #[getter]
    pub fn seat(&self) -> u8 {
        self.seat
    }
    #[getter]
    pub fn tiles(&self) -> Vec<u32> {
        self.tiles.iter().map(|&t| t as u32).collect()
    }
    #[getter]
    pub fn melds(&self) -> Vec<Meld> {
        self.melds.clone()
    }
    #[getter]
    pub fn agari_tile(&self) -> u32 {
        self.agari_tile as u32
    }
    #[getter]
    pub fn dora_indicators(&self) -> Vec<u32> {
        self.dora_indicators.iter().map(|&t| t as u32).collect()
    }
    #[getter]
    pub fn ura_indicators(&self) -> Vec<u32> {
        self.ura_indicators.iter().map(|&t| t as u32).collect()
    }
    #[getter]
    pub fn conditions(&self) -> Conditions {
        self.conditions.clone()
    }
    #[getter]
    pub fn expected_yaku(&self) -> Vec<u32> {
        self.expected_yaku.clone()
    }
    #[getter]
    pub fn expected_han(&self) -> u32 {
        self.expected_han
    }
    #[getter]
    pub fn expected_fu(&self) -> u32 {
        self.expected_fu
    }
    #[getter]
    pub fn actual(&self) -> Agari {
        self.actual.clone()
    }

    /// Creates an AgariCalculator initialized with the hand and melds from this context.
    pub fn create_calculator(&self) -> AgariCalculator {
        AgariCalculator::new(self.tiles.clone(), self.melds.clone())
    }

    /// Calculates the agari result using the provided calculator and conditions.
    #[pyo3(signature = (calculator, conditions=None))]
    pub fn calculate(&self, calculator: &AgariCalculator, conditions: Option<Conditions>) -> Agari {
        let cond = conditions.unwrap_or_else(|| self.conditions.clone());
        calculator.calc(
            self.agari_tile,
            self.dora_indicators.clone(),
            self.ura_indicators.clone(),
            Some(cond),
        )
    }
}

pub struct TileConverter {}

impl TileConverter {
    pub fn parse_tile(t: &str) -> (u8, bool) {
        if t.is_empty() {
            return (0, false);
        }
        let (num_str, suit) = t.split_at(1);
        let num: u8 = num_str.parse().unwrap_or(0);
        let is_aka = num == 0;
        let num = if is_aka { 5 } else { num };

        let id_34 = match suit {
            "m" => num - 1,
            "p" => 9 + num - 1,
            "s" => 18 + num - 1,
            "z" => 27 + num - 1,
            _ => 0,
        };

        (id_34, is_aka)
    }

    pub fn parse_tile_34(t: &str) -> (u8, bool) {
        Self::parse_tile(t)
    }

    pub fn parse_tile_136(t: &str) -> u8 {
        let (id_34, is_aka) = Self::parse_tile(t);
        if is_aka {
            match id_34 {
                4 => 16,
                13 => 52,
                22 => 88,
                _ => id_34 * 4,
            }
        } else if id_34 == 4 || id_34 == 13 || id_34 == 22 {
            id_34 * 4 + 1
        } else {
            id_34 * 4
        }
    }

    pub fn to_string(tile: u8) -> String {
        let t34 = tile / 4;
        let is_red = tile == 16 || tile == 52 || tile == 88;
        let suit_idx = t34 / 9;
        let num = t34 % 9 + 1;
        let suit = match suit_idx {
            0 => "m",
            1 => "p",
            2 => "s",
            3 => "z",
            _ => return "?".to_string(),
        };
        if is_red {
            return format!("0{}", suit);
        }
        let res = format!("{}{}", num, suit);
        res
    }

    pub fn match_and_remove_u8(hand: &mut Vec<u8>, target: u8) -> bool {
        if let Some(pos) = hand.iter().position(|x| *x == target) {
            hand.remove(pos);
            return true;
        }
        // Try other 136-ids of the same 34-tile if not found (for robustness)
        let target_34 = target / 4;
        if let Some(pos) = hand.iter().position(|x| *x / 4 == target_34) {
            hand.remove(pos);
            return true;
        }
        false
    }
}
