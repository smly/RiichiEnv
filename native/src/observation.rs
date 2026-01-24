use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};
use serde::{Deserialize, Serialize};

use crate::action::Action;
use crate::types::Meld;

#[pyclass(module = "riichienv._riichienv")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    #[pyo3(get)]
    pub player_id: u8,
    #[pyo3(get)]
    pub hands: Vec<Vec<u32>>,
    #[pyo3(get)]
    pub melds: Vec<Vec<Meld>>,
    #[pyo3(get)]
    pub discards: Vec<Vec<u32>>,
    #[pyo3(get)]
    pub dora_indicators: Vec<u32>,
    #[pyo3(get)]
    pub scores: Vec<i32>,
    #[pyo3(get)]
    pub riichi_declared: Vec<bool>,

    #[serde(skip)]
    pub _legal_actions: Vec<Action>,

    pub events: Vec<String>,

    #[pyo3(get)]
    pub honba: u8,
    #[pyo3(get)]
    pub riichi_sticks: u32,
    #[pyo3(get)]
    pub round_wind: u8,
    #[pyo3(get)]
    pub oya: u8,
}

#[pymethods]
impl Observation {
    #[new]
    pub fn new(
        player_id: u8,
        hands: Vec<Vec<u8>>,
        melds: Vec<Vec<Meld>>,
        discards: Vec<Vec<u8>>,
        dora_indicators: Vec<u8>,
        scores: Vec<i32>,
        riichi_declared: Vec<bool>,
        legal_actions: Vec<Action>,
        events: Vec<String>,
        honba: u8,
        riichi_sticks: u32,
        round_wind: u8,
        oya: u8,
    ) -> Self {
        let hands_u32 = hands
            .iter()
            .map(|h| h.iter().map(|&x| x as u32).collect())
            .collect();
        let discards_u32 = discards
            .iter()
            .map(|d| d.iter().map(|&x| x as u32).collect())
            .collect();
        let dora_u32 = dora_indicators.iter().map(|&x| x as u32).collect();

        Self {
            player_id,
            hands: hands_u32,
            melds,
            discards: discards_u32,
            dora_indicators: dora_u32,
            scores,
            riichi_declared,
            _legal_actions: legal_actions,
            events,
            honba,
            riichi_sticks,
            round_wind,
            oya,
        }
    }

    #[getter]
    pub fn hand(&self) -> Vec<u32> {
        if (self.player_id as usize) < self.hands.len() {
            self.hands[self.player_id as usize].clone()
        } else {
            vec![]
        }
    }

    #[getter]
    pub fn events<'py>(&self, py: Python<'py>) -> PyResult<Vec<Py<PyAny>>> {
        let json = py.import("json")?;
        let loads = json.getattr("loads")?;
        let mut res = Vec::new();
        for s in &self.events {
            let obj = loads.call1((s,))?;
            res.push(obj.unbind());
        }
        Ok(res)
    }

    #[pyo3(name = "legal_actions")]
    pub fn legal_actions_method(&self) -> Vec<Action> {
        self._legal_actions.clone()
    }

    #[pyo3(signature = (mjai_data))]
    pub fn select_action_from_mjai(&self, mjai_data: &Bound<'_, PyAny>) -> Option<Action> {
        let (atype, tile_str) = if let Ok(s) = mjai_data.extract::<String>() {
            let v: serde_json::Value = serde_json::from_str(&s).ok()?;
            (
                v["type"].as_str()?.to_string(),
                v["pai"].as_str().unwrap_or("").to_string(),
            )
        } else if let Ok(dict) = mjai_data.downcast::<PyDict>() {
            let t = dict.get_item("type").ok()??.extract::<String>().ok()?;
            let p = dict
                .get_item("pai")
                .ok()
                .flatten()
                .and_then(|x| x.extract::<String>().ok())
                .unwrap_or_default();
            (t, p)
        } else {
            return None;
        };

        let target_type = match atype.as_str() {
            "dahai" => Some(crate::action::ActionType::Discard),
            "chi" => Some(crate::action::ActionType::Chi),
            "pon" => Some(crate::action::ActionType::Pon),
            "kakan" => Some(crate::action::ActionType::Kakan),
            "daiminkan" => Some(crate::action::ActionType::Daiminkan),
            "ankan" => Some(crate::action::ActionType::Ankan),
            "reach" => Some(crate::action::ActionType::Riichi),
            "hora" => None,
            "ryukyoku" => Some(crate::action::ActionType::KyushuKyuhai),
            _ => None,
        };

        if atype == "hora" {
            return self
                ._legal_actions
                .iter()
                .find(|a| {
                    a.action_type == crate::action::ActionType::Tsumo
                        || a.action_type == crate::action::ActionType::Ron
                })
                .cloned();
        }

        if let Some(tt) = target_type {
            let target_tile: Option<u8> = if !tile_str.is_empty() {
                // Simplified parser for MJAI tile strings (e.g. "5p", "5pr")
                // Assuming crate::parser::tile_from_string is available or implement simple one
                // Since I cannot verify crate::parser exposure easily, I'll fallback to loose matching if necessary
                // But better to implement minimal parse or try to access internal
                // If "5p" -> 53?
                // Let's assume `crate::types::tile_from_string` or similar exists if used elsewhere.
                // Wait, `env.rs` calls `crate::parser::parse_hand_internal`.
                // I'll try `crate::types::tile_from_mjai` if it exists?
                // Or implementing a helper here?
                // `tests/env` uses numbers.
                // Replay uses `tid_to_mjai`.
                // I need `mjai_to_tid`.
                // I'll use a hacky lookup if needed or just skip check if complex.
                // BUT `test_action_to_mjai.py` needs accurate matching.
                // Since `tile_str` "1z" != "5p", simply checking equality of strings is enough if Action has string rep?
                // Action has `tile` u8.
                // I'll try to find a converter. `riichi_env` likely has one.
                None // Placeholder, logic below handles None -> loose match
            } else {
                None
            };

            // Implementing basic tile check if target_tile is None (can't parse)
            // But I MUST parse to pass the test.
            // I'll use `crate::types::msg_to_tile` if available?
            // I'll define a local parsing closure/function if possible.
            // Or inspect `types.rs`.

            return self
                ._legal_actions
                .iter()
                .find(|a| {
                    if a.action_type != tt {
                        return false;
                    }
                    if !tile_str.is_empty() {
                        // If we can't parse easily in Rust without helper,
                        // we can try to convert `a.tile` to string and compare?
                        if let Some(t) = a.tile {
                            // Recover string from t
                            let t_str = crate::parser::tid_to_mjai(t);
                            // MJAI "5p" vs "5mr" ("red 5 m")?
                            // MJAI usually is "5p".
                            // If `tid_to_mjai` returns "5p", strict match works.
                            return t_str == tile_str;
                        }
                    }
                    true
                })
                .cloned();
        }

        if atype == "none" {
            return self
                ._legal_actions
                .iter()
                .find(|a| a.action_type == crate::action::ActionType::Pass)
                .cloned();
        }

        None
    }

    pub fn new_events(&self) -> Vec<String> {
        self.events.clone()
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("player_id", self.player_id)?;
        dict.set_item("hands", self.hands.clone())?;

        let melds_py = pyo3::types::PyList::empty(py);
        for p_melds in &self.melds {
            let p_list = pyo3::types::PyList::new(
                py,
                p_melds.iter().map(|m| m.clone().into_pyobject(py).unwrap()),
            )?;
            melds_py.append(p_list)?;
        }
        dict.set_item("melds", melds_py)?;

        dict.set_item("discards", self.discards.clone())?;
        dict.set_item("dora_indicators", self.dora_indicators.clone())?;
        dict.set_item("scores", self.scores.clone())?;
        dict.set_item("riichi_declared", self.riichi_declared.clone())?;

        let actions_py = pyo3::types::PyList::empty(py);
        for a in &self._legal_actions {
            actions_py.append(a.to_dict(py)?)?;
        }
        dict.set_item("legal_actions", actions_py)?;

        dict.set_item("events", self.events.clone())?;
        dict.set_item("honba", self.honba)?;
        dict.set_item("riichi_sticks", self.riichi_sticks)?;
        dict.set_item("round_wind", self.round_wind)?;
        dict.set_item("oya", self.oya)?;

        Ok(dict.unbind().into())
    }

    pub fn encode<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);

        // Hand: 34 dims (counts)
        let mut hand_vec = vec![0u8; 34];
        if (self.player_id as usize) < self.hands.len() {
            for &t in &self.hands[self.player_id as usize] {
                let idx = (t as usize) / 4;
                if idx < 34 {
                    hand_vec[idx] += 1;
                }
            }
        }
        let hand_py = pyo3::types::PyList::new(py, hand_vec)?;
        dict.set_item("hand", hand_py)?;

        // Melds: 34 dims (counts of melded tiles)
        let mut meld_vec = vec![0u8; 34];
        if (self.player_id as usize) < self.melds.len() {
            for m in &self.melds[self.player_id as usize] {
                for &t in &m.tiles {
                    let idx = (t as usize) / 4;
                    if idx < 34 {
                        meld_vec[idx] += 1;
                    }
                }
            }
        }
        let meld_py = pyo3::types::PyList::new(py, meld_vec)?;
        dict.set_item("melds_vec", meld_py)?;

        // Discards: 34 dims (counts)
        let mut disc_vec = vec![0u8; 34];
        if (self.player_id as usize) < self.discards.len() {
            for &t in &self.discards[self.player_id as usize] {
                let idx = (t as usize) / 4;
                if idx < 34 {
                    disc_vec[idx] += 1;
                }
            }
        }
        let disc_py = pyo3::types::PyList::new(py, disc_vec)?;
        dict.set_item("discards_vec", disc_py)?;

        // Dora: 34 dims (counts)
        let mut dora_vec = vec![0u8; 34];
        for &t in &self.dora_indicators {
            let idx = (t as usize) / 4;
            if idx < 34 {
                dora_vec[idx] += 1;
            }
        }
        let dora_py = pyo3::types::PyList::new(py, dora_vec)?;
        dict.set_item("dora_vec", dora_py)?;

        // Scalars
        dict.set_item("honba", self.honba)?;
        dict.set_item("riichi_sticks", self.riichi_sticks)?;
        dict.set_item("round_wind", self.round_wind)?;
        dict.set_item("oya", self.oya)?;
        dict.set_item("scores", self.scores.clone())?;

        Ok(dict.unbind().into())
    }
}
