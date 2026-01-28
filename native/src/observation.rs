use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};
use serde::{Deserialize, Serialize};

use crate::action::Action;
use crate::types::Meld;
use ndarray::prelude::*;

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
    #[pyo3(get)]
    pub kyoku_index: u8,
    #[pyo3(get)]
    pub waits: Vec<u8>,
    #[pyo3(get)]
    pub is_tenpai: bool,
}

#[pymethods]
impl Observation {
    #[new]
    #[allow(clippy::too_many_arguments)]
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
        kyoku_index: u8,
        waits: Vec<u8>,
        is_tenpai: bool,
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
            kyoku_index,
            waits,
            is_tenpai,
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

    #[pyo3(name = "mask")]
    pub fn mask_method<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut mask = [0u8; 82];
        for action in &self._legal_actions {
            if let Ok(idx) = action.encode() {
                if (idx as usize) < mask.len() {
                    mask[idx as usize] = 1;
                }
            }
        }
        Ok(pyo3::types::PyBytes::new(py, &mask))
    }

    #[pyo3(signature = (action_id))]
    pub fn find_action(&self, action_id: usize) -> Option<Action> {
        self._legal_actions
            .iter()
            .find(|a| {
                if let Ok(idx) = a.encode() {
                    (idx as usize) == action_id
                } else {
                    false
                }
            })
            .cloned()
    }

    #[pyo3(signature = (mjai_data))]
    pub fn select_action_from_mjai(&self, mjai_data: &Bound<'_, PyAny>) -> Option<Action> {
        let (atype, tile_str) = if let Ok(s) = mjai_data.extract::<String>() {
            let v: serde_json::Value = serde_json::from_str(&s).ok()?;
            (
                v["type"].as_str()?.to_string(),
                v["pai"].as_str().unwrap_or("").to_string(),
            )
        } else if let Ok(dict) = mjai_data.cast::<PyDict>() {
            let type_str: String = dict
                .get_item("type")
                .ok()
                .flatten()
                .and_then(|x| x.extract::<String>().ok())
                .unwrap_or_default();
            let _args_list: Vec<String> = dict
                .get_item("args")
                .ok()
                .flatten()
                .and_then(|x| x.extract::<Vec<String>>().ok())
                .unwrap_or_default();
            let _who: i8 = dict
                .get_item("who")
                .ok()
                .flatten()
                .and_then(|x| x.extract::<i8>().ok())
                .unwrap_or(-1);
            // For now, tile is string "3m" etc.
            let tile_str: String = dict
                .get_item("pai")
                .ok()
                .flatten()
                .or_else(|| dict.get_item("tile").ok().flatten())
                .and_then(|x| x.extract::<String>().ok())
                .unwrap_or_default();
            (type_str, tile_str)
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
            return self
                ._legal_actions
                .iter()
                .find(|a| {
                    if a.action_type != tt {
                        return false;
                    }
                    if !tile_str.is_empty() {
                        if let Some(t) = a.tile {
                            let t_str = crate::parser::tid_to_mjai(t);
                            // MJAI sometimes uses "5p" for red 5 distinctively or "5pr"
                            // If mismatched, try to match the base.
                            if t_str == tile_str {
                                return true;
                            }
                            // Allow "5p" to match "5pr" if strict match failed?
                            // Or better: if mjai says "5p", it means non-red 5.
                            // If mjai says "5pr" (or however it denotes red), it means red 5.
                            // tid_to_mjai outputs "5mr" for red.
                            // If input is "5m", it shouldn't match "5mr".
                            return false;
                        } else {
                            // Action has no tile (e.g. Riichi), but tile_str provided?
                            // Should ideally not happen for Discard/Chi/Pon.
                            // But if it does, mismatch.
                            return false;
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

    pub fn encode<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        // Total Channels:
        // 0-3: Hand (1,2,3,4)
        // 4: Red (Hand)
        // 5-8: Melds (Self)
        // 9: Dora Indicators
        // 10-13: Discards (Self, Last 4) (History)
        // 14-25: Discards (Opponents, Last 4 each) (History)
        // 26: Riichi (Self)
        // 27-29: Riichi (Opponents)
        // 30: Round Wind
        // 31: Self Wind
        // 32: Honba
        // 33: Riichi Sticks
        // 34-37: Scores (P0-P3, normalized)
        // 38: Waits (1 channel)
        // 39: Is Tenpai (1 channel)
        // 40-43: Rank (One-hot)
        // 44: Kyoku Index (Normalized)
        // 45: Tiles Seen (Normalized)

        let num_channels = 46;
        let mut arr = Array2::<f32>::zeros((num_channels, 34));

        // 1. Hand (0-3), 2. Red (4)
        if (self.player_id as usize) < self.hands.len() {
            let mut counts = [0u8; 34];
            for &t in &self.hands[self.player_id as usize] {
                let idx = (t as usize) / 4;
                if idx < 34 {
                    counts[idx] += 1;
                    if t == 16 || t == 52 || t == 88 {
                        arr[[4, idx]] = 1.0;
                    }
                }
            }
            for i in 0..34 {
                let c = counts[i];
                if c >= 1 {
                    arr[[0, i]] = 1.0;
                }
                if c >= 2 {
                    arr[[1, i]] = 1.0;
                }
                if c >= 3 {
                    arr[[2, i]] = 1.0;
                }
                if c >= 4 {
                    arr[[3, i]] = 1.0;
                }
            }
        }

        // 3. Melds (Self) (5-8)
        if (self.player_id as usize) < self.melds.len() {
            for (m_idx, meld) in self.melds[self.player_id as usize].iter().enumerate() {
                if m_idx >= 4 {
                    break;
                }
                for &t in &meld.tiles {
                    let idx = (t as usize) / 4;
                    if idx < 34 {
                        arr[[5 + m_idx, idx]] = 1.0;
                    }
                }
            }
        }

        // 4. Dora Indicators (9)
        for &t in &self.dora_indicators {
            let idx = (t as usize) / 4;
            if idx < 34 {
                arr[[9, idx]] = 1.0;
            }
        }

        // 5. Discards (Self) (10-13)
        if (self.player_id as usize) < self.discards.len() {
            let discs = &self.discards[self.player_id as usize];
            for (i, &t) in discs.iter().rev().take(4).enumerate() {
                let idx = (t as usize) / 4;
                if idx < 34 {
                    arr[[10 + i, idx]] = 1.0;
                }
            }
        }

        // 6. Discards (Opponents) (14-25)
        for i in 1..4 {
            let opp_id = (self.player_id + i) % 4;
            if (opp_id as usize) < self.discards.len() {
                let discs = &self.discards[opp_id as usize];
                for (j, &t) in discs.iter().rev().take(4).enumerate() {
                    let idx = (t as usize) / 4;
                    if idx < 34 {
                        let ch_base = 14 + (i as usize - 1) * 4;
                        arr[[ch_base + j, idx]] = 1.0;
                    }
                }
            }
        }

        // 7. Riichi (26-29)
        if (self.player_id as usize) < self.riichi_declared.len()
            && self.riichi_declared[self.player_id as usize]
        {
            for i in 0..34 {
                arr[[26, i]] = 1.0;
            }
        }
        for i in 1..4 {
            let opp_id = (self.player_id + i) % 4;
            if (opp_id as usize) < self.riichi_declared.len()
                && self.riichi_declared[opp_id as usize]
            {
                for k in 0..34 {
                    arr[[27 + (i as usize - 1), k]] = 1.0;
                }
            }
        }

        // 8. Winds (30-31)
        let rw = self.round_wind as usize;
        if 27 + rw < 34 {
            arr[[30, 27 + rw]] = 1.0;
        }
        let seat = (self.player_id + 4 - self.oya) % 4;
        if 27 + (seat as usize) < 34 {
            arr[[31, 27 + (seat as usize)]] = 1.0;
        }

        // 9. Honba/Sticks (32-33)
        let honba_norm = (self.honba as f32) / 10.0;
        let sticks_norm = (self.riichi_sticks as f32) / 5.0;
        for i in 0..34 {
            arr[[32, i]] = honba_norm;
            arr[[33, i]] = sticks_norm;
        }

        // 10. Scores (34-37)
        for i in 0..4 {
            if i < self.scores.len() {
                let score_norm = (self.scores[i] as f32) / 50000.0;
                for k in 0..34 {
                    arr[[34 + i, k]] = score_norm;
                }
            }
        }

        // 11. Waits (38)
        for &t in &self.waits {
            if (t as usize) < 34 {
                arr[[38, t as usize]] = 1.0;
            }
        }

        // 12. Is Tenpai (39)
        let tenpai_val = if self.is_tenpai { 1.0 } else { 0.0 };
        for i in 0..34 {
            arr[[39, i]] = tenpai_val;
        }

        // 13. Rank (40-43)
        // Scores are sorted to find rank?
        // Rank 0 = Highest score.
        let my_score = self
            .scores
            .get(self.player_id as usize)
            .copied()
            .unwrap_or(0);
        let mut rank = 0;
        for &s in &self.scores {
            if s > my_score {
                rank += 1;
            }
        }
        // If tied, logic? Simple > check means same score = same rank (or lower rank if strict).
        // Let's assume strict > means we are 0 if we are max.
        // If tied, we might share rank.
        // Just broadcast 1 to channel (40 + rank).
        if rank < 4 {
            for i in 0..34 {
                arr[[40 + rank, i]] = 1.0;
            }
        }

        // 14. Kyoku (44)
        let k_norm = (self.kyoku_index as f32) / 8.0; // Approx max 8 (East 1-4, South 1-4).
        for i in 0..34 {
            arr[[44, i]] = k_norm;
        }

        // 15. Tiles Seen (45)
        let mut seen = [0u8; 34];
        // Hand
        if (self.player_id as usize) < self.hands.len() {
            for &t in &self.hands[self.player_id as usize] {
                seen[(t as usize) / 4] += 1;
            }
        }
        // Melds (All)
        for mlist in &self.melds {
            for m in mlist {
                for &t in &m.tiles {
                    seen[(t as usize) / 4] += 1;
                }
            }
        }
        // Discards (All)
        for dlist in &self.discards {
            for &t in dlist {
                seen[(t as usize) / 4] += 1;
            }
        }
        // Dora Indicators
        for &t in &self.dora_indicators {
            seen[(t as usize) / 4] += 1;
        }

        for i in 0..34 {
            // 4 visible max (usually). Red 5 counts as 5.
            let norm_seen = (seen[i] as f32) / 4.0;
            arr[[45, i]] = norm_seen;
        }

        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }
}
