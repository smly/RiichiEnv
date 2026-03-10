use ndarray::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

use crate::action::{Action, Action3P, ActionEncoder, ActionType};
use crate::shanten;
use crate::types::{Meld, MeldType};
use crate::yaku_checker;

use super::Observation3P;
use super::helpers::{TILE_DIM_3P, get_next_tile_sanma, tile34_to_compact};

const NP: usize = 3;
const TOTAL_TILES: u32 = 108;

#[pymethods]
impl Observation3P {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (player_id, hands, melds, discards, dora_indicators, scores, riichi_declared, legal_actions, events, honba, riichi_sticks, round_wind, oya, kyoku_index, waits, is_tenpai, riichi_sutehais, last_tedashis, last_discard))]
    pub fn py_new(
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
        riichi_sutehais: Vec<Option<u8>>,
        last_tedashis: Vec<Option<u8>>,
        last_discard: Option<u32>,
    ) -> Self {
        let hands: [Vec<u8>; 3] = hands.try_into().expect("expected 3 hands");
        let melds: [Vec<Meld>; 3] = melds.try_into().expect("expected 3 melds");
        let discards: [Vec<u8>; 3] = discards.try_into().expect("expected 3 discards");
        let scores: [i32; 3] = scores.try_into().expect("expected 3 scores");
        let riichi_declared: [bool; 3] = riichi_declared
            .try_into()
            .expect("expected 3 riichi_declared");
        let riichi_sutehais: [Option<u8>; 3] = riichi_sutehais
            .try_into()
            .expect("expected 3 riichi_sutehais");
        let last_tedashis: [Option<u8>; 3] =
            last_tedashis.try_into().expect("expected 3 last_tedashis");
        Self::new(
            player_id,
            hands,
            melds,
            discards,
            dora_indicators,
            scores,
            riichi_declared,
            legal_actions,
            events,
            honba,
            riichi_sticks,
            round_wind,
            oya,
            kyoku_index,
            waits,
            is_tenpai,
            riichi_sutehais,
            last_tedashis,
            last_discard,
        )
    }

    #[getter]
    pub fn hand(&self) -> Vec<u32> {
        self.hands[self.player_id as usize].clone()
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
    pub fn legal_actions_method_py(&self) -> Vec<Action3P> {
        self.legal_actions_method()
    }

    #[pyo3(name = "mask")]
    pub fn mask_method<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let size = ActionEncoder::ThreePlayer.action_space_size();
        let mut mask = vec![0u8; size];
        for action in &self._legal_actions {
            if let Ok(idx) = action.encode()
                && (idx as usize) < mask.len()
            {
                mask[idx as usize] = 1;
            }
        }
        Ok(pyo3::types::PyBytes::new(py, &mask))
    }

    #[getter]
    pub fn action_space_size(&self) -> usize {
        ActionEncoder::ThreePlayer.action_space_size()
    }

    #[pyo3(name = "find_action", signature = (action_id))]
    pub fn find_action_py(&self, action_id: usize) -> Option<Action3P> {
        self.find_action(action_id)
    }

    #[pyo3(signature = (mjai_data))]
    pub fn select_action_from_mjai(&self, mjai_data: &Bound<'_, PyAny>) -> Option<Action3P> {
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
            "kita" => Some(crate::action::ActionType::Kita),
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
                            if t_str == tile_str {
                                return true;
                            }
                            return false;
                        } else {
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

    #[pyo3(name = "new_events")]
    pub fn new_events_py(&self) -> Vec<String> {
        self.new_events()
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
        dict.set_item("scores", self.scores)?;
        dict.set_item("riichi_declared", self.riichi_declared)?;

        let actions_py = pyo3::types::PyList::empty(py);
        for a in &self._legal_actions {
            actions_py.append(a.to_dict_py(py)?)?;
        }
        dict.set_item("legal_actions", actions_py)?;

        dict.set_item("events", self.events.clone())?;
        dict.set_item("honba", self.honba)?;
        dict.set_item("riichi_sticks", self.riichi_sticks)?;
        dict.set_item("round_wind", self.round_wind)?;
        dict.set_item("oya", self.oya)?;

        Ok(dict.unbind().into())
    }

    /// Serialize this Observation3P to a base64-encoded JSON string.
    #[pyo3(name = "serialize_to_base64")]
    pub fn serialize_to_base64_py(&self) -> PyResult<String> {
        self.serialize_to_base64().map_err(Into::into)
    }

    /// Deserialize an Observation3P from a base64-encoded JSON string.
    #[staticmethod]
    #[pyo3(name = "deserialize_from_base64")]
    pub fn deserialize_from_base64_py(s: &str) -> PyResult<Self> {
        Self::deserialize_from_base64(s).map_err(Into::into)
    }

    /// Encode discard history with exponential decay weighting.
    #[pyo3(name = "encode_discard_history_decay", signature = (decay_rate=None))]
    pub fn encode_discard_history_decay<'py>(
        &self,
        py: Python<'py>,
        decay_rate: Option<f32>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let decay_rate = decay_rate.unwrap_or(0.2);
        let mut arr = Array2::<f32>::zeros((NP, TILE_DIM_3P));

        for (ch_idx, &abs_idx) in self.rel_order().iter().enumerate() {
            let discs = &self.discards[abs_idx];
            let max_len = discs.len();

            if max_len == 0 {
                continue;
            }

            for (turn, &tile) in discs.iter().enumerate() {
                let tile34 = (tile as usize) / 4;
                if let Some(idx) = tile34_to_compact(tile34) {
                    let age = (max_len - 1 - turn) as f32;
                    let weight = (-decay_rate * age).exp();
                    arr[[ch_idx, idx]] += weight;
                }
            }
        }

        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encode furiten-aware ron possibility based on tsumogiri patterns.
    #[pyo3(name = "encode_furiten_ron_possibility")]
    pub fn encode_furiten_ron_possibility<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        const NUM_YAKU: usize = 21;
        let mut arr = Array2::<f32>::ones((NP, NUM_YAKU));

        for player_idx in 0..NP {
            let flags = &self.tsumogiri_flags[player_idx];
            if flags.is_empty() {
                continue;
            }

            let mut consecutive_tsumogiri = 0;
            for &flag in flags.iter().rev() {
                if flag {
                    consecutive_tsumogiri += 1;
                } else {
                    break;
                }
            }

            if consecutive_tsumogiri >= 3 {
                for yaku_idx in 0..NUM_YAKU {
                    arr[[player_idx, yaku_idx]] = 0.0;
                }
            }
        }

        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encode yaku (winning hand patterns) possibility for each player.
    #[pyo3(name = "encode_yaku_possibility")]
    pub fn encode_yaku_possibility<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        const NUM_YAKU: usize = 21;
        let mut arr = Array3::<f32>::ones((NP, NUM_YAKU, 2));

        let all_discards: [Vec<u32>; 3] = self.discards.clone();

        for player_idx in 0..NP {
            let melds = &self.melds[player_idx];
            let discards = &all_discards[player_idx];

            // Yaku 0: Tanyao
            let tanyao = yaku_checker::check_tanyao(melds);
            arr[[player_idx, 0, 0]] = tanyao.to_f32();
            arr[[player_idx, 0, 1]] = tanyao.to_f32();

            // Yaku 1-3: Yakuhai (dragons: White=31, Green=32, Red=33)
            for (yaku_idx, &tile_type) in [31, 32, 33].iter().enumerate() {
                let yakuhai =
                    yaku_checker::check_yakuhai(tile_type, melds, discards, &self.dora_indicators);
                arr[[player_idx, 1 + yaku_idx, 0]] = yakuhai.to_f32();
                arr[[player_idx, 1 + yaku_idx, 1]] = yakuhai.to_f32();
            }

            // Yaku 4: Yakuhai (round wind)
            let round_wind_type = 27 + self.round_wind as usize;
            let yakuhai_round = yaku_checker::check_yakuhai(
                round_wind_type,
                melds,
                discards,
                &self.dora_indicators,
            );
            arr[[player_idx, 4, 0]] = yakuhai_round.to_f32();
            arr[[player_idx, 4, 1]] = yakuhai_round.to_f32();

            // Yaku 5: Yakuhai (seat wind)
            let seat = (player_idx as u8 + NP as u8 - self.oya) % NP as u8;
            let seat_wind_type = 27 + seat as usize;
            let yakuhai_seat =
                yaku_checker::check_yakuhai(seat_wind_type, melds, discards, &self.dora_indicators);
            arr[[player_idx, 5, 0]] = yakuhai_seat.to_f32();
            arr[[player_idx, 5, 1]] = yakuhai_seat.to_f32();

            // Yaku 6-7: Honitsu, Chinitsu
            let (honitsu, chinitsu) = yaku_checker::check_flush(melds);
            arr[[player_idx, 6, 0]] = honitsu.to_f32();
            arr[[player_idx, 6, 1]] = honitsu.to_f32();
            arr[[player_idx, 7, 0]] = chinitsu.to_f32();
            arr[[player_idx, 7, 1]] = chinitsu.to_f32();

            // Yaku 8: Toitoi
            let toitoi = yaku_checker::check_toitoi(melds);
            arr[[player_idx, 8, 0]] = toitoi.to_f32();
            arr[[player_idx, 8, 1]] = toitoi.to_f32();

            // Yaku 9: Chiitoitsu
            let chiitoitsu = yaku_checker::check_chiitoitsu(melds);
            arr[[player_idx, 9, 0]] = chiitoitsu.to_f32();
            arr[[player_idx, 9, 1]] = chiitoitsu.to_f32();

            // Yaku 10: Shousangen
            let shousangen = yaku_checker::check_shousangen(melds, discards, &self.dora_indicators);
            arr[[player_idx, 10, 0]] = shousangen.to_f32();
            arr[[player_idx, 10, 1]] = shousangen.to_f32();

            // Yaku 11: Daisangen
            let daisangen = yaku_checker::check_daisangen(melds, discards, &self.dora_indicators);
            arr[[player_idx, 11, 0]] = daisangen.to_f32();
            arr[[player_idx, 11, 1]] = daisangen.to_f32();

            // Yaku 12: Tsuuiisou
            let tsuuiisou = yaku_checker::check_tsuuiisou(melds);
            arr[[player_idx, 12, 0]] = tsuuiisou.to_f32();
            arr[[player_idx, 12, 1]] = tsuuiisou.to_f32();

            // Yaku 13: Chinroutou
            let chinroutou = yaku_checker::check_chinroutou(melds);
            arr[[player_idx, 13, 0]] = chinroutou.to_f32();
            arr[[player_idx, 13, 1]] = chinroutou.to_f32();

            // Yaku 14: Honroutou
            let honroutou = yaku_checker::check_honroutou(melds);
            arr[[player_idx, 14, 0]] = honroutou.to_f32();
            arr[[player_idx, 14, 1]] = honroutou.to_f32();

            // Yaku 15: Kokushi
            let kokushi = yaku_checker::check_kokushi(melds, discards, &self.dora_indicators);
            arr[[player_idx, 15, 0]] = kokushi.to_f32();
            arr[[player_idx, 15, 1]] = kokushi.to_f32();

            // Yaku 16: Chanta
            let chanta = yaku_checker::check_chanta(melds);
            arr[[player_idx, 16, 0]] = chanta.to_f32();
            arr[[player_idx, 16, 1]] = chanta.to_f32();

            // Yaku 17: Junchan
            let junchan = yaku_checker::check_junchan(melds);
            arr[[player_idx, 17, 0]] = junchan.to_f32();
            arr[[player_idx, 17, 1]] = junchan.to_f32();

            // Yaku 18: Sanshoku doujun
            let sanshoku = yaku_checker::check_sanshoku_doujun(melds);
            arr[[player_idx, 18, 0]] = sanshoku.to_f32();
            arr[[player_idx, 18, 1]] = sanshoku.to_f32();

            // Yaku 19: Iipeikou
            let iipeikou = yaku_checker::check_iipeikou(melds);
            arr[[player_idx, 19, 0]] = iipeikou.to_f32();
            arr[[player_idx, 19, 1]] = iipeikou.to_f32();

            // Yaku 20: Ittsu
            let ittsu = yaku_checker::check_ittsu(melds);
            arr[[player_idx, 20, 0]] = ittsu.to_f32();
            arr[[player_idx, 20, 1]] = ittsu.to_f32();
        }

        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    pub fn encode<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let num_channels = 74;
        let mut arr = Array2::<f32>::zeros((num_channels, TILE_DIM_3P));

        // 1. Hand (0-3), 2. Red (4)
        let mut counts = [0u8; TILE_DIM_3P];
        for &t in &self.hands[self.player_id as usize] {
            let idx34 = (t as usize) / 4;
            if let Some(idx) = tile34_to_compact(idx34) {
                counts[idx] += 1;
                if t == 16 || t == 52 || t == 88 {
                    arr[[4, idx]] = 1.0;
                }
            }
        }
        for i in 0..TILE_DIM_3P {
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

        // 3. Melds (Self) (5-8)
        for (m_idx, meld) in self.melds[self.player_id as usize].iter().enumerate() {
            if m_idx >= 4 {
                break;
            }
            for &t in &meld.tiles {
                let idx34 = (t as usize) / 4;
                if let Some(idx) = tile34_to_compact(idx34) {
                    arr[[5 + m_idx, idx]] = 1.0;
                }
            }
        }

        // 4. Dora Indicators (9)
        for &t in &self.dora_indicators {
            let idx34 = (t as usize) / 4;
            if let Some(idx) = tile34_to_compact(idx34) {
                arr[[9, idx]] = 1.0;
            }
        }

        // 5. Discards (Self) (10-13)
        let discs = &self.discards[self.player_id as usize];
        for (i, &t) in discs.iter().rev().take(4).enumerate() {
            let idx34 = (t as usize) / 4;
            if let Some(idx) = tile34_to_compact(idx34) {
                arr[[10 + i, idx]] = 1.0;
            }
        }

        // 6. Discards (Opponents) (14-21 for 2 opponents)
        for i in 1..NP {
            let opp_id = (self.player_id as usize + i) % NP;
            let discs = &self.discards[opp_id];
            for (j, &t) in discs.iter().rev().take(4).enumerate() {
                let idx34 = (t as usize) / 4;
                if let Some(idx) = tile34_to_compact(idx34) {
                    let ch_base = 14 + (i - 1) * 4;
                    arr[[ch_base + j, idx]] = 1.0;
                }
            }
        }

        // 7. Discard Counts (26-28 for 3 players, relative order)
        for (ch_idx, &abs_idx) in self.rel_order().iter().enumerate() {
            let count_norm = (self.discards[abs_idx].len() as f32) / 24.0;
            for k in 0..TILE_DIM_3P {
                arr[[26 + ch_idx, k]] = count_norm;
            }
        }

        // 8. Tiles Left in Wall (30) - 108 tiles for sanma
        let mut tiles_used = 0;
        for discs in &self.discards {
            tiles_used += discs.len();
        }
        for melds_list in &self.melds {
            for meld in melds_list {
                tiles_used += meld.tiles.len();
            }
        }
        tiles_used += self.hands[self.player_id as usize].len();
        tiles_used += self.dora_indicators.len();
        let tiles_left = (TOTAL_TILES as i32 - tiles_used as i32).max(0) as f32;
        let tiles_left_norm = tiles_left / 70.0;
        for k in 0..TILE_DIM_3P {
            arr[[30, k]] = tiles_left_norm;
        }

        // 9. Riichi (31-33: self + 2 opponents)
        if self.riichi_declared[self.player_id as usize] {
            for i in 0..TILE_DIM_3P {
                arr[[31, i]] = 1.0;
            }
        }
        for i in 1..NP {
            let opp_id = (self.player_id as usize + i) % NP;
            if self.riichi_declared[opp_id] {
                for k in 0..TILE_DIM_3P {
                    arr[[32 + (i - 1), k]] = 1.0;
                }
            }
        }

        // 10. Winds (35-36)
        // tile34=27-30 (winds) → compact=20-23
        let rw = self.round_wind as usize;
        if let Some(compact_wind) = tile34_to_compact(27 + rw) {
            arr[[35, compact_wind]] = 1.0;
        }
        let seat = (self.player_id + NP as u8 - self.oya) % NP as u8;
        if let Some(compact_wind) = tile34_to_compact(27 + (seat as usize)) {
            arr[[36, compact_wind]] = 1.0;
        }

        // 11. Honba/Sticks (37-38)
        let honba_norm = (self.honba as f32) / 10.0;
        let sticks_norm = (self.riichi_sticks as f32) / 5.0;
        for i in 0..TILE_DIM_3P {
            arr[[37, i]] = honba_norm;
            arr[[38, i]] = sticks_norm;
        }

        // 12. Scores (39-41) normalized 0-100000, relative order
        for (ch_idx, &abs_idx) in self.rel_order().iter().enumerate() {
            let score_norm = (self.scores[abs_idx].clamp(0, 100000) as f32) / 100000.0;
            for k in 0..TILE_DIM_3P {
                arr[[39 + ch_idx, k]] = score_norm;
            }
        }

        // 13. Scores (43-45) normalized 0-30000, relative order
        for (ch_idx, &abs_idx) in self.rel_order().iter().enumerate() {
            let score_norm = (self.scores[abs_idx].clamp(0, 30000) as f32) / 30000.0;
            for k in 0..TILE_DIM_3P {
                arr[[43 + ch_idx, k]] = score_norm;
            }
        }

        // 14. Waits (47)
        for &t in &self.waits {
            if let Some(idx) = tile34_to_compact(t as usize) {
                arr[[47, idx]] = 1.0;
            }
        }

        // 15. Is Tenpai (48)
        let tenpai_val = if self.is_tenpai { 1.0 } else { 0.0 };
        for i in 0..TILE_DIM_3P {
            arr[[48, i]] = tenpai_val;
        }

        // 16. Rank (49-51 for 3 players)
        let my_score = self.scores[self.player_id as usize];
        let mut rank = 0;
        for &s in &self.scores {
            if s > my_score {
                rank += 1;
            }
        }
        if rank < NP {
            for i in 0..TILE_DIM_3P {
                arr[[49 + rank, i]] = 1.0;
            }
        }

        // 17. Kyoku (53)
        let k_norm = (self.kyoku_index as f32) / 8.0;
        for i in 0..TILE_DIM_3P {
            arr[[53, i]] = k_norm;
        }

        // 18. Round Progress (54)
        let round_progress = (self.round_wind as f32) * 4.0 + (self.kyoku_index as f32);
        let round_progress_norm = round_progress / 7.0;
        for i in 0..TILE_DIM_3P {
            arr[[54, i]] = round_progress_norm;
        }

        // 19. Dora Count (55-57 for 3 players)
        let mut dora_counts = [0u8; NP];
        for (player_idx, dora_count) in dora_counts.iter_mut().enumerate() {
            for meld in &self.melds[player_idx] {
                for &tile in &meld.tiles {
                    for &dora_ind in &self.dora_indicators {
                        let dora_tile = get_next_tile_sanma(dora_ind);
                        if (tile / 4) == (dora_tile / 4) {
                            *dora_count += 1;
                        }
                    }
                }
            }
            for &tile in &self.discards[player_idx] {
                for &dora_ind in &self.dora_indicators {
                    let dora_tile = get_next_tile_sanma(dora_ind);
                    if ((tile / 4) as u8) == (dora_tile / 4) {
                        *dora_count += 1;
                    }
                }
            }
        }
        for &tile in &self.hands[self.player_id as usize] {
            for &dora_ind in &self.dora_indicators {
                let dora_tile = get_next_tile_sanma(dora_ind);
                if ((tile / 4) as u8) == (dora_tile / 4) {
                    dora_counts[self.player_id as usize] += 1;
                }
            }
        }
        for (ch_idx, &abs_idx) in self.rel_order().iter().enumerate() {
            let dora_norm = (dora_counts[abs_idx] as f32) / 12.0;
            for k in 0..TILE_DIM_3P {
                arr[[55 + ch_idx, k]] = dora_norm;
            }
        }

        // 20. Melds Count (59-61 for 3 players, relative order)
        for (ch_idx, &abs_idx) in self.rel_order().iter().enumerate() {
            let meld_count_norm = (self.melds[abs_idx].len() as f32) / 4.0;
            for k in 0..TILE_DIM_3P {
                arr[[59 + ch_idx, k]] = meld_count_norm;
            }
        }

        // 21. Tiles Seen (63)
        let mut seen = [0u8; TILE_DIM_3P];
        for &t in &self.hands[self.player_id as usize] {
            if let Some(idx) = tile34_to_compact((t as usize) / 4) {
                seen[idx] += 1;
            }
        }
        for mlist in &self.melds {
            for m in mlist {
                for &t in &m.tiles {
                    if let Some(idx) = tile34_to_compact((t as usize) / 4) {
                        seen[idx] += 1;
                    }
                }
            }
        }
        for dlist in &self.discards {
            for &t in dlist {
                if let Some(idx) = tile34_to_compact((t as usize) / 4) {
                    seen[idx] += 1;
                }
            }
        }
        for &t in &self.dora_indicators {
            if let Some(idx) = tile34_to_compact((t as usize) / 4) {
                seen[idx] += 1;
            }
        }
        for i in 0..TILE_DIM_3P {
            let norm_seen = (seen[i] as f32) / 4.0;
            arr[[63, i]] = norm_seen;
        }

        // 22-24. Extended Discard History (64-69)
        let discs = &self.discards[self.player_id as usize];
        for (i, &t) in discs.iter().rev().skip(4).take(4).enumerate() {
            let idx34 = (t as usize) / 4;
            if let Some(idx) = tile34_to_compact(idx34) {
                arr[[64 + i, idx]] = 1.0;
            }
        }

        let opp1_id = (self.player_id as usize + 1) % NP;
        let discs = &self.discards[opp1_id];
        for (i, &t) in discs.iter().rev().skip(4).take(2).enumerate() {
            let idx34 = (t as usize) / 4;
            if let Some(idx) = tile34_to_compact(idx34) {
                arr[[68 + i, idx]] = 1.0;
            }
        }

        // 25. Tsumogiri flags (70-72 for 3 players, relative order)
        for (ch_idx, &abs_idx) in self.rel_order().iter().enumerate() {
            if !self.tsumogiri_flags[abs_idx].is_empty() {
                let last_tsumogiri = *self.tsumogiri_flags[abs_idx].last().unwrap_or(&false);
                let val = if last_tsumogiri { 1.0 } else { 0.0 };
                for k in 0..TILE_DIM_3P {
                    arr[[70 + ch_idx, k]] = val;
                }
            }
        }

        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encode shanten number and tile efficiency features.
    #[pyo3(name = "encode_shanten_efficiency")]
    pub fn encode_shanten_efficiency<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut arr = Array2::<f32>::zeros((NP, 4));

        let mut all_visible: Vec<u32> = Vec::new();
        for discs in &self.discards {
            all_visible.extend(discs.iter().copied());
        }
        for melds_list in &self.melds {
            for meld in melds_list {
                all_visible.extend(meld.tiles.iter().map(|&x| x as u32));
            }
        }
        all_visible.extend(self.dora_indicators.iter().copied());

        for (ch_idx, &abs_idx) in self.rel_order().iter().enumerate() {
            let hand = &self.hands[abs_idx];

            if abs_idx == self.player_id as usize {
                let shanten = crate::shanten::calculate_shanten_3p(hand);
                let effective = crate::shanten::calculate_effective_tiles_3p(hand);
                let best_ukeire = crate::shanten::calculate_best_ukeire_3p(hand, &all_visible);

                arr[[ch_idx, 0]] = (shanten as f32).max(0.0) / 8.0;
                arr[[ch_idx, 1]] = (effective as f32) / 27.0;
                arr[[ch_idx, 2]] = (best_ukeire as f32) / 80.0;
            } else {
                arr[[ch_idx, 0]] = 0.5;
                arr[[ch_idx, 1]] = 0.5;
                arr[[ch_idx, 2]] = 0.5;
            }

            let turn_count = self.discards[abs_idx].len() as f32;
            arr[[ch_idx, 3]] = (turn_count / 18.0).min(1.0);
        }

        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encode kawa (discard pile) overview for all players
    /// Returns a (3, 7, 27) array: 3 players x 7 channels x 27 tile types
    #[pyo3(name = "encode_kawa_overview")]
    pub fn encode_kawa_overview<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut arr = Array3::<f32>::zeros((NP, 7, TILE_DIM_3P));

        for (player_idx, discards) in self.discards.iter().enumerate() {
            let mut tile_counts = [0u8; TILE_DIM_3P];
            let mut aka_flags = [false; 3];

            for &tile in discards {
                let tile34 = (tile / 4) as usize;
                if let Some(idx) = tile34_to_compact(tile34) {
                    let count_idx = tile_counts[idx].min(3) as usize;
                    arr[[player_idx, count_idx, idx]] = 1.0;
                    tile_counts[idx] = tile_counts[idx].saturating_add(1);
                }

                match tile {
                    20 => aka_flags[0] = true, // 5mr - dead in sanma (tile34=5 → None)
                    24 => aka_flags[1] = true, // 5pr
                    28 => aka_flags[2] = true, // 5sr
                    _ => {}
                }
            }

            // aka_flags[0] = 5mr: tile34=4 (5m) → excluded in sanma, skip
            // aka_flags[1] = 5pr: tile34=13 → compact=6
            if aka_flags[1]
                && let Some(idx) = tile34_to_compact(13)
            {
                arr[[player_idx, 5, idx]] = 1.0;
            }
            // aka_flags[2] = 5sr: tile34=22 → compact=15
            if aka_flags[2]
                && let Some(idx) = tile34_to_compact(22)
            {
                arr[[player_idx, 6, idx]] = 1.0;
            }
        }

        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encode fuuro (meld) overview for all players
    /// Returns a (3, 4, 5, 27) array: 3 players x 4 melds x 5 channels x 27 tile types
    #[pyo3(name = "encode_fuuro_overview")]
    pub fn encode_fuuro_overview<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut arr = Array4::<f32>::zeros((NP, 4, 5, TILE_DIM_3P));

        for (ch_idx, &abs_idx) in self.rel_order().iter().enumerate() {
            for (meld_idx, meld) in self.melds[abs_idx].iter().enumerate() {
                if meld_idx >= 4 {
                    break;
                }

                for (tile_slot_idx, &tile) in meld.tiles.iter().enumerate() {
                    if tile_slot_idx >= 4 {
                        break;
                    }

                    let tile34 = (tile / 4) as usize;
                    if let Some(idx) = tile34_to_compact(tile34) {
                        arr[[ch_idx, meld_idx, tile_slot_idx, idx]] = 1.0;
                    }

                    let is_aka = matches!(tile, 16 | 52 | 88);
                    if is_aka && let Some(idx) = tile34_to_compact((tile / 4) as usize) {
                        arr[[ch_idx, meld_idx, 4, idx]] = 1.0;
                    }
                }
            }
        }

        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encode ankan (concealed kan) overview for all players
    /// Returns a (3, 27) array: 3 players x 27 tile types
    #[pyo3(name = "encode_ankan_overview")]
    pub fn encode_ankan_overview<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut arr = Array2::<f32>::zeros((NP, TILE_DIM_3P));

        for (ch_idx, &abs_idx) in self.rel_order().iter().enumerate() {
            for meld in &self.melds[abs_idx] {
                if matches!(meld.meld_type, MeldType::Ankan)
                    && let Some(&tile) = meld.tiles.first()
                {
                    let tile34 = (tile / 4) as usize;
                    if let Some(idx) = tile34_to_compact(tile34) {
                        arr[[ch_idx, idx]] = 1.0;
                    }
                }
            }
        }

        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encode action availability flags
    /// Returns a (11,) array
    #[pyo3(name = "encode_action_availability")]
    pub fn encode_action_availability<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut arr = Array1::<f32>::zeros(11);

        for action in &self._legal_actions {
            match action.action_type {
                ActionType::Riichi => arr[0] = 1.0,
                ActionType::Chi => {
                    // Chi shouldn't happen in 3P, but handle gracefully
                    let tiles = &action.consume_tiles;
                    if tiles.len() == 2 {
                        let t0 = tiles[0] / 4;
                        let t1 = tiles[1] / 4;
                        let diff = (t1 as i32 - t0 as i32).abs();

                        if diff == 1 {
                            if t0 < t1 {
                                arr[1] = 1.0;
                            } else {
                                arr[3] = 1.0;
                            }
                        } else if diff == 2 {
                            arr[2] = 1.0;
                        }
                    }
                }
                ActionType::Pon => arr[4] = 1.0,
                ActionType::Daiminkan => arr[5] = 1.0,
                ActionType::Ankan => arr[6] = 1.0,
                ActionType::Kakan => arr[7] = 1.0,
                ActionType::Tsumo | ActionType::Ron => arr[8] = 1.0,
                ActionType::KyushuKyuhai => arr[9] = 1.0,
                ActionType::Pass => arr[10] = 1.0,
                _ => {}
            }
        }

        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encodes riichi sutehais for opponents
    /// Returns: (2, 3) array - 2 opponents x 3 channels
    #[pyo3(name = "encode_riichi_sutehais")]
    pub fn encode_riichi_sutehais<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut arr = Array2::<f32>::zeros((NP - 1, 3));

        let dora_tiles: Vec<u8> = self
            .dora_indicators
            .iter()
            .map(|&indicator| get_next_tile_sanma(indicator))
            .collect();

        let mut opponent_idx = 0;
        for player_id in 0..NP {
            if player_id == self.player_id as usize {
                continue;
            }

            if let Some(tile) = self.riichi_sutehais[player_id] {
                let tile34 = (tile / 4) as usize;
                if let Some(compact) = tile34_to_compact(tile34) {
                    arr[[opponent_idx, 0]] = compact as f32 / 26.0;
                }
                let is_aka = matches!(tile, 16 | 52 | 88);
                arr[[opponent_idx, 1]] = if is_aka { 1.0 } else { 0.0 };
                let is_dora = dora_tiles.contains(&tile);
                arr[[opponent_idx, 2]] = if is_dora { 1.0 } else { 0.0 };
            }

            opponent_idx += 1;
        }

        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encodes last tedashis for opponents
    /// Returns: (2, 3) array - 2 opponents x 3 channels
    #[pyo3(name = "encode_last_tedashis")]
    pub fn encode_last_tedashis<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut arr = Array2::<f32>::zeros((NP - 1, 3));

        let dora_tiles: Vec<u8> = self
            .dora_indicators
            .iter()
            .map(|&indicator| get_next_tile_sanma(indicator))
            .collect();

        let mut opponent_idx = 0;
        for player_id in 0..NP {
            if player_id == self.player_id as usize {
                continue;
            }

            if let Some(tile) = self.last_tedashis[player_id] {
                let tile34 = (tile / 4) as usize;
                if let Some(compact) = tile34_to_compact(tile34) {
                    arr[[opponent_idx, 0]] = compact as f32 / 26.0;
                }
                let is_aka = matches!(tile, 16 | 52 | 88);
                arr[[opponent_idx, 1]] = if is_aka { 1.0 } else { 0.0 };
                let is_dora = dora_tiles.contains(&tile);
                arr[[opponent_idx, 2]] = if is_dora { 1.0 } else { 0.0 };
            }

            opponent_idx += 1;
        }

        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encodes pass context
    /// Returns: (3,) array: [tile_type, is_aka, is_dora]
    #[pyo3(name = "encode_pass_context")]
    pub fn encode_pass_context<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut arr = Array1::<f32>::zeros(3);

        if let Some(tile) = self.last_discard {
            let tile34 = (tile / 4) as usize;
            if let Some(compact) = tile34_to_compact(tile34) {
                arr[0] = compact as f32 / 26.0;
            }
            let is_aka = matches!(tile, 16 | 52 | 88);
            arr[1] = if is_aka { 1.0 } else { 0.0 };
            let dora_tiles: Vec<u8> = self
                .dora_indicators
                .iter()
                .map(|&indicator| get_next_tile_sanma(indicator))
                .collect();
            let is_dora = dora_tiles.contains(&(tile as u8));
            arr[2] = if is_dora { 1.0 } else { 0.0 };
        }

        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encodes discard candidates detail
    /// Returns: (5,) array
    #[pyo3(name = "encode_discard_candidates")]
    pub fn encode_discard_candidates<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut arr = Array1::<f32>::zeros(5);

        let player_idx = self.player_id as usize;

        let hand = &self.hands[player_idx];
        let current_shanten = shanten::calculate_shanten_3p(hand);

        arr[0] = hand.len() as f32 / 34.0;

        let mut keep_shanten_count = 0;
        let mut increase_shanten_count = 0;

        for (idx, _tile) in hand.iter().enumerate() {
            let new_hand: Vec<u32> = hand
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(_, &t)| t)
                .collect();

            let new_shanten = shanten::calculate_shanten_3p(&new_hand);

            if new_shanten == current_shanten {
                keep_shanten_count += 1;
            } else if new_shanten > current_shanten {
                increase_shanten_count += 1;
            }
        }

        if !hand.is_empty() {
            arr[1] = keep_shanten_count as f32 / hand.len() as f32;
        }
        if !hand.is_empty() {
            arr[2] = increase_shanten_count as f32 / hand.len() as f32;
        }

        arr[3] = if current_shanten == -1 { 1.0 } else { 0.0 };
        arr[4] = if self.riichi_declared[player_idx] {
            1.0
        } else {
            0.0
        };

        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encode all 215 channels of Extended features in a single call.
    #[pyo3(name = "encode_extended")]
    pub fn encode_extended<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let total = 215 * TILE_DIM_3P;
        let mut buf = vec![0.0f32; total];

        self.encode_base_into(&mut buf, 0);
        self.encode_discard_decay_into(&mut buf, 74);
        self.encode_shanten_into(&mut buf, 78);
        self.encode_ankan_into(&mut buf, 94);
        self.encode_fuuro_into(&mut buf, 98);
        self.encode_action_avail_into(&mut buf, 178);
        self.encode_discard_cand_into(&mut buf, 189);
        self.encode_pass_ctx_into(&mut buf, 194);
        self.encode_last_ted_into(&mut buf, 197);
        self.encode_riichi_sute_into(&mut buf, 206);

        let byte_len = total * std::mem::size_of::<f32>();
        let byte_slice = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }
}
