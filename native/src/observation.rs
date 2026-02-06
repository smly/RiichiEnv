use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};
use serde::{Deserialize, Serialize};

use crate::action::{Action, ActionType};
use crate::shanten;
use crate::types::{Meld, MeldType};
use crate::yaku_checker;
use ndarray::prelude::*;

/// Get the next tile in sequence for dora calculation
/// Tile type: 0-33 (m1-m9: 0-8, p1-p9: 9-17, s1-s9: 18-26, winds: 27-30, dragons: 31-33)
fn get_next_tile(tile: u32) -> u8 {
    let tile_type = (tile / 4) / 9; // 0=man, 1=pin, 2=sou, 3+=honors
    let tile_num = (tile / 4) % 9;

    if tile_type < 3 {
        // Number tiles (1-9)
        let next_num = if tile_num == 8 { 0 } else { tile_num + 1 };
        ((tile_type * 9 + next_num) * 4) as u8
    } else {
        // Honor tiles (winds 27-30, dragons 31-33)
        let base = tile / 4;
        if (27..31).contains(&base) {
            // Winds: E->S->W->N->E
            let wind_idx = base - 27;
            let next_wind = (wind_idx + 1) % 4;
            ((27 + next_wind) * 4) as u8
        } else if (31..34).contains(&base) {
            // Dragons: White->Green->Red->White
            let dragon_idx = base - 31;
            let next_dragon = (dragon_idx + 1) % 3;
            ((31 + next_dragon) * 4) as u8
        } else {
            tile as u8 // Fallback
        }
    }
}

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
    #[pyo3(get)]
    pub tsumogiri_flags: Vec<Vec<bool>>,
    #[pyo3(get)]
    pub riichi_sutehais: Vec<Option<u8>>,
    #[pyo3(get)]
    pub last_tedashis: Vec<Option<u8>>,
    #[pyo3(get)]
    pub last_discard: Option<u32>,
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
        riichi_sutehais: Vec<Option<u8>>,
        last_tedashis: Vec<Option<u8>>,
        last_discard: Option<u32>,
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
            tsumogiri_flags: vec![vec![]; 4], // Initialize with empty vectors
            riichi_sutehais,
            last_tedashis,
            last_discard,
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

    /// Encode discard history with exponential decay weighting.
    ///
    /// Returns a (4, 34) array where:
    /// - Row 0: Self discard history with exponential decay
    /// - Row 1-3: Opponent discard history with exponential decay
    ///
    /// Each tile type gets a weighted sum of all discards of that type,
    /// with more recent discards weighted higher:
    /// weight = exp(-decay_rate × age)
    ///
    /// Parameters:
    /// - decay_rate: Rate of exponential decay (default 0.2, as used in Mortal)
    #[pyo3(name = "encode_discard_history_decay", signature = (decay_rate=None))]
    pub fn encode_discard_history_decay<'py>(
        &self,
        py: Python<'py>,
        decay_rate: Option<f32>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let decay_rate = decay_rate.unwrap_or(0.2);
        let mut arr = Array2::<f32>::zeros((4, 34));

        // Encode discard history for all 4 players
        for player_idx in 0..4 {
            if player_idx >= self.discards.len() {
                continue;
            }

            let discs = &self.discards[player_idx];
            let max_len = discs.len();

            if max_len == 0 {
                continue;
            }

            // Iterate through all discards, applying exponential decay
            for (turn, &tile) in discs.iter().enumerate() {
                let tile_idx = (tile as usize) / 4;
                if tile_idx < 34 {
                    // Age = how many turns ago this discard happened
                    // Most recent discard has age 0
                    let age = (max_len - 1 - turn) as f32;
                    let weight = (-decay_rate * age).exp();

                    // Add weighted value (accumulates if same tile discarded multiple times)
                    arr[[player_idx, tile_idx]] += weight;
                }
            }
        }

        // Convert to bytes
        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encode furiten-aware ron possibility based on tsumogiri patterns.
    ///
    /// Returns a (4, 21) array where:
    /// - Dimension 0 (4): Player index (0=self, 1-3=opponents)
    /// - Dimension 1 (21): Yaku type (same indices as encode_yaku_possibility)
    ///
    /// Values:
    /// - 1.0: Ron is possible (player is not in obvious furiten state for recent discards)
    /// - 0.0: Ron is likely impossible (player has been tsumogiri, suggesting no hand changes)
    ///
    /// Logic: If a player has been consistently doing tsumogiri (auto-discard), their hand
    /// hasn't changed, so they likely cannot ron on tiles in their own river (furiten rule).
    #[pyo3(name = "encode_furiten_ron_possibility")]
    pub fn encode_furiten_ron_possibility<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        const NUM_YAKU: usize = 21;
        let mut arr = Array2::<f32>::ones((4, NUM_YAKU));

        // For each player, check if they've been doing tsumogiri consecutively
        for player_idx in 0..4 {
            if player_idx >= self.tsumogiri_flags.len() {
                continue;
            }

            let flags = &self.tsumogiri_flags[player_idx];
            if flags.is_empty() {
                continue;
            }

            // Count consecutive tsumogiri from the end
            let mut consecutive_tsumogiri = 0;
            for &flag in flags.iter().rev() {
                if flag {
                    consecutive_tsumogiri += 1;
                } else {
                    break;
                }
            }

            // If 3+ consecutive tsumogiri, assume hand hasn't changed
            // This means ron on their own discards is likely furiten
            if consecutive_tsumogiri >= 3 {
                // Mark all yaku as having reduced ron possibility
                for yaku_idx in 0..NUM_YAKU {
                    arr[[player_idx, yaku_idx]] = 0.0;
                }
            }
        }

        // Convert to bytes
        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encode yaku (winning hand patterns) possibility for each player.
    ///
    /// Returns a (4, N, 2) array where:
    /// - Dimension 0 (4): Player index (0=self, 1-3=opponents)
    /// - Dimension 1 (N): Yaku type (see YAKU_INDICES below)
    /// - Dimension 2 (2): [Tsumo possibility, Ron possibility]
    ///
    /// Values:
    /// - 1.0: Yaku is possible or unknown (conservative estimate)
    /// - 0.0: Yaku is definitely impossible based on visible information
    ///
    /// Yaku indices:
    /// 0: Tanyao (all simples)
    /// 1-5: Yakuhai (honor tiles): White, Green, Red, Round wind, Seat wind
    /// 6: Honitsu (half flush)
    /// 7: Chinitsu (full flush)
    /// 8: Toitoi (all triplets)
    /// 9: Chiitoitsu (seven pairs)
    /// 10: Shousangen (small three dragons)
    /// 11: Daisangen (big three dragons)
    /// 12: Tsuuiisou (all honors)
    /// 13: Chinroutou (all terminals)
    /// 14: Honroutou (terminals and honors)
    /// 15: Kokushi (thirteen orphans)
    /// 16: Chanta (outside hand)
    /// 17: Junchan (pure outside hand)
    /// 18: Sanshoku doujun (three colored straight)
    /// 19: Iipeikou (pure double sequence)
    /// 20: Ittsu (straight)
    ///
    /// Total: 21 yaku types
    #[pyo3(name = "encode_yaku_possibility")]
    pub fn encode_yaku_possibility<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        const NUM_YAKU: usize = 21;
        let mut arr = Array3::<f32>::ones((4, NUM_YAKU, 2));

        // Combine all visible tiles for each player
        let mut all_discards: Vec<Vec<u32>> = Vec::with_capacity(4);
        for player_idx in 0..4 {
            if player_idx < self.discards.len() {
                all_discards.push(self.discards[player_idx].clone());
            } else {
                all_discards.push(Vec::new());
            }
        }

        // Encode for each player
        for player_idx in 0..4 {
            if player_idx >= self.melds.len() {
                continue;
            }

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
            let seat = (player_idx as u8 + 4 - self.oya) % 4;
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

            // Yaku 15: Kokushi (thirteen orphans)
            let kokushi = yaku_checker::check_kokushi(melds, discards, &self.dora_indicators);
            arr[[player_idx, 15, 0]] = kokushi.to_f32();
            arr[[player_idx, 15, 1]] = kokushi.to_f32();

            // Yaku 16: Chanta (outside hand)
            let chanta = yaku_checker::check_chanta(melds);
            arr[[player_idx, 16, 0]] = chanta.to_f32();
            arr[[player_idx, 16, 1]] = chanta.to_f32();

            // Yaku 17: Junchan (pure outside hand)
            let junchan = yaku_checker::check_junchan(melds);
            arr[[player_idx, 17, 0]] = junchan.to_f32();
            arr[[player_idx, 17, 1]] = junchan.to_f32();

            // Yaku 18: Sanshoku doujun (three colored straight)
            let sanshoku = yaku_checker::check_sanshoku_doujun(melds);
            arr[[player_idx, 18, 0]] = sanshoku.to_f32();
            arr[[player_idx, 18, 1]] = sanshoku.to_f32();

            // Yaku 19: Iipeikou (pure double sequence)
            let iipeikou = yaku_checker::check_iipeikou(melds);
            arr[[player_idx, 19, 0]] = iipeikou.to_f32();
            arr[[player_idx, 19, 1]] = iipeikou.to_f32();

            // Yaku 20: Ittsu (straight)
            let ittsu = yaku_checker::check_ittsu(melds);
            arr[[player_idx, 20, 0]] = ittsu.to_f32();
            arr[[player_idx, 20, 1]] = ittsu.to_f32();
        }

        // Convert to bytes
        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    pub fn encode<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        // Total Channels (Expanded with Mortal features):
        // 0-3: Hand (1,2,3,4)
        // 4: Red (Hand)
        // 5-8: Melds (Self)
        // 9: Dora Indicators
        // 10-13: Discards (Self, Last 4) (History)
        // 14-25: Discards (Opponents, Last 4 each) (History)
        // 26-29: Discard Counts (All players, normalized)
        // 30: Tiles Left in Wall (normalized)
        // 31: Riichi (Self)
        // 32-34: Riichi (Opponents)
        // 35: Round Wind
        // 36: Self Wind
        // 37: Honba
        // 38: Riichi Sticks
        // 39-42: Scores (P0-P3, normalized 0-100000)
        // 43-46: Scores (P0-P3, normalized 0-30000)
        // 47: Waits (1 channel)
        // 48: Is Tenpai (1 channel)
        // 49-52: Rank (One-hot)
        // 53: Kyoku Index (Normalized)
        // 54: Round Progress (combined: (round_wind)*4 + kyoku_idx, normalized)
        // 55-58: Dora Count (per player, normalized)
        // 59-62: Melds Count (per player)
        // 63: Tiles Seen (Normalized)
        // 64-67: Extended discards (Self, 5-8)
        // 68-69: Extended discards (Opponent 1, 5-6)
        // 70-73: Tsumogiri flags (All players, last discard)
        // Total: 74 channels

        let num_channels = 74;
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

        // 7. Discard Counts (All players, normalized) (26-29)
        for (player_idx, discs) in self.discards.iter().enumerate() {
            let count_norm = (discs.len() as f32) / 24.0; // Max ~24 discards
            for k in 0..34 {
                arr[[26 + player_idx, k]] = count_norm;
            }
        }

        // 8. Tiles Left in Wall (30)
        // Estimate tiles left: 136 total - discards - melds - hands visible
        let mut tiles_used = 0;
        for discs in &self.discards {
            tiles_used += discs.len();
        }
        for melds_list in &self.melds {
            for meld in melds_list {
                tiles_used += meld.tiles.len();
            }
        }
        // Visible hands (just self)
        if (self.player_id as usize) < self.hands.len() {
            tiles_used += self.hands[self.player_id as usize].len();
        }
        tiles_used += self.dora_indicators.len();
        let tiles_left = (136_i32 - tiles_used as i32).max(0) as f32;
        let tiles_left_norm = tiles_left / 70.0; // Max ~70 tiles left in wall
        for k in 0..34 {
            arr[[30, k]] = tiles_left_norm;
        }

        // 9. Riichi (31-34)
        if (self.player_id as usize) < self.riichi_declared.len()
            && self.riichi_declared[self.player_id as usize]
        {
            for i in 0..34 {
                arr[[31, i]] = 1.0;
            }
        }
        for i in 1..4 {
            let opp_id = (self.player_id + i) % 4;
            if (opp_id as usize) < self.riichi_declared.len()
                && self.riichi_declared[opp_id as usize]
            {
                for k in 0..34 {
                    arr[[32 + (i as usize - 1), k]] = 1.0;
                }
            }
        }

        // 10. Winds (35-36)
        let rw = self.round_wind as usize;
        if 27 + rw < 34 {
            arr[[35, 27 + rw]] = 1.0;
        }
        let seat = (self.player_id + 4 - self.oya) % 4;
        if 27 + (seat as usize) < 34 {
            arr[[36, 27 + (seat as usize)]] = 1.0;
        }

        // 11. Honba/Sticks (37-38)
        let honba_norm = (self.honba as f32) / 10.0;
        let sticks_norm = (self.riichi_sticks as f32) / 5.0;
        for i in 0..34 {
            arr[[37, i]] = honba_norm;
            arr[[38, i]] = sticks_norm;
        }

        // 12. Scores (39-42) normalized 0-100000
        for i in 0..4 {
            if i < self.scores.len() {
                let score_norm = (self.scores[i].clamp(0, 100000) as f32) / 100000.0;
                for k in 0..34 {
                    arr[[39 + i, k]] = score_norm;
                }
            }
        }

        // 13. Scores (43-46) normalized 0-30000
        for i in 0..4 {
            if i < self.scores.len() {
                let score_norm = (self.scores[i].clamp(0, 30000) as f32) / 30000.0;
                for k in 0..34 {
                    arr[[43 + i, k]] = score_norm;
                }
            }
        }

        // 14. Waits (47)
        for &t in &self.waits {
            if (t as usize) < 34 {
                arr[[47, t as usize]] = 1.0;
            }
        }

        // 15. Is Tenpai (48)
        let tenpai_val = if self.is_tenpai { 1.0 } else { 0.0 };
        for i in 0..34 {
            arr[[48, i]] = tenpai_val;
        }

        // 16. Rank (49-52)
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
        // Just broadcast 1 to channel (49 + rank).
        if rank < 4 {
            for i in 0..34 {
                arr[[49 + rank, i]] = 1.0;
            }
        }

        // 17. Kyoku (53)
        let k_norm = (self.kyoku_index as f32) / 8.0; // Approx max 8 (East 1-4, South 1-4).
        for i in 0..34 {
            arr[[53, i]] = k_norm;
        }

        // 18. Round Progress (54) - Combined: (round_wind) * 4 + kyoku_idx
        let round_progress = (self.round_wind as f32) * 4.0 + (self.kyoku_index as f32);
        let round_progress_norm = round_progress / 7.0; // 0-7 for East 1-4, South 1-4
        for i in 0..34 {
            arr[[54, i]] = round_progress_norm;
        }

        // 19. Dora Count (55-58) - per player, rescaled
        let mut dora_counts = [0u8; 4];
        // Count dora in each player's visible tiles (melds + discards)
        for (player_idx, dora_count) in dora_counts.iter_mut().enumerate() {
            // Melds
            if player_idx < self.melds.len() {
                for meld in &self.melds[player_idx] {
                    for &tile in &meld.tiles {
                        // Check if tile is dora
                        for &dora_ind in &self.dora_indicators {
                            let dora_tile = get_next_tile(dora_ind);
                            if (tile / 4) == (dora_tile / 4) {
                                *dora_count += 1;
                            }
                        }
                    }
                }
            }
            // Discards
            if player_idx < self.discards.len() {
                for &tile in &self.discards[player_idx] {
                    // tile is u32 from Vec<u32>
                    for &dora_ind in &self.dora_indicators {
                        let dora_tile = get_next_tile(dora_ind);
                        if ((tile / 4) as u8) == (dora_tile / 4) {
                            *dora_count += 1;
                        }
                    }
                }
            }
        }
        // Self hand
        if (self.player_id as usize) < self.hands.len() {
            for &tile in &self.hands[self.player_id as usize] {
                // tile is u32 from Vec<u32>
                for &dora_ind in &self.dora_indicators {
                    let dora_tile = get_next_tile(dora_ind);
                    if ((tile / 4) as u8) == (dora_tile / 4) {
                        dora_counts[self.player_id as usize] += 1;
                    }
                }
            }
        }
        for i in 0..4 {
            let dora_norm = (dora_counts[i] as f32) / 12.0; // Cap at 12
            for k in 0..34 {
                arr[[55 + i, k]] = dora_norm;
            }
        }

        // 20. Melds Count (59-62) - per player, normalized
        for (player_idx, melds_list) in self.melds.iter().enumerate() {
            let meld_count_norm = (melds_list.len() as f32) / 4.0; // Max 4 melds
            for k in 0..34 {
                arr[[59 + player_idx, k]] = meld_count_norm;
            }
        }

        // 21. Tiles Seen (63)
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
            arr[[63, i]] = norm_seen;
        }

        // 22-24. Extended Discard History (64-69)
        // Self: Last 18 discards (64-67, 4 channels)
        if (self.player_id as usize) < self.discards.len() {
            let discs = &self.discards[self.player_id as usize];
            for (i, &t) in discs.iter().rev().skip(4).take(4).enumerate() {
                let idx = (t as usize) / 4;
                if idx < 34 {
                    arr[[64 + i, idx]] = 1.0;
                }
            }
        }

        // Opponents: Last 4-7 discards (68-69, 2 channels)
        // Simplified: only track 2 more channels for first opponent
        if self.discards.len() > ((self.player_id + 1) % 4) as usize {
            let opp_id = (self.player_id + 1) % 4;
            let discs = &self.discards[opp_id as usize];
            for (i, &t) in discs.iter().rev().skip(4).take(2).enumerate() {
                let idx = (t as usize) / 4;
                if idx < 34 {
                    arr[[68 + i, idx]] = 1.0;
                }
            }
        }

        // 25. Tsumogiri flags (70-73) - broadcast per player
        // 1.0 if last discard was tsumogiri (drawn and immediately discarded)
        for player_idx in 0..4 {
            if player_idx < self.tsumogiri_flags.len()
                && !self.tsumogiri_flags[player_idx].is_empty()
            {
                let last_tsumogiri = *self.tsumogiri_flags[player_idx].last().unwrap_or(&false);
                let val = if last_tsumogiri { 1.0 } else { 0.0 };
                for k in 0..34 {
                    arr[[70 + player_idx, k]] = val;
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
    ///
    /// Returns a (4, 4) array where:
    /// - Dimension 0 (4): Player index (0=self, 1-3=opponents)
    /// - Dimension 1 (4): [shanten, effective_tiles, best_ukeire, normalized_turn]
    ///
    /// Features:
    /// - shanten: Normalized shanten number (/ 8.0, where 8 is max typical shanten)
    /// - effective_tiles: Number of tile types that reduce shanten (/ 34.0)
    /// - best_ukeire: Best ukeire count after optimal discard (/ 80.0)
    /// - normalized_turn: Current turn / max_turns (approx / 18.0)
    #[pyo3(name = "encode_shanten_efficiency")]
    pub fn encode_shanten_efficiency<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut arr = Array2::<f32>::zeros((4, 4));

        // Collect all visible tiles for ukire calculation
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

        // Calculate features for each player
        for player_idx in 0..4 {
            if player_idx >= self.hands.len() {
                continue;
            }

            let hand = &self.hands[player_idx];

            // For self, we have full information
            // For opponents, we can only estimate based on visible info
            if player_idx == self.player_id as usize {
                // Self: full calculation
                let shanten = crate::shanten::calculate_shanten(hand);
                let effective = crate::shanten::calculate_effective_tiles(hand);
                let best_ukeire = crate::shanten::calculate_best_ukeire(hand, &all_visible);

                // Normalize features
                arr[[player_idx, 0]] = (shanten as f32).max(0.0) / 8.0;
                arr[[player_idx, 1]] = (effective as f32) / 34.0;
                arr[[player_idx, 2]] = (best_ukeire as f32) / 80.0;
            } else {
                // Opponents: estimate or use conservative values
                // We don't know their hand, so set to unknown (0.5)
                arr[[player_idx, 0]] = 0.5; // Unknown shanten
                arr[[player_idx, 1]] = 0.5; // Unknown effective tiles
                arr[[player_idx, 2]] = 0.5; // Unknown ukeire
            }

            // Turn count (same for all players)
            let turn_count = if player_idx < self.discards.len() {
                self.discards[player_idx].len() as f32
            } else {
                0.0
            };
            arr[[player_idx, 3]] = turn_count / 18.0; // Normalize by typical max turns
        }

        // Convert to bytes
        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encode kawa (discard pile) overview for all players
    /// Returns a (4, 7, 34) array: 4 players × 7 channels × 34 tile types
    /// Channels: [count1, count2, count3, count4, aka5m, aka5p, aka5s]
    #[pyo3(name = "encode_kawa_overview")]
    pub fn encode_kawa_overview<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut arr = Array3::<f32>::zeros((4, 7, 34));

        for (player_idx, discards) in self.discards.iter().enumerate() {
            if player_idx >= 4 {
                break;
            }

            // Count each tile type (up to 4 copies)
            let mut tile_counts = [0u8; 34];
            let mut aka_flags = [false; 3]; // aka 5m/5p/5s

            for &tile in discards {
                let tile_type = (tile / 4) as usize;
                if tile_type < 34 {
                    let count_idx = tile_counts[tile_type].min(3) as usize;
                    arr[[player_idx, count_idx, tile_type]] = 1.0;
                    tile_counts[tile_type] = tile_counts[tile_type].saturating_add(1);
                }

                // Check for aka tiles (20, 24, 28 are aka 5m/5p/5s in 136-tile encoding)
                match tile {
                    20 => aka_flags[0] = true, // aka 5m
                    24 => aka_flags[1] = true, // aka 5p
                    28 => aka_flags[2] = true, // aka 5s
                    _ => {}
                }
            }

            // Set aka flags
            for (i, &has_aka) in aka_flags.iter().enumerate() {
                if has_aka {
                    arr[[player_idx, 4 + i, 5 + i * 9]] = 1.0; // 5m=5, 5p=14, 5s=23
                }
            }
        }

        // Convert to bytes
        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encode fuuro (meld) overview for all players
    /// Returns a (4, 4, 5, 34) array: 4 players × 4 melds × 5 channels × 34 tile types
    /// Channels: [tile1, tile2, tile3, tile4, aka]
    #[pyo3(name = "encode_fuuro_overview")]
    pub fn encode_fuuro_overview<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut arr = Array4::<f32>::zeros((4, 4, 5, 34));

        for (player_idx, melds) in self.melds.iter().enumerate() {
            if player_idx >= 4 {
                break;
            }

            for (meld_idx, meld) in melds.iter().enumerate() {
                if meld_idx >= 4 {
                    break;
                }

                // Encode each tile in the meld
                for (tile_slot_idx, &tile) in meld.tiles.iter().enumerate() {
                    if tile_slot_idx >= 4 {
                        break;
                    }

                    let tile_type = (tile / 4) as usize;
                    if tile_type < 34 {
                        arr[[player_idx, meld_idx, tile_slot_idx, tile_type]] = 1.0;
                    }

                    // Check for aka
                    let is_aka = matches!(tile, 20 | 24 | 28);
                    if is_aka {
                        arr[[player_idx, meld_idx, 4, tile_type]] = 1.0;
                    }
                }
            }
        }

        // Convert to bytes
        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encode ankan (concealed kan) overview for all players
    /// Returns a (4, 34) array: 4 players × 34 tile types
    #[pyo3(name = "encode_ankan_overview")]
    pub fn encode_ankan_overview<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut arr = Array2::<f32>::zeros((4, 34));

        for (player_idx, melds) in self.melds.iter().enumerate() {
            if player_idx >= 4 {
                break;
            }

            for meld in melds {
                // Check if this is an ankan (concealed kan)
                if matches!(meld.meld_type, MeldType::Angang) {
                    // Use the first tile to determine type
                    if let Some(&tile) = meld.tiles.first() {
                        let tile_type = (tile / 4) as usize;
                        if tile_type < 34 {
                            arr[[player_idx, tile_type]] = 1.0;
                        }
                    }
                }
            }
        }

        // Convert to bytes
        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encode action availability flags
    /// Returns a (11,) array: [can_riichi, can_chi_low, can_chi_mid, can_chi_high,
    ///                          can_pon, can_daiminkan, can_ankan, can_kakan,
    ///                          can_agari, can_ryukyoku, can_pass]
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
                    // Determine chi type by consumed tiles
                    let tiles = &action.consume_tiles;
                    if tiles.len() == 2 {
                        let t0 = tiles[0] / 4;
                        let t1 = tiles[1] / 4;
                        let diff = (t1 as i32 - t0 as i32).abs();

                        if diff == 1 {
                            // Could be low or high
                            if t0 < t1 {
                                arr[1] = 1.0; // low (call tile is highest)
                            } else {
                                arr[3] = 1.0; // high (call tile is lowest)
                            }
                        } else if diff == 2 {
                            arr[2] = 1.0; // mid
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

        // Convert to bytes
        let slice = arr.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
        })?;
        let byte_len = std::mem::size_of_val(slice);
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Ok(pyo3::types::PyBytes::new(py, byte_slice))
    }

    /// Encodes riichi sutehais (tiles discarded when declaring riichi) for opponents
    /// Returns: (3, 3) array
    /// - 3 opponents (excluding self)
    /// - 3 channels per opponent: [tile_type, is_aka, is_dora]
    #[pyo3(name = "encode_riichi_sutehais")]
    pub fn encode_riichi_sutehais<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut arr = Array2::<f32>::zeros((3, 3));

        let dora_tiles: Vec<u8> = self
            .dora_indicators
            .iter()
            .map(|&indicator| get_next_tile(indicator))
            .collect();

        let mut opponent_idx = 0;
        for player_id in 0..4 {
            if player_id == self.player_id as usize {
                continue;
            }

            if let Some(tile) = self.riichi_sutehais[player_id] {
                let tile_type = (tile / 4) as usize;
                let tile_u32 = tile as u32;

                // Channel 0: tile type (0-33)
                arr[[opponent_idx, 0]] = tile_type as f32 / 33.0;

                // Channel 1: is aka (red five)
                let is_aka = matches!(tile_u32, 20 | 24 | 28);
                arr[[opponent_idx, 1]] = if is_aka { 1.0 } else { 0.0 };

                // Channel 2: is dora
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

    /// Encodes last tedashis (last hand discards, not tsumogiri) for opponents
    /// Returns: (3, 3) array
    /// - 3 opponents (excluding self)
    /// - 3 channels per opponent: [tile_type, is_aka, is_dora]
    #[pyo3(name = "encode_last_tedashis")]
    pub fn encode_last_tedashis<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut arr = Array2::<f32>::zeros((3, 3));

        let dora_tiles: Vec<u8> = self
            .dora_indicators
            .iter()
            .map(|&indicator| get_next_tile(indicator))
            .collect();

        let mut opponent_idx = 0;
        for player_id in 0..4 {
            if player_id == self.player_id as usize {
                continue;
            }

            if let Some(tile) = self.last_tedashis[player_id] {
                let tile_type = (tile / 4) as usize;
                let tile_u32 = tile as u32;

                // Channel 0: tile type (0-33)
                arr[[opponent_idx, 0]] = tile_type as f32 / 33.0;

                // Channel 1: is aka (red five)
                let is_aka = matches!(tile_u32, 20 | 24 | 28);
                arr[[opponent_idx, 1]] = if is_aka { 1.0 } else { 0.0 };

                // Channel 2: is dora
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

    /// Encodes pass context (current offer tile for chi/pon/kan/ron)
    /// Returns: (3,) array: [tile_type, is_aka, is_dora]
    #[pyo3(name = "encode_pass_context")]
    pub fn encode_pass_context<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut arr = Array1::<f32>::zeros(3);

        if let Some(tile) = self.last_discard {
            let tile_type = (tile / 4) as usize;

            // Channel 0: tile type (0-33)
            arr[0] = tile_type as f32 / 33.0;

            // Channel 1: is aka (red five)
            let is_aka = matches!(tile, 20 | 24 | 28);
            arr[1] = if is_aka { 1.0 } else { 0.0 };

            // Channel 2: is dora
            let dora_tiles: Vec<u8> = self
                .dora_indicators
                .iter()
                .map(|&indicator| get_next_tile(indicator))
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
    /// - Channel 0: number of tiles that can be discarded
    /// - Channel 1: number of discards that keep current shanten
    /// - Channel 2: number of discards that increase shanten
    /// - Channel 3: whether already in tenpai (unconditional tenpai check)
    /// - Channel 4: whether riichi is declared
    #[pyo3(name = "encode_discard_candidates")]
    pub fn encode_discard_candidates<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut arr = Array1::<f32>::zeros(5);

        let player_idx = self.player_id as usize;
        if player_idx >= self.hands.len() {
            let slice = arr.as_slice().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Array not contiguous")
            })?;
            let byte_len = std::mem::size_of_val(slice);
            let byte_slice =
                unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
            return Ok(pyo3::types::PyBytes::new(py, byte_slice));
        }

        let hand = &self.hands[player_idx];
        let current_shanten = shanten::calculate_shanten(hand);

        // Channel 0: number of tiles that can be discarded (normalized by 34)
        arr[0] = hand.len() as f32 / 34.0;

        // Analyze each possible discard
        let mut keep_shanten_count = 0;
        let mut increase_shanten_count = 0;

        for (idx, _tile) in hand.iter().enumerate() {
            let new_hand: Vec<u32> = hand
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(_, &t)| t)
                .collect();

            let new_shanten = shanten::calculate_shanten(&new_hand);

            if new_shanten == current_shanten {
                keep_shanten_count += 1;
            } else if new_shanten > current_shanten {
                increase_shanten_count += 1;
            }
        }

        // Channel 1: discards that keep shanten (normalized by hand size)
        if !hand.is_empty() {
            arr[1] = keep_shanten_count as f32 / hand.len() as f32;
        }

        // Channel 2: discards that increase shanten (normalized by hand size)
        if !hand.is_empty() {
            arr[2] = increase_shanten_count as f32 / hand.len() as f32;
        }

        // Channel 3: unconditional tenpai (shanten == -1)
        arr[3] = if current_shanten == -1 { 1.0 } else { 0.0 };

        // Channel 4: riichi declared
        arr[4] = if player_idx < self.riichi_declared.len() && self.riichi_declared[player_idx] {
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
}
