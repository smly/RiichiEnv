#[cfg(feature = "python")]
use flate2::read::GzDecoder;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
#[cfg(feature = "python")]
use serde_json::Value;
#[cfg(feature = "python")]
use std::fs::File;
#[cfg(feature = "python")]
use std::io::{BufReader, Read};
#[cfg(feature = "python")]
use std::sync::Arc;

#[cfg(feature = "python")]
use crate::replay::{Action, HuleData, LogKyoku, TileConverter, WinResultContextIterator};
#[cfg(feature = "python")]
use crate::types::MeldType;

#[cfg(feature = "python")]
#[pyclass(module = "riichienv._riichienv")]
pub struct MjSoulReplay {
    pub rounds: Vec<LogKyoku>,
}

#[cfg(feature = "python")]
#[derive(Debug)]
#[pyclass(module = "riichienv._riichienv")]
pub struct KyokuIterator {
    game: Py<MjSoulReplay>,
    index: usize,
    len: usize,
}

#[cfg(feature = "python")]
#[pymethods]
impl KyokuIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<LogKyoku> {
        if slf.index >= slf.len {
            return None;
        }

        let kyoku = {
            let game = slf.game.borrow(slf.py());
            game.rounds[slf.index].clone()
        };
        slf.index += 1;

        Some(kyoku)
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(tag = "name", content = "data")]
pub enum RawAction {
    #[serde(rename = "NewRound")]
    NewRound {
        scores: Vec<i32>,
        doras: Option<Vec<String>>,
        dora_indicators: Option<Vec<String>>,
        dora_marker: Option<String>,
        tiles0: Vec<String>,
        tiles1: Vec<String>,
        tiles2: Vec<String>,
        tiles3: Vec<String>,
        chang: u8,
        ju: u8,
        ben: Option<u8>,
        honba: Option<u8>,
        liqibang: u8,
        left_tile_count: Option<u8>,
        ura_doras: Option<Vec<String>>,
        paishan: Option<String>,
    },
    #[serde(rename = "DiscardTile")]
    DiscardTile {
        seat: usize,
        tile: String,
        #[serde(default)]
        is_liqi: bool,
        #[serde(default)]
        is_wliqi: bool,
        #[serde(default)]
        doras: Vec<String>,
    },
    #[serde(rename = "DealTile")]
    DealTile {
        seat: usize,
        tile: String,
        #[serde(default)]
        doras: Vec<String>,
        dora_marker: Option<String>,
        left_tile_count: Option<u8>,
    },
    #[serde(rename = "ChiPengGang")]
    ChiPengGang {
        seat: usize,
        #[serde(rename = "type")]
        meld_type: u64,
        tiles: Vec<String>,
        froms: Vec<usize>,
    },
    #[serde(rename = "AnGangAddGang")]
    AnGangAddGang {
        seat: usize,
        #[serde(rename = "type")]
        meld_type: u64,
        tiles: String,
    },
    #[serde(rename = "Hule")]
    Hule {
        hules: Vec<HuleDataRaw>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        old_scores: Option<Vec<i32>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        delta_scores: Option<Vec<i32>>,
    },
    #[serde(rename = "dora")]
    Dora { dora_marker: String },
    #[serde(rename = "NoTile")]
    NoTile {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        old_scores: Option<Vec<i32>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        delta_scores: Option<Vec<i32>>,
    },
    #[serde(rename = "BaBei")]
    BaBei {
        seat: usize,
        #[serde(default)]
        moqie: bool,
        #[serde(default)]
        doras: Vec<String>,
    },
    #[serde(rename = "LiuJu")]
    LiuJu {
        #[serde(rename = "type", default)]
        lj_type: u8,
        #[serde(default)]
        seat: usize,
        #[serde(default)]
        tiles: Vec<String>,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct HuleDataRaw {
    pub seat: usize,
    pub hu_tile: String,
    pub zimo: bool,
    pub count: u32,
    pub fu: u32,
    pub fans: Vec<FanRaw>,
    pub hand: Vec<String>,
    pub ura_dora_indicators: Option<Vec<String>>,
    pub li_doras: Option<Vec<String>>,
    pub yiman: bool,
    pub point_rong: u32,
    pub point_zimo_qin: u32,
    pub point_zimo_xian: u32,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct FanRaw {
    pub id: u32,
    #[serde(default)]
    pub val: u32,
}

#[derive(Deserialize, Serialize)]
pub struct GameLog {
    pub rounds: Vec<Vec<RawAction>>,
}

#[cfg(feature = "python")]
type ParsedRawRounds = (
    Vec<Vec<RawAction>>,
    Vec<Vec<Option<bool>>>,
    Vec<Vec<Option<Vec<String>>>>,
);

/// Parses the legacy public `RawAction` shape while retaining optional source
/// metadata (`data.moqie` and dora snapshots on variants that lack a public
/// field) in crate-private parallel vectors. Unknown JSON fields remain
/// accepted by serde exactly as they were in v0.4.8.
#[cfg(feature = "python")]
fn parse_raw_rounds_with_metadata(rounds_value: Value) -> Result<ParsedRawRounds, String> {
    let round_values = rounds_value
        .as_array()
        .ok_or_else(|| "rounds must be an array".to_string())?;
    let mut action_tsumogiri = Vec::with_capacity(round_values.len());
    let mut action_dora_snapshots = Vec::with_capacity(round_values.len());
    for (round_index, round) in round_values.iter().enumerate() {
        let actions = round
            .as_array()
            .ok_or_else(|| format!("round {round_index} must be an array"))?;
        let mut round_tsumogiri = Vec::with_capacity(actions.len());
        let mut round_dora_snapshots = Vec::with_capacity(actions.len());
        for (action_index, action) in actions.iter().enumerate() {
            let name = action.get("name").and_then(Value::as_str);
            let source = if action.get("name").and_then(Value::as_str) == Some("DiscardTile") {
                match action.get("data").and_then(|data| data.get("moqie")) {
                    None | Some(Value::Null) => None,
                    Some(Value::Bool(value)) => Some(*value),
                    Some(_) => {
                        return Err(format!(
                            "round {round_index} action {action_index} DiscardTile moqie must be boolean or null"
                        ));
                    }
                }
            } else {
                None
            };
            let private_dora_snapshot = if matches!(name, Some("AnGangAddGang" | "BaBei")) {
                match action.get("data").and_then(|data| data.get("doras")) {
                    None | Some(Value::Null) => None,
                    Some(Value::Array(values)) if values.is_empty() => None,
                    Some(Value::Array(values)) => {
                        let mut tiles = Vec::with_capacity(values.len());
                        for value in values {
                            let tile = value.as_str().ok_or_else(|| {
                                format!(
                                    "round {round_index} action {action_index} doras must contain strings"
                                )
                            })?;
                            validate_mjsoul_tile(tile, "private action doras")?;
                            tiles.push(tile.to_string());
                        }
                        validate_dora_snapshot_len(tiles.len(), "private action doras")?;
                        Some(tiles)
                    }
                    Some(_) => {
                        return Err(format!(
                            "round {round_index} action {action_index} doras must be an array or null"
                        ));
                    }
                }
            } else {
                None
            };
            round_tsumogiri.push(source);
            round_dora_snapshots.push(private_dora_snapshot);
        }
        action_tsumogiri.push(round_tsumogiri);
        action_dora_snapshots.push(round_dora_snapshots);
    }

    let rounds = serde_json::from_value(rounds_value)
        .map_err(|error| format!("Failed to parse rounds: {error}"))?;
    Ok((rounds, action_tsumogiri, action_dora_snapshots))
}

#[cfg(feature = "python")]
fn validate_mjsoul_tile(tile: &str, field: &str) -> Result<(), String> {
    let bytes = tile.as_bytes();
    let valid = bytes.len() == 2
        && match bytes[1] {
            b'm' | b'p' | b's' => bytes[0].is_ascii_digit(),
            b'z' => matches!(bytes[0], b'1'..=b'7'),
            _ => false,
        };
    if valid {
        Ok(())
    } else {
        Err(format!("{field} contains invalid tile {tile:?}"))
    }
}

#[cfg(feature = "python")]
fn validate_mjsoul_tiles(tiles: &[String], field: &str) -> Result<(), String> {
    for tile in tiles {
        validate_mjsoul_tile(tile, field)?;
    }
    Ok(())
}

#[cfg(feature = "python")]
fn validate_dora_snapshot_len(len: usize, field: &str) -> Result<(), String> {
    if len <= 5 {
        Ok(())
    } else {
        Err(format!(
            "{field} contains {len} indicators; at most five are valid"
        ))
    }
}

#[cfg(feature = "python")]
fn validate_cumulative_dora_snapshot(
    current: &[u8],
    next: &[u8],
    field: &str,
) -> Result<(), String> {
    if next.len() < current.len() || !next.starts_with(current) {
        Err(format!(
            "{field} must preserve the complete prefix of the previous cumulative snapshot"
        ))
    } else {
        Ok(())
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl MjSoulReplay {
    #[staticmethod]
    fn from_json(path: String) -> PyResult<Self> {
        let file = File::open(&path)
            .map_err(|e| PyValueError::new_err(format!("Failed to open file: {}", e)))?;
        let reader = BufReader::with_capacity(65536, file);
        let mut decoder = GzDecoder::new(reader);
        let mut buffer = Vec::with_capacity(128 * 1024);

        decoder
            .read_to_end(&mut buffer)
            .map_err(|e| PyValueError::new_err(format!("Failed to decompress: {}", e)))?;

        let value: Value = serde_json::from_slice(&buffer)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse JSON: {}", e)))?;
        let rounds_value = value
            .get("rounds")
            .cloned()
            .ok_or_else(|| PyValueError::new_err("Invalid replay: missing rounds"))?;
        let (rounds_raw, action_tsumogiri, action_dora_snapshots) =
            parse_raw_rounds_with_metadata(rounds_value).map_err(PyValueError::new_err)?;

        let mut rounds: Vec<LogKyoku> = Vec::with_capacity(rounds_raw.len());
        for ((r_raw, round_tsumogiri), round_dora_snapshots) in rounds_raw
            .into_iter()
            .zip(action_tsumogiri)
            .zip(action_dora_snapshots)
        {
            let kyoku = Self::kyoku_from_raw_actions(r_raw, round_tsumogiri, round_dora_snapshots)
                .map_err(|error| PyValueError::new_err(format!("Invalid round: {error}")))?;
            if let Some(first) = rounds.first()
                && first.scores.len() != kyoku.scores.len()
            {
                return Err(PyValueError::new_err(format!(
                    "Invalid replay: player count changed from {} to {}",
                    first.scores.len(),
                    kyoku.scores.len()
                )));
            }
            rounds.push(kyoku);
        }

        // Populate end_scores based on next round's start scores
        for i in 0..rounds.len().saturating_sub(1) {
            rounds[i].end_scores = rounds[i + 1].scores.clone();
        }

        Ok(MjSoulReplay { rounds })
    }

    #[staticmethod]
    fn from_dict(py: Python, paifu: Py<PyAny>) -> PyResult<Self> {
        let json = py.import("json")?;
        let s: String = json.call_method1("dumps", (paifu,))?.extract()?;
        let v: serde_json::Value = serde_json::from_str(&s)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse JSON: {}", e)))?;

        let (rounds_value, _rule) = if let Some(obj) = v.as_object() {
            if let Some(data) = obj.get("data") {
                // assume Paifu struct { header, data }
                // TODO: Parse header for rule if converting from Paifu
                (data.clone(), crate::rule::GameRule::default_mjsoul())
            } else {
                // maybe just dict of rounds? Unlikely given usage.
                return Err(PyValueError::new_err("Invalid dict format: missing 'data'"));
            }
        } else if v.is_array() {
            (v, crate::rule::GameRule::default_mjsoul())
        } else {
            return Err(PyValueError::new_err(
                "Invalid input format: expected dict or list",
            ));
        };
        let (rounds_raw, action_tsumogiri, action_dora_snapshots) =
            parse_raw_rounds_with_metadata(rounds_value).map_err(PyValueError::new_err)?;

        // Detect 3P from the first round's scores length
        let is_3p = rounds_raw
            .first()
            .and_then(|round| round.first())
            .and_then(|action| {
                if let RawAction::NewRound { scores, .. } = action {
                    Some(scores.len() == 3)
                } else {
                    None
                }
            })
            .unwrap_or(false);
        let rule = if is_3p {
            crate::rule::GameRule::default_mjsoul()
        } else {
            _rule
        };

        let mut rounds: Vec<LogKyoku> = Vec::with_capacity(rounds_raw.len());
        for ((r_raw, round_tsumogiri), round_dora_snapshots) in rounds_raw
            .into_iter()
            .zip(action_tsumogiri)
            .zip(action_dora_snapshots)
        {
            let mut kyoku =
                Self::kyoku_from_raw_actions(r_raw, round_tsumogiri, round_dora_snapshots)
                    .map_err(|error| PyValueError::new_err(format!("Invalid round: {error}")))?;
            if let Some(first) = rounds.first()
                && first.scores.len() != kyoku.scores.len()
            {
                return Err(PyValueError::new_err(format!(
                    "Invalid replay: player count changed from {} to {}",
                    first.scores.len(),
                    kyoku.scores.len()
                )));
            }
            kyoku.rule = rule;
            rounds.push(kyoku);
        }

        // Populate end_scores based on next round's start scores
        for i in 0..rounds.len().saturating_sub(1) {
            rounds[i].end_scores = rounds[i + 1].scores.clone();
        }

        // Calculate game end scores using the last round
        let is_3p = rounds.first().map(|r| r.scores.len() == 3).unwrap_or(false);

        let game_end_scores = if let Some(last) = rounds.last_mut() {
            if last.end_scores != last.scores {
                // ActionHule/NoTile supplied old_scores + delta_scores.
                Some(last.end_scores.clone())
            } else if is_3p {
                // For 3P, simulate using GameState3P
                let mut state = crate::state_3p::GameState3P::new(0, false, None, 0, last.rule);
                let initial_scores: [i32; 3] = last.scores.clone().try_into().unwrap_or([35000; 3]);
                let oya = last.ju % 3;
                let bakaze = match last.chang {
                    0 => crate::types::Wind::East,
                    1 => crate::types::Wind::South,
                    2 => crate::types::Wind::West,
                    3 => crate::types::Wind::North,
                    _ => crate::types::Wind::East,
                } as u8;
                state._initialize_round(
                    oya,
                    bakaze,
                    last.ben,
                    last.liqibang as u32,
                    None,
                    Some(initial_scores.to_vec()),
                );
                // Replace the randomly-dealt hands with the actual replay
                // hands so that tenpai detection in NoTile is correct.
                for (i, hand) in last.hands.iter().enumerate() {
                    if i < state.players.len() {
                        state.players[i].hand = hand.clone();
                        state.players[i].hand.sort();
                    }
                }
                for (action_index, action) in last.actions.iter().enumerate() {
                    let source_tsumogiri =
                        last.action_tsumogiri.get(action_index).copied().flatten();
                    let dora_snapshot = last
                        .action_dora_snapshots
                        .get(action_index)
                        .and_then(Option::as_deref);
                    state.apply_log_action_with_metadata(action, source_tsumogiri, dora_snapshot);
                }
                last.end_scores = state.players.iter().map(|p| p.score).collect();
                Some(last.end_scores.clone())
            } else {
                // 4P path
                let mut state = crate::state::GameState::new(0, false, None, 0, last.rule);
                let initial_scores: [i32; 4] = last.scores.clone().try_into().unwrap_or([25000; 4]);
                let oya = last.ju % 4;
                let bakaze = match last.chang {
                    0 => crate::types::Wind::East,
                    1 => crate::types::Wind::South,
                    2 => crate::types::Wind::West,
                    3 => crate::types::Wind::North,
                    _ => crate::types::Wind::East,
                } as u8;
                state._initialize_round(
                    oya,
                    bakaze,
                    last.ben,
                    last.liqibang as u32,
                    None,
                    Some(initial_scores.to_vec()),
                );
                // Replace the randomly-dealt hands with the actual replay
                // hands so that tenpai detection in NoTile is correct.
                for (i, hand) in last.hands.iter().enumerate() {
                    if i < state.players.len() {
                        state.players[i].hand = hand.clone();
                        state.players[i].hand.sort();
                    }
                }
                for (action_index, action) in last.actions.iter().enumerate() {
                    let source_tsumogiri =
                        last.action_tsumogiri.get(action_index).copied().flatten();
                    let dora_snapshot = last
                        .action_dora_snapshots
                        .get(action_index)
                        .and_then(Option::as_deref);
                    state.apply_log_action_with_metadata(action, source_tsumogiri, dora_snapshot);
                }
                last.end_scores = state.players.iter().map(|p| p.score).collect();
                Some(last.end_scores.clone())
            }
        } else {
            None
        };

        // Set game_end_scores for all rounds
        if let Some(ges) = game_end_scores {
            for r in &mut rounds {
                r.game_end_scores = Some(ges.clone());
            }
        }

        Ok(MjSoulReplay { rounds })
    }

    fn num_rounds(&self) -> usize {
        self.rounds.len()
    }

    fn take_kyokus(slf: Py<Self>, py: Python<'_>) -> PyResult<KyokuIterator> {
        let logs_len = slf.borrow(py).rounds.len();
        Ok(KyokuIterator {
            game: slf,
            index: 0,
            len: logs_len,
        })
    }

    fn verify(&self) -> (usize, usize) {
        let mut total_agari = 0;
        let mut total_mismatches = 0;

        for kyoku in &self.rounds {
            let mut iter = WinResultContextIterator::new(kyoku.clone());

            while let Some(ctx) = iter.do_next() {
                total_agari += 1;

                let sim_han = ctx.actual.han;
                let sim_fu = ctx.actual.fu;
                let sim_yaku = ctx.actual.yaku.clone();

                let exp_han = ctx.expected_han;
                let exp_fu = ctx.expected_fu;
                let exp_yaku = ctx.expected_yaku.clone();

                // IGNORED: 31 (Dora), 32 (Aka), 33 (Ura)
                let ignored = [31, 32, 33];
                let yakuman_ids: Vec<u32> = (35..51).collect();

                let mut sim_filtered: Vec<u32> = sim_yaku
                    .iter()
                    .filter(|y| !ignored.contains(y))
                    .cloned()
                    .collect();
                let mut exp_filtered: Vec<u32> = exp_yaku
                    .iter()
                    .filter(|y| !ignored.contains(y))
                    .cloned()
                    .collect();

                let mut normalized_exp_han = exp_han;
                let is_yakuman = exp_yaku.iter().any(|y| yakuman_ids.contains(y));
                if is_yakuman && exp_han < 13 {
                    normalized_exp_han = exp_han * 13;
                }

                let mut mismatch = false;
                sim_filtered.sort();
                exp_filtered.sort();

                if sim_filtered != exp_filtered {
                    mismatch = true;
                } else {
                    let sim_ignored_han =
                        sim_yaku.iter().filter(|y| ignored.contains(y)).count() as u32;
                    let exp_ignored_han =
                        exp_yaku.iter().filter(|y| ignored.contains(y)).count() as u32;
                    let expected_sim_han =
                        normalized_exp_han as i32 - exp_ignored_han as i32 + sim_ignored_han as i32;

                    if normalized_exp_han < 13 && sim_han as i32 != expected_sim_han {
                        if sim_han != normalized_exp_han {
                            mismatch = true;
                        }
                    } else if (sim_han >= 13) != (normalized_exp_han >= 13) {
                        mismatch = true;
                    }

                    if !mismatch && normalized_exp_han < 13 && sim_fu != exp_fu {
                        mismatch = true;
                    }
                }

                if mismatch {
                    total_mismatches += 1;
                    println!(
                        "Mismatch: seat={}, han=(sim={}, exp={}), fu=(sim={}, exp={})",
                        ctx.seat, sim_han, exp_han, sim_fu, exp_fu
                    );
                    println!("  Expected Yaku: {:?}", exp_yaku);
                    println!("  Actual Yaku: {:?}", sim_yaku);
                    println!("  Conditions: {:?}", ctx.conditions);
                }
            }
        }
        (total_agari, total_mismatches)
    }
}

#[cfg(feature = "python")]
impl MjSoulReplay {
    fn kyoku_from_raw_actions(
        raw_actions: Vec<RawAction>,
        action_tsumogiri: Vec<Option<bool>>,
        action_dora_snapshots: Vec<Option<Vec<String>>>,
    ) -> Result<LogKyoku, String> {
        if raw_actions.is_empty() {
            return Err("round contains no actions".to_string());
        }
        if raw_actions.len() != action_tsumogiri.len() {
            return Err(format!(
                "action metadata length {} does not match action length {}",
                action_tsumogiri.len(),
                raw_actions.len()
            ));
        }
        if raw_actions.len() != action_dora_snapshots.len() {
            return Err(format!(
                "dora metadata length {} does not match action length {}",
                action_dora_snapshots.len(),
                raw_actions.len()
            ));
        }
        let num_players = match &raw_actions[0] {
            RawAction::NewRound { scores, .. } if matches!(scores.len(), 3 | 4) => scores.len(),
            RawAction::NewRound { scores, .. } => {
                return Err(format!(
                    "NewRound scores must contain 3 or 4 players, got {}",
                    scores.len()
                ));
            }
            _ => return Err("round must start with NewRound".to_string()),
        };
        for (index, (action, private_doras)) in
            raw_actions.iter().zip(&action_dora_snapshots).enumerate()
        {
            if index > 0 && matches!(action, RawAction::NewRound { .. }) {
                return Err("round contains more than one NewRound action".to_string());
            }
            Self::validate_raw_action(action, num_players)?;
            if let Some(snapshot) = private_doras {
                validate_mjsoul_tiles(snapshot, "private action doras")?;
                validate_dora_snapshot_len(snapshot.len(), "private action doras")?;
            }
        }
        let mut scores = Vec::new();
        let mut doras = Vec::new();
        let mut hands = vec![Vec::new(); 4];
        let mut chang = 0;
        let mut ju = 0;
        let mut ben = 0;
        let mut liqibang = 0;
        let mut left_tile_count = if num_players == 3 { 55 } else { 70 };
        let mut ura_doras = Vec::new();
        let mut paishan = None;
        let mut recorded_end_scores = None;

        if let RawAction::NewRound {
            scores: s,
            doras: d_opt,
            dora_indicators,
            dora_marker,
            tiles0,
            tiles1,
            tiles2,
            tiles3,
            chang: c,
            ju: j,
            ben: b,
            honba,
            liqibang: l,
            left_tile_count: lc,
            ura_doras: ud,
            paishan: p,
        } = &raw_actions[0]
        {
            scores = s.clone();
            if let Some(da) = dora_indicators.as_ref().or(d_opt.as_ref()) {
                for v in da {
                    doras.push(TileConverter::parse_tile_136(v));
                }
            } else if let Some(dm) = dora_marker {
                doras.push(TileConverter::parse_tile_136(dm));
            }
            hands = vec![
                tiles0
                    .iter()
                    .map(|v| TileConverter::parse_tile_136(v))
                    .collect(),
                tiles1
                    .iter()
                    .map(|v| TileConverter::parse_tile_136(v))
                    .collect(),
                tiles2
                    .iter()
                    .map(|v| TileConverter::parse_tile_136(v))
                    .collect(),
                tiles3
                    .iter()
                    .map(|v| TileConverter::parse_tile_136(v))
                    .collect(),
            ];
            hands.truncate(scores.len());
            chang = *c;
            ju = *j;
            ben = b.or(*honba).unwrap_or(0);
            liqibang = *l;
            left_tile_count = lc.unwrap_or(if num_players == 3 { 55 } else { 70 });
            if let Some(uda) = ud {
                ura_doras = uda
                    .iter()
                    .map(|v| TileConverter::parse_tile_136(v))
                    .collect();
            }
            paishan = p.clone();
        }

        // Result records contain the only authoritative terminal score for
        // the final round (there is no following NewRound to copy it from).
        for action in &raw_actions {
            let score_fields = match action {
                RawAction::Hule {
                    old_scores,
                    delta_scores,
                    ..
                }
                | RawAction::NoTile {
                    old_scores,
                    delta_scores,
                } => Some((old_scores, delta_scores)),
                _ => None,
            };
            if let Some((old_scores, Some(delta_scores))) = score_fields {
                let base = old_scores.as_ref().unwrap_or(&scores);
                if base.len() != scores.len() || delta_scores.len() != scores.len() {
                    return Err(format!(
                        "result score vectors must contain {} players (old={}, delta={})",
                        scores.len(),
                        base.len(),
                        delta_scores.len()
                    ));
                }
                recorded_end_scores = Some(
                    base.iter()
                        .zip(delta_scores)
                        .map(|(score, delta)| {
                            score
                                .checked_add(*delta)
                                .ok_or_else(|| "result score overflows i32".to_string())
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
        }

        let mut actions = Vec::with_capacity(raw_actions.len());
        let mut parsed_action_dora_snapshots = Vec::with_capacity(raw_actions.len());
        let has_sequential_dora_updates =
            raw_actions
                .iter()
                .zip(&action_dora_snapshots)
                .any(|(action, private_snapshot)| {
                    private_snapshot.is_some()
                        || match action {
                            RawAction::DealTile {
                                doras, dora_marker, ..
                            } => !doras.is_empty() || dora_marker.is_some(),
                            RawAction::DiscardTile { doras, .. } => !doras.is_empty(),
                            RawAction::Dora { .. } => true,
                            _ => false,
                        }
                });
        // Some legacy inputs store the complete round-level indicator list in
        // NewRound while also carrying chronological action snapshots.  The
        // replay contract starts such streams from the initial marker, so the
        // cumulative-prefix validator must use that same baseline.
        let mut current_doras = if has_sequential_dora_updates {
            doras.first().copied().into_iter().collect()
        } else {
            doras.clone()
        };
        for (ma, private_doras) in raw_actions.into_iter().zip(action_dora_snapshots) {
            // Mahjong Soul's repeated `doras` field is a cumulative snapshot.
            // The legacy singular `dora_marker` compatibility field is not
            // present in the current public schema, so preserve its historical
            // incremental meaning by canonicalizing it to a cumulative
            // snapshot before the action reaches the state reducer.
            let deal_dora_update = match &ma {
                RawAction::DealTile {
                    doras, dora_marker, ..
                } => Some((doras.clone(), dora_marker.clone())),
                _ => None,
            };
            let discard_dora_snapshot = match &ma {
                RawAction::DiscardTile { doras, .. } if !doras.is_empty() => Some(doras.clone()),
                _ => None,
            };
            let explicit_dora = match &ma {
                RawAction::Dora { dora_marker } => Some(TileConverter::parse_tile_136(dora_marker)),
                _ => None,
            };

            let mut action = Self::parse_raw_action(ma);
            if let Some((snapshot, marker)) = deal_dora_update {
                let canonical = if !snapshot.is_empty() {
                    let parsed = snapshot
                        .iter()
                        .map(|tile| TileConverter::parse_tile_136(tile))
                        .collect::<Vec<_>>();
                    validate_cumulative_dora_snapshot(&current_doras, &parsed, "DealTile doras")?;
                    current_doras = parsed;
                    Some(current_doras.clone())
                } else if let Some(marker) = marker {
                    if current_doras.len() >= 5 {
                        return Err(
                            "legacy DealTile dora_marker would exceed five indicators".to_string()
                        );
                    }
                    current_doras.push(TileConverter::parse_tile_136(&marker));
                    Some(current_doras.clone())
                } else {
                    None
                };
                if let Action::DealTile { doras, .. } = &mut action {
                    *doras = canonical;
                }
            } else if let Some(snapshot) = discard_dora_snapshot {
                let parsed = snapshot
                    .iter()
                    .map(|tile| TileConverter::parse_tile_136(tile))
                    .collect::<Vec<_>>();
                validate_cumulative_dora_snapshot(&current_doras, &parsed, "DiscardTile doras")?;
                current_doras = parsed;
            } else if let Some(marker) = explicit_dora {
                if current_doras.len() >= 5 {
                    return Err("Dora action would exceed five indicators".to_string());
                }
                current_doras.push(marker);
            }
            let private_doras = match private_doras {
                Some(snapshot) => {
                    let parsed = snapshot
                        .iter()
                        .map(|tile| TileConverter::parse_tile_136(tile))
                        .collect::<Vec<_>>();
                    validate_cumulative_dora_snapshot(
                        &current_doras,
                        &parsed,
                        "private action doras",
                    )?;
                    Some(parsed)
                }
                None => None,
            };
            if let Some(snapshot) = &private_doras {
                current_doras.clone_from(snapshot);
            }
            actions.push(action);
            parsed_action_dora_snapshots.push(private_doras);
        }

        let end_scores = recorded_end_scores.unwrap_or_else(|| scores.clone());

        let mut wliqi = vec![false; scores.len()];
        for action in &actions {
            if let Action::DiscardTile { seat, is_wliqi, .. } = action
                && *is_wliqi
            {
                wliqi[*seat] = true;
            }
        }

        Ok(LogKyoku {
            scores,
            end_scores,
            doras,
            ura_doras,
            hands,
            chang,
            ju,
            ben,
            liqibang,
            left_tile_count,
            wliqi,
            paishan,
            actions: Arc::from(actions),
            action_tsumogiri: Arc::from(action_tsumogiri),
            action_dora_snapshots: Arc::from(parsed_action_dora_snapshots),
            rule: crate::rule::GameRule::default_mjsoul(),
            game_end_scores: None,
        })
    }

    fn validate_raw_action(action: &RawAction, num_players: usize) -> Result<(), String> {
        let validate_seat = |seat: usize, field: &str| {
            if seat < num_players {
                Ok(())
            } else {
                Err(format!(
                    "{field} seat {seat} is out of range for {num_players} players"
                ))
            }
        };

        match action {
            RawAction::NewRound {
                scores,
                doras,
                dora_indicators,
                dora_marker,
                tiles0,
                tiles1,
                tiles2,
                tiles3,
                ura_doras,
                chang,
                ju,
                ..
            } => {
                if scores.len() != num_players {
                    return Err(format!(
                        "NewRound scores must contain {num_players} players, got {}",
                        scores.len()
                    ));
                }
                if usize::from(*ju) >= num_players {
                    return Err(format!(
                        "NewRound ju {ju} is out of range for {num_players} players"
                    ));
                }
                if *chang > 3 {
                    return Err(format!(
                        "NewRound chang must be between 0 and 3, got {chang}"
                    ));
                }
                for (field, tiles) in [
                    ("NewRound tiles0", tiles0),
                    ("NewRound tiles1", tiles1),
                    ("NewRound tiles2", tiles2),
                    ("NewRound tiles3", tiles3),
                ] {
                    validate_mjsoul_tiles(tiles, field)?;
                }
                if let Some(tiles) = doras {
                    validate_mjsoul_tiles(tiles, "NewRound doras")?;
                    validate_dora_snapshot_len(tiles.len(), "NewRound doras")?;
                }
                if let Some(tiles) = dora_indicators {
                    validate_mjsoul_tiles(tiles, "NewRound dora_indicators")?;
                    validate_dora_snapshot_len(tiles.len(), "NewRound dora_indicators")?;
                }
                if let Some(tile) = dora_marker {
                    validate_mjsoul_tile(tile, "NewRound dora_marker")?;
                }
                if let Some(tiles) = ura_doras {
                    validate_mjsoul_tiles(tiles, "NewRound ura_doras")?;
                }
            }
            RawAction::DiscardTile {
                seat, tile, doras, ..
            } => {
                validate_seat(*seat, "DiscardTile")?;
                validate_mjsoul_tile(tile, "DiscardTile tile")?;
                validate_mjsoul_tiles(doras, "DiscardTile doras")?;
                validate_dora_snapshot_len(doras.len(), "DiscardTile doras")?;
            }
            RawAction::DealTile {
                seat,
                tile,
                doras,
                dora_marker,
                ..
            } => {
                validate_seat(*seat, "DealTile")?;
                validate_mjsoul_tile(tile, "DealTile tile")?;
                validate_mjsoul_tiles(doras, "DealTile doras")?;
                validate_dora_snapshot_len(doras.len(), "DealTile doras")?;
                if let Some(marker) = dora_marker {
                    validate_mjsoul_tile(marker, "DealTile dora_marker")?;
                }
            }
            RawAction::AnGangAddGang {
                seat,
                meld_type,
                tiles,
            } => {
                validate_seat(*seat, "AnGangAddGang")?;
                if !matches!(*meld_type, 2 | 3) {
                    return Err(format!("unsupported AnGangAddGang type {meld_type}"));
                }
                validate_mjsoul_tile(tiles, "AnGangAddGang tiles")?;
            }
            RawAction::ChiPengGang {
                seat,
                meld_type,
                tiles,
                froms,
            } => {
                validate_seat(*seat, "ChiPengGang")?;
                if !matches!(*meld_type, 0..=3) {
                    return Err(format!("unsupported ChiPengGang type {meld_type}"));
                }
                let expected_tiles = if *meld_type < 2 { 3 } else { 4 };
                if tiles.len() != expected_tiles || froms.len() != expected_tiles {
                    return Err(format!(
                        "ChiPengGang type {meld_type} requires {expected_tiles} tiles/froms, got {}/{}",
                        tiles.len(),
                        froms.len()
                    ));
                }
                for &from in froms {
                    validate_seat(from, "ChiPengGang from")?;
                }
                validate_mjsoul_tiles(tiles, "ChiPengGang tiles")?;
            }
            RawAction::Hule { hules, .. } => {
                if hules.is_empty() {
                    return Err("Hule must contain at least one winner".to_string());
                }
                for hule in hules {
                    validate_seat(hule.seat, "Hule")?;
                    validate_mjsoul_tile(&hule.hu_tile, "Hule hu_tile")?;
                    validate_mjsoul_tiles(&hule.hand, "Hule hand")?;
                    if let Some(tiles) = &hule.ura_dora_indicators {
                        validate_mjsoul_tiles(tiles, "Hule ura_dora_indicators")?;
                    }
                    if let Some(tiles) = &hule.li_doras {
                        validate_mjsoul_tiles(tiles, "Hule li_doras")?;
                    }
                }
            }
            RawAction::Dora { dora_marker } => {
                validate_mjsoul_tile(dora_marker, "Dora dora_marker")?;
            }
            RawAction::BaBei { seat, doras, .. } => {
                if num_players != 3 {
                    return Err("BaBei is only valid in a three-player round".to_string());
                }
                validate_seat(*seat, "BaBei")?;
                validate_mjsoul_tiles(doras, "BaBei doras")?;
                validate_dora_snapshot_len(doras.len(), "BaBei doras")?;
            }
            RawAction::LiuJu { seat, tiles, .. } => {
                validate_seat(*seat, "LiuJu")?;
                validate_mjsoul_tiles(tiles, "LiuJu tiles")?;
            }
            RawAction::NoTile { .. } | RawAction::Other => {}
        }
        Ok(())
    }

    fn parse_raw_action(ma: RawAction) -> Action {
        match ma {
            RawAction::DiscardTile {
                seat,
                tile,
                is_liqi,
                is_wliqi,
                doras,
            } => Action::DiscardTile {
                seat,
                tile: TileConverter::parse_tile_136(&tile),
                is_liqi,
                is_wliqi,
                doras: if doras.is_empty() {
                    None
                } else {
                    Some(
                        doras
                            .iter()
                            .map(|v| TileConverter::parse_tile_136(v))
                            .collect(),
                    )
                },
            },
            RawAction::DealTile {
                seat,
                tile,
                doras,
                dora_marker,
                left_tile_count,
            } => {
                let mut d_res = if doras.is_empty() {
                    None
                } else {
                    Some(
                        doras
                            .iter()
                            .map(|v| TileConverter::parse_tile_136(v))
                            .collect(),
                    )
                };
                if d_res.is_none()
                    && let Some(dm) = dora_marker
                {
                    d_res = Some(vec![TileConverter::parse_tile_136(&dm)]);
                }
                Action::DealTile {
                    seat,
                    tile: TileConverter::parse_tile_136(&tile),
                    doras: d_res,
                    left_tile_count,
                }
            }
            RawAction::ChiPengGang {
                seat,
                meld_type,
                tiles,
                froms,
            } => {
                let m_type = match meld_type {
                    0 => MeldType::Chi,
                    1 => MeldType::Pon,
                    2 => MeldType::Daiminkan,
                    3 => MeldType::Ankan,
                    _ => MeldType::Chi,
                };
                Action::ChiPengGang {
                    seat,
                    meld_type: m_type,
                    tiles: tiles
                        .iter()
                        .map(|v| TileConverter::parse_tile_136(v))
                        .collect(),
                    froms,
                }
            }
            RawAction::AnGangAddGang {
                seat,
                meld_type,
                tiles,
            } => {
                let m_type = if meld_type == 3 {
                    MeldType::Ankan
                } else {
                    MeldType::Kakan
                };
                let tile_raw_id = TileConverter::parse_tile_34(&tiles).0;
                Action::AnGangAddGang {
                    seat,
                    meld_type: m_type,
                    tiles: vec![TileConverter::parse_tile_136(&tiles)],
                    tile_raw_id,
                    doras: None, // Will be updated by Dora action or DealTile
                }
            }
            RawAction::Hule { hules, .. } => {
                let hules_typed = hules
                    .into_iter()
                    .map(|h| HuleData {
                        seat: h.seat,
                        hu_tile: TileConverter::parse_tile_136(&h.hu_tile),
                        zimo: h.zimo,
                        count: h.count,
                        fu: h.fu,
                        fans: h.fans.iter().filter(|f| f.val > 0).map(|f| f.id).collect(),
                        fan_values: h.fans.iter().map(|f| (f.id, f.val)).collect(),
                        li_doras: h
                            .ura_dora_indicators
                            .or(h.li_doras)
                            .map(|a| a.iter().map(|v| TileConverter::parse_tile_136(v)).collect()),
                        yiman: h.yiman,
                        point_rong: h.point_rong,
                        point_zimo_qin: h.point_zimo_qin,
                        point_zimo_xian: h.point_zimo_xian,
                        riichi_sticks: None,
                    })
                    .collect();
                Action::Hule { hules: hules_typed }
            }
            RawAction::Dora { dora_marker } => Action::Dora {
                dora_marker: TileConverter::parse_tile_136(&dora_marker),
            },
            RawAction::NoTile { .. } => Action::NoTile,
            RawAction::BaBei {
                seat,
                moqie,
                doras: _,
            } => Action::BaBei { seat, moqie },
            RawAction::LiuJu {
                lj_type,
                seat,
                tiles,
            } => Action::LiuJu {
                lj_type,
                seat,
                tiles: tiles
                    .iter()
                    .map(|v| TileConverter::parse_tile_136(v))
                    .collect(),
            },
            _ => Action::Other("Other".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RawAction;
    #[cfg(feature = "python")]
    use super::{MjSoulReplay, parse_raw_rounds_with_metadata};
    #[cfg(feature = "python")]
    use crate::replay::{Action, TileConverter};
    #[cfg(feature = "python")]
    use crate::{rule::GameRule, state_3p::GameState3P};
    use serde_json::json;

    #[test]
    fn absent_result_score_fields_do_not_add_nulls_to_the_wire_format() {
        let hule = serde_json::to_value(RawAction::Hule {
            hules: Vec::new(),
            old_scores: None,
            delta_scores: None,
        })
        .unwrap();
        let no_tile = serde_json::to_value(RawAction::NoTile {
            old_scores: None,
            delta_scores: None,
        })
        .unwrap();

        assert_eq!(hule, json!({"name": "Hule", "data": {"hules": []}}));
        assert_eq!(no_tile, json!({"name": "NoTile", "data": {}}));
    }

    #[cfg(feature = "python")]
    #[test]
    fn discard_moqie_is_retained_in_the_private_action_sidecar() {
        let rounds = json!([[{
            "name": "NewRound",
            "data": {
                "scores": [25000, 25000, 25000, 25000],
                "doras": ["1m"],
                "tiles0": [], "tiles1": [], "tiles2": [], "tiles3": [],
                "chang": 0, "ju": 0, "ben": 0, "liqibang": 0,
                "left_tile_count": 70
            }
        }, {
            "name": "DiscardTile",
            "data": {
                "seat": 0, "tile": "5p", "is_liqi": false,
                "is_wliqi": false, "moqie": true, "doras": []
            }
        }]]);

        let (raw_rounds, metadata, dora_metadata) = parse_raw_rounds_with_metadata(rounds).unwrap();
        assert!(matches!(
            raw_rounds[0][1],
            RawAction::DiscardTile { seat: 0, .. }
        ));
        assert_eq!(metadata[0], vec![None, Some(true)]);

        let kyoku = MjSoulReplay::kyoku_from_raw_actions(
            raw_rounds.into_iter().next().unwrap(),
            metadata.into_iter().next().unwrap(),
            dora_metadata.into_iter().next().unwrap(),
        )
        .unwrap();
        assert_eq!(&*kyoku.action_tsumogiri, &[None, Some(true)]);
    }

    #[cfg(feature = "python")]
    #[test]
    fn legacy_singular_dora_marker_is_canonicalized_as_incremental() {
        let rounds = json!([[{
            "name": "NewRound",
            "data": {
                "scores": [25000, 25000, 25000, 25000],
                "doras": ["1p", "2p", "3p"],
                "tiles0": [], "tiles1": [], "tiles2": [], "tiles3": [],
                "chang": 0, "ju": 0, "ben": 0, "liqibang": 0,
                "left_tile_count": 70
            }
        }, {
            "name": "DealTile",
            "data": {
                "seat": 0, "tile": "1m", "doras": [],
                "dora_marker": "2p", "left_tile_count": 69
            }
        }, {
            "name": "DealTile",
            "data": {
                "seat": 1, "tile": "2m", "doras": ["1p", "2p", "3p"],
                "left_tile_count": 68
            }
        }]]);

        let (raw_rounds, metadata, dora_metadata) = parse_raw_rounds_with_metadata(rounds).unwrap();
        let kyoku = MjSoulReplay::kyoku_from_raw_actions(
            raw_rounds.into_iter().next().unwrap(),
            metadata.into_iter().next().unwrap(),
            dora_metadata.into_iter().next().unwrap(),
        )
        .unwrap();
        let expected_incremental = vec![
            TileConverter::parse_tile_136("1p"),
            TileConverter::parse_tile_136("2p"),
        ];
        let expected_snapshot = vec![
            TileConverter::parse_tile_136("1p"),
            TileConverter::parse_tile_136("2p"),
            TileConverter::parse_tile_136("3p"),
        ];
        assert!(matches!(
            &kyoku.actions[1],
            Action::DealTile {
                doras: Some(doras),
                ..
            } if *doras == expected_incremental
        ));
        assert!(matches!(
            &kyoku.actions[2],
            Action::DealTile {
                doras: Some(doras),
                ..
            } if *doras == expected_snapshot
        ));
    }

    #[cfg(feature = "python")]
    #[test]
    fn round_level_complete_doras_accept_chronological_initial_snapshot() {
        let rounds = json!([[{
            "name": "NewRound",
            "data": {
                "scores": [25000, 25000, 25000, 25000],
                "doras": ["1p", "2p"],
                "tiles0": [], "tiles1": [], "tiles2": [], "tiles3": [],
                "chang": 0, "ju": 0, "ben": 0, "liqibang": 0,
                "left_tile_count": 70
            }
        }, {
            "name": "DealTile",
            "data": {
                "seat": 0, "tile": "1m", "doras": ["1p"],
                "left_tile_count": 69
            }
        }, {
            "name": "DealTile",
            "data": {
                "seat": 1, "tile": "2m", "doras": ["1p", "2p"],
                "left_tile_count": 68
            }
        }]]);

        let (raw_rounds, metadata, dora_metadata) = parse_raw_rounds_with_metadata(rounds).unwrap();
        let kyoku = MjSoulReplay::kyoku_from_raw_actions(
            raw_rounds.into_iter().next().unwrap(),
            metadata.into_iter().next().unwrap(),
            dora_metadata.into_iter().next().unwrap(),
        )
        .unwrap();
        let first = vec![TileConverter::parse_tile_136("1p")];
        let second = vec![
            TileConverter::parse_tile_136("1p"),
            TileConverter::parse_tile_136("2p"),
        ];
        assert!(matches!(
            &kyoku.actions[1],
            Action::DealTile {
                doras: Some(doras),
                ..
            } if *doras == first
        ));
        assert!(matches!(
            &kyoku.actions[2],
            Action::DealTile {
                doras: Some(doras),
                ..
            } if *doras == second
        ));
    }

    #[cfg(feature = "python")]
    #[test]
    fn cumulative_dora_snapshots_reject_shrink_and_prefix_rewrite() {
        for snapshot in [json!(["1p"]), json!(["3p", "2p"])] {
            let rounds = json!([[{
                "name": "NewRound",
                "data": {
                    "scores": [25000, 25000, 25000, 25000],
                    "doras": ["1p"],
                    "tiles0": [], "tiles1": [], "tiles2": [], "tiles3": [],
                    "chang": 0, "ju": 0, "ben": 0, "liqibang": 0,
                    "left_tile_count": 70
                }
            }, {
                "name": "DealTile",
                "data": {
                    "seat": 0, "tile": "1m", "doras": ["1p", "2p"],
                    "left_tile_count": 69
                }
            }, {
                "name": "DealTile",
                "data": {
                    "seat": 1, "tile": "2m", "doras": snapshot,
                    "left_tile_count": 68
                }
            }]]);

            let (raw_rounds, metadata, dora_metadata) =
                parse_raw_rounds_with_metadata(rounds).unwrap();
            let result = MjSoulReplay::kyoku_from_raw_actions(
                raw_rounds.into_iter().next().unwrap(),
                metadata.into_iter().next().unwrap(),
                dora_metadata.into_iter().next().unwrap(),
            );
            let error = match result {
                Ok(_) => panic!("malformed cumulative dora snapshot was accepted"),
                Err(error) => error,
            };
            assert!(error.contains("complete prefix"), "{error}");
        }
    }

    #[cfg(feature = "python")]
    #[test]
    fn private_kan_and_kita_dora_snapshots_reach_the_state_reducer() {
        let rounds = json!([[{
            "name": "NewRound",
            "data": {
                "scores": [35000, 35000, 35000],
                "doras": ["1p"],
                "tiles0": ["1s", "1s", "1s", "1s", "4z"],
                "tiles1": [], "tiles2": [], "tiles3": [],
                "chang": 0, "ju": 0, "ben": 0, "liqibang": 0,
                "left_tile_count": 55
            }
        }, {
            "name": "AnGangAddGang",
            "data": {
                "seat": 0, "type": 3, "tiles": "1s",
                "doras": ["1p", "2p"]
            }
        }, {
            "name": "BaBei",
            "data": {
                "seat": 0, "moqie": false,
                "doras": ["1p", "2p", "3p"]
            }
        }]]);

        let (raw_rounds, tsumogiri, dora_metadata) =
            parse_raw_rounds_with_metadata(rounds).unwrap();
        let kyoku = MjSoulReplay::kyoku_from_raw_actions(
            raw_rounds.into_iter().next().unwrap(),
            tsumogiri.into_iter().next().unwrap(),
            dora_metadata.into_iter().next().unwrap(),
        )
        .unwrap();
        assert_eq!(kyoku.action_dora_snapshots[0], None);
        assert_eq!(kyoku.action_dora_snapshots[1].as_deref().unwrap().len(), 2);
        assert_eq!(kyoku.action_dora_snapshots[2].as_deref().unwrap().len(), 3);
        assert_eq!(kyoku.initial_doras_for_action_replay().len(), 1);

        let mut state = GameState3P::new(0, false, None, 0, GameRule::default_mjsoul());
        state.wall.dora_indicators = kyoku.initial_doras_for_action_replay();
        for (index, action) in kyoku.actions.iter().enumerate().skip(1) {
            state.apply_log_action_with_metadata(
                action,
                kyoku.action_tsumogiri[index],
                kyoku.action_dora_snapshots[index].as_deref(),
            );
            assert_eq!(state.wall.dora_indicators.len(), index + 1);
        }
    }
}
