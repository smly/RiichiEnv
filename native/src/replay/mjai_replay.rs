use flate2::read::GzDecoder;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;

use crate::replay::{Action, HuleData, Kyoku, TileConverter};
use crate::types::MeldType;

#[pyclass]
pub struct MjaiReplay {
    pub rounds: Vec<Kyoku>,
}

#[derive(Debug)]
#[pyclass]
pub struct KyokuIterator {
    game: Py<MjaiReplay>,
    index: usize,
    len: usize,
}

#[pymethods]
impl KyokuIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<Kyoku> {
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

// MJAI Event Definitions
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(tag = "type")]
pub enum MjaiEvent {
    #[serde(rename = "start_game")]
    StartGame {
        names: Option<Vec<String>>,
        id: Option<String>,
    },
    #[serde(rename = "start_kyoku")]
    StartKyoku {
        bakaze: String,
        kyoku: u8,
        honba: u8,
        #[serde(alias = "kyotaku")]
        kyoutaku: u8,
        oya: u8,
        scores: Vec<i32>,
        dora_marker: String,
        tehais: Vec<Vec<String>>,
    },
    #[serde(rename = "tsumo")]
    Tsumo { actor: usize, pai: String },
    #[serde(rename = "dahai")]
    Dahai {
        actor: usize,
        pai: String,
        tsumogiri: bool,
    },
    #[serde(rename = "pon")]
    Pon {
        actor: usize,
        target: usize,
        pai: String,
        consumed: Vec<String>,
    },
    #[serde(rename = "chi")]
    Chi {
        actor: usize,
        target: usize,
        pai: String,
        consumed: Vec<String>,
    },
    #[serde(rename = "kan")]
    Kan {
        actor: usize,
        target: usize,
        pai: String,
        consumed: Vec<String>,
    },
    #[serde(rename = "kakan")]
    Kakan { actor: usize, pai: String },
    #[serde(rename = "ankan")]
    Ankan { actor: usize, consumed: Vec<String> },
    #[serde(rename = "dora")]
    Dora { dora_marker: String },
    #[serde(rename = "reach")]
    Reach { actor: usize },
    #[serde(rename = "reach_accepted")]
    ReachAccepted { actor: usize },
    #[serde(rename = "hora")]
    Hora {
        actor: usize,
        target: usize,
        pai: Option<String>, // Winning tile (optional in some logs)
        uradora_markers: Option<Vec<String>>,
        #[serde(default)]
        yaku: Option<Vec<(String, u32)>>, // List of [yaku_name, han_value]
        fu: Option<u32>,
        han: Option<u32>,
        #[serde(default)]
        scores: Option<Vec<i32>>, // Scores AFTER hor
        #[serde(alias = "deltas")]
        delta: Option<Vec<i32>>,
    },
    #[serde(rename = "ryukyoku")]
    Ryukyoku {
        reason: Option<String>,
        tehais: Option<Vec<Vec<String>>>, // Revealed hands
        #[serde(alias = "deltas")]
        delta: Option<Vec<i32>>,
        scores: Option<Vec<i32>>,
    },
    #[serde(rename = "end_game")]
    EndGame,
    #[serde(rename = "end_kyoku")]
    EndKyoku,
    #[serde(other)]
    Other,
}

// State for building a Kyoku from MJAI stream
struct KyokuBuilder {
    actions: Vec<Action>,
    scores: Vec<i32>,
    end_scores: Vec<i32>,
    hands: Vec<Vec<u8>>,
    doras: Vec<u8>,
    chang: u8,
    ju: u8,
    ben: u8,
    liqibang: u8,
    left_tile_count: u8,
    ura_doras: Vec<u8>,

    // Internal tracking
    liqi_flags: Vec<bool>, // Who has declared reach (to set `is_liqi` on discard)
    wliqi_flags: Vec<bool>, // Who has effectively achieved Double Riichi
    reach_accepted: Vec<bool>,
    first_discard: Vec<bool>,
    has_calls: bool,
}

impl KyokuBuilder {
    fn new(
        bakaze: String,
        kyoku: u8,
        honba: u8,
        kyoutaku: u8,
        scores: Vec<i32>,
        dora_marker: String,
        tehais: Vec<Vec<String>>,
    ) -> Self {
        let chang = match bakaze.as_str() {
            "S" => 1,
            "W" => 2,
            "N" => 3,
            _ => 0, // "E" or default
        };
        let ju = kyoku - 1;

        let mut hands = vec![Vec::new(); 4];
        for (i, tehai_strs) in tehais.iter().enumerate() {
            if i < 4 {
                hands[i] = tehai_strs
                    .iter()
                    .map(|s| TileConverter::parse_tile_136(s))
                    .collect();
            }
        }

        let first_dora = TileConverter::parse_tile_136(&dora_marker);
        let end_scores = scores.clone();

        KyokuBuilder {
            actions: Vec::new(),
            scores,
            end_scores,
            hands,
            doras: vec![first_dora],
            chang,
            ju,
            ben: honba,
            liqibang: kyoutaku,
            left_tile_count: 70, // Standard starting count?
            ura_doras: Vec::new(),
            liqi_flags: vec![false; 4],
            wliqi_flags: vec![false; 4],
            reach_accepted: vec![false; 4],
            first_discard: vec![true; 4],
            has_calls: false,
        }
    }

    fn build(self) -> Kyoku {
        Kyoku {
            scores: self.scores,
            end_scores: self.end_scores,
            doras: self.doras,
            ura_doras: self.ura_doras,
            hands: self.hands,
            chang: self.chang,
            ju: self.ju,
            ben: self.ben,
            liqibang: self.liqibang,
            left_tile_count: self.left_tile_count,
            wliqi: self.wliqi_flags,
            paishan: None, // MJAI usually doesn't have full paishan
            actions: Arc::from(self.actions),
        }
    }
}

#[pymethods]
impl MjaiReplay {
    #[staticmethod]
    pub fn from_jsonl(path: String) -> PyResult<Self> {
        let is_gzip = path.ends_with(".gz");
        let file = File::open(&path)
            .map_err(|e| PyValueError::new_err(format!("Failed to open file: {}", e)))?;

        // Setup reader
        let reader: Box<dyn BufRead> = if is_gzip {
            let decoder = GzDecoder::new(file);
            Box::new(BufReader::new(decoder))
        } else {
            Box::new(BufReader::new(file))
        };

        let mut rounds = Vec::new();
        let mut builder: Option<KyokuBuilder> = None;

        for line in reader.lines() {
            let line = line.map_err(|e| PyValueError::new_err(format!("Read error: {}", e)))?;
            if line.trim().is_empty() {
                continue;
            }
            let event: MjaiEvent = serde_json::from_str(&line)
                .map_err(|e| PyValueError::new_err(format!("Parse error: {}", e)))?;

            match event {
                MjaiEvent::StartKyoku {
                    bakaze,
                    kyoku,
                    honba,
                    kyoutaku,
                    scores,
                    dora_marker,
                    tehais,
                    ..
                } => {
                    // Start new Kyoku
                    if let Some(b) = builder.take() {
                        rounds.push(b.build());
                    }
                    builder = Some(KyokuBuilder::new(
                        bakaze,
                        kyoku,
                        honba,
                        kyoutaku,
                        scores,
                        dora_marker,
                        tehais,
                    ));
                }
                MjaiEvent::EndKyoku | MjaiEvent::EndGame => {
                    if let Some(b) = builder.take() {
                        rounds.push(b.build());
                    }
                }
                _ => {
                    if let Some(ref mut b) = builder {
                        Self::process_event(b, event);
                    }
                }
            }
        }

        // Final flush if unexpected end
        if let Some(b) = builder.take() {
            rounds.push(b.build());
        }

        Ok(MjaiReplay { rounds })
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
}

impl MjaiReplay {
    fn process_event(builder: &mut KyokuBuilder, event: MjaiEvent) {
        match event {
            MjaiEvent::Tsumo { actor, pai } => {
                let tile = TileConverter::parse_tile_136(&pai);
                builder.actions.push(Action::DealTile {
                    seat: actor,
                    tile,
                    doras: None,
                    left_tile_count: None, // Could decrement builder.left_tile_count?
                });
                if builder.left_tile_count > 0 {
                    builder.left_tile_count -= 1;
                }
            }
            MjaiEvent::Dahai {
                actor,
                pai,
                tsumogiri: _,
            } => {
                let tile = TileConverter::parse_tile_136(&pai);
                let is_liqi = builder.liqi_flags[actor];

                let is_wliqi = is_liqi && builder.first_discard[actor] && !builder.has_calls;
                if is_wliqi {
                    builder.wliqi_flags[actor] = true;
                }

                builder.actions.push(Action::DiscardTile {
                    seat: actor,
                    tile,
                    is_liqi,
                    is_wliqi,
                    doras: None,
                });

                builder.first_discard[actor] = false;

                if is_liqi {
                    builder.liqibang += 1;
                    builder.liqi_flags[actor] = false;
                }
            }
            MjaiEvent::Reach { actor } => {
                builder.liqi_flags[actor] = true;
            }
            MjaiEvent::ReachAccepted { actor } => {
                builder.reach_accepted[actor] = true;
            }
            MjaiEvent::Chi {
                actor,
                pai,
                consumed,
                ..
            } => {
                builder.has_calls = true;
                let mut tiles = vec![TileConverter::parse_tile_136(&pai)];
                for c in consumed {
                    tiles.push(TileConverter::parse_tile_136(&c));
                }
                builder.actions.push(Action::ChiPengGang {
                    seat: actor,
                    meld_type: MeldType::Chi,
                    tiles,
                    froms: vec![],
                });
            }
            MjaiEvent::Pon {
                actor,
                pai,
                consumed,
                ..
            } => {
                builder.has_calls = true;
                let mut tiles = vec![TileConverter::parse_tile_136(&pai)];
                for c in consumed {
                    tiles.push(TileConverter::parse_tile_136(&c));
                }
                builder.actions.push(Action::ChiPengGang {
                    seat: actor,
                    meld_type: MeldType::Peng,
                    tiles,
                    froms: vec![],
                });
            }
            MjaiEvent::Kan {
                actor,
                pai,
                consumed,
                ..
            } => {
                builder.has_calls = true;
                // Daiminkan
                let mut tiles = vec![TileConverter::parse_tile_136(&pai)];
                for c in consumed {
                    tiles.push(TileConverter::parse_tile_136(&c));
                }
                builder.actions.push(Action::ChiPengGang {
                    seat: actor,
                    meld_type: MeldType::Gang,
                    tiles,
                    froms: vec![],
                });
            }
            MjaiEvent::Ankan { actor, consumed } => {
                builder.has_calls = true;
                let tiles: Vec<u8> = consumed
                    .iter()
                    .map(|s| TileConverter::parse_tile_136(s))
                    .collect();
                builder.actions.push(Action::AnGangAddGang {
                    seat: actor,
                    meld_type: MeldType::Angang,
                    tiles,
                    tile_raw_id: 0,
                    doras: None,
                });
            }
            MjaiEvent::Kakan { actor, pai } => {
                builder.has_calls = true;
                let tile = TileConverter::parse_tile_136(&pai);
                builder.actions.push(Action::AnGangAddGang {
                    seat: actor,
                    meld_type: MeldType::Addgang,
                    tiles: vec![tile],
                    tile_raw_id: 0,
                    doras: None,
                });
            }
            MjaiEvent::Dora { dora_marker } => {
                let marker = TileConverter::parse_tile_136(&dora_marker);
                builder.doras.push(marker);
                builder.actions.push(Action::Dora {
                    dora_marker: marker,
                });
            }
            MjaiEvent::Hora {
                actor,
                target,
                pai,
                uradora_markers,
                yaku: _,
                fu,
                han,
                scores,
                delta,
            } => {
                let hu_tile_id = if let Some(p) = pai {
                    TileConverter::parse_tile_136(&p)
                } else {
                    // Try to infer from last action
                    // If Tsumo (actor == target), last action should be DealTile for actor
                    // If Ron (actor != target), last action should be DiscardTile
                    // Simplifying assumption: look at last action
                    if let Some(last_action) = builder.actions.last() {
                        match last_action {
                            Action::DealTile { tile, .. } => *tile,
                            Action::DiscardTile { tile, .. } => *tile,
                            // Check for AddGang too (Chankan)?
                            Action::AnGangAddGang { tiles, .. } => tiles[0], // AddGang
                            _ => 0, // Fallback, though ideally shouldn't happen
                        }
                    } else {
                        0
                    }
                };

                let mut hule_data = HuleData {
                    seat: actor,
                    hu_tile: hu_tile_id,
                    zimo: actor == target, // If actor is target, it's Tsumo
                    count: han.unwrap_or(0),
                    fu: fu.unwrap_or(0),
                    fans: Vec::new(),
                    li_doras: None,
                    yiman: false,
                    point_rong: 0,
                    point_zimo_qin: 0,
                    point_zimo_xian: 0,
                };

                if let Some(uras) = uradora_markers {
                    let ud: Vec<u8> = uras
                        .iter()
                        .map(|s| TileConverter::parse_tile_136(s))
                        .collect();
                    builder.ura_doras = ud.clone();
                    hule_data.li_doras = Some(ud);
                }

                // Update end_scores
                if let Some(s) = scores {
                    builder.end_scores = s;
                } else if let Some(d) = delta {
                    for (i, val) in d.iter().enumerate() {
                        if i < builder.end_scores.len() {
                            let riichi_cost = if builder.reach_accepted[i] { 1000 } else { 0 };
                            builder.end_scores[i] = builder.scores[i] + val - riichi_cost;
                        }
                    }
                }

                builder.actions.push(Action::Hule {
                    hules: vec![hule_data],
                });
            }
            MjaiEvent::Ryukyoku { delta, scores, .. } => {
                if let Some(s) = scores {
                    builder.end_scores = s;
                } else if let Some(d) = delta {
                    for (i, val) in d.iter().enumerate() {
                        if i < builder.end_scores.len() {
                            let riichi_cost = if builder.reach_accepted[i] { 1000 } else { 0 };
                            builder.end_scores[i] = builder.scores[i] + val - riichi_cost;
                        }
                    }
                }
                builder.actions.push(Action::NoTile);
            }
            _ => {}
        }
    }
}

/*
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    // Test disabled due to linking issues in CI environment (missing python symbols).
    // To run locally, ensure binding env is set up.
    // #[test]
    fn test_mjai_parsing() {
        let json_data = r#"
{"type":"start_game"}
{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyoutaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["1s","1s","1s","2s","3s","4s","5s","6s","7s","8s","9s","9s","9s"],["1s","1s","1s","2s","3s","4s","5s","6s","7s","8s","9s","9s","9s"],["1s","1s","1s","2s","3s","4s","5s","6s","7s","8s","9s","9s","9s"],["1s","1s","1s","2s","3s","4s","5s","6s","7s","8s","9s","9s","9s"]]}
{"type":"tsumo","actor":0,"pai":"2m"}
{"type":"dahai","actor":0,"pai":"2m","tsumogiri":false}
{"type":"ryukyoku","reason":"fanpai"}
{"type":"end_kyoku"}
{"type":"end_game"}
"#;
        let mut path = std::env::temp_dir();
        path.push("test_mjai.jsonl");
        let mut file = File::create(&path).unwrap();
        writeln!(file, "{}", json_data.trim()).unwrap();

        let path_str = path.to_str().unwrap().to_string();

        let replay = MjaiReplay::from_jsonl(path_str.clone()).expect("Failed to parse MJAI");
        assert_eq!(replay.rounds.len(), 1);
        let kyoku = &replay.rounds[0];
        assert_eq!(kyoku.actions.len(), 3);

        // ... (assertions)

        let _ = std::fs::remove_file(path);
    }
}
*/
