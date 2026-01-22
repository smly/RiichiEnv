use flate2::read::GzDecoder;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, Read};
use std::sync::Arc;

use crate::replay::{Action, AgariContextIterator, HuleData, Kyoku, TileConverter};
use crate::types::MeldType;

#[pyclass]
pub struct MjSoulReplay {
    pub rounds: Vec<Kyoku>,
}

#[derive(Debug)]
#[pyclass]
pub struct KyokuIterator {
    game: Py<MjSoulReplay>,
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
    Hule { hules: Vec<HuleDataRaw> },
    #[serde(rename = "dora")]
    Dora { dora_marker: String },
    #[serde(rename = "NoTile")]
    NoTile {},
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

        let log: GameLog = serde_json::from_slice(&buffer)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse JSON: {}", e)))?;

        let mut rounds = Vec::with_capacity(log.rounds.len());
        for r_raw in log.rounds {
            rounds.push(Self::kyoku_from_raw_actions(r_raw));
        }

        // Populate end_scores based on next round's start scores
        for i in 0..rounds.len().saturating_sub(1) {
            rounds[i].end_scores = rounds[i + 1].scores.clone();
        }

        Ok(MjSoulReplay { rounds })
    }

    #[staticmethod]
    fn from_paifu(json_str: String) -> PyResult<Self> {
        let rounds_raw: Vec<Vec<RawAction>> = serde_json::from_str(&json_str)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse JSON: {}", e)))?;

        let mut rounds = Vec::with_capacity(rounds_raw.len());
        for r_raw in rounds_raw {
            rounds.push(Self::kyoku_from_raw_actions(r_raw));
        }

        // Populate end_scores based on next round's start scores
        for i in 0..rounds.len().saturating_sub(1) {
            rounds[i].end_scores = rounds[i + 1].scores.clone();
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
            let mut iter = AgariContextIterator::new(kyoku.clone());

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

impl MjSoulReplay {
    fn kyoku_from_raw_actions(raw_actions: Vec<RawAction>) -> Kyoku {
        let mut scores = Vec::new();
        let mut doras = Vec::new();
        let mut hands = vec![Vec::new(); 4];
        let mut chang = 0;
        let mut ju = 0;
        let mut ben = 0;
        let mut liqibang = 0;
        let mut left_tile_count = 70;
        let mut ura_doras = Vec::new();
        let mut paishan = None;

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
            chang = *c;
            ju = *j;
            ben = b.or(*honba).unwrap_or(0);
            liqibang = *l;
            left_tile_count = lc.unwrap_or(70);
            if let Some(uda) = ud {
                ura_doras = uda
                    .iter()
                    .map(|v| TileConverter::parse_tile_136(v))
                    .collect();
            }
            paishan = p.clone();
        }

        let mut actions = Vec::with_capacity(raw_actions.len());
        for ma in raw_actions {
            actions.push(Self::parse_raw_action(ma));
        }

        let end_scores = scores.clone();

        let mut wliqi = vec![false; 4];
        for action in &actions {
            if let Action::DiscardTile { seat, is_wliqi, .. } = action {
                if *is_wliqi {
                    wliqi[*seat] = true;
                }
            }
        }

        Kyoku {
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
        }
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
                if d_res.is_none() {
                    if let Some(dm) = dora_marker {
                        d_res = Some(vec![TileConverter::parse_tile_136(&dm)]);
                    }
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
                    1 => MeldType::Peng,
                    2 => MeldType::Gang,
                    3 => MeldType::Angang,
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
                    MeldType::Angang
                } else {
                    MeldType::Addgang
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
            RawAction::Hule { hules } => {
                let hules_typed = hules
                    .into_iter()
                    .map(|h| HuleData {
                        seat: h.seat,
                        hu_tile: TileConverter::parse_tile_136(&h.hu_tile),
                        zimo: h.zimo,
                        count: h.count,
                        fu: h.fu,
                        fans: h.fans.iter().filter(|f| f.val > 0).map(|f| f.id).collect(),
                        li_doras: h
                            .ura_dora_indicators
                            .or(h.li_doras)
                            .map(|a| a.iter().map(|v| TileConverter::parse_tile_136(v)).collect()),
                        yiman: h.yiman,
                        point_rong: h.point_rong,
                        point_zimo_qin: h.point_zimo_qin,
                        point_zimo_xian: h.point_zimo_xian,
                    })
                    .collect();
                Action::Hule { hules: hules_typed }
            }
            RawAction::Dora { dora_marker } => Action::Dora {
                dora_marker: TileConverter::parse_tile_136(&dora_marker),
            },
            RawAction::NoTile {} => Action::NoTile,
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
