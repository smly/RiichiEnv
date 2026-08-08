#[cfg(feature = "python")]
use flate2::read::MultiGzDecoder;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[cfg(feature = "python")]
use std::fs::File;
use std::io::BufRead;
#[cfg(feature = "python")]
use std::io::BufReader;
use std::sync::Arc;

use crate::errors::{RiichiError, RiichiResult};
use crate::parser::mjai_to_tid;
use crate::replay::{Action, HuleData, LogKyoku};
use crate::rule::GameRule;
use crate::types::MeldType;

fn parse_mjai_tile(s: &str) -> RiichiResult<u8> {
    mjai_to_tid(s).ok_or_else(|| RiichiError::Parse {
        input: "MJAI tile".to_string(),
        message: format!("invalid tile string {s:?}"),
    })
}

/// Pure-Rust typed replay built from an MJAI event stream.
///
/// Unlike the legacy Python adapter this type is independent of filesystem
/// paths, compression, and Python.  An unfinished kyoku at EOF is deliberately
/// omitted; callers that need live prefixes should retain the accompanying
/// `EventJournal` and append more events before rebuilding the typed view.
#[derive(Clone, Default)]
pub struct ReplayLog {
    rounds: Vec<LogKyoku>,
}

impl ReplayLog {
    pub fn from_jsonl(jsonl: &str, rule: GameRule) -> RiichiResult<Self> {
        Self::from_jsonl_reader(std::io::Cursor::new(jsonl.as_bytes()), rule)
    }

    pub fn from_jsonl_reader(reader: impl BufRead, rule: GameRule) -> RiichiResult<Self> {
        Self::from_jsonl_reader_with_eof_policy(reader, rule, false)
    }

    pub fn from_events(
        events: impl IntoIterator<Item = MjaiEvent>,
        rule: GameRule,
    ) -> RiichiResult<Self> {
        Self::from_typed_events(events, rule, false)
    }

    pub fn rounds(&self) -> &[LogKyoku] {
        &self.rounds
    }

    pub fn len(&self) -> usize {
        self.rounds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rounds.is_empty()
    }

    pub fn cursor(&self) -> ReplayCursor<'_> {
        ReplayCursor {
            rounds: &self.rounds,
            position: 0,
        }
    }

    pub fn into_rounds(self) -> Vec<LogKyoku> {
        self.rounds
    }
}

/// Borrowed, seekable cursor over typed kyokus in a `ReplayLog`.
#[derive(Clone)]
pub struct ReplayCursor<'a> {
    rounds: &'a [LogKyoku],
    position: usize,
}

impl ReplayCursor<'_> {
    pub fn position(&self) -> usize {
        self.position
    }

    pub fn remaining(&self) -> usize {
        self.rounds.len().saturating_sub(self.position)
    }

    pub fn seek(&mut self, position: usize) -> RiichiResult<()> {
        if position > self.rounds.len() {
            return Err(RiichiError::InvalidState {
                message: format!(
                    "replay cursor {position} exceeds replay length {}",
                    self.rounds.len()
                ),
            });
        }
        self.position = position;
        Ok(())
    }
}

impl<'a> Iterator for ReplayCursor<'a> {
    type Item = &'a LogKyoku;

    fn next(&mut self) -> Option<Self::Item> {
        let round = self.rounds.get(self.position)?;
        self.position += 1;
        Some(round)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining();
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ReplayCursor<'_> {}

#[cfg(feature = "python")]
#[pyclass]
pub struct MjaiReplay {
    pub rounds: Vec<LogKyoku>,
}

#[cfg(feature = "python")]
#[derive(Debug)]
#[pyclass]
pub struct KyokuIterator {
    game: Py<MjaiReplay>,
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

// MJAI Event Definitions
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(tag = "type")]
pub enum MjaiEvent {
    #[serde(rename = "start_game")]
    StartGame {
        names: Option<Vec<String>>,
        #[serde(default, deserialize_with = "deserialize_optional_game_id")]
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
    #[serde(rename = "kan", alias = "daiminkan")]
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
        #[serde(alias = "ura_markers")]
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
    #[serde(rename = "kita")]
    Kita { actor: usize },
    #[serde(rename = "end_game")]
    EndGame,
    #[serde(rename = "end_kyoku")]
    EndKyoku,
    #[serde(other)]
    Other,
}

fn deserialize_optional_game_id<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(id)) => Ok(Some(id)),
        Some(Value::Number(id)) => Ok(Some(id.to_string())),
        Some(_) => Err(serde::de::Error::custom(
            "start_game id must be a string or number",
        )),
    }
}

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
    rule: GameRule,

    // Internal tracking
    liqi_flags: Vec<bool>, // Who has declared reach (to set `is_liqi` on discard)
    wliqi_flags: Vec<bool>, // Who has effectively achieved Double Riichi
    reach_accepted: Vec<bool>,
    reached: Vec<bool>, // Tracks if player declared reach (for riichi cost in end_scores)
    first_discard: Vec<bool>,
    has_calls: bool,
    pending_hule: Vec<HuleData>, // Buffer for batching consecutive hora events (double/triple ron)
}

impl KyokuBuilder {
    #[allow(clippy::too_many_arguments)]
    fn new(
        bakaze: String,
        kyoku: u8,
        honba: u8,
        kyoutaku: u8,
        scores: Vec<i32>,
        dora_marker: String,
        tehais: Vec<Vec<String>>,
        rule: GameRule,
    ) -> RiichiResult<Self> {
        if !matches!(scores.len(), 3 | 4) {
            return Err(RiichiError::InvalidState {
                message: format!(
                    "start_kyoku has {} scores; expected three or four",
                    scores.len()
                ),
            });
        }
        if tehais.len() != scores.len() {
            return Err(RiichiError::InvalidState {
                message: format!(
                    "start_kyoku has {} hands for {} players",
                    tehais.len(),
                    scores.len()
                ),
            });
        }
        let chang = match bakaze.as_str() {
            "S" => 1,
            "W" => 2,
            "N" => 3,
            _ => 0, // "E" or default
        };
        let ju = kyoku
            .checked_sub(1)
            .ok_or_else(|| RiichiError::InvalidState {
                message: "start_kyoku kyoku must be at least one".to_string(),
            })?;

        let np = scores.len();
        let mut hands = vec![Vec::new(); np];
        for (i, tehai_strs) in tehais.iter().enumerate() {
            if i < np {
                hands[i] = tehai_strs
                    .iter()
                    .map(|s| parse_mjai_tile(s))
                    .collect::<RiichiResult<_>>()?;
            }
        }

        let first_dora = parse_mjai_tile(&dora_marker)?;
        let end_scores = scores.clone();

        // Standard tile counts: 4p = 136, 3p = 108 (excludes 2m-8m)
        // left_tile_count = total - non_drawable_reserve(14) - dealt(13*np)
        let left_tile_count = if np == 3 { 55u8 } else { 70u8 };

        Ok(KyokuBuilder {
            actions: Vec::new(),
            scores,
            end_scores,
            hands,
            doras: vec![first_dora],
            chang,
            ju,
            ben: honba,
            liqibang: kyoutaku,
            left_tile_count,
            ura_doras: Vec::new(),
            rule,
            liqi_flags: vec![false; np],
            wliqi_flags: vec![false; np],
            reach_accepted: vec![false; np],
            reached: vec![false; np],
            first_discard: vec![true; np],
            has_calls: false,
            pending_hule: Vec::new(),
        })
    }

    fn flush_pending_hule(&mut self) {
        if !self.pending_hule.is_empty() {
            let hules = std::mem::take(&mut self.pending_hule);
            self.actions.push(Action::Hule { hules });
        }
    }

    fn build(mut self) -> LogKyoku {
        self.flush_pending_hule();
        LogKyoku {
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
            rule: self.rule,
            game_end_scores: None,
        }
    }

    fn validate_player(&self, player: usize, field: &str) -> RiichiResult<()> {
        if player >= self.scores.len() {
            return Err(RiichiError::InvalidState {
                message: format!(
                    "{field} player {player} is out of range for {} players",
                    self.scores.len()
                ),
            });
        }
        Ok(())
    }
}

impl ReplayLog {
    fn from_jsonl_reader_with_eof_policy(
        reader: impl BufRead,
        rule: GameRule,
        include_incomplete_kyoku: bool,
    ) -> RiichiResult<Self> {
        let mut events = Vec::new();
        for (line_index, line) in reader.lines().enumerate() {
            let line = line.map_err(|error| RiichiError::Serialization {
                message: format!("failed to read MJAI line {}: {error}", line_index + 1),
            })?;
            if line.trim().is_empty() {
                continue;
            }
            let event = serde_json::from_str(&line).map_err(|error| RiichiError::Parse {
                input: format!("line {}", line_index + 1),
                message: format!("invalid MJAI event: {error}"),
            })?;
            events.push(event);
        }
        Self::from_typed_events(events, rule, include_incomplete_kyoku)
    }

    fn from_typed_events(
        events: impl IntoIterator<Item = MjaiEvent>,
        rule: GameRule,
        include_incomplete_kyoku: bool,
    ) -> RiichiResult<Self> {
        let mut rounds = Vec::new();
        let mut builder: Option<KyokuBuilder> = None;

        for event in events {
            match event {
                MjaiEvent::StartKyoku {
                    bakaze,
                    kyoku,
                    honba,
                    kyoutaku,
                    oya,
                    scores,
                    dora_marker,
                    tehais,
                } => {
                    if let Some(previous) = builder.take()
                        && include_incomplete_kyoku
                    {
                        rounds.push(previous.build());
                    }
                    if oya as usize >= scores.len() {
                        return Err(RiichiError::InvalidState {
                            message: format!(
                                "start_kyoku dealer {oya} is out of range for {} players",
                                scores.len()
                            ),
                        });
                    }
                    builder = Some(KyokuBuilder::new(
                        bakaze,
                        kyoku,
                        honba,
                        kyoutaku,
                        scores,
                        dora_marker,
                        tehais,
                        rule,
                    )?);
                }
                MjaiEvent::EndKyoku => {
                    if let Some(completed) = builder.take() {
                        rounds.push(completed.build());
                    }
                }
                MjaiEvent::EndGame => {
                    if let Some(incomplete) = builder.take()
                        && include_incomplete_kyoku
                    {
                        rounds.push(incomplete.build());
                    }
                }
                event => {
                    if let Some(builder) = &mut builder {
                        builder.process_event(event)?;
                    }
                }
            }
        }

        if let Some(incomplete) = builder
            && include_incomplete_kyoku
        {
            rounds.push(incomplete.build());
        }

        // The following round's start scores are the authoritative post-round
        // scores when the event itself omitted them.
        for index in 0..rounds.len().saturating_sub(1) {
            rounds[index].end_scores = rounds[index + 1].scores.clone();
        }

        Ok(Self { rounds })
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl MjaiReplay {
    #[staticmethod]
    #[pyo3(signature = (path, rule=None))]
    pub fn from_jsonl(py: Python<'_>, path: String, rule: Option<String>) -> PyResult<Self> {
        let game_rule = python_game_rule(rule.as_deref())?;
        let replay = py.detach(move || {
            let file = File::open(&path).map_err(|error| RiichiError::Serialization {
                message: format!("failed to open replay file: {error}"),
            })?;
            let mut buf_reader = BufReader::new(file);

            // Detect gzip by magic bytes rather than the path suffix.
            let is_gzip = {
                let buf = buf_reader
                    .fill_buf()
                    .map_err(|error| RiichiError::Serialization {
                        message: format!("failed to inspect replay file: {error}"),
                    })?;
                buf.len() >= 2 && buf[0] == 0x1f && buf[1] == 0x8b
            };

            let reader: Box<dyn BufRead> = if is_gzip {
                Box::new(BufReader::new(MultiGzDecoder::new(buf_reader)))
            } else {
                Box::new(buf_reader)
            };
            ReplayLog::from_jsonl_reader_with_eof_policy(reader, game_rule, true)
        })?;
        Ok(MjaiReplay {
            rounds: replay.into_rounds(),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (jsonl, rule=None))]
    pub fn from_jsonl_text(py: Python<'_>, jsonl: String, rule: Option<String>) -> PyResult<Self> {
        let game_rule = python_game_rule(rule.as_deref())?;
        let replay = py.detach(move || {
            ReplayLog::from_jsonl_reader_with_eof_policy(
                std::io::Cursor::new(jsonl.into_bytes()),
                game_rule,
                true,
            )
        })?;
        Ok(Self {
            rounds: replay.into_rounds(),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (events, rule=None))]
    pub fn from_events(
        py: Python<'_>,
        events: Vec<String>,
        rule: Option<String>,
    ) -> PyResult<Self> {
        let game_rule = python_game_rule(rule.as_deref())?;
        let replay = py.detach(move || {
            ReplayLog::from_jsonl_reader_with_eof_policy(
                std::io::Cursor::new(events.join("\n").into_bytes()),
                game_rule,
                true,
            )
        })?;
        Ok(Self {
            rounds: replay.into_rounds(),
        })
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

#[cfg(feature = "python")]
fn python_game_rule(rule: Option<&str>) -> PyResult<GameRule> {
    match rule {
        Some("tenhou") | None => Ok(GameRule::default_tenhou()),
        Some("mjsoul") => Ok(GameRule::default_mjsoul()),
        Some(other) => Err(PyValueError::new_err(format!(
            "Unknown rule: '{other}'. Expected 'tenhou' or 'mjsoul'"
        ))),
    }
}

impl KyokuBuilder {
    fn process_event(&mut self, event: MjaiEvent) -> RiichiResult<()> {
        // Flush pending hora batch before any non-Hora event
        if !matches!(event, MjaiEvent::Hora { .. }) {
            self.flush_pending_hule();
        }

        match event {
            MjaiEvent::Tsumo { actor, pai } => {
                self.validate_player(actor, "tsumo actor")?;
                let tile = parse_mjai_tile(&pai)?;
                self.actions.push(Action::DealTile {
                    seat: actor,
                    tile,
                    doras: None,
                    left_tile_count: None,
                });
                if self.left_tile_count > 0 {
                    self.left_tile_count -= 1;
                }
            }
            MjaiEvent::Dahai {
                actor,
                pai,
                tsumogiri: _,
            } => {
                self.validate_player(actor, "dahai actor")?;
                let tile = parse_mjai_tile(&pai)?;
                let is_liqi = self.liqi_flags[actor];

                let is_wliqi = is_liqi && self.first_discard[actor] && !self.has_calls;
                if is_wliqi {
                    self.wliqi_flags[actor] = true;
                }

                self.actions.push(Action::DiscardTile {
                    seat: actor,
                    tile,
                    is_liqi,
                    is_wliqi,
                    doras: None,
                });

                self.first_discard[actor] = false;
                if is_liqi {
                    self.liqi_flags[actor] = false;
                }
            }
            MjaiEvent::Reach { actor } => {
                self.validate_player(actor, "reach actor")?;
                self.liqi_flags[actor] = true;
                self.reached[actor] = true;
            }
            MjaiEvent::ReachAccepted { actor } => {
                self.validate_player(actor, "reach_accepted actor")?;
                self.reach_accepted[actor] = true;
            }
            MjaiEvent::Chi {
                actor,
                target,
                pai,
                consumed,
            } => {
                self.validate_player(actor, "chi actor")?;
                self.validate_player(target, "chi target")?;
                self.has_calls = true;
                let mut tiles = vec![parse_mjai_tile(&pai)?];
                let mut froms = vec![target];
                for c in &consumed {
                    tiles.push(parse_mjai_tile(c)?);
                    froms.push(actor);
                }
                self.actions.push(Action::ChiPengGang {
                    seat: actor,
                    meld_type: MeldType::Chi,
                    tiles,
                    froms,
                });
            }
            MjaiEvent::Pon {
                actor,
                target,
                pai,
                consumed,
            } => {
                self.validate_player(actor, "pon actor")?;
                self.validate_player(target, "pon target")?;
                self.has_calls = true;
                let mut tiles = vec![parse_mjai_tile(&pai)?];
                let mut froms = vec![target];
                for c in &consumed {
                    tiles.push(parse_mjai_tile(c)?);
                    froms.push(actor);
                }
                self.actions.push(Action::ChiPengGang {
                    seat: actor,
                    meld_type: MeldType::Pon,
                    tiles,
                    froms,
                });
            }
            MjaiEvent::Kan {
                actor,
                target,
                pai,
                consumed,
            } => {
                self.validate_player(actor, "kan actor")?;
                self.validate_player(target, "kan target")?;
                self.has_calls = true;
                let mut tiles = vec![parse_mjai_tile(&pai)?];
                let mut froms = vec![target];
                for c in &consumed {
                    tiles.push(parse_mjai_tile(c)?);
                    froms.push(actor);
                }
                self.actions.push(Action::ChiPengGang {
                    seat: actor,
                    meld_type: MeldType::Daiminkan,
                    tiles,
                    froms,
                });
            }
            MjaiEvent::Ankan { actor, consumed } => {
                self.validate_player(actor, "ankan actor")?;
                self.has_calls = true;
                let tiles = consumed
                    .iter()
                    .map(|s| parse_mjai_tile(s))
                    .collect::<RiichiResult<Vec<_>>>()?;
                self.actions.push(Action::AnGangAddGang {
                    seat: actor,
                    meld_type: MeldType::Ankan,
                    tiles,
                    tile_raw_id: 0,
                    doras: None,
                });
            }
            MjaiEvent::Kakan { actor, pai } => {
                self.validate_player(actor, "kakan actor")?;
                self.has_calls = true;
                let tile = parse_mjai_tile(&pai)?;
                self.actions.push(Action::AnGangAddGang {
                    seat: actor,
                    meld_type: MeldType::Kakan,
                    tiles: vec![tile],
                    tile_raw_id: 0,
                    doras: None,
                });
            }
            MjaiEvent::Dora { dora_marker } => {
                let marker = parse_mjai_tile(&dora_marker)?;
                self.doras.push(marker);
                self.actions.push(Action::Dora {
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
                self.validate_player(actor, "hora actor")?;
                self.validate_player(target, "hora target")?;
                let hu_tile_id = if let Some(p) = pai {
                    parse_mjai_tile(&p)?
                } else {
                    // Try to infer from last action
                    // If Tsumo (actor == target), last action should be DealTile for actor
                    // If Ron (actor != target), last action should be DiscardTile
                    // Simplifying assumption: look at last action
                    if let Some(last_action) = self.actions.last() {
                        match last_action {
                            Action::DealTile { tile, .. } => *tile,
                            Action::DiscardTile { tile, .. } => *tile,
                            Action::AnGangAddGang { tiles, .. } => {
                                tiles.first().copied().unwrap_or(0)
                            }
                            _ => 0,
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
                    let ud = uras
                        .iter()
                        .map(|s| parse_mjai_tile(s))
                        .collect::<RiichiResult<Vec<_>>>()?;
                    self.ura_doras = ud.clone();
                    hule_data.li_doras = Some(ud);
                }

                // Update end_scores: accumulate deltas for double/triple ron
                if let Some(s) = scores {
                    self.validate_scores(&s, "hora scores")?;
                    self.end_scores = s;
                } else if let Some(d) = delta {
                    self.validate_scores(&d, "hora delta")?;
                    let is_first_hora = self.pending_hule.is_empty();
                    for (i, val) in d.iter().enumerate() {
                        self.end_scores[i] = if is_first_hora {
                            // A declaration tile that is immediately ronned
                            // never pays the riichi deposit.
                            let cost = if self.reach_accepted[i] { 1000 } else { 0 };
                            checked_score_delta(self.scores[i], *val, cost)?
                        } else {
                            self.end_scores[i].checked_add(*val).ok_or_else(|| {
                                RiichiError::InvalidState {
                                    message: "hora score delta overflows i32".to_string(),
                                }
                            })?
                        };
                    }
                }

                // Buffer the hora for batching (double/triple ron)
                self.pending_hule.push(hule_data);
            }
            MjaiEvent::Kita { actor } => {
                self.validate_player(actor, "kita actor")?;
                self.actions.push(Action::BaBei {
                    seat: actor,
                    moqie: false,
                });
            }
            MjaiEvent::Ryukyoku { delta, scores, .. } => {
                if let Some(s) = scores {
                    self.validate_scores(&s, "ryukyoku scores")?;
                    self.end_scores = s;
                } else if let Some(d) = delta {
                    self.validate_scores(&d, "ryukyoku delta")?;
                    for (i, val) in d.iter().enumerate() {
                        let cost = if self.reached[i] { 1000 } else { 0 };
                        self.end_scores[i] = checked_score_delta(self.scores[i], *val, cost)?;
                    }
                }
                self.actions.push(Action::NoTile);
            }
            _ => {}
        }
        Ok(())
    }

    fn validate_scores(&self, scores: &[i32], field: &str) -> RiichiResult<()> {
        if scores.len() != self.scores.len() {
            return Err(RiichiError::InvalidState {
                message: format!(
                    "{field} has {} values for {} players",
                    scores.len(),
                    self.scores.len()
                ),
            });
        }
        Ok(())
    }
}

fn checked_score_delta(score: i32, delta: i32, cost: i32) -> RiichiResult<i32> {
    score
        .checked_add(delta)
        .and_then(|value| value.checked_sub(cost))
        .ok_or_else(|| RiichiError::InvalidState {
            message: "replay score delta overflows i32".to_string(),
        })
}

#[cfg(test)]
mod replay_log_tests {
    use super::*;

    const COMPLETE_LOG: &str = r#"{"type":"start_game"}
{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyoutaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["1m"],["2m"],["3m"],["4m"]]}
{"type":"tsumo","actor":0,"pai":"5m"}
{"type":"dahai","actor":0,"pai":"5m","tsumogiri":true}
{"type":"ryukyoku","reason":"test"}
{"type":"end_kyoku"}
{"type":"end_game"}"#;

    #[test]
    fn pure_replay_log_parses_without_python_or_a_path() {
        let replay = ReplayLog::from_jsonl(COMPLETE_LOG, GameRule::default_tenhou()).unwrap();
        assert_eq!(replay.len(), 1);
        assert_eq!(replay.rounds()[0].actions().len(), 3);
        assert!(matches!(
            replay.rounds()[0].actions()[0],
            Action::DealTile { seat: 0, .. }
        ));
    }

    #[test]
    fn unfinished_eof_is_not_promoted_to_a_completed_kyoku() {
        let truncated = COMPLETE_LOG.lines().take(4).collect::<Vec<_>>().join("\n");
        let replay = ReplayLog::from_jsonl(&truncated, GameRule::default_tenhou()).unwrap();
        assert!(replay.is_empty());
    }

    #[test]
    fn replay_cursor_is_borrowed_and_bounds_checked() {
        let replay = ReplayLog::from_jsonl(COMPLETE_LOG, GameRule::default_tenhou()).unwrap();
        let mut cursor = replay.cursor();
        assert_eq!(cursor.remaining(), 1);
        assert_eq!(cursor.next().unwrap().ju, 0);
        assert_eq!(cursor.position(), 1);
        assert!(cursor.seek(2).is_err());
        cursor.seek(0).unwrap();
        assert_eq!(cursor.len(), 1);
    }

    #[test]
    fn malformed_player_ids_return_errors_instead_of_panicking() {
        let invalid = COMPLETE_LOG.replace(
            "{\"type\":\"tsumo\",\"actor\":0",
            "{\"type\":\"tsumo\",\"actor\":4",
        );
        assert!(ReplayLog::from_jsonl(&invalid, GameRule::default_tenhou()).is_err());
    }

    #[test]
    fn tracked_replay_corpus_is_available_in_pure_core() {
        let replay = ReplayLog::from_jsonl(
            include_str!("../../../tests/data/126_204_0_mjai.jsonl"),
            GameRule::default_tenhou(),
        )
        .unwrap();
        assert_eq!(replay.len(), 12);
        assert!(
            replay
                .rounds()
                .iter()
                .all(|round| !round.actions().is_empty())
        );
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
