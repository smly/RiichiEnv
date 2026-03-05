#[cfg(feature = "python")]
mod encode;
#[cfg(feature = "python")]
pub(crate) mod helpers;
#[cfg(feature = "python")]
mod python;
#[cfg(feature = "python")]
pub(crate) mod sequence_features;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde::{Deserialize, Serialize};

use crate::action::{Action, ActionEncoder};
use crate::errors::{RiichiError, RiichiResult};
use crate::types::Meld;

#[cfg_attr(
    feature = "python",
    pyo3::pyclass(module = "riichienv._riichienv", get_all, from_py_object)
)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub player_id: u8,
    pub hands: [Vec<u32>; 4],
    pub melds: [Vec<Meld>; 4],
    pub discards: [Vec<u32>; 4],
    pub dora_indicators: Vec<u32>,
    pub scores: [i32; 4],
    pub riichi_declared: [bool; 4],

    pub(crate) _legal_actions: Vec<Action>,

    pub(crate) events: Vec<String>,

    /// Pre-computed progression tuples (set by GameState for O(1) access).
    /// When Some, encode_seq_progression() returns this directly.
    #[serde(skip)]
    pub(crate) cached_progression: Option<Vec<[u16; 5]>>,

    pub honba: u8,
    pub riichi_sticks: u32,
    pub round_wind: u8,
    pub oya: u8,
    pub kyoku_index: u8,
    pub waits: Vec<u8>,
    pub is_tenpai: bool,
    pub tsumogiri_flags: [Vec<bool>; 4],
    pub riichi_sutehais: [Option<u8>; 4],
    pub last_tedashis: [Option<u8>; 4],
    pub last_discard: Option<u32>,
}

/// Pure Rust methods (no PyO3 dependency).
impl Observation {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        player_id: u8,
        hands: [Vec<u8>; 4],
        melds: [Vec<Meld>; 4],
        discards: [Vec<u8>; 4],
        dora_indicators: Vec<u8>,
        scores: [i32; 4],
        riichi_declared: [bool; 4],
        legal_actions: Vec<Action>,
        events: Vec<String>,
        honba: u8,
        riichi_sticks: u32,
        round_wind: u8,
        oya: u8,
        kyoku_index: u8,
        waits: Vec<u8>,
        is_tenpai: bool,
        riichi_sutehais: [Option<u8>; 4],
        last_tedashis: [Option<u8>; 4],
        last_discard: Option<u32>,
    ) -> Self {
        let hands_u32 = hands.map(|h| h.into_iter().map(|x| x as u32).collect());
        let discards_u32 = discards.map(|d| d.into_iter().map(|x| x as u32).collect());
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
            cached_progression: None,
            honba,
            riichi_sticks,
            round_wind,
            oya,
            kyoku_index,
            waits,
            is_tenpai,
            tsumogiri_flags: Default::default(),
            riichi_sutehais,
            last_tedashis,
            last_discard,
        }
    }

    pub fn legal_actions_method(&self) -> Vec<Action> {
        self._legal_actions.clone()
    }

    pub fn find_action(&self, action_id: usize) -> Option<Action> {
        let encoder = ActionEncoder::FourPlayer;
        self._legal_actions
            .iter()
            .find(|a| {
                if let Ok(idx) = encoder.encode(a) {
                    (idx as usize) == action_id
                } else {
                    false
                }
            })
            .cloned()
    }

    /// Return absolute player indices in relative order: [self, shimocha, toimen, kamicha].
    pub(crate) fn rel_order(&self) -> [usize; 4] {
        let pid = self.player_id as usize;
        [pid, (pid + 1) % 4, (pid + 2) % 4, (pid + 3) % 4]
    }

    pub fn new_events(&self) -> Vec<String> {
        self.events.clone()
    }

    /// Serialize this Observation to a base64-encoded JSON string.
    pub fn serialize_to_base64(&self) -> RiichiResult<String> {
        let json = serde_json::to_vec(self).map_err(|e| RiichiError::Serialization {
            message: format!("serialization failed: {e}"),
        })?;
        Ok(BASE64.encode(&json))
    }

    /// Deserialize an Observation from a base64-encoded JSON string.
    pub fn deserialize_from_base64(s: &str) -> RiichiResult<Self> {
        let bytes = BASE64.decode(s).map_err(|e| RiichiError::Serialization {
            message: format!("base64 decode failed: {e}"),
        })?;
        let obs: Observation =
            serde_json::from_slice(&bytes).map_err(|e| RiichiError::Serialization {
                message: format!("JSON deserialize failed: {e}"),
            })?;
        Ok(obs)
    }
}
