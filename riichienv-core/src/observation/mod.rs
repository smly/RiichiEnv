mod encode;
pub(crate) mod helpers;
#[cfg(feature = "python")]
mod python;

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde::{Deserialize, Serialize};

use crate::action::{Action, ActionEncoder};
use crate::errors::{RiichiError, RiichiResult};
use crate::types::Meld;

#[cfg_attr(
    feature = "python",
    pyo3::pyclass(module = "riichienv._riichienv", get_all)
)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub player_id: u8,
    pub hands: Vec<Vec<u32>>,
    pub melds: Vec<Vec<Meld>>,
    pub discards: Vec<Vec<u32>>,
    pub dora_indicators: Vec<u32>,
    pub scores: Vec<i32>,
    pub riichi_declared: Vec<bool>,

    pub(crate) _legal_actions: Vec<Action>,

    pub(crate) events: Vec<String>,

    pub honba: u8,
    pub riichi_sticks: u32,
    pub round_wind: u8,
    pub oya: u8,
    pub kyoku_index: u8,
    pub waits: Vec<u8>,
    pub is_tenpai: bool,
    pub tsumogiri_flags: Vec<Vec<bool>>,
    pub riichi_sutehais: Vec<Option<u8>>,
    pub last_tedashis: Vec<Option<u8>>,
    pub last_discard: Option<u32>,
}

/// Pure Rust methods (no PyO3 dependency).
impl Observation {
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
            tsumogiri_flags: vec![vec![]; 4],
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
