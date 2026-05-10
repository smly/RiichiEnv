#[cfg(feature = "python")]
mod encode;
#[cfg(feature = "python")]
pub use encode::OBS_EXTENDED_CHANNELS;
#[cfg(feature = "python")]
pub(crate) mod helpers;
#[cfg(feature = "python")]
pub(crate) mod mjai_select;
#[cfg(feature = "python")]
mod python;
#[cfg(feature = "python")]
pub(crate) mod sequence_features;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde::{Deserialize, Serialize};

use crate::action::{Action, ActionEncoder, ActionType};
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
    #[cfg_attr(not(feature = "python"), allow(dead_code))]
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
    #[serde(default)]
    pub drawn_tile: Option<u8>,
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
        drawn_tile: Option<u8>,
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
            drawn_tile,
        }
    }

    pub fn legal_actions_method(&self) -> Vec<Action> {
        self._legal_actions.clone()
    }

    pub fn find_action(&self, action_id: usize) -> Option<Action> {
        let encoder = ActionEncoder::FourPlayer;
        // Prefer non-red-five candidates so that 5m/5p/5s discards do not
        // accidentally drop the akadora when a normal 5 is also legal.
        let mut fallback: Option<&Action> = None;
        for action in &self._legal_actions {
            let Ok(idx) = encoder.encode(action) else {
                continue;
            };
            if (idx as usize) != action_id {
                continue;
            }
            if is_red_five_discard(action) {
                fallback.get_or_insert(action);
            } else {
                return Some(action.clone());
            }
        }
        fallback.cloned()
    }

    /// Return absolute player indices in relative order: [self, shimocha, toimen, kamicha].
    #[cfg_attr(not(feature = "python"), allow(dead_code))]
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

fn is_red_five_discard(action: &Action) -> bool {
    matches!(action.action_type, ActionType::Discard)
        && matches!(action.tile, Some(16) | Some(52) | Some(88))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs_with_actions(actions: Vec<Action>) -> Observation {
        Observation::new(
            0,
            [vec![], vec![], vec![], vec![]],
            [vec![], vec![], vec![], vec![]],
            [vec![], vec![], vec![], vec![]],
            vec![],
            [25000, 25000, 25000, 25000],
            [false, false, false, false],
            actions,
            vec![],
            0,
            0,
            0,
            0,
            0,
            vec![],
            false,
            [None, None, None, None],
            [None, None, None, None],
            None,
            None,
        )
    }

    fn discard(tile: u8) -> Action {
        Action::new(ActionType::Discard, Some(tile), vec![], Some(0))
    }

    #[test]
    fn find_action_prefers_non_red_5m() {
        // 5m action_id = 16 / 4 = 4; legal actions list red 5m first.
        let obs = obs_with_actions(vec![discard(16), discard(17), discard(18), discard(19)]);
        let chosen = obs.find_action(4).expect("discard 5m should resolve");
        assert_eq!(chosen.tile, Some(17), "non-red 5m must win over red 5m");
    }

    #[test]
    fn find_action_prefers_non_red_5p() {
        // 5p action_id = 52 / 4 = 13.
        let obs = obs_with_actions(vec![discard(52), discard(54)]);
        let chosen = obs.find_action(13).expect("discard 5p should resolve");
        assert_eq!(chosen.tile, Some(54));
    }

    #[test]
    fn find_action_prefers_non_red_5s() {
        // 5s action_id = 88 / 4 = 22.
        let obs = obs_with_actions(vec![discard(88), discard(90)]);
        let chosen = obs.find_action(22).expect("discard 5s should resolve");
        assert_eq!(chosen.tile, Some(90));
    }

    #[test]
    fn find_action_falls_back_to_red_when_only_red_legal() {
        // Only the red 5m is in the hand; we still need a working discard.
        let obs = obs_with_actions(vec![discard(16)]);
        let chosen = obs.find_action(4).expect("red-only 5m must still resolve");
        assert_eq!(chosen.tile, Some(16));
    }

    #[test]
    fn find_action_unaffected_for_non_five_tiles() {
        // 4m action_id = 12 / 4 = 3.
        let obs = obs_with_actions(vec![discard(12), discard(13)]);
        let chosen = obs.find_action(3).expect("discard 4m should resolve");
        assert_eq!(chosen.tile, Some(12), "first match wins for non-red ties");
    }
}
