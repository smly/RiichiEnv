//! Shared helpers for mapping Mjai messages to a legal `Action`.
//!
//! Used by both `Observation::select_action_from_mjai` (4P) and
//! `Observation3P::select_action_from_mjai` (3P).

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

use crate::action::{Action, ActionType};
use crate::parser::tid_to_mjai;

pub(crate) struct ParsedMjai {
    pub type_str: String,
    pub tile_str: String,
    pub tsumogiri: Option<bool>,
    pub consumed: Option<Vec<String>>,
}

pub(crate) fn parse_mjai_message(mjai_data: &Bound<'_, PyAny>) -> Option<ParsedMjai> {
    if let Ok(s) = mjai_data.extract::<String>() {
        let v: serde_json::Value = serde_json::from_str(&s).ok()?;
        let type_str = v["type"].as_str()?.to_string();
        let tile_str = v["pai"].as_str().unwrap_or("").to_string();
        let tsumogiri = v.get("tsumogiri").and_then(|x| x.as_bool());
        let consumed = v.get("consumed").and_then(|x| x.as_array()).map(|arr| {
            arr.iter()
                .filter_map(|e| e.as_str().map(|s| s.to_string()))
                .collect::<Vec<_>>()
        });
        Some(ParsedMjai {
            type_str,
            tile_str,
            tsumogiri,
            consumed,
        })
    } else if let Ok(dict) = mjai_data.cast::<PyDict>() {
        let type_str: String = dict
            .get_item("type")
            .ok()
            .flatten()
            .and_then(|x| x.extract::<String>().ok())
            .unwrap_or_default();
        let tile_str: String = dict
            .get_item("pai")
            .ok()
            .flatten()
            .or_else(|| dict.get_item("tile").ok().flatten())
            .and_then(|x| x.extract::<String>().ok())
            .unwrap_or_default();
        let tsumogiri = dict
            .get_item("tsumogiri")
            .ok()
            .flatten()
            .and_then(|x| x.extract::<bool>().ok());
        let consumed = dict
            .get_item("consumed")
            .ok()
            .flatten()
            .and_then(|x| x.extract::<Vec<String>>().ok());
        Some(ParsedMjai {
            type_str,
            tile_str,
            tsumogiri,
            consumed,
        })
    } else {
        None
    }
}

fn consumed_matches(action_consume: &[u8], expected: &[String]) -> bool {
    if action_consume.len() != expected.len() {
        return false;
    }
    let mut a: Vec<String> = action_consume.iter().map(|&t| tid_to_mjai(t)).collect();
    let mut b: Vec<String> = expected.to_vec();
    a.sort();
    b.sort();
    a == b
}

/// Select a matching `Action` from a slice of legal actions for a parsed Mjai
/// message.
///
/// `three_player` controls whether 3P-only types (`kita`) are recognized; chi
/// is rejected when set.
pub(crate) fn select_action<'a>(
    legal_actions: &'a [Action],
    parsed: &ParsedMjai,
    drawn_tile: Option<u8>,
    three_player: bool,
) -> Option<&'a Action> {
    let atype = parsed.type_str.as_str();

    if atype == "hora" {
        return legal_actions
            .iter()
            .find(|a| matches!(a.action_type, ActionType::Tsumo | ActionType::Ron));
    }

    if atype == "none" {
        return legal_actions
            .iter()
            .find(|a| a.action_type == ActionType::Pass);
    }

    let target_type = match atype {
        "dahai" => Some(ActionType::Discard),
        "chi" if !three_player => Some(ActionType::Chi),
        "pon" => Some(ActionType::Pon),
        "kakan" => Some(ActionType::Kakan),
        "daiminkan" => Some(ActionType::Daiminkan),
        "ankan" => Some(ActionType::Ankan),
        "kita" if three_player => Some(ActionType::Kita),
        "reach" => Some(ActionType::Riichi),
        "ryukyoku" => Some(ActionType::KyushuKyuhai),
        _ => None,
    };

    let tt = target_type?;

    // Special-case Discard: filter by mjai pai then disambiguate via tsumogiri.
    if tt == ActionType::Discard {
        let candidates: Vec<&Action> = legal_actions
            .iter()
            .filter(|a| {
                a.action_type == ActionType::Discard
                    && a.tile.is_some_and(|t| tid_to_mjai(t) == parsed.tile_str)
            })
            .collect();

        if candidates.is_empty() {
            return None;
        }

        if let (Some(tsumogiri), Some(drawn)) = (parsed.tsumogiri, drawn_tile) {
            let preferred = candidates.iter().find(|a| {
                let is_drawn = a.tile == Some(drawn);
                if tsumogiri { is_drawn } else { !is_drawn }
            });
            if let Some(a) = preferred {
                return Some(*a);
            }
        }

        return Some(candidates[0]);
    }

    legal_actions.iter().find(|a| {
        if a.action_type != tt {
            return false;
        }

        if let Some(consumed) = parsed.consumed.as_ref() {
            if !consumed_matches(&a.consume_tiles, consumed) {
                return false;
            }
            // If pai is also given, double-check tile match for actions that
            // carry a meaningful tile (chi/pon/daiminkan/kakan).
            if !parsed.tile_str.is_empty()
                && matches!(
                    tt,
                    ActionType::Chi | ActionType::Pon | ActionType::Daiminkan | ActionType::Kakan
                )
            {
                if let Some(t) = a.tile {
                    if tid_to_mjai(t) != parsed.tile_str {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            return true;
        }

        // No consumed field: fall back to pai-based match.
        if !parsed.tile_str.is_empty() {
            if let Some(t) = a.tile {
                return tid_to_mjai(t) == parsed.tile_str;
            }
            return false;
        }
        true
    })
}
