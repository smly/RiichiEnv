use wasm_bindgen::prelude::*;

use riichienv_core::hand_evaluator::HandEvaluator;
use riichienv_core::parser::{mjai_to_tid, tid_to_mjai};
use riichienv_core::shanten::calculate_shanten;
use riichienv_core::types::{Conditions, Meld, MeldType, Wind};

/// Input format for melds passed from JavaScript.
#[derive(serde::Deserialize)]
struct MeldInput {
    meld_type: String,
    tiles: Vec<u8>,
}

impl MeldInput {
    fn to_meld(&self) -> Meld {
        let meld_type = match self.meld_type.as_str() {
            "chi" => MeldType::Chi,
            "pon" => MeldType::Pon,
            "daiminkan" => MeldType::Daiminkan,
            "ankan" => MeldType::Ankan,
            "kakan" => MeldType::Kakan,
            _ => MeldType::Chi,
        };
        Meld::new(
            meld_type,
            self.tiles.clone(),
            meld_type != MeldType::Ankan,
            -1,
            None,
        )
    }
}

/// Input format for scoring conditions passed from JavaScript.
#[derive(Default, serde::Deserialize)]
#[serde(default)]
struct ConditionsInput {
    tsumo: bool,
    riichi: bool,
    double_riichi: bool,
    ippatsu: bool,
    haitei: bool,
    houtei: bool,
    rinshan: bool,
    chankan: bool,
    tsumo_first_turn: bool,
    player_wind: u8,
    round_wind: u8,
    honba: u32,
}

impl ConditionsInput {
    fn to_conditions(&self) -> Conditions {
        Conditions {
            tsumo: self.tsumo,
            riichi: self.riichi,
            double_riichi: self.double_riichi,
            ippatsu: self.ippatsu,
            haitei: self.haitei,
            houtei: self.houtei,
            rinshan: self.rinshan,
            chankan: self.chankan,
            tsumo_first_turn: self.tsumo_first_turn,
            player_wind: Wind::from(self.player_wind),
            round_wind: Wind::from(self.round_wind),
            riichi_sticks: 0,
            honba: self.honba,
            kita_count: 0,
            is_sanma: false,
            num_players: 4,
        }
    }
}

/// Output format for scoring results returned to JavaScript.
#[derive(serde::Serialize)]
struct ScoreResult {
    is_win: bool,
    yakuman: bool,
    han: u32,
    fu: u32,
    ron_agari: u32,
    tsumo_agari_oya: u32,
    tsumo_agari_ko: u32,
    yaku: Vec<u32>,
}

/// Calculate wait tiles (machi) for a given hand.
///
/// Input: JSON array of tile IDs (136-encoding) for hand tiles.
/// Melds: JSON array of meld objects (optional).
/// Returns: JSON array of wait tile types (34-encoding).
#[wasm_bindgen]
pub fn calc_waits(tiles_json: &str, melds_json: &str) -> Result<JsValue, JsValue> {
    let tiles: Vec<u8> = serde_json::from_str(tiles_json)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse tiles: {}", e)))?;

    let meld_inputs: Vec<MeldInput> = serde_json::from_str(melds_json)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse melds: {}", e)))?;

    let melds: Vec<Meld> = meld_inputs.iter().map(|m| m.to_meld()).collect();

    let evaluator = HandEvaluator::new(tiles, melds);
    let waits = evaluator.get_waits_u8();

    serde_wasm_bindgen::to_value(&waits)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Calculate shanten number for a given hand.
///
/// Input: JSON array of tile IDs (136-encoding).
/// Returns: shanten number (-1 = tenpai, 0 = iishanten, etc.)
#[wasm_bindgen]
pub fn calc_shanten(tiles_json: &str) -> Result<i32, JsValue> {
    let tiles: Vec<u32> = serde_json::from_str(tiles_json)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse tiles: {}", e)))?;

    Ok(calculate_shanten(&tiles))
}

/// Calculate score for a winning hand.
///
/// Input: hand tiles (136-encoding), win tile, dora indicators, and conditions.
/// Returns: score result as JSON.
#[wasm_bindgen]
pub fn calc_score(
    tiles_json: &str,
    melds_json: &str,
    win_tile: u8,
    dora_json: &str,
    ura_json: &str,
    conditions_json: &str,
) -> Result<JsValue, JsValue> {
    let tiles: Vec<u8> = serde_json::from_str(tiles_json)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse tiles: {}", e)))?;

    let meld_inputs: Vec<MeldInput> = serde_json::from_str(melds_json)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse melds: {}", e)))?;

    let dora_indicators: Vec<u8> = serde_json::from_str(dora_json)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse dora: {}", e)))?;

    let ura_indicators: Vec<u8> = serde_json::from_str(ura_json)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse ura: {}", e)))?;

    let cond_input: ConditionsInput = serde_json::from_str(conditions_json)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse conditions: {}", e)))?;

    let melds: Vec<Meld> = meld_inputs.iter().map(|m| m.to_meld()).collect();
    let evaluator = HandEvaluator::new(tiles, melds);

    let result = evaluator.calc(
        win_tile,
        dora_indicators,
        ura_indicators,
        Some(cond_input.to_conditions()),
    );

    let score = ScoreResult {
        is_win: result.is_win,
        yakuman: result.yakuman,
        han: result.han,
        fu: result.fu,
        ron_agari: result.ron_agari,
        tsumo_agari_oya: result.tsumo_agari_oya,
        tsumo_agari_ko: result.tsumo_agari_ko,
        yaku: result.yaku,
    };

    serde_wasm_bindgen::to_value(&score)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Convert MJAI tile string to 136-encoding tile ID.
#[wasm_bindgen]
pub fn mjai_to_tile_id(mjai: &str) -> Option<u8> {
    mjai_to_tid(mjai)
}

/// Convert 136-encoding tile ID to MJAI tile string.
#[wasm_bindgen]
pub fn tile_id_to_mjai(tid: u8) -> String {
    tid_to_mjai(tid)
}

/// Check if a hand is tenpai (one tile away from winning).
///
/// Input: JSON array of tile IDs (136-encoding) for hand tiles.
/// Returns: true if tenpai.
#[wasm_bindgen]
pub fn is_tenpai(tiles_json: &str, melds_json: &str) -> Result<bool, JsValue> {
    let tiles: Vec<u8> = serde_json::from_str(tiles_json)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse tiles: {}", e)))?;

    let meld_inputs: Vec<MeldInput> = serde_json::from_str(melds_json)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse melds: {}", e)))?;

    let melds: Vec<Meld> = meld_inputs.iter().map(|m| m.to_meld()).collect();
    let evaluator = HandEvaluator::new(tiles, melds);

    Ok(evaluator.is_tenpai())
}
