use wasm_bindgen::prelude::*;

use riichienv_core::drev::{self, DrevInput, encode_drev};
use riichienv_core::hand_evaluator::HandEvaluator;
use riichienv_core::hand_evaluator_3p::HandEvaluator3P;
use riichienv_core::parser::{mjai_to_tid, tid_to_mjai};
use riichienv_core::sp::{self, SpInput, encode_sp};
use riichienv_core::types::{Conditions, Meld, MeldType, TILE_MAX, Wind};
use riichienv_core::{score, yaku};

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
    kita_count: u8,
    is_sanma: bool,
    is_kokushi_musou_13machi_double: bool,
    is_suuankou_tanki_double: bool,
    is_junsei_chuurenpoutou_double: bool,
    is_daisuushii_double: bool,
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
            kita_count: self.kita_count,
            is_sanma: self.is_sanma,
            num_players: if self.is_sanma { 3 } else { 4 },
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

fn apply_double_yakuman_rules(score: &mut ScoreResult, conditions: &ConditionsInput) {
    if !score.yakuman || score.han <= 13 {
        return;
    }

    let mut cap = 0u32;
    for &y in &score.yaku {
        match y {
            yaku::ID_JUNSEI_CHUUREN if !conditions.is_junsei_chuurenpoutou_double => cap += 13,
            yaku::ID_SUANKO_TANKI if !conditions.is_suuankou_tanki_double => cap += 13,
            yaku::ID_KOKUSHI_13 if !conditions.is_kokushi_musou_13machi_double => cap += 13,
            yaku::ID_DAISUUSHI if !conditions.is_daisuushii_double => cap += 13,
            _ => {}
        }
    }

    if cap == 0 {
        return;
    }

    score.han = score.han.saturating_sub(cap).max(13);
    let score_res = score::calculate_score(
        score.han as u8,
        0,
        conditions.player_wind % 4 == Wind::East as u8,
        conditions.tsumo,
        conditions.honba,
        if conditions.is_sanma { 3 } else { 4 },
    );
    score.ron_agari = score_res.pay_ron;
    score.tsumo_agari_oya = score_res.pay_tsumo_oya;
    score.tsumo_agari_ko = score_res.pay_tsumo_ko;
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
    let conditions = cond_input.to_conditions();

    let result = if cond_input.is_sanma {
        let evaluator = HandEvaluator3P::new(tiles, melds);
        evaluator.calc(win_tile, dora_indicators, ura_indicators, Some(conditions))
    } else {
        let evaluator = HandEvaluator::new(tiles, melds);
        evaluator.calc(win_tile, dora_indicators, ura_indicators, Some(conditions))
    };

    let mut score = ScoreResult {
        is_win: result.is_win,
        yakuman: result.yakuman,
        han: result.han,
        fu: result.fu,
        ron_agari: result.ron_agari,
        tsumo_agari_oya: result.tsumo_agari_oya,
        tsumo_agari_ko: result.tsumo_agari_ko,
        yaku: result.yaku,
    };
    apply_double_yakuman_rules(&mut score, &cond_input);

    serde_wasm_bindgen::to_value(&score)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Convert MJAI tile string to 136-encoding tile ID.
#[wasm_bindgen]
pub fn mjai_to_tile_id(mjai: &str) -> Option<u8> {
    mjai_to_tid(mjai)
}

#[derive(serde::Serialize)]
struct SpCandidateJs {
    tile: u8,
    exp_value: f32,
    tenpai_prob: f32,
    win_prob: f32,
    num_required_tiles: f32,
    num_yaku_progress_tiles: f32,
    min_point: f32,
    mean_point: f32,
    max_point: f32,
    yaku_mask: u32,
    required_tiles: Vec<f32>,
    yaku_progress_tiles: Vec<f32>,
    point_achievement_probs: Vec<f32>,
    tenpai_series: Vec<f32>,
    win_series: Vec<f32>,
    ev_series: Vec<f32>,
}

#[derive(serde::Serialize)]
struct SpFeaturesJs {
    sp_channels: usize,
    tile_max: usize,
    encoded: Vec<f32>,
    candidates: Vec<SpCandidateJs>,
}

#[derive(serde::Serialize)]
struct DrevFeaturesJs {
    drev_channels: usize,
    tile_max: usize,
    encoded: Vec<f32>,
    anpai_norm: Vec<f32>,
    suji_norm: Vec<f32>,
    kabe: Vec<f32>,
    reach_genbutsu_norm: Vec<f32>,
    n_reach_norm: f32,
    opp_tenpai_prob: Vec<f32>,
    threat: Vec<f32>,
}

/// Compute SP features for a hand state.
///
/// Input JSON matches `SpInput`. Returns `{ sp_channels, tile_max, encoded,
/// candidates }`. `encoded` is the row-major `[SP_CHANNELS][TILE_MAX]`
/// channel buffer (length 178 * 34); `candidates` is a per-discard summary
/// sorted by descending immediate EV (matches the in-engine ordering).
#[wasm_bindgen]
pub fn calc_sp_features(input_json: &str) -> Result<JsValue, JsValue> {
    let input: SpInput = serde_json::from_str(input_json)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse SpInput: {}", e)))?;

    let result = sp::calculate_sp(&input);
    let encoded = encode_sp(&result);

    let candidates: Vec<SpCandidateJs> = result
        .candidates
        .iter()
        .map(|c| SpCandidateJs {
            tile: c.tile,
            exp_value: c.exp_values[0],
            tenpai_prob: c.tenpai_probs[0],
            win_prob: c.win_probs[0],
            num_required_tiles: c.num_required_tiles,
            num_yaku_progress_tiles: c.num_yaku_progress_tiles,
            min_point: c.min_point,
            mean_point: c.mean_point,
            max_point: c.max_point,
            yaku_mask: c.yaku_mask,
            required_tiles: c.required_tiles.to_vec(),
            yaku_progress_tiles: c.yaku_progress_tiles.to_vec(),
            point_achievement_probs: c.point_achievement_probs.to_vec(),
            tenpai_series: c.tenpai_probs.to_vec(),
            win_series: c.win_probs.to_vec(),
            ev_series: c.exp_values.to_vec(),
        })
        .collect();

    let out = SpFeaturesJs {
        sp_channels: sp::SP_CHANNELS,
        tile_max: TILE_MAX,
        encoded,
        candidates,
    };
    serde_wasm_bindgen::to_value(&out)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Compute Deal-in Risk EV (DREV) features for a hand state.
///
/// Input JSON matches `DrevInput`. Returns `{ drev_channels, tile_max,
/// encoded, anpai_norm, suji_norm, kabe, reach_genbutsu_norm, n_reach_norm,
/// opp_tenpai_prob, threat }`. `encoded` is the row-major
/// `[DREV_CHANNELS][TILE_MAX]` channel buffer (length 9 * 34).
#[wasm_bindgen]
pub fn calc_drev_features(input_json: &str) -> Result<JsValue, JsValue> {
    let input: DrevInput = serde_json::from_str(input_json)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse DrevInput: {}", e)))?;

    let result = drev::calculate_drev(&input);
    let encoded = encode_drev(&result);

    let out = DrevFeaturesJs {
        drev_channels: drev::DREV_CHANNELS,
        tile_max: TILE_MAX,
        encoded,
        anpai_norm: result.anpai_norm.to_vec(),
        suji_norm: result.suji_norm.to_vec(),
        kabe: result.kabe.to_vec(),
        reach_genbutsu_norm: result.reach_genbutsu_norm.to_vec(),
        n_reach_norm: result.n_reach_norm,
        opp_tenpai_prob: result.opp_tenpai_prob.to_vec(),
        threat: result.threat.to_vec(),
    };
    serde_wasm_bindgen::to_value(&out)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Convert 136-encoding tile ID to MJAI tile string.
#[wasm_bindgen]
pub fn tile_id_to_mjai(tid: u8) -> String {
    tid_to_mjai(tid)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn double_yakuman_score(yaku_id: u32) -> ScoreResult {
        ScoreResult {
            is_win: true,
            yakuman: true,
            han: 26,
            fu: 0,
            ron_agari: 64000,
            tsumo_agari_oya: 32000,
            tsumo_agari_ko: 16000,
            yaku: vec![yaku_id],
        }
    }

    #[test]
    fn tenhou_default_caps_suuankou_tanki_to_single_yakuman() {
        let conditions = ConditionsInput {
            player_wind: Wind::South as u8,
            ..Default::default()
        };
        let mut score = double_yakuman_score(yaku::ID_SUANKO_TANKI);

        apply_double_yakuman_rules(&mut score, &conditions);

        assert_eq!(score.han, 13);
        assert_eq!(score.ron_agari, 32000);
    }

    #[test]
    fn enabled_suuankou_tanki_rule_keeps_double_yakuman() {
        let conditions = ConditionsInput {
            player_wind: Wind::South as u8,
            is_suuankou_tanki_double: true,
            ..Default::default()
        };
        let mut score = double_yakuman_score(yaku::ID_SUANKO_TANKI);

        apply_double_yakuman_rules(&mut score, &conditions);

        assert_eq!(score.han, 26);
        assert_eq!(score.ron_agari, 64000);
    }
}
