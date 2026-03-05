#![allow(clippy::useless_conversion)]
use crate::agari;
use crate::errors::RiichiResult;
use crate::score;
use crate::types::{Conditions, Hand, Meld, MeldType, WinResult, Wind};
use crate::yaku;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg_attr(feature = "python", pyclass)]
pub struct HandEvaluator {
    pub hand: Hand,      // Normalised for agari detection
    pub full_hand: Hand, // Full counts for dora/yaku
    pub melds: Vec<Meld>,
    pub aka_dora_count: u8,
}

impl HandEvaluator {
    pub fn hand_from_text(text: &str) -> RiichiResult<Self> {
        let (tiles, melds) = crate::parser::parse_hand_internal(text)?;
        Ok(Self::new(tiles, melds))
    }

    pub fn new(tiles_136: Vec<u8>, melds: Vec<Meld>) -> Self {
        let mut aka_dora_count = 0;
        let mut tiles_34 = Vec::with_capacity(tiles_136.len());

        for &t in &tiles_136 {
            if t == 16 || t == 52 || t == 88 {
                aka_dora_count += 1;
            }
            tiles_34.push(t / 4);
        }

        let mut full_hand = Hand::new(Some(tiles_34));
        let mut hand = full_hand.clone();

        let mut internal_melds = Vec::with_capacity(melds.len());

        for meld in &melds {
            let mut new_meld = meld.clone();

            if new_meld.meld_type == MeldType::Daiminkan
                || new_meld.meld_type == MeldType::Ankan
                || new_meld.meld_type == MeldType::Kakan
            {
                let t_34 = new_meld.tiles[0] / 4;
                if hand.counts[t_34 as usize] == 4 {
                    hand.counts[t_34 as usize] = 3;
                }
            }

            let mut meld_tiles_34 = Vec::with_capacity(new_meld.tiles.len());
            for &t in &new_meld.tiles {
                if t == 16 || t == 52 || t == 88 {
                    aka_dora_count += 1;
                }
                let t_34 = t / 4;
                meld_tiles_34.push(t_34);
                full_hand.add(t_34);
            }
            new_meld.tiles = meld_tiles_34;
            if new_meld.meld_type == MeldType::Chi {
                new_meld.tiles.sort();
            }
            internal_melds.push(new_meld);
        }

        Self {
            hand,
            full_hand,
            melds: internal_melds,
            aka_dora_count,
        }
    }

    pub fn calc(
        &self,
        win_tile: u8,
        dora_indicators: Vec<u8>,
        ura_indicators: Vec<u8>,
        conditions: Option<Conditions>,
    ) -> WinResult {
        let win_tile_136 = win_tile;
        let conditions = conditions.unwrap_or_default();
        let win_tile_34 = win_tile_136 / 4;

        let mut hand_14 = self.hand.clone();
        let mut full_hand_14 = self.full_hand.clone();

        let current_total: u8 = hand_14.counts.iter().sum::<u8>() + (self.melds.len() as u8 * 3);

        if current_total == 13 {
            hand_14.add(win_tile_34);
            full_hand_14.add(win_tile_34);
        }

        let is_agari = agari::is_agari(&mut hand_14);

        if !is_agari {
            return WinResult::new(false, false, 0, 0, 0, vec![], 0, 0, None, false);
        }

        let mut dora_count = 0;
        for &indicator_136 in &dora_indicators {
            let next_tile_34 = get_next_tile(indicator_136 / 4);
            dora_count += full_hand_14.counts[next_tile_34 as usize];
        }

        let mut ura_dora_count = 0;
        for &indicator_136 in &ura_indicators {
            let next_tile_34 = get_next_tile(indicator_136 / 4);
            ura_dora_count += full_hand_14.counts[next_tile_34 as usize];
        }

        let mut aka_dora = self.aka_dora_count;
        if current_total == 13 && (win_tile_136 == 16 || win_tile_136 == 52 || win_tile_136 == 88) {
            aka_dora += 1;
        }

        let ctx = yaku::YakuContext {
            is_tsumo: conditions.tsumo,
            is_reach: conditions.riichi,
            is_daburu_reach: conditions.double_riichi,
            is_ippatsu: conditions.ippatsu,
            is_haitei: conditions.haitei,
            is_houtei: conditions.houtei,
            is_rinshan: conditions.rinshan,
            is_chankan: conditions.chankan,
            is_tsumo_first_turn: conditions.tsumo_first_turn,
            dora_count,
            aka_dora,
            ura_dora_count,
            round_wind: 27 + conditions.round_wind as u8,
            seat_wind: 27 + conditions.player_wind as u8,
            is_menzen: self.melds.iter().all(|m| !m.opened),
        };

        let _divisions = agari::find_divisions(&hand_14);
        let yaku_res = yaku::calculate_yaku(&hand_14, &self.melds, &ctx, win_tile_34);

        let is_oya = conditions.player_wind == Wind::East;
        // Kazoe yakuman: cap han at 13 for scoring (single yakuman)
        let scoring_han = if yaku_res.yakuman_count == 0 && yaku_res.han >= 13 {
            13
        } else {
            yaku_res.han
        };
        let score_res = score::calculate_score(
            scoring_han,
            yaku_res.fu,
            is_oya,
            conditions.tsumo,
            conditions.honba,
            4, // Always 4 players
        );
        let has_yaku = yaku_res
            .yaku_ids
            .iter()
            .any(|&id| id != yaku::ID_DORA && id != yaku::ID_AKADORA && id != yaku::ID_URADORA);

        let official_yaku: Vec<u32> = yaku_res.yaku_ids.into_iter().collect();

        WinResult {
            is_win: (has_yaku || yaku_res.yakuman_count > 0) && yaku_res.han >= 1,
            yakuman: yaku_res.yakuman_count > 0,
            ron_agari: score_res.pay_ron,
            tsumo_agari_oya: score_res.pay_tsumo_oya,
            tsumo_agari_ko: score_res.pay_tsumo_ko,
            yaku: official_yaku,
            han: yaku_res.han as u32,
            fu: yaku_res.fu as u32,
            pao_payer: None,
            has_win_shape: true,
        }
    }

    pub fn is_tenpai(&self) -> bool {
        let current_total: u8 = self.hand.counts.iter().sum::<u8>() + (self.melds.len() as u8 * 3);
        if current_total != 13 {
            return false;
        }
        let mut hand_14 = self.hand.clone();
        for i in 0..crate::types::TILE_MAX {
            if hand_14.counts[i] < 4 {
                hand_14.add(i as u8);
                if agari::is_agari(&mut hand_14) {
                    return true;
                }
                hand_14.remove(i as u8);
            }
        }
        false
    }

    pub fn get_waits_u8(&self) -> Vec<u8> {
        let mut waits = Vec::new();
        let current_total: u8 = self.hand.counts.iter().sum::<u8>() + (self.melds.len() as u8 * 3);
        if current_total != 13 {
            return waits;
        }
        let mut hand_14 = self.hand.clone();
        for i in 0..crate::types::TILE_MAX {
            if hand_14.counts[i] < 4 {
                hand_14.add(i as u8);
                if crate::agari::is_agari(&mut hand_14) {
                    waits.push(i as u8);
                }
                hand_14.remove(i as u8);
            }
        }
        waits
    }

    pub fn get_waits(&self) -> Vec<u32> {
        self.get_waits_u8().iter().map(|&x| x as u32).collect()
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl HandEvaluator {
    #[staticmethod]
    #[pyo3(name = "hand_from_text")]
    pub fn hand_from_text_py(text: &str) -> PyResult<Self> {
        Self::hand_from_text(text).map_err(Into::into)
    }

    #[new]
    #[pyo3(signature = (tiles_136, melds=vec![]))]
    pub fn py_new(tiles_136: Vec<u8>, melds: Vec<Meld>) -> Self {
        Self::new(tiles_136, melds)
    }

    #[pyo3(signature = (win_tile, dora_indicators=vec![], ura_indicators=vec![], conditions=None))]
    #[pyo3(name = "calc")]
    pub fn calc_py(
        &self,
        win_tile: u8,
        dora_indicators: Vec<u8>,
        ura_indicators: Vec<u8>,
        conditions: Option<Conditions>,
    ) -> WinResult {
        self.calc(win_tile, dora_indicators, ura_indicators, conditions)
    }

    #[pyo3(name = "is_tenpai")]
    pub fn is_tenpai_py(&self) -> bool {
        self.is_tenpai()
    }

    #[pyo3(name = "get_waits_u8")]
    pub fn get_waits_u8_py(&self) -> Vec<u8> {
        self.get_waits_u8()
    }

    #[pyo3(name = "get_waits")]
    pub fn get_waits_py(&self) -> Vec<u32> {
        self.get_waits()
    }
}

pub fn check_riichi_candidates(tiles_136: Vec<u8>) -> Vec<u32> {
    let mut candidates = Vec::new();
    // Convert to 34-tile hand
    let mut tiles_34 = Vec::with_capacity(tiles_136.len());
    for t in &tiles_136 {
        tiles_34.push(t / 4);
    }

    for (i, &t_discard) in tiles_136.iter().enumerate() {
        let mut hand = crate::types::Hand::default();
        for (j, &t) in tiles_34.iter().enumerate() {
            if i != j {
                hand.add(t);
            }
        }

        if agari::is_tenpai(&mut hand) {
            candidates.push(t_discard as u32);
        }
    }
    candidates
}

fn get_next_tile(tile: u8) -> u8 {
    if tile < 9 {
        if tile == 8 { 0 } else { tile + 1 }
    } else if tile < 18 {
        if tile == 17 { 9 } else { tile + 1 }
    } else if tile < 27 {
        if tile == 26 { 18 } else { tile + 1 }
    } else if tile < 31 {
        if tile == 30 { 27 } else { tile + 1 }
    } else if tile == 33 {
        31
    } else {
        tile + 1
    }
}
