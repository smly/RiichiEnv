#![allow(clippy::useless_conversion)]
use crate::agari;
use crate::score;
use crate::types::{Agari, Conditions, Hand, Meld, MeldType, Wind};
use crate::yaku;
use pyo3::prelude::*;

#[pyclass]
pub struct AgariCalculator {
    pub hand: Hand,      // Normalised for agari detection
    pub full_hand: Hand, // Full counts for dora/yaku
    pub melds: Vec<Meld>,
    pub aka_dora_count: u8,
}

#[pymethods]
impl AgariCalculator {
    #[staticmethod]
    pub fn hand_from_text(text: &str) -> PyResult<Self> {
        let (tiles, melds) = crate::parser::parse_hand_internal(text)?;
        Ok(Self::new(tiles, melds))
    }
    #[new]
    #[pyo3(signature = (tiles_136, melds=vec![]))]
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

        // Clone melds to avoid mutating the Python objects passed in
        let mut internal_melds = Vec::with_capacity(melds.len());

        for meld in &melds {
            let mut new_meld = meld.clone();

            // Reduce Kongs to triplets for agari detection
            if new_meld.meld_type == MeldType::Gang
                || new_meld.meld_type == MeldType::Angang
                || new_meld.meld_type == MeldType::Addgang
            {
                let t_34 = new_meld.tiles[0] / 4;
                if hand.counts[t_34 as usize] == 4 {
                    hand.counts[t_34 as usize] = 3;
                }
            }

            // Convert meld tiles to 34-tile IDs
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

    #[pyo3(signature = (win_tile, dora_indicators=vec![], ura_indicators=vec![], conditions=None))]
    pub fn calc(
        &self,
        win_tile: u8, // Renamed from win_tile_136
        dora_indicators: Vec<u8>,
        ura_indicators: Vec<u8>,
        conditions: Option<Conditions>,
    ) -> Agari {
        let win_tile_136 = win_tile;
        let conditions = conditions.unwrap_or_default();
        let win_tile_34 = win_tile_136 / 4;

        // Clone and add win tile to create 14-tile hands for check
        let mut hand_14 = self.hand.clone();
        let mut full_hand_14 = self.full_hand.clone();

        // Total tiles in agari-equivalent hand (Kans reduced to 3)
        let current_total: u8 = hand_14.counts.iter().sum::<u8>() + (self.melds.len() as u8 * 3);

        if current_total == 13 {
            hand_14.add(win_tile_34);
            full_hand_14.add(win_tile_34);
        } else if current_total != 14 {
            // Unexpected hand size, but we'll try detection anyway or return false?
            // Usually it should be 14. If it's 11 (2 melds missing?), etc.
            // But let's assume it should be 14 for calc.
        }

        if std::env::var("DEBUG").is_ok() {
            eprintln!(
                "DEBUG RUST: AgariCalculator::calc win_tile_136={} houtei={} haitei={} tsumo={}",
                win_tile_136, conditions.houtei, conditions.haitei, conditions.tsumo
            );
        }
        let is_agari = agari::is_agari(&mut hand_14);

        if !is_agari {
            return Agari::new(false, false, 0, 0, 0, vec![], 0, 0);
        }

        // Count normal doras in 14-tile hand
        let mut dora_count = 0;
        for &indicator_136 in &dora_indicators {
            let next_tile_34 = get_next_tile(indicator_136 / 4);
            dora_count += full_hand_14.counts[next_tile_34 as usize];
        }

        // Count ura doras in 14-tile hand
        let mut ura_dora_count = 0;
        for &indicator_136 in &ura_indicators {
            let next_tile_34 = get_next_tile(indicator_136 / 4);
            ura_dora_count += full_hand_14.counts[next_tile_34 as usize];
        }

        // Handle red win_tile
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
            bakaze: 27 + conditions.round_wind as u8,
            jikaze: 27 + conditions.player_wind as u8,
            is_menzen: self.melds.iter().all(|m| !m.opened),
        };

        let _divisions = agari::find_divisions(&hand_14);
        let yaku_res = yaku::calculate_yaku(&hand_14, &self.melds, &ctx, win_tile_34);

        let is_oya = conditions.player_wind == Wind::East;
        let score_res = score::calculate_score(yaku_res.han, yaku_res.fu, is_oya, conditions.tsumo);

        let has_yaku = yaku_res
            .yaku_ids
            .iter()
            .any(|&id| id != yaku::ID_DORA && id != yaku::ID_AKADORA && id != yaku::ID_URADORA);

        Agari {
            agari: (has_yaku || yaku_res.yakuman_count > 0) && yaku_res.han >= 1, // Ensure at least 1 han even if just from Yaku (implicit)
            yakuman: yaku_res.yakuman_count > 0,
            ron_agari: score_res.pay_ron,
            tsumo_agari_oya: score_res.pay_tsumo_oya,
            tsumo_agari_ko: score_res.pay_tsumo_ko,
            yaku: yaku_res.yaku_ids,
            han: yaku_res.han as u32,
            fu: yaku_res.fu as u32,
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

fn get_next_tile(tile: u8) -> u8 {
    if tile < 9 {
        // man
        if tile == 8 {
            0
        } else {
            tile + 1
        }
    } else if tile < 18 {
        // pin
        if tile == 17 {
            9
        } else {
            tile + 1
        }
    } else if tile < 27 {
        // sou
        if tile == 26 {
            18
        } else {
            tile + 1
        }
    } else if tile < 31 {
        // winds
        if tile == 30 {
            27
        } else {
            tile + 1
        }
    } else {
        // dragons
        if tile == 33 {
            31
        } else {
            tile + 1
        }
    }
}
