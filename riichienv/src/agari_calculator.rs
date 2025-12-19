use crate::types::{Hand, Meld, Conditions, Agari, MeldType};
use crate::agari;
use crate::score;
use crate::yaku;
use pyo3::prelude::*;

#[pyclass]
pub struct AgariCalculator {
    pub hand: Hand, // Normalised for agari detection
    pub full_hand: Hand, // Full counts for dora/yaku
    pub melds: Vec<Meld>,
    pub aka_dora_count: u8,
}

#[pymethods]
impl AgariCalculator {
    #[new]
    #[pyo3(signature = (tiles_136, melds=vec![]))]
    pub fn new(tiles_136: Vec<u8>, mut melds: Vec<Meld>) -> Self {
        let mut aka_dora_count = 0;
        let mut tiles_34 = Vec::with_capacity(tiles_136.len());
        
        for t in tiles_136 {
            if t == 16 || t == 52 || t == 88 {
                aka_dora_count += 1;
            }
            tiles_34.push(t / 4);
        }

        let mut full_hand = Hand::new(Some(tiles_34));
        let mut hand = full_hand.clone();

        for meld in &mut melds {
            // Reduce Kongs to triplets for agari detection
            if meld.meld_type == MeldType::Gang || meld.meld_type == MeldType::Angang || meld.meld_type == MeldType::Addgang {
                let t_34 = meld.tiles[0] / 4;
                if hand.counts[t_34 as usize] == 4 {
                    hand.counts[t_34 as usize] = 3;
                }
            }

            // Convert meld tiles to 34-tile IDs
            let mut meld_tiles_34 = Vec::with_capacity(meld.tiles.len());
            for &t in &meld.tiles {
                if t == 16 || t == 52 || t == 88 {
                    aka_dora_count += 1;
                }
                let t_34 = t / 4;
                meld_tiles_34.push(t_34);
                full_hand.add(t_34);
            }
            meld.tiles = meld_tiles_34;
        }

        Self {
            hand,
            full_hand,
            melds,
            aka_dora_count,
        }
    }

    pub fn calc(&self, win_tile_136: u8, dora_indicators_136: Vec<u8>, conditions: Conditions) -> Agari {
        let win_tile_34 = win_tile_136 / 4;
        let is_agari = agari::is_agari(&self.hand);
        if !is_agari {
            return Agari::new(false, false, 0, 0, 0, vec![], 0, 0);
        }

        let mut ctx = yaku::YakuContext::default();
        ctx.is_tsumo = conditions.tsumo;
        ctx.is_reach = conditions.riichi;
        ctx.is_daburu_reach = conditions.double_riichi;
        ctx.is_ippatsu = conditions.ippatsu;
        ctx.is_haitei = conditions.haitei;
        ctx.is_houtei = conditions.houtei;
        ctx.is_rinshan = conditions.rinshan;
        ctx.is_chankan = conditions.chankan;
        
        // Count doras
        let mut dora_count = 0;
        for &indicator_136 in &dora_indicators_136 {
            let next_tile_34 = get_next_tile(indicator_136 / 4);
            dora_count += self.full_hand.counts[next_tile_34 as usize];
        }
        ctx.dora_count = dora_count;
        ctx.aka_dora = self.aka_dora_count;

        ctx.bakaze = 27 + conditions.round_wind;
        ctx.jikaze = 27 + conditions.player_wind;
        ctx.is_menzen = self.melds.iter().all(|m| !m.opened);

        let yaku_res = yaku::calculate_yaku(&self.hand, &self.melds, &ctx, win_tile_34);

        let is_oya = conditions.player_wind == 0;
        let score_res = score::calculate_score(yaku_res.han, yaku_res.fu, is_oya, conditions.tsumo);

        Agari {
            agari: true,
            yakuman: yaku_res.han >= 13,
            ron_agari: score_res.pay_ron,
            tsumo_agari_oya: score_res.pay_tsumo_oya,
            tsumo_agari_ko: score_res.pay_tsumo_ko,
            yaku: yaku_res.yaku_ids,
            han: yaku_res.han as u32,
            fu: yaku_res.fu as u32,
        }
    }
}

fn get_next_tile(tile: u8) -> u8 {
    if tile < 9 { // man
        if tile == 8 { 0 } else { tile + 1 }
    } else if tile < 18 { // pin
        if tile == 17 { 9 } else { tile + 1 }
    } else if tile < 27 { // sou
        if tile == 26 { 18 } else { tile + 1 }
    } else if tile < 31 { // winds
        if tile == 30 { 27 } else { tile + 1 }
    } else { // dragons
        if tile == 33 { 31 } else { tile + 1 }
    }
}
