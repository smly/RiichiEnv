use std::collections::HashMap;

use crate::types::Meld;

#[derive(Debug, Clone)]
pub struct PlayerState {
    pub hand: Vec<u8>,
    pub melds: Vec<Meld>,
    pub discards: Vec<u8>,
    pub discard_from_hand: Vec<bool>,
    pub discard_is_riichi: Vec<bool>,
    pub riichi_declaration_index: Option<usize>,
    pub score: i32,
    pub score_delta: i32,
    pub riichi_declared: bool,
    pub riichi_stage: bool,
    pub double_riichi_declared: bool,
    pub missed_agari_riichi: bool,
    pub missed_agari_doujun: bool,
    pub nagashi_eligible: bool,
    pub ippatsu_cycle: bool,
    pub pao: HashMap<u8, u8>,
    pub forbidden_discards: Vec<u8>,
    pub mjai_log: Vec<String>,
}

impl PlayerState {
    pub fn new(starting_score: i32) -> Self {
        Self {
            hand: Vec::new(),
            melds: Vec::new(),
            discards: Vec::new(),
            discard_from_hand: Vec::new(),
            discard_is_riichi: Vec::new(),
            riichi_declaration_index: None,
            score: starting_score,
            score_delta: 0,
            riichi_declared: false,
            riichi_stage: false,
            double_riichi_declared: false,
            missed_agari_riichi: false,
            missed_agari_doujun: false,
            nagashi_eligible: true,
            ippatsu_cycle: false,
            pao: HashMap::new(),
            forbidden_discards: Vec::new(),
            mjai_log: Vec::new(),
        }
    }

    pub fn reset_round(&mut self) {
        self.hand.clear();
        self.melds.clear();
        self.discards.clear();
        self.discard_from_hand.clear();
        self.discard_is_riichi.clear();
        self.riichi_declaration_index = None;
        self.riichi_declared = false;
        self.riichi_stage = false;
        self.double_riichi_declared = false;
        self.missed_agari_riichi = false;
        self.missed_agari_doujun = false;
        self.nagashi_eligible = true;
        self.ippatsu_cycle = false;
        self.forbidden_discards.clear();
        self.mjai_log.clear();
        // pao is usually cleared? Original code: self.pao = [HashMap::new(); 4]; in _initialize_round
        self.pao.clear();
    }
}
