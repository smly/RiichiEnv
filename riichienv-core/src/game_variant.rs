use crate::rule::GameRule;
use crate::state::GameState;
use crate::state_3p::GameState3P;

#[derive(Debug, Clone)]
pub enum GameStateVariant {
    FourPlayer(GameState),
    ThreePlayer(GameState3P),
}

impl GameStateVariant {
    pub fn new(
        game_mode: u8,
        skip_mjai_logging: bool,
        seed: Option<u64>,
        round_wind: u8,
        rule: GameRule,
    ) -> Self {
        if game_mode >= 3 {
            GameStateVariant::ThreePlayer(GameState3P::new(
                game_mode,
                skip_mjai_logging,
                seed,
                round_wind,
                rule,
            ))
        } else {
            GameStateVariant::FourPlayer(GameState::new(
                game_mode,
                skip_mjai_logging,
                seed,
                round_wind,
                rule,
            ))
        }
    }

    pub fn num_players(&self) -> u8 {
        match self {
            GameStateVariant::FourPlayer(s) => s.mode.num_players(),
            GameStateVariant::ThreePlayer(_) => 3,
        }
    }

    pub fn is_three_player(&self) -> bool {
        matches!(self, GameStateVariant::ThreePlayer(_))
    }
}
