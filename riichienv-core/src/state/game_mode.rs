use crate::rule::GameRule;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GameSubMode {
    Single = 0,
    East = 1,
    Half = 2,
}

/// 4P-only game mode configuration.
/// 3P games use `state_3p::GameState3P` instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameModeConfig {
    pub sub_mode: GameSubMode,
    pub rule: GameRule,
}

impl GameModeConfig {
    pub fn from_game_mode(mode: u8, rule: GameRule) -> Self {
        let sub_mode = match mode {
            0 => GameSubMode::Single,
            1 => GameSubMode::East,
            2 => GameSubMode::Half,
            _ => GameSubMode::East,
        };
        GameModeConfig { sub_mode, rule }
    }

    pub fn num_players(&self) -> u8 {
        4
    }

    pub fn starting_score(&self) -> i32 {
        25000
    }

    pub fn tenpai_pool(&self) -> i32 {
        3000
    }

    pub fn get_next_dora_tile(&self, tile: u8) -> u8 {
        standard_next_dora_tile(tile)
    }

    pub fn rule(&self) -> &GameRule {
        &self.rule
    }

    pub fn game_mode_id(&self) -> u8 {
        self.sub_mode as u8
    }

    pub fn sub_mode(&self) -> &GameSubMode {
        &self.sub_mode
    }
}

// Re-export shared utilities from types module
pub use crate::types::{is_sanma_excluded_tile, standard_next_dora_tile};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::GameRule;

    #[test]
    fn test_game_mode_config_four_player() {
        let mode = GameModeConfig::from_game_mode(2, GameRule::default_tenhou());
        assert_eq!(mode.num_players(), 4);
        assert_eq!(mode.starting_score(), 25000);
        assert_eq!(mode.tenpai_pool(), 3000);
        assert_eq!(mode.game_mode_id(), 2);
    }

    #[test]
    fn test_sanma_excluded_tiles() {
        assert!(!is_sanma_excluded_tile(0));
        assert!(!is_sanma_excluded_tile(3));
        assert!(is_sanma_excluded_tile(4));
        assert!(is_sanma_excluded_tile(7));
        assert!(is_sanma_excluded_tile(28));
        assert!(is_sanma_excluded_tile(31));
        assert!(!is_sanma_excluded_tile(32));
        assert!(!is_sanma_excluded_tile(35));
        assert!(!is_sanma_excluded_tile(36));
        assert!(!is_sanma_excluded_tile(72));
        assert!(!is_sanma_excluded_tile(108));
        assert!(!is_sanma_excluded_tile(135));
    }

    #[test]
    fn test_four_player_dora_wrapping() {
        let mode = GameModeConfig::from_game_mode(0, GameRule::default_tenhou());
        assert_eq!(mode.get_next_dora_tile(0), 1); // 1m -> 2m
        assert_eq!(mode.get_next_dora_tile(8), 0); // 9m -> 1m
        assert_eq!(mode.get_next_dora_tile(27), 28); // E -> S
        assert_eq!(mode.get_next_dora_tile(30), 27); // N -> E
        assert_eq!(mode.get_next_dora_tile(31), 32); // Haku -> Hatsu
        assert_eq!(mode.get_next_dora_tile(33), 31); // Chun -> Haku
    }
}
