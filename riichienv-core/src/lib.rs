mod agari;
pub mod errors;
pub mod hand_evaluator;
pub mod hand_evaluator_3p;
pub mod score;
mod tests;
pub mod types;
mod yaku;
mod yaku_3p;

pub mod action;
pub mod game_variant;
pub mod observation;
pub mod observation_3p;
pub mod parser;
pub mod replay;
pub mod rule;
mod shanten;
pub mod state;
pub mod state_3p;
pub mod win_projection;
#[cfg(feature = "python")]
mod yaku_checker;

pub use hand_evaluator::check_riichi_candidates;
