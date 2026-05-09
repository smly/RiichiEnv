pub mod agari;
pub mod agari_table;
pub mod sp_yaku;
pub mod errors;
pub mod hand_evaluator;
pub mod hand_evaluator_3p;
pub mod score;
mod tests;
pub mod types;
pub mod yaku;
mod yaku_3p;

pub mod action;
pub mod game_variant;
pub mod observation;
pub mod observation_3p;
pub mod parser;
pub mod replay;
pub mod rule;
pub mod shanten;
pub mod sp;
pub mod state;

// DEBUG-ONLY: links against AGPL Mortal libriichi for SP feature numerical
// comparison. Gated behind `debug_mortal_sp` feature; never enabled in
// release/wheel builds. See src/debug_only_mortal_sp/.
#[cfg(feature = "debug_mortal_sp")]
mod debug_only_mortal_sp;
pub mod state_3p;
#[cfg(feature = "python")]
mod yaku_checker;

pub use hand_evaluator::check_riichi_candidates;
