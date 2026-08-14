use pyo3::prelude::*;

use riichienv_core::drev_validation::validate_drev_replay;
use riichienv_core::replay::ReplayLog;
use riichienv_core::rule::GameRule;

/// Validate a complete MJAI JSONL replay against DREV-v2's hidden-state
/// oracle and return a machine-readable JSON report. Attributed excerpts may
/// opt in to retaining one empty trailing `start_kyoku` header; any event in
/// that trailing kyoku is still rejected.
#[pyfunction]
#[pyo3(
    name = "validate_drev_replay_jsonl",
    signature = (jsonl, rule=None, allow_trailing_start_kyoku=false)
)]
pub fn validate_drev_replay_jsonl_py(
    py: Python<'_>,
    jsonl: &str,
    rule: Option<GameRule>,
    allow_trailing_start_kyoku: bool,
) -> PyResult<String> {
    let jsonl = jsonl.to_owned();
    py.detach(move || {
        let rule = rule.unwrap_or_default();
        let replay = if allow_trailing_start_kyoku {
            ReplayLog::from_jsonl_strict_allowing_trailing_start_kyoku(&jsonl, rule)?
        } else {
            ReplayLog::from_jsonl_strict(&jsonl, rule)?
        };
        let report = validate_drev_replay(&replay)?;
        serde_json::to_string(&report).map_err(|error| {
            riichienv_core::errors::RiichiError::InvalidState {
                message: format!("failed to serialize DREV validation report: {error}"),
            }
        })
    })
    .map_err(Into::into)
}
