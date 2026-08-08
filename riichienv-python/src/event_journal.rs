use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use flate2::read::MultiGzDecoder;
use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use riichienv_core::replay::{
    EVENT_JOURNAL_SCHEMA_VERSION, EventCursor, EventJournal as CoreEventJournal,
};

type KyokuSpanTuple = (usize, usize, Option<String>, Option<u8>, Option<u8>);

/// Append-only MJAI event journal for replay and delayed spectators.
#[pyclass(name = "EventJournal", module = "riichienv._riichienv")]
#[derive(Debug, Default)]
pub struct EventJournal {
    inner: CoreEventJournal,
}

impl EventJournal {
    pub(crate) fn from_core(inner: CoreEventJournal) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl EventJournal {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Load a plain or gzip-compressed MJAI JSONL file.
    #[staticmethod]
    fn from_jsonl(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let inner = py.detach(|| load_jsonl(&path))?;
        Ok(Self { inner })
    }

    /// Build a journal from MJAI JSONL text.
    #[staticmethod]
    fn from_jsonl_text(py: Python<'_>, jsonl: String) -> PyResult<Self> {
        let inner = py
            .detach(|| CoreEventJournal::from_jsonl(&jsonl))
            .map_err(PyErr::from)?;
        Ok(Self { inner })
    }

    /// Build a journal from raw MJAI JSON event strings.
    #[staticmethod]
    fn from_events(py: Python<'_>, events: Vec<String>) -> PyResult<Self> {
        let inner = py
            .detach(|| CoreEventJournal::from_events(events))
            .map_err(PyErr::from)?;
        Ok(Self { inner })
    }

    /// Append one raw MJAI JSON event and return the new revision cursor.
    fn push_json(&mut self, py: Python<'_>, event: String) -> PyResult<usize> {
        py.detach(|| self.inner.push_json(event))
            .map(EventCursor::index)
            .map_err(PyErr::from)
    }

    #[getter]
    fn revision(&self) -> usize {
        self.inner.revision().index()
    }

    #[getter]
    fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }

    #[getter]
    fn has_in_progress_kyoku(&self) -> bool {
        self.inner.has_in_progress_kyoku()
    }

    #[getter]
    fn current_kyoku_start(&self) -> Option<usize> {
        self.inner.current_kyoku_start().map(EventCursor::index)
    }

    #[getter]
    fn events(&self, py: Python<'_>) -> Vec<String> {
        py.detach(|| self.inner.events().to_vec())
    }

    /// Completed kyokus as `(start, end, bakaze, kyoku, honba)` tuples.
    #[getter]
    fn completed_kyokus(&self, py: Python<'_>) -> Vec<KyokuSpanTuple> {
        py.detach(|| {
            self.inner
                .completed_kyokus()
                .iter()
                .map(|span| {
                    (
                        span.start.index(),
                        span.end.index(),
                        span.key.bakaze.clone(),
                        span.key.kyoku,
                        span.key.honba,
                    )
                })
                .collect()
        })
    }

    fn events_since(&self, py: Python<'_>, cursor: usize) -> PyResult<Vec<String>> {
        py.detach(|| {
            self.inner
                .events_since(EventCursor(cursor))
                .map(<[String]>::to_vec)
        })
        .map_err(PyErr::from)
    }

    /// Schema-v1 full-log cursor envelope shared with WASM and TypeScript.
    fn event_batch_since(&self, py: Python<'_>, cursor: usize) -> PyResult<Py<PyAny>> {
        let (from, to, complete, events) = py
            .detach(|| {
                self.inner
                    .event_batch_since(EventCursor(cursor))
                    .map(|batch| {
                        (
                            batch.from.index(),
                            batch.to.index(),
                            batch.complete,
                            batch.events.to_vec(),
                        )
                    })
            })
            .map_err(PyErr::from)?;
        batch_dict(py, from, to, complete, &events)
    }

    fn events_for_kyoku(&self, py: Python<'_>, index: usize) -> PyResult<Vec<String>> {
        py.detach(|| self.inner.events_for_kyoku(index).map(<[String]>::to_vec))
            .map_err(PyErr::from)
    }

    fn prefix_through_completed_kyoku(
        &self,
        py: Python<'_>,
        index: usize,
    ) -> PyResult<Vec<String>> {
        py.detach(|| {
            self.inner
                .prefix_through_completed_kyoku(index)
                .map(<[String]>::to_vec)
        })
        .map_err(PyErr::from)
    }

    #[pyo3(signature = (delay_kyokus=1))]
    fn spectator_end(&self, delay_kyokus: usize) -> usize {
        self.inner.spectator_end(delay_kyokus).index()
    }

    #[pyo3(signature = (delay_kyokus=1))]
    fn spectator_prefix(&self, py: Python<'_>, delay_kyokus: usize) -> Vec<String> {
        py.detach(|| self.inner.spectator_prefix(delay_kyokus).to_vec())
    }

    #[pyo3(signature = (cursor, delay_kyokus=1))]
    fn spectator_events_since(
        &self,
        py: Python<'_>,
        cursor: usize,
        delay_kyokus: usize,
    ) -> PyResult<Vec<String>> {
        py.detach(|| {
            self.inner
                .spectator_events_since(EventCursor(cursor), delay_kyokus)
                .map(<[String]>::to_vec)
        })
        .map_err(PyErr::from)
    }

    /// Schema-v1 delayed-spectator cursor envelope.
    #[pyo3(signature = (cursor, delay_kyokus=1))]
    fn spectator_batch_since(
        &self,
        py: Python<'_>,
        cursor: usize,
        delay_kyokus: usize,
    ) -> PyResult<Py<PyAny>> {
        let (from, to, complete, events) = py
            .detach(|| {
                self.inner
                    .spectator_batch_since(EventCursor(cursor), delay_kyokus)
                    .map(|batch| {
                        (
                            batch.from.index(),
                            batch.to.index(),
                            batch.complete,
                            batch.events.to_vec(),
                        )
                    })
            })
            .map_err(PyErr::from)?;
        batch_dict(py, from, to, complete, &events)
    }

    /// Return the newly visible spectator events and their end cursor.
    #[pyo3(signature = (cursor, delay_kyokus=1))]
    fn spectator_delta(
        &self,
        py: Python<'_>,
        cursor: usize,
        delay_kyokus: usize,
    ) -> PyResult<(usize, Vec<String>)> {
        py.detach(|| {
            let end = self.inner.spectator_end(delay_kyokus);
            let events = self
                .inner
                .spectator_events_since(EventCursor(cursor), delay_kyokus)?
                .to_vec();
            Ok::<_, riichienv_core::errors::RiichiError>((end.index(), events))
        })
        .map_err(PyErr::from)
    }

    fn to_jsonl(&self, py: Python<'_>) -> String {
        py.detach(|| self.inner.to_jsonl())
    }

    fn __len__(&self) -> usize {
        self.inner.events().len()
    }

    fn __repr__(&self) -> String {
        format!(
            "EventJournal(revision={}, completed_kyokus={}, is_complete={})",
            self.inner.revision().index(),
            self.inner.completed_kyokus().len(),
            self.inner.is_complete()
        )
    }
}

fn batch_dict(
    py: Python<'_>,
    from: usize,
    to: usize,
    complete: bool,
    events: &[String],
) -> PyResult<Py<PyAny>> {
    let result = PyDict::new(py);
    result.set_item("schemaVersion", EVENT_JOURNAL_SCHEMA_VERSION)?;
    result.set_item("complete", complete)?;
    result.set_item("from", from)?;
    result.set_item("to", to)?;
    result.set_item("events", events)?;
    Ok(result.unbind().into())
}

fn load_jsonl(path: &Path) -> PyResult<CoreEventJournal> {
    let file = File::open(path).map_err(|error| {
        PyOSError::new_err(format!(
            "failed to open JSONL file '{}': {error}",
            path.display()
        ))
    })?;
    let mut reader = BufReader::new(file);
    let is_gzip = reader
        .fill_buf()
        .map_err(|error| {
            PyOSError::new_err(format!(
                "failed to inspect JSONL file '{}': {error}",
                path.display()
            ))
        })?
        .starts_with(&[0x1f, 0x8b]);

    if is_gzip {
        CoreEventJournal::from_jsonl_reader(BufReader::new(MultiGzDecoder::new(reader)))
            .map_err(PyErr::from)
    } else {
        CoreEventJournal::from_jsonl_reader(reader).map_err(PyErr::from)
    }
}
