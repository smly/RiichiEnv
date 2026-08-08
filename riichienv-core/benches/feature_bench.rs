use std::collections::HashMap;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use riichienv_core::action::Action;
use riichienv_core::engine::{
    EngineConfig, EventLogPolicy, GameEngine, GameMode, ObservationVariant,
};
use riichienv_core::features::{
    BASE_3P_V0, BASE_4P_V0, DREV_4P_V0, EXTENDED_3P_V0, EXTENDED_4P_V0, EXTENDED_SP_DREV_4P_V0,
    SP_4P_V0, encode_base_3p_batch, encode_base_3p_batch_into, encode_base_4p_batch,
    encode_base_4p_batch_into, encode_drev_4p_batch, encode_drev_4p_batch_into,
    encode_extended_3p_batch, encode_extended_3p_batch_into, encode_extended_4p_batch,
    encode_extended_4p_batch_into, encode_extended_sp_drev_4p_batch,
    encode_extended_sp_drev_4p_batch_into, encode_sp_4p_batch, encode_sp_4p_batch_into,
};
use riichienv_core::observation::Observation;
use riichienv_core::observation_3p::Observation3P;

const CORPUS_SIZE: usize = 16;
const MAX_DECISIONS_PER_GAME: usize = 512;

/// Build deterministic, progressed observations through the public engine
/// facade. Picking the lowest legal action ID permits response phases and
/// calls instead of benchmarking only a manually constructed opening hand.
fn collect_observations(mode: GameMode, count: usize) -> Vec<ObservationVariant> {
    let mut observations = Vec::with_capacity(count);

    for game_index in 0..64_u64 {
        if observations.len() == count {
            break;
        }

        let config = EngineConfig::new(mode)
            .with_seed(10_000 + game_index)
            .with_event_log(EventLogPolicy::Off);
        let mut engine = GameEngine::new(config).expect("feature benchmark engine must initialize");
        let mut decisions = engine.decisions();

        for _ in 0..MAX_DECISIONS_PER_GAME {
            if decisions.is_empty() || observations.len() == count {
                break;
            }

            let mut actions = HashMap::<u8, Action>::with_capacity(decisions.len());
            for decision in decisions {
                let observation = decision.observation;
                let action_id = observation
                    .legal_action_ids()
                    .expect("benchmark observation must expose legal action IDs")
                    .into_iter()
                    .min()
                    .expect("pending decision must have a legal action");
                let action = observation
                    .select_action(action_id)
                    .expect("selected benchmark action ID must be legal");
                if observations.len() < count {
                    observations.push(observation);
                }
                actions.insert(decision.player_id, action);
            }

            let outcome = engine.step(&actions);
            assert!(
                outcome.error.is_none(),
                "deterministic feature benchmark rollout failed: {:?}",
                outcome.error
            );
            decisions = outcome.decisions;
        }
    }

    assert_eq!(
        observations.len(),
        count,
        "could not collect the requested feature benchmark corpus"
    );
    observations
}

fn four_player_corpus() -> Vec<Observation> {
    collect_observations(GameMode::FourPlayerSingle, CORPUS_SIZE)
        .into_iter()
        .map(|observation| match observation {
            ObservationVariant::FourPlayer(observation) => observation,
            ObservationVariant::ThreePlayer(_) => unreachable!(),
        })
        .collect()
}

fn three_player_corpus() -> Vec<Observation3P> {
    collect_observations(GameMode::ThreePlayerSingle, CORPUS_SIZE)
        .into_iter()
        .map(|observation| match observation {
            ObservationVariant::ThreePlayer(observation) => observation,
            ObservationVariant::FourPlayer(_) => unreachable!(),
        })
        .collect()
}

fn bench_four_player_features(c: &mut Criterion) {
    let observations = four_player_corpus();
    let mut group = c.benchmark_group(format!("features/4p/batch_{CORPUS_SIZE}"));
    group.throughput(Throughput::Elements(CORPUS_SIZE as u64));

    group.bench_function("base/allocate", |b| {
        b.iter(|| black_box(encode_base_4p_batch(black_box(&observations)).unwrap()))
    });
    let mut base_output = vec![0.0; BASE_4P_V0.values_for_batch(CORPUS_SIZE)];
    group.bench_function("base/into", |b| {
        b.iter(|| {
            encode_base_4p_batch_into(black_box(&observations), black_box(&mut base_output))
                .unwrap();
            black_box(&base_output);
        })
    });

    group.bench_function("extended/allocate", |b| {
        b.iter(|| black_box(encode_extended_4p_batch(black_box(&observations)).unwrap()))
    });
    let mut extended_output = vec![0.0; EXTENDED_4P_V0.values_for_batch(CORPUS_SIZE)];
    group.bench_function("extended/into", |b| {
        b.iter(|| {
            encode_extended_4p_batch_into(
                black_box(&observations),
                black_box(&mut extended_output),
            )
            .unwrap();
            black_box(&extended_output);
        })
    });

    group.bench_function("sp/allocate", |b| {
        b.iter(|| black_box(encode_sp_4p_batch(black_box(&observations)).unwrap()))
    });
    let mut sp_output = vec![0.0; SP_4P_V0.values_for_batch(CORPUS_SIZE)];
    group.bench_function("sp/into", |b| {
        b.iter(|| {
            encode_sp_4p_batch_into(black_box(&observations), black_box(&mut sp_output)).unwrap();
            black_box(&sp_output);
        })
    });

    group.bench_function("drev/allocate", |b| {
        b.iter(|| black_box(encode_drev_4p_batch(black_box(&observations)).unwrap()))
    });
    let mut drev_output = vec![0.0; DREV_4P_V0.values_for_batch(CORPUS_SIZE)];
    group.bench_function("drev/into", |b| {
        b.iter(|| {
            encode_drev_4p_batch_into(black_box(&observations), black_box(&mut drev_output))
                .unwrap();
            black_box(&drev_output);
        })
    });

    group.bench_function("extended_sp_drev/allocate", |b| {
        b.iter(|| black_box(encode_extended_sp_drev_4p_batch(black_box(&observations)).unwrap()))
    });
    let mut combined_output = vec![0.0; EXTENDED_SP_DREV_4P_V0.values_for_batch(CORPUS_SIZE)];
    group.bench_function("extended_sp_drev/into", |b| {
        b.iter(|| {
            encode_extended_sp_drev_4p_batch_into(
                black_box(&observations),
                black_box(&mut combined_output),
            )
            .unwrap();
            black_box(&combined_output);
        })
    });

    group.finish();
}

fn bench_three_player_features(c: &mut Criterion) {
    let observations = three_player_corpus();
    let mut group = c.benchmark_group(format!("features/3p/batch_{CORPUS_SIZE}"));
    group.throughput(Throughput::Elements(CORPUS_SIZE as u64));

    group.bench_function("base/allocate", |b| {
        b.iter(|| black_box(encode_base_3p_batch(black_box(&observations)).unwrap()))
    });
    let mut base_output = vec![0.0; BASE_3P_V0.values_for_batch(CORPUS_SIZE)];
    group.bench_function("base/into", |b| {
        b.iter(|| {
            encode_base_3p_batch_into(black_box(&observations), black_box(&mut base_output))
                .unwrap();
            black_box(&base_output);
        })
    });

    group.bench_function("extended/allocate", |b| {
        b.iter(|| black_box(encode_extended_3p_batch(black_box(&observations)).unwrap()))
    });
    let mut extended_output = vec![0.0; EXTENDED_3P_V0.values_for_batch(CORPUS_SIZE)];
    group.bench_function("extended/into", |b| {
        b.iter(|| {
            encode_extended_3p_batch_into(
                black_box(&observations),
                black_box(&mut extended_output),
            )
            .unwrap();
            black_box(&extended_output);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_four_player_features,
    bench_three_player_features
);
criterion_main!(benches);
