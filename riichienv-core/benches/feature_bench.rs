use std::collections::HashMap;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use riichienv_core::action::Action;
use riichienv_core::engine::{
    EngineConfig, EventLogPolicy, GameEngine, GameMode, ObservationVariant,
};
use riichienv_core::features::{
    BASE_3P_V0, BASE_4P_V0, DREV_3P_V1, DREV_3P_V2, DREV_4P_V1, DREV_4P_V2, EXTENDED_3P_V0,
    EXTENDED_4P_V0, EXTENDED_SP_DREV_3P_V1, EXTENDED_SP_DREV_3P_V2, EXTENDED_SP_DREV_4P_V1,
    EXTENDED_SP_DREV_4P_V2, SP_3P_V1, SP_4P_V0, encode_base_3p_batch, encode_base_3p_batch_into,
    encode_base_4p_batch, encode_base_4p_batch_into, encode_drev_3p_batch,
    encode_drev_3p_batch_into, encode_drev_4p_batch, encode_drev_4p_batch_into,
    encode_drev_v2_3p_batch, encode_drev_v2_3p_batch_into, encode_drev_v2_4p_batch,
    encode_drev_v2_4p_batch_into, encode_extended_3p_batch, encode_extended_3p_batch_into,
    encode_extended_4p_batch, encode_extended_4p_batch_into, encode_extended_sp_drev_3p_batch,
    encode_extended_sp_drev_3p_batch_into, encode_extended_sp_drev_4p_batch,
    encode_extended_sp_drev_4p_batch_into, encode_extended_sp_drev_v2_3p_batch,
    encode_extended_sp_drev_v2_3p_batch_into, encode_extended_sp_drev_v2_4p_batch,
    encode_extended_sp_drev_v2_4p_batch_into, encode_sp_3p_batch, encode_sp_3p_batch_into,
    encode_sp_4p_batch, encode_sp_4p_batch_into,
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

fn four_player_corpus(count: usize) -> Vec<Observation> {
    collect_observations(GameMode::FourPlayerSingle, count)
        .into_iter()
        .map(|observation| match observation {
            ObservationVariant::FourPlayer(observation) => observation,
            ObservationVariant::ThreePlayer(_) => unreachable!(),
        })
        .collect()
}

fn three_player_corpus(count: usize) -> Vec<Observation3P> {
    collect_observations(GameMode::ThreePlayerSingle, count)
        .into_iter()
        .map(|observation| match observation {
            ObservationVariant::ThreePlayer(observation) => observation,
            ObservationVariant::FourPlayer(_) => unreachable!(),
        })
        .collect()
}

fn evenly_sample<T: Clone>(corpus: &[T], count: usize) -> Vec<T> {
    assert!(count > 0 && count <= corpus.len());
    if count == 1 {
        return vec![corpus[corpus.len() / 2].clone()];
    }
    (0..count)
        .map(|index| corpus[index * (corpus.len() - 1) / (count - 1)].clone())
        .collect()
}

fn report_history_lengths(label: &str, mut lengths: Vec<usize>) {
    lengths.sort_unstable();
    let median = lengths[lengths.len() / 2];
    let p95 = lengths[(lengths.len() - 1) * 95 / 100];
    println!(
        "{label} public-discard history: min={}, median={median}, p95={p95}, max={}",
        lengths[0],
        lengths[lengths.len() - 1]
    );
}

fn bench_four_player_features(c: &mut Criterion) {
    let observations = four_player_corpus(CORPUS_SIZE);
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
    let mut drev_output = vec![0.0; DREV_4P_V1.values_for_batch(CORPUS_SIZE)];
    group.bench_function("drev/into", |b| {
        b.iter(|| {
            encode_drev_4p_batch_into(black_box(&observations), black_box(&mut drev_output))
                .unwrap();
            black_box(&drev_output);
        })
    });

    group.bench_function("drev_v2/allocate", |b| {
        b.iter(|| black_box(encode_drev_v2_4p_batch(black_box(&observations)).unwrap()))
    });
    let mut drev_v2_output = vec![0.0; DREV_4P_V2.values_for_batch(CORPUS_SIZE)];
    group.bench_function("drev_v2/into", |b| {
        b.iter(|| {
            encode_drev_v2_4p_batch_into(black_box(&observations), black_box(&mut drev_v2_output))
                .unwrap();
            black_box(&drev_v2_output);
        })
    });

    group.bench_function("extended_sp_drev/allocate", |b| {
        b.iter(|| black_box(encode_extended_sp_drev_4p_batch(black_box(&observations)).unwrap()))
    });
    let mut combined_output = vec![0.0; EXTENDED_SP_DREV_4P_V1.values_for_batch(CORPUS_SIZE)];
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

    group.bench_function("extended_sp_drev_v2/allocate", |b| {
        b.iter(|| black_box(encode_extended_sp_drev_v2_4p_batch(black_box(&observations)).unwrap()))
    });
    let mut combined_v2_output = vec![0.0; EXTENDED_SP_DREV_4P_V2.values_for_batch(CORPUS_SIZE)];
    group.bench_function("extended_sp_drev_v2/into", |b| {
        b.iter(|| {
            encode_extended_sp_drev_v2_4p_batch_into(
                black_box(&observations),
                black_box(&mut combined_v2_output),
            )
            .unwrap();
            black_box(&combined_v2_output);
        })
    });

    group.finish();
}

fn bench_three_player_features(c: &mut Criterion) {
    let observations = three_player_corpus(CORPUS_SIZE);
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

    group.bench_function("sp/allocate", |b| {
        b.iter(|| black_box(encode_sp_3p_batch(black_box(&observations)).unwrap()))
    });
    let mut sp_output = vec![0.0; SP_3P_V1.values_for_batch(CORPUS_SIZE)];
    group.bench_function("sp/into", |b| {
        b.iter(|| {
            encode_sp_3p_batch_into(black_box(&observations), black_box(&mut sp_output)).unwrap();
            black_box(&sp_output);
        })
    });

    group.bench_function("drev/allocate", |b| {
        b.iter(|| black_box(encode_drev_3p_batch(black_box(&observations)).unwrap()))
    });
    let mut drev_output = vec![0.0; DREV_3P_V1.values_for_batch(CORPUS_SIZE)];
    group.bench_function("drev/into", |b| {
        b.iter(|| {
            encode_drev_3p_batch_into(black_box(&observations), black_box(&mut drev_output))
                .unwrap();
            black_box(&drev_output);
        })
    });

    group.bench_function("drev_v2/allocate", |b| {
        b.iter(|| black_box(encode_drev_v2_3p_batch(black_box(&observations)).unwrap()))
    });
    let mut drev_v2_output = vec![0.0; DREV_3P_V2.values_for_batch(CORPUS_SIZE)];
    group.bench_function("drev_v2/into", |b| {
        b.iter(|| {
            encode_drev_v2_3p_batch_into(black_box(&observations), black_box(&mut drev_v2_output))
                .unwrap();
            black_box(&drev_v2_output);
        })
    });

    group.bench_function("extended_sp_drev/allocate", |b| {
        b.iter(|| black_box(encode_extended_sp_drev_3p_batch(black_box(&observations)).unwrap()))
    });
    let mut combined_output = vec![0.0; EXTENDED_SP_DREV_3P_V1.values_for_batch(CORPUS_SIZE)];
    group.bench_function("extended_sp_drev/into", |b| {
        b.iter(|| {
            encode_extended_sp_drev_3p_batch_into(
                black_box(&observations),
                black_box(&mut combined_output),
            )
            .unwrap();
            black_box(&combined_output);
        })
    });

    group.bench_function("extended_sp_drev_v2/allocate", |b| {
        b.iter(|| black_box(encode_extended_sp_drev_v2_3p_batch(black_box(&observations)).unwrap()))
    });
    let mut combined_v2_output = vec![0.0; EXTENDED_SP_DREV_3P_V2.values_for_batch(CORPUS_SIZE)];
    group.bench_function("extended_sp_drev_v2/into", |b| {
        b.iter(|| {
            encode_extended_sp_drev_v2_3p_batch_into(
                black_box(&observations),
                black_box(&mut combined_v2_output),
            )
            .unwrap();
            black_box(&combined_v2_output);
        })
    });

    group.finish();
}

fn bench_drev_v2_scaling(c: &mut Criterion) {
    const MAX_BATCH: usize = 256;
    let four_player = four_player_corpus(MAX_BATCH);
    let three_player = three_player_corpus(MAX_BATCH);
    report_history_lengths(
        "4P batch-16 corpus",
        four_player[..CORPUS_SIZE]
            .iter()
            .map(|observation| observation.public_discard_actors().len())
            .collect(),
    );
    report_history_lengths(
        "4P batch-256 corpus",
        four_player
            .iter()
            .map(|observation| observation.public_discard_actors().len())
            .collect(),
    );
    report_history_lengths(
        "3P batch-16 corpus",
        three_player[..CORPUS_SIZE]
            .iter()
            .map(|observation| observation.public_discard_actors().len())
            .collect(),
    );
    report_history_lengths(
        "3P batch-256 corpus",
        three_player
            .iter()
            .map(|observation| observation.public_discard_actors().len())
            .collect(),
    );

    for batch_size in [1usize, 16, MAX_BATCH] {
        // Spread each batch over the same full progressed corpus. This keeps
        // public-history depth representative instead of making batch 1 an
        // opening-only observation and batch 256 a late-round observation.
        let observations = evenly_sample(&four_player, batch_size);
        let mut group = c.benchmark_group(format!("features/4p/drev_v2/batch_{batch_size}"));
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_function("allocate", |b| {
            b.iter(|| black_box(encode_drev_v2_4p_batch(black_box(&observations)).unwrap()))
        });
        let mut output = vec![0.0; DREV_4P_V2.values_for_batch(batch_size)];
        group.bench_function("into", |b| {
            b.iter(|| {
                encode_drev_v2_4p_batch_into(black_box(&observations), black_box(&mut output))
                    .unwrap();
                black_box(&output);
            })
        });
        group.finish();

        let observations = evenly_sample(&three_player, batch_size);
        let mut group = c.benchmark_group(format!("features/3p/drev_v2/batch_{batch_size}"));
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_function("allocate", |b| {
            b.iter(|| black_box(encode_drev_v2_3p_batch(black_box(&observations)).unwrap()))
        });
        let mut output = vec![0.0; DREV_3P_V2.values_for_batch(batch_size)];
        group.bench_function("into", |b| {
            b.iter(|| {
                encode_drev_v2_3p_batch_into(black_box(&observations), black_box(&mut output))
                    .unwrap();
                black_box(&output);
            })
        });
        group.finish();
    }
}

criterion_group!(
    benches,
    bench_four_player_features,
    bench_three_player_features,
    bench_drev_v2_scaling
);
criterion_main!(benches);
