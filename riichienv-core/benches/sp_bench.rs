use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use riichienv_core::drev::DrevInput;
use riichienv_core::engine::{EngineConfig, GameEngine, GameMode, ObservationVariant};
use riichienv_core::feature_context::FeatureContext;
use riichienv_core::shanten;
use riichienv_core::sp::{SP_CHANNELS, SpInput, calculate_sp, encode_sp, encode_sp_into};
use riichienv_core::types::{Meld, MeldType, TILE_MAX};

fn input_from_tiles(tile_types: &[u8], tsumos_left: u8) -> SpInput {
    let mut tehai = [0u8; TILE_MAX];
    let mut seen = [0u8; TILE_MAX];
    for &tile in tile_types {
        tehai[tile as usize] += 1;
        seen[tile as usize] += 1;
    }
    SpInput {
        tehai,
        akas_in_hand: [false; 3],
        tiles_seen: seen,
        akas_seen: [false; 3],
        dora_indicators: vec![],
        melds: Vec::<Meld>::new(),
        bakaze: 27,
        jikaze: 27,
        is_menzen: true,
        can_riichi: true,
        can_double_riichi: false,
        tsumos_left,
        discard_candidates: vec![],
    }
}

fn best_post_discard_shanten(input: &SpInput) -> i32 {
    input
        .tehai
        .iter()
        .enumerate()
        .filter_map(|(discard, &count)| {
            if count == 0
                || (!input.discard_candidates.is_empty()
                    && !input.discard_candidates.contains(&(discard as u8)))
            {
                return None;
            }

            let mut hand = Vec::with_capacity(input.tehai.iter().map(|&c| c as usize).sum());
            for (tile, &tile_count) in input.tehai.iter().enumerate() {
                let adjusted = tile_count - u8::from(tile == discard);
                for copy in 0..adjusted {
                    hand.push((tile as u32) * 4 + copy as u32);
                }
            }
            Some(shanten::calculate_shanten(&hand))
        })
        .min()
        .expect("SP benchmark fixture must have a legal discard")
}

fn assert_fixture(name: &str, input: &SpInput, expected: i32) {
    assert_eq!(
        best_post_discard_shanten(input),
        expected,
        "{name} benchmark fixture drifted away from its labeled shanten"
    );
}

fn s0_closed(tsumos_left: u8) -> SpInput {
    input_from_tiles(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 18, 22], tsumos_left)
}

fn s1_closed(tsumos_left: u8) -> SpInput {
    input_from_tiles(
        &[1, 2, 5, 6, 7, 10, 10, 10, 11, 11, 13, 13, 20, 31],
        tsumos_left,
    )
}

fn s2_closed(tsumos_left: u8) -> SpInput {
    input_from_tiles(
        &[0, 0, 0, 1, 4, 5, 9, 10, 11, 11, 19, 24, 25, 27],
        tsumos_left,
    )
}

fn s3_closed(tsumos_left: u8) -> SpInput {
    input_from_tiles(
        &[0, 1, 2, 2, 3, 6, 7, 13, 14, 17, 23, 24, 26, 30],
        tsumos_left,
    )
}

fn s5_closed(tsumos_left: u8) -> SpInput {
    input_from_tiles(
        &[0, 2, 5, 8, 9, 12, 15, 18, 21, 24, 27, 29, 31, 33],
        tsumos_left,
    )
}

fn s0_open_aka_dora(tsumos_left: u8) -> SpInput {
    // Open 123m plus concealed 456789m 12p 11s and an extra 5s. The red 5m
    // remains concealed; 2p is the dora indicator, making the 3p wait dora.
    let mut input = input_from_tiles(&[3, 4, 5, 6, 7, 8, 9, 10, 18, 18, 22], tsumos_left);
    input
        .melds
        .push(Meld::new(MeldType::Chi, vec![0, 4, 8], true, 3, Some(8)));
    for tile in 0..=2 {
        input.tiles_seen[tile] += 1;
    }
    input.akas_in_hand[0] = true;
    input.akas_seen[0] = true;
    input.dora_indicators.push(40);
    input.tiles_seen[10] += 1;
    input.is_menzen = false;
    input.can_riichi = false;
    input.can_double_riichi = false;
    input
}

fn bench_calculate_sp(c: &mut Criterion) {
    let fixtures = [
        ("s0_closed", s0_closed(10), 0),
        ("s1_closed", s1_closed(10), 1),
        ("s2_closed", s2_closed(10), 2),
        ("s3_closed", s3_closed(10), 3),
        ("s5_closed_light", s5_closed(10), 5),
        ("s0_open_aka_dora", s0_open_aka_dora(10), 0),
    ];
    for (name, input, expected) in &fixtures {
        assert_fixture(name, input, *expected);
    }

    let mut group = c.benchmark_group("sp/calculate");
    group.throughput(Throughput::Elements(1));
    for (name, input, _) in &fixtures {
        group.bench_with_input(BenchmarkId::from_parameter(name), input, |b, input| {
            b.iter(|| black_box(calculate_sp(black_box(input))))
        });
    }
    group.finish();
}

fn bench_calculate_sp_by_horizon(c: &mut Criterion) {
    let h3 = s1_closed(3);
    let h10 = s1_closed(10);
    let h17 = s1_closed(17);
    assert_fixture("s1_closed_h3", &h3, 1);
    assert_fixture("s1_closed_h10", &h10, 1);
    assert_fixture("s1_closed_h17", &h17, 1);

    c.bench_function("sp/horizon/s1_closed_h3", |b| {
        b.iter(|| black_box(calculate_sp(black_box(&h3))))
    });
    c.bench_function("sp/horizon/s1_closed_h10", |b| {
        b.iter(|| black_box(calculate_sp(black_box(&h10))))
    });
    c.bench_function("sp/horizon/s1_closed_h17", |b| {
        b.iter(|| black_box(calculate_sp(black_box(&h17))))
    });
}

fn bench_shanten_baseline(c: &mut Criterion) {
    let input = s1_closed(10);
    assert_fixture("s1_closed_shanten_baseline", &input, 1);
    let hands: Vec<Vec<u32>> = input
        .tehai
        .iter()
        .enumerate()
        .filter_map(|(discard, &count)| {
            if count == 0 {
                return None;
            }
            let mut tiles = Vec::new();
            for (tile, &tile_count) in input.tehai.iter().enumerate() {
                let adjusted = tile_count - u8::from(tile == discard);
                for copy in 0..adjusted {
                    tiles.push((tile as u32) * 4 + copy as u32);
                }
            }
            Some(tiles)
        })
        .collect();

    c.bench_function("sp/shanten/discard_candidates", |b| {
        b.iter(|| {
            for hand in &hands {
                black_box(shanten::calculate_shanten(black_box(hand)));
            }
        })
    });
}

fn bench_encode_sp(c: &mut Criterion) {
    let input = s0_closed(10);
    assert_fixture("s0_closed_encode", &input, 0);
    let result = calculate_sp(&input);
    assert_eq!(encode_sp(&result).len(), SP_CHANNELS * TILE_MAX);

    c.bench_function("sp/encode_only", |b| {
        b.iter(|| black_box(encode_sp(black_box(&result))))
    });
    let mut encode_buf = vec![0.0f32; SP_CHANNELS * TILE_MAX];
    c.bench_function("sp/encode_into_only", |b| {
        b.iter(|| {
            encode_sp_into(black_box(&result), black_box(&mut encode_buf), 0);
            black_box(&encode_buf);
        })
    });
    c.bench_function("sp/calculate_and_encode", |b| {
        b.iter(|| {
            let result = calculate_sp(black_box(&input));
            black_box(encode_sp(&result))
        })
    });
}

fn bench_feature_context(c: &mut Criterion) {
    let mut engine =
        GameEngine::new(EngineConfig::new(GameMode::FourPlayerSingle).with_seed(42)).unwrap();
    let observation = match engine.decisions().remove(0).observation {
        ObservationVariant::FourPlayer(observation) => observation,
        ObservationVariant::ThreePlayer(_) => unreachable!(),
    };

    c.bench_function("features/context/build", |b| {
        b.iter(|| black_box(FeatureContext::new(black_box(&observation)).unwrap()))
    });
    c.bench_function("features/context/sp_drev_inputs_independent", |b| {
        b.iter(|| {
            black_box(SpInput::from_observation(black_box(&observation)));
            black_box(DrevInput::from_observation(black_box(&observation)));
        })
    });
    c.bench_function("features/context/sp_drev_inputs_shared", |b| {
        b.iter(|| {
            let context = FeatureContext::new(black_box(&observation)).unwrap();
            black_box(SpInput::from_feature_context(&context));
            black_box(DrevInput::from_feature_context(&context));
        })
    });
}

criterion_group!(
    benches,
    bench_calculate_sp,
    bench_calculate_sp_by_horizon,
    bench_shanten_baseline,
    bench_encode_sp,
    bench_feature_context
);
criterion_main!(benches);
