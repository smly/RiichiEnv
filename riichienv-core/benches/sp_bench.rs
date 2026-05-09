use criterion::{Criterion, black_box, criterion_group, criterion_main};
use riichienv_core::shanten;
use riichienv_core::sp::{SP_CHANNELS, SpInput, calculate_sp, encode_sp};
use riichienv_core::types::{Meld, TILE_MAX};

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

fn bench_calculate_sp(c: &mut Criterion) {
    let tenpai = input_from_tiles(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 18, 22], 10);
    let one_shanten = input_from_tiles(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 18, 25, 26], 10);
    let two_or_three = input_from_tiles(&[0, 1, 4, 5, 8, 9, 10, 13, 18, 19, 22, 27, 28, 31], 10);
    let four_plus = input_from_tiles(&[0, 2, 5, 8, 9, 12, 15, 18, 21, 24, 27, 29, 31, 33], 10);

    c.bench_function("sp/calculate/tenpai_dp", |b| {
        b.iter(|| black_box(calculate_sp(black_box(&tenpai))))
    });
    c.bench_function("sp/calculate/one_shanten_dp", |b| {
        b.iter(|| black_box(calculate_sp(black_box(&one_shanten))))
    });
    c.bench_function("sp/calculate/two_or_three_shanten_dp", |b| {
        b.iter(|| black_box(calculate_sp(black_box(&two_or_three))))
    });
    c.bench_function("sp/calculate/four_plus_light", |b| {
        b.iter(|| black_box(calculate_sp(black_box(&four_plus))))
    });
}

fn bench_calculate_sp_by_horizon(c: &mut Criterion) {
    let h3 = input_from_tiles(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 18, 25, 26], 3);
    let h10 = input_from_tiles(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 18, 25, 26], 10);
    let h17 = input_from_tiles(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 18, 25, 26], 17);

    c.bench_function("sp/calculate/one_shanten_h3", |b| {
        b.iter(|| black_box(calculate_sp(black_box(&h3))))
    });
    c.bench_function("sp/calculate/one_shanten_h10", |b| {
        b.iter(|| black_box(calculate_sp(black_box(&h10))))
    });
    c.bench_function("sp/calculate/one_shanten_h17", |b| {
        b.iter(|| black_box(calculate_sp(black_box(&h17))))
    });
}

fn bench_shanten_baseline(c: &mut Criterion) {
    let input = input_from_tiles(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 18, 25, 26], 10);
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
    let input = input_from_tiles(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 18, 22], 10);
    let result = calculate_sp(&input);
    assert_eq!(encode_sp(&result).len(), SP_CHANNELS * TILE_MAX);

    c.bench_function("sp/encode_only", |b| {
        b.iter(|| black_box(encode_sp(black_box(&result))))
    });
    c.bench_function("sp/calculate_and_encode", |b| {
        b.iter(|| {
            let result = calculate_sp(black_box(&input));
            black_box(encode_sp(&result))
        })
    });
}

criterion_group!(
    benches,
    bench_calculate_sp,
    bench_calculate_sp_by_horizon,
    bench_shanten_baseline,
    bench_encode_sp
);
criterion_main!(benches);
