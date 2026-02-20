/// Helper: write a scalar value broadcast across 34 tile positions into a flat buffer.
/// buf layout: channel-major, i.e. buf[(ch_offset + ch) * 34 + tile] = val
#[inline]
pub(crate) fn broadcast_scalar(buf: &mut [f32], ch_offset: usize, ch: usize, val: f32) {
    let start = (ch_offset + ch) * 34;
    for j in 0..34 {
        buf[start + j] = val;
    }
}

/// Helper: set a single value in the flat buffer.
#[inline]
pub(crate) fn set_val(buf: &mut [f32], ch_offset: usize, ch: usize, tile: usize, val: f32) {
    buf[(ch_offset + ch) * 34 + tile] = val;
}

/// Helper: add a value in the flat buffer.
#[inline]
pub(crate) fn add_val(buf: &mut [f32], ch_offset: usize, ch: usize, tile: usize, val: f32) {
    buf[(ch_offset + ch) * 34 + tile] += val;
}

/// Sanma dora indicator -> dora tile mapping (tile IDs, i.e. tile/4 in 0..34).
/// In sanma: 1m(0)->9m(8), 9m(8)->1m(0), no 2m-8m exist.
pub(crate) fn get_next_tile_sanma(tile: u32) -> u8 {
    let tile34 = tile / 4;
    match tile34 {
        0 => (8 * 4) as u8,  // 1m -> 9m
        8 => (0 * 4) as u8,  // 9m -> 1m
        1..=7 => tile as u8, // shouldn't appear in sanma
        _ => get_next_tile(tile),
    }
}

/// Standard next tile for dora calculation (non-manzu suits and honors).
fn get_next_tile(tile: u32) -> u8 {
    let tile_type = (tile / 4) / 9;
    let tile_num = (tile / 4) % 9;

    if tile_type < 3 {
        let next_num = if tile_num == 8 { 0 } else { tile_num + 1 };
        ((tile_type * 9 + next_num) * 4) as u8
    } else {
        let base = tile / 4;
        if (27..31).contains(&base) {
            let wind_idx = base - 27;
            let next_wind = (wind_idx + 1) % 4;
            ((27 + next_wind) * 4) as u8
        } else if (31..34).contains(&base) {
            let dragon_idx = base - 31;
            let next_dragon = (dragon_idx + 1) % 3;
            ((31 + next_dragon) * 4) as u8
        } else {
            tile as u8
        }
    }
}
