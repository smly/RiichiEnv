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

/// Get the next tile in sequence for dora calculation
/// Tile type: 0-33 (m1-m9: 0-8, p1-p9: 9-17, s1-s9: 18-26, winds: 27-30, dragons: 31-33)
pub(crate) fn get_next_tile(tile: u32) -> u8 {
    let tile_type = (tile / 4) / 9; // 0=man, 1=pin, 2=sou, 3+=honors
    let tile_num = (tile / 4) % 9;

    if tile_type < 3 {
        // Number tiles (1-9)
        let next_num = if tile_num == 8 { 0 } else { tile_num + 1 };
        ((tile_type * 9 + next_num) * 4) as u8
    } else {
        // Honor tiles (winds 27-30, dragons 31-33)
        let base = tile / 4;
        if (27..31).contains(&base) {
            // Winds: E->S->W->N->E
            let wind_idx = base - 27;
            let next_wind = (wind_idx + 1) % 4;
            ((27 + next_wind) * 4) as u8
        } else if (31..34).contains(&base) {
            // Dragons: White->Green->Red->White
            let dragon_idx = base - 31;
            let next_dragon = (dragon_idx + 1) % 3;
            ((31 + next_dragon) * 4) as u8
        } else {
            tile as u8 // Fallback
        }
    }
}
