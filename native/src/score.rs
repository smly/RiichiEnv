use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone)]
pub struct Score {
    #[pyo3(get)]
    pub total: u32,
    #[pyo3(get)]
    pub pay_ron: u32,
    #[pyo3(get)]
    pub pay_tsumo_oya: u32,
    #[pyo3(get)]
    pub pay_tsumo_ko: u32,
}

#[pyfunction]
pub fn calculate_score(han: u8, fu: u8, is_oya: bool, is_tsumo: bool, honba: u32) -> Score {
    let mut s = if han >= 5 {
        let base_points = match han {
            5 => 2000,                     // Mangan
            6 | 7 => 3000,                 // Haneman
            8..=10 => 4000,                // Baiman
            11 | 12 => 6000,               // Sanbaiman
            _ => 8000 * (han as u32 / 13), // Yakuman (13, 26, 39, ...)
        };
        make_score_result(base_points, is_oya, is_tsumo)
    } else {
        let fu = round_up_fu(fu);
        let bp = (fu as u32) * (2u32.pow(2 + han as u32));
        if bp > 2000 {
            make_score_result(2000, is_oya, is_tsumo)
        } else {
            make_score_result(bp, is_oya, is_tsumo)
        }
    };

    // Add Honba (300 per honba stack)
    if is_tsumo {
        s.pay_tsumo_oya += honba * 100;
        s.pay_tsumo_ko += honba * 100;
        s.total += honba * 300;
    } else {
        s.pay_ron += honba * 300;
        s.total += honba * 300;
    }
    s
}

fn make_score_result(base_points: u32, is_oya: bool, is_tsumo: bool) -> Score {
    let total_ron = if is_oya {
        ceil_100(base_points * 6)
    } else {
        ceil_100(base_points * 4)
    };

    let (pay_oya, pay_ko) = if is_oya {
        (0, ceil_100(base_points * 2))
    } else {
        (ceil_100(base_points * 2), ceil_100(base_points))
    };

    let total_tsumo = if is_oya {
        pay_ko * 3
    } else {
        pay_oya + pay_ko * 2
    };

    if is_tsumo {
        Score {
            total: total_tsumo,
            pay_ron: 0,
            pay_tsumo_oya: pay_oya,
            pay_tsumo_ko: pay_ko,
        }
    } else {
        Score {
            total: total_ron,
            pay_ron: total_ron,
            pay_tsumo_oya: 0,
            pay_tsumo_ko: 0,
        }
    }
}

fn round_up_fu(fu: u8) -> u8 {
    if fu == 25 {
        return 25; // Chiitoitsu fixed
    }
    fu.div_ceil(10) * 10
}

fn ceil_100(val: u32) -> u32 {
    val.div_ceil(100) * 100
}
