//! Reference integrals for Kuhn's measured P22 response (IEEE S&P 2002, eqs. 8–10).
//! The renderer's positive reservoirs are checked against these analytic curves.
#[cfg(test)]
pub fn energy(channel: usize, begin: f64, end: f64) -> f64 {
    let series: &[(f64, f64)] = match channel {
        0 => &[(4.,360.),(1.75,1600.),(2.,8000.),(2.25,25000.),(15.,700000.),(29.,7000000.)],
        1 => &[(37.,150000.),(100.,700000.),(90.,5000000.)],
        _ => &[(75.,100000.),(1000.,1100000.),(1100.,4000000.)],
    };
    let mut area = 0.; let mut total = 0.;
    for &(amplitude, frequency) in series {
        let rate = std::f64::consts::TAU * frequency;
        total += amplitude / rate;
        area += amplitude / rate * ((-rate * begin).exp() - (-rate * end).exp());
    }
    if channel > 0 {
        let (a, alpha, beta): (f64, f64, f64) = if channel == 1 { (210e-6, 5.5e-6, 1.1) } else { (190e-6, 5e-6, 1.11) };
        total += a * alpha.powf(1. - beta) / (beta - 1.);
        area += a * ((begin + alpha).powf(1. - beta) - (end + alpha).powf(1. - beta)) / (beta - 1.);
    }
    area / total
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reservoir_fit_matches_published_integrals_and_conserves_energy() {
        let data: serde_json::Value = serde_json::from_str(include_str!("../docs/phosphor-response.json")).unwrap();
        let terms = data["terms"].as_array().unwrap();
        // Also pin the compiled shader coefficients to the reviewed dataset.
        for (column, key) in [("PHOS_RATE", "rate_per_second"), ("PHOS_ENERGY", "energy_fraction")] {
            let text = include_str!("phosphor.wgsl").split(column).nth(1).unwrap();
            let body = text.split("(\n").nth(1).unwrap().split(");").next().unwrap();
            let rows: Vec<_> = body.lines().filter(|l| l.contains("vec3<f32>(")).collect();
            assert_eq!(rows.len(), terms.len());
            for (row, term) in rows.iter().zip(terms) {
                let numbers: Vec<f64> = row.split('(').nth(1).unwrap().split(')').next().unwrap()
                    .split(',').map(|s| s.trim().parse().unwrap()).collect();
                for c in 0..3 {
                    let expected = term[key][c].as_f64().unwrap();
                    assert!((numbers[c] - expected).abs() <= expected.abs() * 1e-8 + 1e-20);
                }
            }
        }
        for c in 0..3 {
            let total: f64 = terms.iter().map(|t| t["energy_fraction"][c].as_f64().unwrap()).sum();
            assert!((total - 1.).abs() < 1e-6);
            for i in 0..120 {
                let begin = 10_f64.powf(-6. + i as f64 / 20.);
                let end = begin * 1.1;
                let expected = energy(c, begin, end);
                let actual: f64 = terms.iter().map(|t| {
                    let rate = t["rate_per_second"][c].as_f64().unwrap();
                    t["energy_fraction"][c].as_f64().unwrap() * ((-rate*begin).exp() - (-rate*end).exp())
                }).sum();
                if expected > 1e-12 { assert!((actual/expected-1.).abs() < 0.002, "channel {c} at {begin}: {actual} vs {expected}"); }
            }
        }
    }
}
