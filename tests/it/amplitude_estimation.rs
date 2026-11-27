//! Amplitude estimation of rollout values: exact outcomes, the error bound and the error
//! against classical Monte Carlo at the same query count.

use core::f64::consts::PI;

use quantum_planning::SplitMix64;
use quantum_planning::amplitude_estimation::{
    EstimationError, classical_estimate, estimate, outcome_value, outcomes,
};

use crate::to_f64;

/// Rollout values of eight seeds, as win probabilities from one MCTS node.
const VALUES: [f64; 8] = [0.1, 0.9, 0.3, 0.6, 0.0, 1.0, 0.45, 0.2];

/// Returns the mean of [`VALUES`].
fn mean() -> f64 {
    VALUES.iter().sum::<f64>() / 8.0
}

/// When the mean is exactly `sin²(π·j/K)`, only the outcomes `j` and `K − j` occur, each
/// with probability `1/2`.
#[test]
fn exact_amplitudes_give_exact_outcomes() {
    let outcomes_count = 32;

    for j in [1, 3, 7, 12] {
        let a = outcome_value(j, outcomes_count);
        let spread = a.min(1.0 - a) / 2.0;
        let values = [a - spread, a + spread];

        let probabilities = outcomes(1, |seed| values[seed], 5).unwrap();

        assert!((probabilities[j] - 0.5).abs() < 1e-9, "j={j}");
        assert!(
            (probabilities[outcomes_count - j] - 0.5).abs() < 1e-9,
            "j={j}"
        );
        assert!((outcome_value(j, outcomes_count) - a).abs() < 1e-12);
    }
}

/// With probability at least `8/π²`, the error is within `2π·√(a(1−a))/K + π²/K²`.
#[test]
fn meets_the_brassard_error_bound() {
    let a = mean();

    for bits in 3..=7 {
        let k = to_f64(1 << bits);
        let bound = 2.0 * PI * (a * (1.0 - a)).sqrt() / k + PI * PI / (k * k);

        let probabilities = outcomes(3, |seed| VALUES[seed], bits).unwrap();
        let within: f64 = probabilities
            .iter()
            .enumerate()
            .filter(|&(y, _)| (outcome_value(y, 1 << bits) - a).abs() <= bound)
            .map(|(_, probability)| probability)
            .sum();

        assert!(within >= 8.0 / (PI * PI), "m={bits}: {within}");
    }
}

/// Returns the median of `|estimate − a|` under the outcome distribution of `bits`.
fn quantum_median_error(a: f64, bits: usize) -> f64 {
    let probabilities = outcomes(3, |seed| VALUES[seed], bits).unwrap();
    let mut errors: Vec<(f64, f64)> = probabilities
        .iter()
        .enumerate()
        .map(|(y, &probability)| ((outcome_value(y, 1 << bits) - a).abs(), probability))
        .collect();
    errors.sort_by(|x, y| x.0.total_cmp(&y.0));

    let mut cumulative = 0.0;
    errors
        .into_iter()
        .find(|&(_, probability)| {
            cumulative += probability;
            cumulative >= 0.5
        })
        .map_or(1.0, |(error, _)| error)
}

/// Returns the median of `|estimate − a|` over `runs` classical estimates of `samples` each.
fn classical_median_error(a: f64, samples: usize, runs: usize, seed: u64) -> f64 {
    let mut rng = SplitMix64::new(seed);
    let mut errors: Vec<f64> = (0..runs)
        .map(|_| {
            let run = classical_estimate(3, |seed| VALUES[seed], samples, &mut rng).unwrap();

            (run.value - a).abs()
        })
        .collect();
    errors.sort_by(f64::total_cmp);

    errors[runs / 2]
}

/// At the same query count, the median error of amplitude estimation is below that of
/// classical Monte Carlo from 255 queries on, and the gap grows with the queries. At 63
/// queries (`m = 5`), Monte Carlo still wins on these values: 0.041 against 0.029.
#[test]
fn beats_monte_carlo_at_equal_queries() {
    let a = mean();
    let mut ratios = Vec::new();

    for bits in [7, 9] {
        let queries = 2 * (1 << bits) - 1;
        let run = estimate(3, |seed| VALUES[seed], bits, &mut SplitMix64::new(3)).unwrap();
        assert_eq!(run.queries, queries);

        let quantum = quantum_median_error(a, bits);
        let classical = classical_median_error(a, queries, 401, 9);

        assert!(quantum < classical, "m={bits}: {quantum} vs {classical}");
        ratios.push(classical / quantum);
    }

    assert!(ratios[1] > ratios[0], "the advantage grows: {ratios:?}");
}

/// Values outside `[0, 1]` and registers above the limit are errors.
#[test]
fn rejects_invalid_problems() {
    let mut rng = SplitMix64::new(1);

    assert_eq!(
        estimate(1, |seed| [0.5, 1.5][seed], 3, &mut rng),
        Err(EstimationError::InvalidValue {
            seed: 1,
            value: 1.5
        })
    );
    assert!(classical_estimate(1, |_| f64::NAN, 3, &mut rng).is_err());
    assert!(matches!(
        outcomes(20, |_| 0.5, 4),
        Err(EstimationError::TooManyQubits(_))
    ));
}
