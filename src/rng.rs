//! A small seeded random number generator for reproducible measurement and BBHT rounds.

/// The `SplitMix64` generator: fast, seedable and reproducible on every platform.
///
/// It is not cryptographically secure. Use it only to sample measurements and to pick
/// iteration counts.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    /// The generator state. Each call adds the golden-ratio increment to it.
    state: u64,
}

impl SplitMix64 {
    /// Makes a generator from `seed`. The same seed gives the same sequence.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns the next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);

        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);

        z ^ (z >> 31)
    }

    /// Returns a uniform value in `[0, 1)` with 53 random bits.
    pub fn next_f64(&mut self) -> f64 {
        /// The weight of the lowest of the 53 mantissa bits, `2^-53`.
        const UNIT: f64 = 1.0 / 9_007_199_254_740_992.0;

        #[expect(
            clippy::cast_precision_loss,
            reason = "the value has 53 bits after the shift, so f64 holds it exactly"
        )]
        let bits = (self.next_u64() >> 11) as f64;

        bits * UNIT
    }

    /// Returns a uniform integer in `0..bound`, or `0` when `bound` is `0`.
    ///
    /// shortcut: multiply-high without rejection, so the bias is at most `bound / 2^64`; add
    /// rejection if a caller ever needs exact uniformity.
    pub fn below(&mut self, bound: usize) -> usize {
        let bound = bound as u64;
        let wide = u128::from(self.next_u64()) * u128::from(bound);

        #[expect(
            clippy::cast_possible_truncation,
            reason = "the high word is below bound, so it fits in usize"
        )]
        let value = (wide >> 64) as usize;

        value
    }
}

/// Unit tests for the generator.
#[cfg(test)]
mod tests {
    use super::SplitMix64;

    /// The same seed gives the same sequence, and the values stay in range.
    #[test]
    fn is_reproducible_and_in_range() {
        let mut a = SplitMix64::new(7);
        let mut b = SplitMix64::new(7);

        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
            assert!((0.0..1.0).contains(&a.next_f64()));
            assert!(a.below(10) < 10);

            b.next_f64();
            b.below(10);
        }

        assert_eq!(a.below(0), 0);
    }
}
