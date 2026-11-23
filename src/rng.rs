//! Random draws over [`rand_core::Rng`], and a small seeded generator.
//!
//! The simulator and the searches take any `R: Rng`, so a fixed seed gives reproducible
//! measurements and BBHT rounds. [`SplitMix64`] is a dependency-free generator for callers
//! that have none.

use core::convert::Infallible;

use rand_core::{Rng, TryRng};

/// The `SplitMix64` generator: fast, seedable and reproducible on every platform.
///
/// It is not cryptographically secure. Use it only to sample measurements and to pick
/// iteration counts.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    /// The generator state. Each step adds the golden-ratio increment to it.
    state: u64,
}

impl SplitMix64 {
    /// Makes a generator from `seed`. The same seed gives the same sequence.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advances the state and returns the next 64 random bits.
    fn step(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);

        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);

        z ^ (z >> 31)
    }
}

/// The generator cannot fail, so it is an [`Rng`] through the blanket impl of `rand_core`.
impl TryRng for SplitMix64 {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        // The upper half of the output has the better bits; the shift makes the cast lossless.
        let upper = (self.step() >> 32) as u32;

        Ok(upper)
    }

    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        Ok(self.step())
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
        for chunk in dst.chunks_mut(8) {
            let bytes = self.step().to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }

        Ok(())
    }
}

/// Returns a uniform value in `[0, 1)` with 53 random bits from `rng`.
pub(crate) fn unit_f64<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    /// The weight of the lowest of the 53 mantissa bits, `2^-53`.
    const UNIT: f64 = 1.0 / 9_007_199_254_740_992.0;

    #[expect(
        clippy::cast_precision_loss,
        reason = "the value has 53 bits after the shift, so f64 holds it exactly"
    )]
    let bits = (rng.next_u64() >> 11) as f64;

    bits * UNIT
}

/// Returns a uniform integer in `0..bound` from `rng`, or `0` when `bound` is `0`.
///
/// shortcut: multiply-high without rejection, so the bias is at most `bound / 2^64`; add
/// rejection if a caller ever needs exact uniformity.
pub(crate) fn below<R: Rng + ?Sized>(rng: &mut R, bound: usize) -> usize {
    let wide = u128::from(rng.next_u64()) * bound as u128;

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the high word is below bound, so it fits in usize"
    )]
    let value = (wide >> 64) as usize;

    value
}

/// Unit tests for the generator and the draws.
#[cfg(test)]
mod tests {
    use super::{SplitMix64, below, unit_f64};
    use rand_core::Rng;

    /// The same seed gives the same sequence, and the draws stay in range.
    #[test]
    fn is_reproducible_and_in_range() {
        let mut a = SplitMix64::new(7);
        let mut b = SplitMix64::new(7);

        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
            assert!((0.0..1.0).contains(&unit_f64(&mut a)));
            assert!(below(&mut a, 10) < 10);

            unit_f64(&mut b);
            below(&mut b, 10);
        }

        assert_eq!(below(&mut a, 0), 0);
    }

    /// `fill_bytes` writes the little-endian words of the sequence, also into a partial chunk.
    #[test]
    fn fills_bytes_from_the_sequence() {
        let mut words = SplitMix64::new(1);
        let first = words.next_u64().to_le_bytes();
        let second = words.next_u64().to_le_bytes();

        let mut bytes = [0_u8; 11];
        SplitMix64::new(1).fill_bytes(&mut bytes);

        assert_eq!(bytes[..8], first);
        assert_eq!(bytes[8..], second[..3]);
        assert_eq!(
            SplitMix64::new(1).next_u32(),
            u32::from_le_bytes([first[4], first[5], first[6], first[7]])
        );
    }
}
