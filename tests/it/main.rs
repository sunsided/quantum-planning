//! Integration tests for the simulator and the Grover searches.

mod bbht;
mod grover;
mod plan;

use quantum_planning::Oracle;

/// Marks `count` of `states` basis indices, spread over the index range.
///
/// Index `i` is marked when `(37·i + 11) mod states < count`. For a power-of-two `states`,
/// this map is a bijection, so exactly `count` indices are marked.
pub struct Spread {
    /// The number of basis states, a power of two.
    pub states: usize,
    /// The number of marked states.
    pub count: usize,
}

impl Oracle<usize> for Spread {
    fn accepts(&self, index: &usize) -> bool {
        (37 * index + 11) % self.states < self.count
    }
}

/// Converts a small count to `f64`.
///
/// # Panics
///
/// Panics when `count` does not fit in `u32`; test counts stay far below that.
#[must_use]
pub fn to_f64(count: usize) -> f64 {
    f64::from(u32::try_from(count).expect("test counts fit in u32"))
}
