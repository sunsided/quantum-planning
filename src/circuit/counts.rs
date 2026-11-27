//! Measurement counts from hardware, mapped back to candidate indices.

use alloc::vec::Vec;
use core::fmt;

use crate::MAX_QUBITS;

/// The error for a bitstring that is not a measurement of the expected register.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InvalidBitstring {
    /// The bitstring has a different length than the register.
    Length {
        /// The number of measured qubits.
        expected: usize,
        /// The length of the bitstring.
        found: usize,
    },
    /// The bitstring has a character other than `0` and `1`.
    Character {
        /// The character position, counted from the left in characters.
        position: usize,
        /// The character.
        found: char,
    },
}

impl fmt::Display for InvalidBitstring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length { expected, found } => {
                write!(f, "bitstring of length {found}, expected {expected}")
            }
            Self::Character { position, found } => {
                write!(f, "bitstring has {found:?} at position {position}")
            }
        }
    }
}

impl core::error::Error for InvalidBitstring {}

/// Returns the candidate index of a measured bitstring over `qubits` qubits.
///
/// The bitstring prints the highest qubit first, as hardware results usually do, so its
/// last character is qubit 0. Read as a binary number it is the index; see the
/// [module docs](super) for the qubit order.
///
/// # Errors
///
/// Returns [`InvalidBitstring`] when the bitstring is not `qubits` characters of `0` and `1`.
///
/// # Panics
///
/// Panics when `qubits` is more than [`MAX_QUBITS`], the largest register of a circuit.
///
/// # Examples
///
/// ```
/// use quantum_planning::circuit::bitstring_index;
///
/// // c[2] = 1, c[1] = 0, c[0] = 0: qubit 2 is set, so the index is 4.
/// assert_eq!(bitstring_index("100", 3), Ok(4));
/// ```
pub fn bitstring_index(bitstring: &str, qubits: usize) -> Result<usize, InvalidBitstring> {
    assert!(
        qubits <= MAX_QUBITS,
        "{qubits} qubits exceed every register"
    );

    let found = bitstring.chars().count();

    if found != qubits {
        return Err(InvalidBitstring::Length {
            expected: qubits,
            found,
        });
    }

    bitstring
        .chars()
        .enumerate()
        .try_fold(0, |index, (position, character)| match character {
            '0' => Ok(index << 1),
            '1' => Ok(index << 1 | 1),
            found => Err(InvalidBitstring::Character { position, found }),
        })
}

/// Maps measurement counts (bitstring and shots) to candidate indices and their shots.
///
/// The result is sorted by index; repeated bitstrings add up, saturating at `u64::MAX`. See [`bitstring_index`].
///
/// # Errors
///
/// Returns the first [`InvalidBitstring`].
///
/// # Panics
///
/// As [`bitstring_index`].
pub fn candidate_counts<'a>(
    counts: impl IntoIterator<Item = (&'a str, u64)>,
    qubits: usize,
) -> Result<Vec<(usize, u64)>, InvalidBitstring> {
    let mut indexed = counts
        .into_iter()
        .map(|(bitstring, shots)| Ok((bitstring_index(bitstring, qubits)?, shots)))
        .collect::<Result<Vec<_>, InvalidBitstring>>()?;

    indexed.sort_unstable_by_key(|&(index, _)| index);
    indexed.dedup_by(|later, kept| {
        let same = later.0 == kept.0;

        if same {
            kept.1 = kept.1.saturating_add(later.1);
        }

        same
    });

    Ok(indexed)
}
