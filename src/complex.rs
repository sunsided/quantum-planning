//! A minimal complex number over `f64`, the amplitude type of the simulator.

use core::ops::{Add, Mul, Neg, Sub};

/// A complex number `re + i·im` with `f64` parts.
///
/// The type has only the operations the simulator needs. Arithmetic follows IEEE 754: NaN
/// and infinities propagate and are not checked.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex {
    /// The real part.
    pub re: f64,
    /// The imaginary part.
    pub im: f64,
}

impl Complex {
    /// The additive identity, `0`.
    pub const ZERO: Self = Self::new(0.0, 0.0);

    /// The multiplicative identity, `1`.
    pub const ONE: Self = Self::new(1.0, 0.0);

    /// Makes the complex number `re + i·im`.
    #[must_use]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Makes the complex number `e^(i·angle)`, with `angle` in radians.
    #[must_use]
    pub fn from_angle(angle: f64) -> Self {
        Self::new(libm::cos(angle), libm::sin(angle))
    }

    /// Returns the squared magnitude `re² + im²`, the measurement probability of an amplitude.
    #[must_use]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Multiplies both parts by the real number `factor`.
    #[must_use]
    pub fn scale(self, factor: f64) -> Self {
        Self::new(self.re * factor, self.im * factor)
    }
}

impl Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl Neg for Complex {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

/// Unit tests for the complex arithmetic.
#[cfg(test)]
mod tests {
    use super::Complex;

    /// `i · i = -1`, and `e^(iπ/2) = i` within rounding.
    #[test]
    fn multiplies_and_rotates() {
        let i = Complex::new(0.0, 1.0);
        let quarter_turn = Complex::from_angle(core::f64::consts::FRAC_PI_2);

        assert_eq!(i * i, -Complex::ONE);
        assert!((quarter_turn - i).norm_sqr() < 1e-30);
        assert_eq!((Complex::new(3.0, 4.0)).norm_sqr(), 25.0);
    }
}
