/// `MarsagliaXOR` random number generator
#[derive(Debug, Clone)]
pub struct MarsagliaXOR {
    seed: u32,
}

impl MarsagliaXOR {
    #[must_use]
    pub const fn new(seed: u32) -> Self {
        Self {
            seed: if seed == 0 { 1 } else { seed },
        }
    }

    /// Generate next random number using `MarsagliaXOR` algorithm
    pub const fn next(&mut self) -> u32 {
        const A: u32 = 123_456_789;
        const M: u32 = 2_147_483_647;
        const Q: u32 = 521_288_629; // M div A
        const R: u32 = 88_675_123; // M mod A

        let hi = self.seed / Q;
        let lo = self.seed % Q;
        let test = A.wrapping_mul(lo).wrapping_sub(R.wrapping_mul(hi));

        self.seed = if test > 0 { test } else { test.wrapping_add(M) };
        self.seed
    }

    /// Generate random number in range [0, max)
    pub const fn range(&mut self, max: u32) -> u32 {
        if max == 0 {
            return 0;
        }
        self.next() % max
    }

    /// Get current seed value
    #[must_use]
    pub const fn seed(&self) -> u32 {
        self.seed
    }

    /// Set seed value
    pub const fn set_seed(&mut self, seed: u32) {
        self.seed = if seed == 0 { 1 } else { seed };
    }
}

impl Default for MarsagliaXOR {
    fn default() -> Self {
        Self::new(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marsaglia_xor_deterministic() {
        let mut rng1 = MarsagliaXOR::new(12345);
        let mut rng2 = MarsagliaXOR::new(12345);

        // Should produce same sequence with same seed
        for _ in 0..100 {
            assert_eq!(rng1.next(), rng2.next());
        }
    }

    #[test]
    fn test_marsaglia_xor_range() {
        let mut rng = MarsagliaXOR::new(42);

        for _ in 0..1000 {
            let val = rng.range(100);
            assert!(val < 100);
        }
    }

    #[test]
    fn test_zero_seed_handling() {
        let rng = MarsagliaXOR::new(0);
        assert_eq!(rng.seed(), 1);
    }
}
