use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;

        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;

        self.state = x;
        x
    }

    #[inline(always)]
    pub fn unit_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 16_777_216.0
    }

    #[inline(always)]
    pub fn range_f32(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.unit_f32()
    }

    #[inline(always)]
    pub fn range_usize(&mut self, upper: usize) -> usize {
        assert!(upper > 0);
        (self.next_u64() % upper as u64) as usize
    }

    /// Ruído aproximadamente normal, com média zero.
    pub fn normal_f32(&mut self) -> f32 {
        let u1 = self.unit_f32().max(f32::MIN_POSITIVE);
        let u2 = self.unit_f32();

        (-2.0 * u1.ln()).sqrt()
            * (std::f32::consts::TAU * u2).cos()
    }
}

pub fn fresh_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E37_79B9_7F4A_7C15)
}