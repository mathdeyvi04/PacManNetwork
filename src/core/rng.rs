//! RNG determinístico (xorshift64) — compartilhado por:
//!
//! * `ai::neural`  → inicialização de pesos e mutação.
//! * `core::ghost` → decisões aleatórias (detour, fuga de loop).
//!
//! **Sem estado global** → seguro entre threads e reprodutível.
//! Cada consumidor recebe seu próprio `Rng` com uma seed explícita.

pub struct Rng {
    state: u64,
}

impl Rng {
    /// Constrói com seed. `0` é remapeado para a constante áurea (xorshift
    /// não pode ter estado 0, senão trava em zero).
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }

    #[inline]
    pub fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniforme em `[0, 1)`.
    #[inline]
    pub fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / ((1u64 << 24) as f32)
    }

    /// Uniforme em `[low, high)`.
    #[inline]
    pub fn range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.unit()
    }

    /// Inteiro uniforme em `[0, n)`. Substitui `gen_range(0, n)`.
    #[inline]
    pub fn range_usize(&mut self, n: usize) -> usize {
        debug_assert!(n > 0, "range_usize requer n > 0");
        let v = (self.next() >> 32) as u64 * n as u64 >> 32;
        v as usize
    }
}