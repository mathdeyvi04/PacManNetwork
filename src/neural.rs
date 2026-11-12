//! Inferência de rede neural (MLP 57 → 64 → 4) jogando Pac-Man.
//!
//! Sem heap no forward pass. Pesos em arrays fixos (≈16 KB).
//! Formato de arquivo próprio (magic + header + f32 LE).

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

use crate::constants::*;
use crate::controller::{Action, Controller};
use crate::game::Game;
use crate::map::Tile;
use crate::player::Direction;

// ---------------------------------------------------------------------------
// Arquitetura
// ---------------------------------------------------------------------------
// 
// Seguinte, ajude-me a resolver alguns pontos da visualização da rede

const VISION: usize = 7;
const HALF: i32 = (VISION as i32) / 2;

pub const INPUT_SIZE:  usize = VISION * VISION + 4 + 2 + 2; // 57
pub const HIDDEN_SIZE: usize = 64;
pub const OUTPUT_SIZE: usize = 4;

const MAGIC: &[u8; 8] = b"PACNET01";
const VERSION: u32 = 1;

pub struct NeuralNet {
    // w1[j][i]  → j em [0,HIDDEN), i em [0,INPUT)
    w1: [f32; HIDDEN_SIZE * INPUT_SIZE],
    b1: [f32; HIDDEN_SIZE],
    // w2[k][j]  → k em [0,OUTPUT), j em [0,HIDDEN)
    w2: [f32; OUTPUT_SIZE * HIDDEN_SIZE],
    b2: [f32; OUTPUT_SIZE],
}

impl NeuralNet {
    pub fn random(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut net = Self {
            w1: [0.0; HIDDEN_SIZE * INPUT_SIZE],
            b1: [0.0; HIDDEN_SIZE],
            w2: [0.0; OUTPUT_SIZE * HIDDEN_SIZE],
            b2: [0.0; OUTPUT_SIZE],
        };
        // Xavier-ish uniforme.
        let s1 = 2.0 / (INPUT_SIZE  as f32).sqrt();
        let s2 = 2.0 / (HIDDEN_SIZE as f32).sqrt();
        for w in net.w1.iter_mut() { *w = rng.range(-s1, s1); }
        for w in net.w2.iter_mut() { *w = rng.range(-s2, s2); }
        // b1, b2 ficam 0.
        net
    }

    #[inline]
    pub fn forward_into(
        &self,
        x: &[f32; INPUT_SIZE],
        h: &mut [f32; HIDDEN_SIZE],
        y: &mut [f32; OUTPUT_SIZE],
    ) {
        for j in 0..HIDDEN_SIZE {
            let row = &self.w1[j * INPUT_SIZE..(j + 1) * INPUT_SIZE];
            let mut s = self.b1[j];
            for i in 0..INPUT_SIZE {
                s += row[i] * x[i];
            }
            h[j] = if s > 0.0 { s } else { 0.0 };
        }
        for k in 0..OUTPUT_SIZE {
            let row = &self.w2[k * HIDDEN_SIZE..(k + 1) * HIDDEN_SIZE];
            let mut s = self.b2[k];
            for j in 0..HIDDEN_SIZE {
                s += row[j] * h[j];
            }
            y[k] = s;
        }
    }

    #[inline]
    pub fn forward(&self, x: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        let mut h = [0.0f32; HIDDEN_SIZE];
        let mut y = [0.0f32; OUTPUT_SIZE];
        self.forward_into(x, &mut h, &mut y);
        y
    }

    // Acessores read-only para a visualização (sem cópia, sem alocação).
    #[inline(always)] pub fn w1(&self) -> &[f32; HIDDEN_SIZE * INPUT_SIZE]  { &self.w1 }
    #[inline(always)] pub fn w2(&self) -> &[f32; OUTPUT_SIZE * HIDDEN_SIZE] { &self.w2 }
    #[inline(always)] pub fn b1(&self) -> &[f32; HIDDEN_SIZE]               { &self.b1 }
    #[inline(always)] pub fn b2(&self) -> &[f32; OUTPUT_SIZE]               { &self.b2 }

    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut f = File::create(path)?;
        f.write_all(MAGIC)?;
        f.write_all(&VERSION.to_le_bytes())?;
        f.write_all(&(INPUT_SIZE  as u32).to_le_bytes())?;
        f.write_all(&(HIDDEN_SIZE as u32).to_le_bytes())?;
        f.write_all(&(OUTPUT_SIZE as u32).to_le_bytes())?;
        for w in self.w1.iter()
            .chain(self.b1.iter())
            .chain(self.w2.iter())
            .chain(self.b2.iter())
        {
            f.write_all(&w.to_le_bytes())?;
        }
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let mut f = File::open(path)?;
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "magic inválido"));
        }
        let version = read_u32(&mut f)?;
        if version != VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "versão incompatível"));
        }
        let i = read_u32(&mut f)? as usize;
        let h = read_u32(&mut f)? as usize;
        let o = read_u32(&mut f)? as usize;
        if i != INPUT_SIZE || h != HIDDEN_SIZE || o != OUTPUT_SIZE {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "arquitetura incompatível"));
        }
        let mut net = Self {
            w1: [0.0; HIDDEN_SIZE * INPUT_SIZE],
            b1: [0.0; HIDDEN_SIZE],
            w2: [0.0; OUTPUT_SIZE * HIDDEN_SIZE],
            b2: [0.0; OUTPUT_SIZE],
        };
        for w in net.w1.iter_mut()
            .chain(net.b1.iter_mut())
            .chain(net.w2.iter_mut())
            .chain(net.b2.iter_mut())
        {
            let mut buf = [0u8; 4];
            f.read_exact(&mut buf)?;
            *w = f32::from_le_bytes(buf);
        }
        Ok(net)
    }
}

#[inline]
fn read_u32(f: &mut File) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

// ---------------------------------------------------------------------------
// Encoding de features
// ---------------------------------------------------------------------------

/// Codifica o estado do jogo em 57 floats, divididos em 4 grupos:
///
/// **Grupo 1 — Visão local 7×7 (`[0..49)`).**
/// Janela 7×7 centrada no tile do Pac-Man. Conteúdo de cada célula:
/// `1.0` parede/túnel, `0.5` pellet, `0.0` vazio, `-1.0` fantasma
/// (fantasmas sobrescrevem o tile). Fora do mapa conta como parede.
///
/// **Grupo 2 — Direção atual (`[49..53)`).**
/// One-hot `[Up, Down, Left, Right]` — exatamente um `1.0`.
///
/// **Grupo 3 — Fantasma mais próximo (`[53..55)`).**
/// `(dx, dy)` em tiles, normalizado pelos lados do grid, apontando do
/// Pac-Man para o centro do tile do fantasma mais próximo.
///
/// **Grupo 4 — Pellet mais próximo (`[55..57)`).**
/// `(dx, dy)` em tiles, normalizado idem, para o pellet mais próximo.
pub fn encode(game: &Game) -> [f32; INPUT_SIZE] {
    let mut f = [0.0f32; INPUT_SIZE];
    let (px, py) = game.player.tile();
    let cx = px as i32;
    let cy = py as i32;

    // 1) Visão local 7×7
    let mut idx = 0usize;
    for dy in -HALF..=HALF {
        for dx in -HALF..=HALF {
            let nx = cx + dx;
            let ny = cy + dy;
            if nx < 0 || ny < 0 || nx >= GRID_WIDTH as i32 || ny >= GRID_HEIGHT as i32 {
                f[idx] = 1.0; // OOB = parede
            } else {
                f[idx] = match game.map.tile_at(nx as usize, ny as usize) {
                    Tile::Wall | Tile::Tunnel => 1.0,
                    Tile::Empty              => 0.0,
                    Tile::Pellet             => 0.5,
                };
            }
            idx += 1;
        }
    }

    // 2) Fantasmas sobrescrevem o tile na visão local
    for g in game.ghosts() {
        let gx = g.pos.x.floor() as i32;
        let gy = g.pos.y.floor() as i32;
        let lx = gx - (cx - HALF);
        let ly = gy - (cy - HALF);
        if (0..VISION as i32).contains(&lx) && (0..VISION as i32).contains(&ly) {
            f[(ly * VISION as i32 + lx) as usize] = -1.0;
        }
    }

    // 3) Direção atual (one-hot)
    let d = match game.player.dir {
        Direction::Up    => 0,
        Direction::Down  => 1,
        Direction::Left  => 2,
        Direction::Right => 3,
    };
    f[idx + d] = 1.0;
    idx += 4;

    // 4) Direção do fantasma mais próximo (normalizada para ~[-1,1])
    let inv_w = 1.0 / GRID_WIDTH  as f32;
    let inv_h = 1.0 / GRID_HEIGHT as f32;
    let mut best_gd2 = i32::MAX;
    let mut gvec = (0.0f32, 0.0f32);
    for g in game.ghosts() {
        let gx = g.pos.x.floor() as i32;
        let gy = g.pos.y.floor() as i32;
        let dx = gx - cx;
        let dy = gy - cy;
        let d2 = dx * dx + dy * dy;
        if d2 < best_gd2 {
            best_gd2 = d2;
            gvec = (dx as f32 * inv_w, dy as f32 * inv_h);
        }
    }
    f[idx]     = gvec.0;
    f[idx + 1] = gvec.1;
    idx += 2;

    // 5) Pellet mais próximo (varredura linear — 1023 iterações, trivial)
    let mut best_pd2 = i32::MAX;
    let mut pvec = (0.0f32, 0.0f32);
    let tiles = game.map.tiles();
    for y in 0..GRID_HEIGHT {
        for x in 0..GRID_WIDTH {
            if tiles[y * GRID_WIDTH + x] == Tile::Pellet {
                let dx = x as i32 - cx;
                let dy = y as i32 - cy;
                let d2 = dx * dx + dy * dy;
                if d2 < best_pd2 {
                    best_pd2 = d2;
                    pvec = (dx as f32 * inv_w, dy as f32 * inv_h);
                }
            }
        }
    }
    f[idx]     = pvec.0;
    f[idx + 1] = pvec.1;

    f
}

// ---------------------------------------------------------------------------
// Controller
// ---------------------------------------------------------------------------

pub struct NeuralController {
    net: NeuralNet,
    reverse_cooldown: u8,
    pub last_input:  [f32; INPUT_SIZE],
    pub last_hidden: [f32; HIDDEN_SIZE],
    pub last_output: [f32; OUTPUT_SIZE],
}

impl NeuralController {
    pub fn new(net: NeuralNet) -> Self {
        Self {
            net,
            reverse_cooldown: 0,
            last_input:  [0.0; INPUT_SIZE],
            last_hidden: [0.0; HIDDEN_SIZE],
            last_output: [0.0; OUTPUT_SIZE],
        }
    }

    #[inline(always)]
    pub fn net(&self) -> &NeuralNet { &self.net }

    pub fn random() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15);
        Self::new(NeuralNet::random(seed))
    }

    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        Ok(Self::new(NeuralNet::load(path)?))
    }
}

impl Controller for NeuralController {
    fn poll(&mut self, game: &Game) -> Action {
        if game.is_finished() {
            return Action::Restart;
        }

        if self.reverse_cooldown > 0 {
            self.reverse_cooldown -= 1;
        }

        let x = encode(game);
        self.last_input = x;
        self.net.forward_into(&x, &mut self.last_hidden, &mut self.last_output);

        // argmax
        let mut best = 0usize;
        let mut best_v = self.last_output[0];
        for k in 1..OUTPUT_SIZE {
            if self.last_output[k] > best_v { best = k; best_v = self.last_output[k]; }
        }
        let dir = match best {
            0 => Direction::Up,
            1 => Direction::Down,
            2 => Direction::Left,
            _ => Direction::Right,
        };

        // Anti-jitter: ignora inversão durante cooldown.
        if dir == game.player.dir.opposite() {
            if self.reverse_cooldown > 0 {
                return Action::Move(game.player.dir);
            }
            self.reverse_cooldown = 8; // ~130 ms @ 60 fps
        }

        Action::Move(dir)
    }
}

// ---------------------------------------------------------------------------
// RNG determinístico (xorshift64)
// ---------------------------------------------------------------------------

struct Rng { state: u64 }

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed } }
    }
    #[inline] fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    #[inline] fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    #[inline] fn range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.unit()
    }
}