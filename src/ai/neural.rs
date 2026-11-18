//! Inferência de rede neural (MLP 22 → 64 → 4) jogando Pac-Man.
//!
//! Sem heap no forward pass. Pesos em arrays fixos (~6.7 kB).
//! Formato de arquivo próprio (magic + header + f32 LE).
//!
//! Arquitetura (v4):
//!   0..16   núcleo sensorial — raycast + fantasmas (v3)
//!   16..18  gradiente de pellets (vetor de atração ponderado)
//!   18..22  memória recorrente (cópia amortecida dos 4 primeiros ocultos)

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

use crate::constants::*;
use crate::core::controller::{Action, Controller};
use crate::core::game::Game;
use crate::core::ghost::Ghost;
use crate::core::map::{Map, Tile};
use crate::core::player::Direction;
use crate::core::rng::Rng;

// ---------------------------------------------------------------------------
// Arquitetura — 22 entradas → 64 ocultos → 4 saídas
//
// Blocos de entrada (ordem semântica, ver `encode` + `poll`):
//   0..4   categoria do primeiro evento por direção [Up, Down, Left, Right]
//   4..8   distância normalizada até esse evento
//   8..12  distância normalizada a cada fantasma
//   12..16 ângulo relativo a cada fantasma, em [-1, 1]
//   16..18 gradiente de pellets (x, y) em [-1, 1] (encode)
//   18..22 memória recorrente (preenchida em `poll`)
// ---------------------------------------------------------------------------

pub const INPUT_SIZE:  usize = 22;  // 16 originais + 2 (gradiente de pellets) + 4 (memória recorrente)
pub const HIDDEN_SIZE: usize = 64;  // ampliado proporcionalmente
pub const OUTPUT_SIZE: usize = 4;

const MAGIC: &[u8; 8] = b"PACNET01";
const VERSION: u32 = 4;             // bump: incompatível com v1/v2/v3

/// Quantos neurônios ocultos compõem a memória recorrente.
pub const CONTEXT_SIZE: usize = 4;

// Codificação escalar de cada categoria (ordem semântica: perigo → recompensa)
const ENC_GHOST:   f32 = -1.0;         // morte iminente
const ENC_WALL:    f32 = -0.5;         // bloqueio permanente
const ENC_NOTHING: f32 =  0.0;         // neutro
const ENC_TUNNEL:  f32 =  0.5;         // fuga segura
const ENC_PELLET:  f32 =  1.0;         // recompensa

#[derive(Clone)]
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

    /// Aplica **mutação uniforme gene a gene**.
    ///
    /// Cada parâmetro (w1, b1, w2, b2, iterados em sequência) tem
    /// `rate` de probabilidade de receber uma perturbação uniforme em
    /// `[-strength, +strength)`. O `seed` inicializa o `Rng` interno —
    /// chamadas com seeds distintas geram streams independentes.
    pub fn mutate(&mut self, rate: f32, strength: f32, seed: u64) {
        let mut rng = Rng::new(seed);
        for w in self.w1.iter_mut()
            .chain(self.b1.iter_mut())
            .chain(self.w2.iter_mut())
            .chain(self.b2.iter_mut())
        {
            if rng.unit() < rate {
                *w += rng.range(-strength, strength);
            }
        }
    }

    /// Crossover **uniforme gene a gene**: cada parâmetro é escolhido
    /// aleatoriamente de `self` ou de `other` com probabilidade 50%.
    ///
    /// Preferido a single-point porque a topologia é fixa (22→64→4) e
    /// recombinação "por camada" quebraria blocos funcionais inteiros.
    /// Uniforme preserva blocos locais e permite que dois pais contribuam
    /// com partes distintas (ex.: "aprendeu a desviar" + "aprendeu a comer").
    pub fn crossover(&self, other: &Self, rng: &mut Rng) -> Self {
        let mut child = Self {
            w1: [0.0; HIDDEN_SIZE * INPUT_SIZE],
            b1: [0.0; HIDDEN_SIZE],
            w2: [0.0; OUTPUT_SIZE * HIDDEN_SIZE],
            b2: [0.0; OUTPUT_SIZE],
        };

        for i in 0..HIDDEN_SIZE * INPUT_SIZE {
            child.w1[i] = if rng.unit() < 0.5 { self.w1[i] } else { other.w1[i] };
        }
        for i in 0..HIDDEN_SIZE {
            child.b1[i] = if rng.unit() < 0.5 { self.b1[i] } else { other.b1[i] };
        }
        for i in 0..OUTPUT_SIZE * HIDDEN_SIZE {
            child.w2[i] = if rng.unit() < 0.5 { self.w2[i] } else { other.w2[i] };
        }
        for i in 0..OUTPUT_SIZE {
            child.b2[i] = if rng.unit() < 0.5 { self.b2[i] } else { other.b2[i] };
        }

        child
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
// Encoding
// ---------------------------------------------------------------------------

/// 18 entradas produzidas por `encode` (as 4 de memória são injetadas em
/// `NeuralController::poll`):
///
/// * `0..4` — **categorias** `[Up, Down, Left, Right]`: primeiro evento
///   significativo do raycast em cada direção —
///   fantasma (-1.0), parede (-0.5), túnel (+0.5), pellet (+1.0), nada (0.0).
///   Fantasma tem prioridade sobre qualquer tile (sobrevivência primeiro).
/// * `4..8` — **distâncias** até esse mesmo evento, normalizadas por
///   `1 / (1 + d)`, mesma ordem `[Up, Down, Left, Right]`.
/// * `8..12` — **distância a cada fantasma** (Blinky, Pinky, Inky, Clyde),
///   Euclidiana em tiles, normalizada por `1 / (1 + d)`.
/// * `12..16` — **ângulo relativo** a cada fantasma, `atan2(dy, dx) / π`
///   no intervalo `[-1, 1]`.
/// * `16..18` — **gradiente de pellets**: centroide ponderado por
///   `1 / (1 + d²)` dos pellets restantes, tomado em relação ao Pac-Man
///   e comprimido por `tanh` para `[-1, 1]`. Aponta na direção do cluster
///   mais "pesado" (próximo + numeroso) de pellets.
/// * `18..22` — deixado em 0.0 por `encode`; preenchido com a memória
///   recorrente do frame anterior em `poll`.
///
/// A ordem das categorias casa com o argmax em `NeuralController::poll`.
pub fn encode(game: &Game) -> [f32; INPUT_SIZE] {
    let (px, py) = game.player.tile();
    const DIRS: [Direction; 4] = [
        Direction::Up,
        Direction::Down,
        Direction::Left,
        Direction::Right,
    ];

    // OTIMIZAÇÃO: pré-calcula as posições inteiras (tile) dos fantasmas
    // uma única vez — evita `floor()` dentro dos loops de raycast.
    let mut ghost_tiles = [(0i32, 0i32); GHOST_COUNT];
    for (i, g) in game.ghosts().iter().enumerate() {
        let (gx, gy) = g.tile();
        ghost_tiles[i] = (gx as i32, gy as i32);
    }

    let mut f = [0.0f32; INPUT_SIZE];

    // ---- 0..8 — raycast direcional --------------------------------------
    for (i, &dir) in DIRS.iter().enumerate() {
        let (event, distance) = sense(&game.map, &ghost_tiles, px, py, dir);
        f[i]     = event;                          // 0..3: categoria
        f[i + 4] = normalize_distance(distance);   // 4..7: distância
    }

    // ---- 8..16 — sensores de fantasmas ----------------------------------
    // Usa posições em tiles (f32) para obter ângulos mais finos que o
    // raycast por tile; a distância acompanha a mesma unidade.
    let ppx = game.player.pos.x;
    let ppy = game.player.pos.y;
    for (i, g) in game.ghosts().iter().enumerate().take(4) {
        let dx = g.pos.x - ppx;
        let dy = g.pos.y - ppy;
        let dist = (dx * dx + dy * dy).sqrt();

        // Distância normalizada — mesma curva usada no raycast.
        f[8 + i] = 1.0 / (1.0 + dist);

        // Ângulo relativo em [-1, 1]. atan2(dy, dx) ∈ [-π, π].
        let angle = dy.atan2(dx);
        f[12 + i] = angle / std::f32::consts::PI;
    }

    // ---- 16..18 — gradiente de pellets ----------------------------------
    // Centroide ponderado por `1 / (1 + d²)` sobre todos os pellets
    // restantes. O peso próximo-1 / longe-0 prioriza clusters próximos,
    // mas ainda leva em conta pellets distantes quando não há nada perto.
    //
    // Nota de custo: varre o grid inteiro (≈1023 tiles) por frame. Em
    // `MAX_STEPS = 12000` × `POPULATION_SIZE = 100` × `K = 5` seeds isso
    // soma ~6B de leituras de tile por geração — aceitável, mas se o
    // perfil mostrar custo relevante, amostrar a cada 2-4 frames basta:
    // o gradiente varia pouco entre frames consecutivos.
    let mut pellet_sum_x = 0.0f32;
    let mut pellet_sum_y = 0.0f32;
    let mut pellet_count = 0.0f32;

    for y in 0..GRID_HEIGHT {
        for x in 0..GRID_WIDTH {
            if matches!(game.map.tile_at(x, y), Tile::Pellet) {
                let dx = x as f32 - ppx;
                let dy = y as f32 - ppy;
                let dist_sq = dx * dx + dy * dy + 0.1;
                // Pondera inversamente pela distância para priorizar clusters próximos
                let weight = 1.0 / dist_sq;
                pellet_sum_x += dx * weight;
                pellet_sum_y += dy * weight;
                pellet_count += weight;
            }
        }
    }

    if pellet_count > 0.0 {
        f[16] = (pellet_sum_x / pellet_count).tanh(); // Normalizado para [-1, 1]
        f[17] = (pellet_sum_y / pellet_count).tanh();
    }

    // ---- 18..22 — memória recorrente ------------------------------------
    // Preenchida em `NeuralController::poll` com `last_context` do frame
    // anterior. `encode` deixa 0.0 para que a função permaneça pura
    // (mesmo `game` → mesmo retorno).

    f
}

/// Normaliza uma distância em tiles para o intervalo `(0, 0.5]`.
///
/// * `d = 1`  → 0.500
/// * `d = 2`  → 0.333
/// * `d = 5`  → 0.167
/// * `d = 10` → 0.091
/// * `d = 30` → 0.032
///
/// Vantagens: (a) cabe no mesmo range das categorias em `[-1, 1]` sem
/// dominar a primeira camada; (b) decaimento suave → mais resolução perto
/// (o que importa para desviar de fantasma e comer pellet); (c) nunca
/// explode para raycasts longos.
#[inline]
fn normalize_distance(d: f32) -> f32 {
    1.0 / (1.0 + d)
}

/// Raycast a partir de `(px, py)` na direção `dir`.
///
/// Retorna `(evento, distância_em_tiles)`.
///
/// * `evento` mantém a codificação categórica atual.
/// * `distância` é a distância até **esse** evento (≥ 1.0).
#[inline]
fn sense(
    map: &Map,
    ghost_tiles: &[(i32, i32); GHOST_COUNT],
    px: usize,
    py: usize,
    dir: Direction,
) -> (f32, f32) {
    let (dx, dy) = match dir {
        Direction::Up    => (0, -1),
        Direction::Down  => (0,  1),
        Direction::Left  => (-1, 0),
        Direction::Right => ( 1, 0),
    };
    let w = GRID_WIDTH  as i32;
    let h = GRID_HEIGHT as i32;
    let mut x = px as i32 + dx;
    let mut y = py as i32 + dy;
    let mut dist = 1.0f32;

    // Caso especial: Pac-Man em cima de um tile de túnel na borda,
    // olhando para fora → wrap-around → túnel (distância 1).
    if (x < 0 || x >= w) && y == TUNNEL_ROW as i32 {
        return (ENC_TUNNEL, dist);
    }

    let mut first_event      = ENC_NOTHING;
    let mut first_event_dist = 0.0f32;

    // A visão só é bloqueada por paredes. O loop continua varrendo a linha
    // para garantir que um fantasma seja detectado mesmo que haja pellets no caminho.
    while x >= 0 && y >= 0 && x < w && y < h {
        // 1. Fantasma tem prioridade absoluta (sobrevivência em primeiro lugar).
        // Ele "atravessa" pellets na visão do sensor.
        for &(gx, gy) in ghost_tiles {
            if gx == x && gy == y {
                return (ENC_GHOST, dist);
            }
        }

        match map.tile_at(x as usize, y as usize) {
            Tile::Wall => {
                // Parede: devolve o primeiro evento anterior se existir,
                // senão devolve a parede com sua própria distância.
                return if first_event == ENC_NOTHING {
                    (ENC_WALL, dist)
                } else {
                    (first_event, first_event_dist)
                };
            }
            Tile::Pellet => {
                // Registra o primeiro pellet visto, mas não interrompe a busca por fantasmas.
                if first_event == ENC_NOTHING {
                    first_event      = ENC_PELLET;
                    first_event_dist = dist;
                }
            }
            Tile::Tunnel => {
                // Registra o primeiro túnel visto.
                if first_event == ENC_NOTHING {
                    first_event      = ENC_TUNNEL;
                    first_event_dist = dist;
                }
            }
            Tile::Empty => {
                // Continua varrendo
            }
        }

        x += dx;
        y += dy;
        dist += 1.0;
    }

    // Se chegou à borda do mapa sem encontrar parede nem fantasma,
    // devolve o primeiro evento com sua distância (ou ENC_NOTHING + dist
    // "grande" — inofensivo após normalização).
    if first_event == ENC_NOTHING {
        (ENC_NOTHING, dist)
    } else {
        (first_event, first_event_dist)
    }
}

// ---------------------------------------------------------------------------
// Controller
// ---------------------------------------------------------------------------

pub struct NeuralController {
    net: NeuralNet,
    reverse_cooldown: u8,
    /// Memória recorrente: cópia amortecida (tanh) dos 4 primeiros ocultos
    /// do frame anterior, injetada em `x[18..22]`.
    last_context: [f32; CONTEXT_SIZE],
    pub last_input:  [f32; INPUT_SIZE],
    pub last_hidden: [f32; HIDDEN_SIZE],
    pub last_output: [f32; OUTPUT_SIZE],
}

impl NeuralController {
    pub fn new(net: NeuralNet) -> Self {
        Self {
            net,
            reverse_cooldown: 0,
            last_context: [0.0; CONTEXT_SIZE],
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

    /// Versão reprodutível de `random`. Mesmo seed → mesma rede inicial.
    /// Usado pelo AG para que um treino com `--seed 42` produza exatamente
    /// os mesmos pesos iniciais em toda a população.
    pub fn random_with_seed(seed: u64) -> Self {
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

        // 1. Codifica o estado atual do jogo (18 entradas)
        let mut x = encode(game);

        // 2. Injeta os 4 valores de memória do frame anterior nas entradas 18..22
        for i in 0..CONTEXT_SIZE {
            x[18 + i] = self.last_context[i];
        }

        self.last_input = x;
        self.net.forward_into(&x, &mut self.last_hidden, &mut self.last_output);

        // 3. Atualiza o contexto para o próximo frame usando uma fração da camada oculta
        for i in 0..CONTEXT_SIZE {
            self.last_context[i] = self.last_hidden[i].tanh();
        }

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