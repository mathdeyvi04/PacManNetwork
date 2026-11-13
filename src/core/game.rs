use std::time::{SystemTime, UNIX_EPOCH};
use super::ghost::{Ghost, GhostKind, GhostMode};
use super::map::Map;
use super::player::{Direction, Player};
use super::controller::Action;
use super::rng::Rng;  
use crate::constants::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GameState { Playing, GameOver, Win }

pub struct Game {
    pub map: Map,
    pub player: Player,
    pub score: u32,
    ghosts: [Ghost; GHOST_COUNT],
    state: GameState,
    mode: GhostMode,
    mode_timer: f32,
    rng: Rng,
    // ── NOVOS: calculados uma vez, reutilizados em reset_in_place ──
    player_spawn: (usize, usize),
    player_init_dir: Direction,
    ghost_spawns: [(usize, usize); GHOST_COUNT],
}

impl Game {
    // O `new()` original agora apenas chama o `with_map`
    pub fn new() -> Self {
        Self::with_map(Map::generate())
    }
    
    /// Construtor para contextos casuais (modo humano, demo). Seed **aleatória**.
    pub fn with_map(map: Map) -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15);
        Self::with_map_and_seed(map, seed)
    }
    
    /// Construtor para treinamento — seed **explícita** para reprodutibilidade.
    /// Mesma rede + mesma seed + mesmo mapa ⇒ mesma trajetória.
    pub fn with_map_and_seed(map: Map, seed: u64) -> Self {
        let player_spawn = find_spawn(&map);
        let player_init_dir = initial_direction(&map, player_spawn.0, player_spawn.1);
        let ghost_spawns = find_ghost_spawns(&map);
    
        let player = Player::new(player_spawn.0, player_spawn.1, player_init_dir);
        let ghosts = spawn_ghosts(&ghost_spawns, seed);   // ← helper novo
    
        Self {
            map, player, score: 0, ghosts,
            state: GameState::Playing,
            mode: GhostMode::Scatter,
            mode_timer: SCATTER_DURATION,
            rng: Rng::new(seed),
            player_spawn,
            player_init_dir,
            ghost_spawns,
        }
    }

    /// Reinicia o jogo **in-place**, sem re-alocar nada.
    /// Usado em `Agent::evaluate_multi` para as seeds 2..K.
    pub fn reset_in_place(&mut self, seed: u64) {
        self.map.reset_pellets();
        self.player = Player::new(self.player_spawn.0, self.player_spawn.1, self.player_init_dir);
        self.ghosts = spawn_ghosts(&self.ghost_spawns, seed);  // ← helper novo
        self.score = 0;
        self.state = GameState::Playing;
        self.mode = GhostMode::Scatter;
        self.mode_timer = SCATTER_DURATION;
        self.rng = Rng::new(seed);
    }

    #[inline(always)] pub fn state(&self)      -> GameState   { self.state }
    #[inline(always)] pub fn mode(&self)       -> GhostMode   { self.mode }
    #[inline(always)] pub fn mode_timer(&self) -> f32         { self.mode_timer }
    #[inline(always)] pub fn ghosts(&self)     -> &[Ghost]    { &self.ghosts }
    // REMOVIDO: pub fn score_text(&self) -> &str

    #[inline(always)]
    pub fn is_game_over(&self) -> bool {
        self.state == GameState::GameOver
    }

    // Apenas soma o número — sem alocação, sem formatação de string.
    fn add_score(&mut self, amount: u32) {
        if amount == 0 { return; }
        self.score += amount;
    }

    pub fn update(&mut self, dt: f32, action: Action) {
        if self.state != GameState::Playing { return; }

        // --- Input (via Action) ---
        if let Action::Move(dir) = action {
            self.player.set_pending(dir);
        }

        // --- Player ---
        self.player.update(&self.map, dt);
        let (tx, ty) = self.player.tile();
        if self.map.eat_pellet(tx, ty) {
            self.add_score(PELLET_SCORE);
            if self.map.pellets_remaining() == 0 {
                self.state = GameState::Win;
                return;                     // não deixa fantasma matar no mesmo frame
            }
        }

        // --- Timer scatter/chase ---
        self.mode_timer -= dt;
        if self.mode_timer <= 0.0 {
            self.mode = match self.mode {
                GhostMode::Scatter => GhostMode::Chase,
                GhostMode::Chase   => GhostMode::Scatter,
            };
            self.mode_timer = match self.mode {
                GhostMode::Scatter => SCATTER_DURATION,
                GhostMode::Chase   => CHASE_DURATION,
            };
            // Reverter todos ao trocar de modo (regra clássica).
            for g in &mut self.ghosts { g.reverse(); }
        }

        // --- Fantasmas ---
        // Snapshot dos dados compartilhados → sem split-borrow manual.
        let player_tile = self.player.tile();
        let player_dir  = self.player.dir;
        let blinky_tile = self.ghosts[0].tile();
        let mode        = self.mode;
        let map_ref     = &self.map;
        for g in &mut self.ghosts {
            g.update(map_ref, player_tile, player_dir, blinky_tile, mode, dt, &mut self.rng);
        }

        // --- Colisão Pac-Man × fantasma (distância², sem sqrt) ---
        let ppos = self.player.pos;
        for g in &self.ghosts {
            let dx = g.pos.x - ppos.x;
            let dy = g.pos.y - ppos.y;
            if dx * dx + dy * dy < CATCH_DISTANCE_SQ {
                self.state = GameState::GameOver;
                return;
            }
        }
    }

    #[inline(always)]
    pub fn is_finished(&self) -> bool {
        self.state != GameState::Playing
    }
}

// ---------------------------------------------------------------------------
// Spawns
// ---------------------------------------------------------------------------

fn find_spawn(map: &Map) -> (usize, usize) {
    for y in (1..GRID_HEIGHT - 1).rev() {
        for x in 1..GRID_WIDTH - 1 {
            if !map.is_wall(x, y) { return (x, y); }
        }
    }
    (1, 1)
}

fn initial_direction(map: &Map, x: usize, y: usize) -> Direction {
    for d in Direction::ALL {
        let (nx, ny) = match d {
            Direction::Right => (x + 1, y), Direction::Left => (x - 1, y),
            Direction::Down  => (x, y + 1), Direction::Up   => (x, y - 1),
        };
        if !map.is_wall(nx, ny) { return d; }
    }
    Direction::Right
}

/// Coleta 4 tiles livres no terço superior — 1 por linha, para
/// dispersar os fantasmas. Duas passadas O(N), zero alocação.
fn find_ghost_spawns(map: &Map) -> [(usize, usize); 4] {
    let mut spawns = [(1usize, 1usize); 4];
    let mut found = 0usize;
    let y_limit = GRID_HEIGHT / 3;

    for y in 1..y_limit {
        for x in 1..GRID_WIDTH - 1 {
            if !map.is_wall(x, y) {
                spawns[found] = (x, y);
                found += 1;
                break; // pega 1 por linha → dispersão natural
            }
        }
        if found == 4 { return spawns; }
    }

    // Fallback (mapa muito hostil no topo): varre o resto.
    'outer: for y in 1..GRID_HEIGHT - 1 {
        for x in 1..GRID_WIDTH - 1 {
            if !map.is_wall(x, y) {
                spawns[found] = (x, y);
                found += 1;
                if found == 4 { break 'outer; }
            }
        }
    }
    spawns
}

/// Cria os 4 fantasmas com direção inicial pseudoaleatória derivada da seed.
///
/// Por que randomizar a direção inicial: o fantasma chega ao primeiro
/// centro já na 1ª atualização; ali ele **não** roda `choose_direction`
/// (não está `blocked` se a direção inicial é livre), então a direção
/// inicial determina os primeiros ~12 frames. Isso já é suficiente para
/// que os 4 fantasmas comecem em trajetórias diferentes entre seeds,
/// mesmo antes de qualquer `GHOST_RANDOM_TURN_CHANCE` entrar em ação.
///
/// Usamos um `Rng` separado (não o `game.rng`) para não consumir o
/// stream de gameplay — assim o número de chamadas de RNG dentro do
/// loop de simulação não depende dessa inicialização.
fn spawn_ghosts(spawns: &[(usize, usize); GHOST_COUNT], seed: u64) -> [Ghost; GHOST_COUNT] {
    let kinds = [GhostKind::Blinky, GhostKind::Pinky, GhostKind::Inky, GhostKind::Clyde];
    let mut dir_rng = Rng::new(seed ^ 0xDEAD_BEEF_CAFE_F00D);

    let mut ghosts: [Ghost; GHOST_COUNT] = [
        Ghost::new(spawns[0].0, spawns[0].1, kinds[0]),
        Ghost::new(spawns[1].0, spawns[1].1, kinds[1]),
        Ghost::new(spawns[2].0, spawns[2].1, kinds[2]),
        Ghost::new(spawns[3].0, spawns[3].1, kinds[3]),
    ];

    for g in &mut ghosts {
        g.dir = Direction::ALL[dir_rng.range_usize(4)];
    }

    ghosts
}