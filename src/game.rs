use macroquad::prelude::*;
use crate::constants::*;
use crate::ghost::{Ghost, GhostKind, GhostMode};
use crate::map::Map;
use crate::player::{Direction, Player};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GameState { Playing, GameOver, Win }

pub struct Game {
    pub map: Map,
    pub player: Player,
    pub score: u32,
    score_text: String,

    // Array fixo em vez de Vec: 4 × ~16 bytes = 64 bytes na stack do Game.
    // Zero indireção, zero alocação, cache-friendly.
    ghosts: [Ghost; GHOST_COUNT],

    state: GameState,
    mode: GhostMode,
    mode_timer: f32,
}

impl Game {
    pub fn new() -> Self {
        let map = Map::generate();
        let (sx, sy) = find_spawn(&map);
        let dir = initial_direction(&map, sx, sy);
        let player = Player::new(sx, sy, dir);

        let spawns = find_ghost_spawns(&map);
        let kinds = [GhostKind::Blinky, GhostKind::Pinky, GhostKind::Inky, GhostKind::Clyde];
        let ghosts = [
            Ghost::new(spawns[0].0, spawns[0].1, kinds[0]),
            Ghost::new(spawns[1].0, spawns[1].1, kinds[1]),
            Ghost::new(spawns[2].0, spawns[2].1, kinds[2]),
            Ghost::new(spawns[3].0, spawns[3].1, kinds[3]),
        ];

        Self {
            map, player, score: 0,
            score_text: String::from("Score: 0"),
            ghosts,
            state: GameState::Playing,
            mode: GhostMode::Scatter,
            mode_timer: SCATTER_DURATION,
        }
    }

    #[inline(always)]
    pub fn is_game_over(&self) -> bool {
        self.state == GameState::GameOver
    }

    fn add_score(&mut self, amount: u32) {
        if amount == 0 { return; }
        self.score += amount;
        self.score_text.clear();
        use std::fmt::Write;
        let _ = write!(self.score_text, "Score: {}", self.score);
    }

    pub fn update(&mut self, dt: f32) {
        if self.state != GameState::Playing { return; }

        // --- Input ---
        if is_key_pressed(KeyCode::Right) || is_key_pressed(KeyCode::D) { self.player.set_pending(Direction::Right); }
        if is_key_pressed(KeyCode::Left)  || is_key_pressed(KeyCode::A) { self.player.set_pending(Direction::Left);  }
        if is_key_pressed(KeyCode::Up)    || is_key_pressed(KeyCode::W) { self.player.set_pending(Direction::Up);    }
        if is_key_pressed(KeyCode::Down)  || is_key_pressed(KeyCode::S) { self.player.set_pending(Direction::Down);  }

        // --- Player ---
        self.player.update(&self.map, dt);
        let (tx, ty) = self.player.tile();
        if self.map.eat_pellet(tx, ty) { 
            self.add_score(PELLET_SCORE); 
            if self.map.eat_pellet(tx, ty) {
                self.add_score(PELLET_SCORE);
                if self.map.pellets_remaining() == 0 {
                    self.state = GameState::Win;
                    return;                     // não deixa fantasma matar no mesmo frame
                }
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
            for g in &mut self.ghosts { g.dir = g.dir.opposite(); }
        }

        // --- Fantasmas ---
        // Snapshot dos dados compartilhados → sem split-borrow manual.
        let player_tile = self.player.tile();
        let player_dir  = self.player.dir;
        let blinky_tile = self.ghosts[0].tile();
        let mode = self.mode;
        let map_ref = &self.map;
        for g in &mut self.ghosts {
            g.update(map_ref, player_tile, player_dir, blinky_tile, mode, dt);
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

    pub fn draw(&self) {
        self.map.draw();
        for g in &self.ghosts { g.draw(); }
        self.player.draw();

        draw_text(&self.score_text, 8.0, 16.0, 22.0, COLOR_SCORE);

        if self.state == GameState::GameOver {
            let (w, h) = (screen_width(), screen_height());
            draw_rectangle(0.0, 0.0, w, h, COLOR_OVERLAY);
            draw_text("GAME OVER", w * 0.5 - 110.0, h * 0.5 - 10.0, 44.0, COLOR_GAME_OVER);
            draw_text("Press SPACE", w * 0.5 - 78.0, h * 0.5 + 40.0, 24.0, WHITE);
        }
        
        if self.state == GameState::Win {
            let (w, h) = (screen_width(), screen_height());
            draw_rectangle(0.0, 0.0, w, h, COLOR_OVERLAY);
            draw_text("YOU WIN!", w * 0.5 - 100.0, h * 0.5 - 10.0, 44.0, COLOR_WIN);
            draw_text(&self.score_text, w * 0.5 - 70.0, h * 0.5 + 30.0, 24.0, WHITE);
            draw_text("Press SPACE", w * 0.5 - 78.0, h * 0.5 + 70.0, 24.0, WHITE);
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