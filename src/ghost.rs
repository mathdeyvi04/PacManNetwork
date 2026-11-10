use macroquad::prelude::*;
use crate::constants::*;
use crate::map::Map;
use crate::player::{Direction, clamp_against_wall};

/// Cada variante mapeia uma cor e um algoritmo de alvo.
/// `#[repr(u8)]` → 1 byte. Total do struct cabe em ~16 bytes.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GhostKind { Blinky, Pinky, Inky, Clyde }

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GhostMode { Scatter, Chase }

pub struct Ghost {
    pub pos: Vec2,
    pub dir: Direction,
    pub kind: GhostKind,
}

impl Ghost {
    pub fn new(x: usize, y: usize, kind: GhostKind) -> Self {
        Self {
            pos: vec2(x as f32 + 0.5, y as f32 + 0.5),
            dir: Direction::Left,   // será corrigida no 1º centro de tile
            kind,
        }
    }

    #[inline(always)]
    pub fn tile(&self) -> (usize, usize) {
        (self.pos.x.floor() as usize, self.pos.y.floor() as usize)
    }

    #[inline(always)]
    fn color(&self, mode: GhostMode) -> Color {
        match (self.kind, mode) {
            (GhostKind::Blinky, GhostMode::Chase)   => COLOR_CHASE_BLINKY,
            (GhostKind::Pinky,  GhostMode::Chase)   => COLOR_CHASE_PINKY,
            (GhostKind::Inky,   GhostMode::Chase)   => COLOR_CHASE_INKY,
            (GhostKind::Clyde,  GhostMode::Chase)   => COLOR_CHASE_CLYDE,
    
            (GhostKind::Blinky, GhostMode::Scatter) => COLOR_SCATTER_BLINKY,
            (GhostKind::Pinky,  GhostMode::Scatter) => COLOR_SCATTER_PINKY,
            (GhostKind::Inky,   GhostMode::Scatter) => COLOR_SCATTER_INKY,
            (GhostKind::Clyde,  GhostMode::Scatter) => COLOR_SCATTER_CLYDE,
        }
    }

    /// Atualiza um fantasma. Tudo passado por valor (Copy) → sem borrow
    /// cruzado entre ghosts; o `Game` pode iterar `iter_mut()` sem split.
    pub fn update(
        &mut self,
        map: &Map,
        player_tile: (usize, usize),
        player_dir: Direction,
        blinky_tile: (usize, usize),
        mode: GhostMode,
        dt: f32,
    ) {
        // Centro do tile mais próximo.
        let center_x = tile_center(self.pos.x);
        let center_y = tile_center(self.pos.y);
    
        // Verifica se o fantasma está centralizado.
        let at_center =
            (self.pos.x - center_x).abs() < TURN_TOLERANCE
            && (self.pos.y - center_y).abs() < TURN_TOLERANCE;
    
        // Evita que o fantasma fique parado quando começa
        // em uma direção cuja passagem está bloqueada.
        if at_center {
            let tile = (
                center_x.floor() as i32,
                center_y.floor() as i32,
            );
    
            let (dx, dy) = match self.dir {
                Direction::Up    => (0, -1),
                Direction::Down  => (0, 1),
                Direction::Left  => (-1, 0),
                Direction::Right => (1, 0),
            };
    
            let nx = tile.0 + dx;
            let ny = tile.1 + dy;
    
            let blocked =
                nx < 0
                || ny < 0
                || nx >= GRID_WIDTH as i32
                || ny >= GRID_HEIGHT as i32
                || map.is_wall_or_tunnel(nx as usize, ny as usize);
    
            if blocked {
                let target = self.compute_target(
                    player_tile,
                    player_dir,
                    blinky_tile,
                    mode,
                );
    
                self.dir = choose_direction(
                    self.dir,
                    tile,
                    target,
                    map,
                );
            }
        }
    
        // Guarda a posição anterior.
        let old_pos = self.pos;
    
        // Movimento normal.
        let new_pos = self.pos + self.dir.vec() * GHOST_SPEED * dt;
    
        self.pos = clamp_against_wall::<true>(new_pos, self.dir, map);
    
        // Detecta o PRÓXIMO centro na direção do movimento.
        // Importante: o centro atual nunca é considerado
        // uma nova passagem.
    
        let (crossed_center, cx, cy) = match self.dir {
    
            Direction::Right => {
                let cx = (old_pos.x - 0.5).floor() + 1.5;
                let cy = tile_center(old_pos.y);
    
                (
                    old_pos.x < cx
                        && self.pos.x >= cx
                        && (old_pos.y - cy).abs() < TURN_TOLERANCE,
                    cx,
                    cy,
                )
            }
    
            Direction::Left => {
                let cx = (old_pos.x - 0.5).ceil() - 0.5;
                let cy = tile_center(old_pos.y);
    
                (
                    old_pos.x > cx
                        && self.pos.x <= cx
                        && (old_pos.y - cy).abs() < TURN_TOLERANCE,
                    cx,
                    cy,
                )
            }
    
            Direction::Down => {
                let cy = (old_pos.y - 0.5).floor() + 1.5;
                let cx = tile_center(old_pos.x);
    
                (
                    old_pos.y < cy
                        && self.pos.y >= cy
                        && (old_pos.x - cx).abs() < TURN_TOLERANCE,
                    cx,
                    cy,
                )
            }
    
            Direction::Up => {
                let cy = (old_pos.y - 0.5).ceil() - 0.5;
                let cx = tile_center(old_pos.x);
    
                (
                    old_pos.y > cy
                        && self.pos.y <= cy
                        && (old_pos.x - cx).abs() < TURN_TOLERANCE,
                    cx,
                    cy,
                )
            }
        };
    
        // Ao alcançar o próximo centro, decide a direção.
        if crossed_center {
            self.pos = vec2(cx, cy);
    
            let tile = (
                cx.floor() as i32,
                cy.floor() as i32,
            );
    
            let target = self.compute_target(
                player_tile,
                player_dir,
                blinky_tile,
                mode,
            );
    
            self.dir = choose_direction(
                self.dir,
                tile,
                target,
                map,
            );
        }
    }


    /// Calcula o tile-alvo segundo a personalidade.
    #[inline]
    fn compute_target(
        &self,
        pt: (usize, usize),
        pd: Direction,
        bt: (usize, usize),
        mode: GhostMode,
    ) -> (i32, i32) {
        if mode == GhostMode::Scatter {
            return self.scatter_corner();
        }
        match self.kind {
            GhostKind::Blinky => (pt.0 as i32, pt.1 as i32),
            GhostKind::Pinky => {
                let v = pd.vec();
                (pt.0 as i32 + (v.x as i32) * 4,
                 pt.1 as i32 + (v.y as i32) * 4)
            }
            GhostKind::Inky => {
                let v = pd.vec();
                let px = pt.0 as i32 + (v.x as i32) * 2;
                let py = pt.1 as i32 + (v.y as i32) * 2;
                (2 * px - bt.0 as i32, 2 * py - bt.1 as i32)
            }
            GhostKind::Clyde => {
                let dx = self.pos.x - (pt.0 as f32 + 0.5);
                let dy = self.pos.y - (pt.1 as f32 + 0.5);
                if dx * dx + dy * dy < 64.0 {   // < 8 tiles²
                    self.scatter_corner()
                } else {
                    (pt.0 as i32, pt.1 as i32)
                }
            }
        }
    }

    #[inline(always)]
    fn scatter_corner(&self) -> (i32, i32) {
        let w = GRID_WIDTH  as i32 - 2;
        let h = GRID_HEIGHT as i32 - 2;
        match self.kind {
            GhostKind::Blinky => (w, 1),
            GhostKind::Pinky  => (1, 1),
            GhostKind::Inky   => (w, h),
            GhostKind::Clyde  => (1, h),
        }
    }

    /// Desenha como quadrado — coerente com o estilo do Pac-Man "sem sprites".
    pub fn draw(&self, blink: bool, mode: GhostMode) {
        let s = GHOST_RADIUS * 2.0;
        let color = if blink { COLOR_GHOST_BLINK } else { self.color(mode) };
        draw_rectangle(
            self.pos.x * TILE_SIZE - GHOST_RADIUS,
            self.pos.y * TILE_SIZE - GHOST_RADIUS,
            s, s, color,
        );
    }
}


#[inline(always)]
fn tile_center(v: f32) -> f32 {
    (v - 0.5).round() + 0.5
}

/// Escolhe a direção que minimiza distância AO QUADRADO até o alvo.
#[inline]
fn choose_direction(
    current: Direction,
    tile: (i32, i32),
    target: (i32, i32),
    map: &Map,
) -> Direction {
    let reverse = current.opposite();
    let mut best = None;
    let mut best_d2 = i32::MAX;

    // Ordem clássica de desempate: Up > Left > Down > Right.
    for d in Direction::ALL {
        if d == reverse {
            continue;
        }
        let (dx, dy) = match d {
            Direction::Up    => (0, -1),
            Direction::Down  => (0,  1),
            Direction::Left  => (-1, 0),
            Direction::Right => (1,  0),
        };
        let nx = tile.0 + dx;
        let ny = tile.1 + dy;
        // Bordas já são paredes; nx, ny sempre em [0, W-1] / [0, H-1].
        if map.is_wall_or_tunnel(nx as usize, ny as usize) {
            continue;
        }
        let ddx = target.0 - nx;
        let ddy = target.1 - ny;
        let d2 = ddx * ddx + ddy * ddy;
        if d2 < best_d2 {
            best_d2 = d2;
            best = Some(d);
        }
    }

    // Beco sem saída: única opção é reverter (regra do arcade).
    best.unwrap_or(reverse)
}