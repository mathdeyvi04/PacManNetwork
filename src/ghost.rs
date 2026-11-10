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
    fn color(&self) -> Color {
        match self.kind {
            GhostKind::Blinky => COLOR_GHOST_BLINKY,
            GhostKind::Pinky  => COLOR_GHOST_PINKY,
            GhostKind::Inky   => COLOR_GHOST_INKY,
            GhostKind::Clyde  => COLOR_GHOST_CLYDE,
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
        // 1) Move na direção atual (clamp contra parede).
        let new_pos = self.pos + self.dir.vec() * GHOST_SPEED * dt;
        self.pos = clamp_against_wall(new_pos, self.dir, map);

        // 2) Se estiver perto do centro do tile, escolhe nova direção.
        let tx = self.pos.x.floor();
        let ty = self.pos.y.floor();
        let cx = tx + 0.5;
        let cy = ty + 0.5;
        if (self.pos.x - cx).abs() < TURN_TOLERANCE
            && (self.pos.y - cy).abs() < TURN_TOLERANCE
        {
            let target = self.compute_target(player_tile, player_dir, blinky_tile, mode);
            let next = choose_direction(
                self.dir,
                (tx as i32, ty as i32),
                target,
                map,
            );
            if next != self.dir {
                self.dir = next;
                self.pos = vec2(cx, cy);   // snap no centro
            }
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
    pub fn draw(&self) {
        let s = GHOST_RADIUS * 2.0;
        draw_rectangle(
            self.pos.x * TILE_SIZE - GHOST_RADIUS,
            self.pos.y * TILE_SIZE - GHOST_RADIUS,
            s, s, self.color(),
        );
    }
}

// ---------------------------------------------------------------------------
// IA: escolhe a direção que minimiza distância AO QUADRADO até o alvo.
// Sem `sqrt`, sem alocação, sem iteração além de 4 direções fixas.
// ---------------------------------------------------------------------------
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
        if map.is_wall(nx as usize, ny as usize) {
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