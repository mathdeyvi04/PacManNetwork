use macroquad::prelude::*;
use macroquad::rand::gen_range;
use crate::constants::*;
use crate::map::Map;
use crate::player::{Direction, clamp_against_wall};

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GhostKind { Blinky, Pinky, Inky, Clyde }

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GhostMode { Scatter, Chase }

pub struct Ghost {
    pub pos: Vec2,
    pub dir: Direction,
    pub kind: GhostKind,
    hist: [(i16, i16, Direction); HIST_SIZE],
    hist_len: u8,
    hist_head: u8,
    detour: u8,
}

impl Ghost {
    pub fn new(x: usize, y: usize, kind: GhostKind) -> Self {
        Self {
            pos: vec2(x as f32 + 0.5, y as f32 + 0.5),
            dir: Direction::Left,
            kind,
            hist: [(0, 0, Direction::Left); HIST_SIZE],
            hist_len: 0,
            hist_head: 0,
            detour: 0,
        }
    }

    #[inline(always)]
    pub fn tile(&self) -> (usize, usize) {
        (self.pos.x.floor() as usize, self.pos.y.floor() as usize)
    }

    /// Reverte a direção e zera o detector de loop.
    /// Chamado pelo Game na troca Scatter ↔ Chase.
    #[inline]
    pub fn reverse(&mut self) {
        self.dir = self.dir.opposite();
        self.hist_len = 0;
        self.hist_head = 0;
        self.detour = 0;
    }

    fn record(&mut self, tile: (i16, i16), dir: Direction) {
        let h = self.hist_head as usize;
        self.hist[h] = (tile.0, tile.1, dir);
        self.hist_head = ((h + 1) % HIST_SIZE) as u8;
        if self.hist_len < HIST_SIZE as u8 {
            self.hist_len += 1;
        }
    }

    fn is_loop(&self, tile: (i16, i16), dir: Direction) -> bool {
        let n = self.hist_len as usize;
        for i in 0..n {
            let (tx, ty, td) = self.hist[i];
            if tx == tile.0 && ty == tile.1 && td == dir {
                return true;
            }
        }
        false
    }

    fn clear_hist(&mut self) {
        self.hist_len = 0;
        self.hist_head = 0;
    }

    /// Decide a nova direção num centro de tile, com detecção de loop.
    fn decide(&mut self, tile: (i16, i16), target: (i32, i32), map: &Map) -> Direction {
        // 1) Detour em curso: passo aleatório (exceto reverso).
        if self.detour > 0 {
            self.detour -= 1;
            let d = random_non_reverse(self.dir, tile, map);
            self.record(tile, d);
            return d;
        }

        // 2) Loop detectado? Reverte + entra em detour.
        if self.is_loop(tile, self.dir) {
            let d = self.dir.opposite();
            self.clear_hist();
            self.record(tile, d);
            self.detour = DETOUR_LEN;
            return d;
        }

        // 3) Caso normal: greedy clássico.
        self.record(tile, self.dir);
        choose_direction(self.dir, (tile.0 as i32, tile.1 as i32), target, map)
    }

    pub fn update(
        &mut self,
        map: &Map,
        player_tile: (usize, usize),
        player_dir: Direction,
        blinky_tile: (usize, usize),
        mode: GhostMode,
        dt: f32,
    ) {
        let center_x = tile_center(self.pos.x);
        let center_y = tile_center(self.pos.y);

        let at_center =
            (self.pos.x - center_x).abs() < TURN_TOLERANCE
            && (self.pos.y - center_y).abs() < TURN_TOLERANCE;

        if at_center {
            let tile = (center_x.floor() as i32, center_y.floor() as i32);
            let (dx, dy) = match self.dir {
                Direction::Up    => (0, -1),
                Direction::Down  => (0, 1),
                Direction::Left  => (-1, 0),
                Direction::Right => (1, 0),
            };
            let nx = tile.0 + dx;
            let ny = tile.1 + dy;
            let blocked =
                nx < 0 || ny < 0
                || nx >= GRID_WIDTH as i32
                || ny >= GRID_HEIGHT as i32
                || map.is_wall_or_tunnel(nx as usize, ny as usize);

            if blocked {
                let target = self.compute_target(player_tile, player_dir, blinky_tile, mode);
                self.dir = choose_direction(self.dir, tile, target, map);
            }
        }

        let old_pos = self.pos;
        let new_pos = self.pos + self.dir.vec() * GHOST_SPEED * dt;
        self.pos = clamp_against_wall::<true>(new_pos, self.dir, map);

        let (crossed_center, cx, cy) = match self.dir {
            Direction::Right => {
                let cx = (old_pos.x - 0.5).floor() + 1.5;
                let cy = tile_center(old_pos.y);
                (
                    old_pos.x < cx
                        && self.pos.x >= cx
                        && (old_pos.y - cy).abs() < TURN_TOLERANCE,
                    cx, cy,
                )
            }
            Direction::Left => {
                let cx = (old_pos.x - 0.5).ceil() - 0.5;
                let cy = tile_center(old_pos.y);
                (
                    old_pos.x > cx
                        && self.pos.x <= cx
                        && (old_pos.y - cy).abs() < TURN_TOLERANCE,
                    cx, cy,
                )
            }
            Direction::Down => {
                let cy = (old_pos.y - 0.5).floor() + 1.5;
                let cx = tile_center(old_pos.x);
                (
                    old_pos.y < cy
                        && self.pos.y >= cy
                        && (old_pos.x - cx).abs() < TURN_TOLERANCE,
                    cx, cy,
                )
            }
            Direction::Up => {
                let cy = (old_pos.y - 0.5).ceil() - 0.5;
                let cx = tile_center(old_pos.x);
                (
                    old_pos.y > cy
                        && self.pos.y <= cy
                        && (old_pos.x - cx).abs() < TURN_TOLERANCE,
                    cx, cy,
                )
            }
        };

        if crossed_center {
            self.pos = vec2(cx, cy);
            let tile_i16 = (cx.floor() as i16, cy.floor() as i16);
            let target = self.compute_target(player_tile, player_dir, blinky_tile, mode);
            self.dir = self.decide(tile_i16, target, map);
        }
    }

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
                if dx * dx + dy * dy < 64.0 {
                    self.scatter_corner()
                } else {
                    (pt.0 as i32, pt.1 as i32)
                }
            }
        }
    }

    #[inline(always)]
    fn scatter_corner(&self) -> (i32, i32) {
        let w = GRID_WIDTH as i32 - 2;
        let h = GRID_HEIGHT as i32 - 2;
        match self.kind {
            GhostKind::Blinky => (w, 1),
            GhostKind::Pinky  => (1, 1),
            GhostKind::Inky   => (w, h),
            GhostKind::Clyde  => (1, h),
        }
    }
}

#[inline(always)]
fn tile_center(v: f32) -> f32 {
    (v - 0.5).round() + 0.5
}

/// Escolhe uma direção não-reversa uniformemente ao acaso.
fn random_non_reverse(current: Direction, tile: (i16, i16), map: &Map) -> Direction {
    let reverse = current.opposite();
    let mut options = [Direction::Up; 3];
    let mut n = 0usize;
    for d in Direction::ALL {
        if d == reverse { continue; }
        let (dx, dy) = match d {
            Direction::Up    => (0, -1),
            Direction::Down  => (0, 1),
            Direction::Left  => (-1, 0),
            Direction::Right => (1, 0),
        };
        let nx = tile.0 + dx;
        let ny = tile.1 + dy;
        if nx < 0 || ny < 0 || nx >= GRID_WIDTH as i16 || ny >= GRID_HEIGHT as i16 {
            continue;
        }
        if map.is_wall_or_tunnel(nx as usize, ny as usize) { continue; }
        options[n] = d;
        n += 1;
    }
    if n == 0 { reverse } else { options[gen_range(0, n)] }
}

/// Greedy clássico: minimiza d² ao alvo sem reverter.
fn choose_direction(
    current: Direction,
    tile: (i32, i32),
    target: (i32, i32),
    map: &Map,
) -> Direction {
    let reverse = current.opposite();
    let mut best = None;
    let mut best_d2 = i32::MAX;

    for d in Direction::ALL {
        if d == reverse { continue; }
        let (dx, dy) = match d {
            Direction::Up    => (0, -1),
            Direction::Down  => (0,  1),
            Direction::Left  => (-1, 0),
            Direction::Right => (1,  0),
        };
        let nx = tile.0 + dx;
        let ny = tile.1 + dy;
        if nx < 0 || ny < 0
            || nx >= GRID_WIDTH as i32
            || ny >= GRID_HEIGHT as i32
        { continue; }
        if map.is_wall_or_tunnel(nx as usize, ny as usize) { continue; }
        let ddx = target.0 - nx;
        let ddy = target.1 - ny;
        let d2 = ddx * ddx + ddy * ddy;
        if d2 < best_d2 {
            best_d2 = d2;
            best = Some(d);
        }
    }
    best.unwrap_or(reverse)
}