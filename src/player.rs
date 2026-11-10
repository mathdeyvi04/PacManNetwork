use macroquad::prelude::*;
use crate::constants::*;
use crate::map::Map;

/// Uma das quatro direções cardeais.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    /// Vetor unitário correspondente em espaço de tile.
    #[inline(always)]
    pub fn vec(self) -> Vec2 {
        match self {
            Direction::Up => vec2(0.0, -1.0),
            Direction::Down => vec2(0.0, 1.0),
            Direction::Left => vec2(-1.0, 0.0),
            Direction::Right => vec2(1.0, 0.0),
        }
    }

    #[inline(always)]
    pub fn opposite(self) -> Direction {
        match self {
            Direction::Up => Direction::Down,
            Direction::Down => Direction::Up,
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
        }
    }

    pub const ALL: [Direction; 4] = [
        Direction::Up,
        Direction::Left,
        Direction::Down,
        Direction::Right,
    ];
}

/// Estado do jogador: posição contínua, direção atual e direção pedida.
pub struct Player {
    /// Posição em espaço de tile (centro do tile = `(x + 0.5, y + 0.5)`).
    pub pos: Vec2,
    /// Direção em que o jogador está de fato se movendo.
    pub dir: Direction,
    /// Direção solicitada pelo input, aguardando uma curva válida.
    pub pending: Option<Direction>,
    /// Velocidade em tiles por segundo.
    pub speed: f32,
}

impl Player {
    pub fn new(x: usize, y: usize, dir: Direction) -> Self {
        Self {
            pos: vec2(x as f32 + 0.5, y as f32 + 0.5),
            dir,
            pending: None,
            speed: PLAYER_SPEED,
        }
    }

    pub fn set_pending(&mut self, dir: Direction) {
        self.pending = Some(dir);
    }

    #[inline(always)]
    pub fn tile(&self) -> (usize, usize) {
        (self.pos.x.floor() as usize, self.pos.y.floor() as usize)
    }

    /// Atualiza direção e posição em um passo de `dt` segundos.
    ///
    /// Curvas só são aplicadas quando o jogador está alinhado ao centro
    /// do tile no eixo perpendicular ao movimento — assim ele "entra"
    /// no corredor em vez de cortar a esquina na diagonal.
    pub fn update(&mut self, map: &Map, dt: f32) {
        if let Some(p) = self.pending {
            if p == self.dir {
                self.pending = None;
            } else if p == self.dir.opposite() {
                self.dir = p;
                self.pending = None;
            } else {
                // Alinha o eixo perpendicular à nova direção antes de checar
                // a parede à frente; sem isso a checagem usaria um tile
                // vizinho "enviesado" e permitiria virar no meio da célula.
                let mut aligned = false;
                let mut snapped_pos = self.pos;

                match p {
                    Direction::Left | Direction::Right => {
                        if let Some(cy) = near_center(self.pos.y) {
                            snapped_pos.y = cy;
                            aligned = true;
                        }
                    }
                    Direction::Up | Direction::Down => {
                        if let Some(cx) = near_center(self.pos.x) {
                            snapped_pos.x = cx;
                            aligned = true;
                        }
                    }
                }

                if aligned {
                    let (tx, ty) = (snapped_pos.x.floor() as i32, snapped_pos.y.floor() as i32);
                    let (nx, ny) = match p {
                        Direction::Up => (tx, ty - 1),
                        Direction::Down => (tx, ty + 1),
                        Direction::Left => (tx - 1, ty),
                        Direction::Right => (tx + 1, ty),
                    };

                    if !map.is_wall(nx as usize, ny as usize) {
                        self.dir = p;
                        self.pos = snapped_pos;
                        self.pending = None;
                    }
                }
            }
        }

        let delta = self.dir.vec() * self.speed * dt;
        let new_pos = self.pos + delta;
        self.pos = clamp_against_wall(new_pos, self.dir, map);
    }

    /// Desenha o Pac-Man com a boca oscilando na direção atual.
    pub fn draw(&self) {
        let cx = self.pos.x * TILE_SIZE;
        let cy = self.pos.y * TILE_SIZE;

        let t = (get_time() as f32 * 8.0).sin().abs();
        let half_mouth = 0.05 + t * 0.35;

        // macroquad: +Y aponta para baixo.
        let base = match self.dir {
            Direction::Right => 0.0,
            Direction::Down => std::f32::consts::FRAC_PI_2,
            Direction::Left => std::f32::consts::PI,
            Direction::Up => -std::f32::consts::FRAC_PI_2,
        };

        // O arco desenhado cobre [base + half_mouth, base + TAU - half_mouth],
        // deixando um vão centrado em `base` — que é o "buraco" da boca.
        const SEGMENTS: u32 = 20;
        let start = base + half_mouth;
        let sweep = std::f32::consts::TAU - 2.0 * half_mouth;
        let step = sweep / SEGMENTS as f32;

        let center = vec2(cx, cy);
        // Cache do ponto anterior: 1 cos + 1 sin por segmento em vez de 2 + 2.
        let mut prev = vec2(
            cx + PLAYER_RADIUS * start.cos(),
            cy + PLAYER_RADIUS * start.sin(),
        );

        for i in 1..=SEGMENTS {
            let a = start + step * i as f32;
            let next = vec2(
                cx + PLAYER_RADIUS * a.cos(),
                cy + PLAYER_RADIUS * a.sin(),
            );
            draw_triangle(center, prev, next, COLOR_PACMAN);
            prev = next;
        }
    }
}

/// Retorna o centro do tile mais próximo se `v` estiver dentro de
/// [`TURN_TOLERANCE`] dele; caso contrário, `None`.
#[inline(always)]
fn near_center(v: f32) -> Option<f32> {
    let center = (v - 0.5).round() + 0.5;
    if (v - center).abs() < TURN_TOLERANCE {
        Some(center)
    } else {
        None
    }
}

/// Impede atravessar paredes: se o tile à frente na direção `dir` for
/// parede, trava o eixo de movimento no centro do tile atual.
#[inline]
pub fn clamp_against_wall(pos: Vec2, dir: Direction, map: &Map) -> Vec2 {
    let mut p = pos;
    let tx = p.x.floor() as i32;
    let ty = p.y.floor() as i32;
    let w = GRID_WIDTH as i32;
    let h = GRID_HEIGHT as i32;

    match dir {
        Direction::Right => {
            if tx + 1 < w && map.is_wall((tx + 1) as usize, ty as usize) {
                p.x = p.x.min(tx as f32 + 0.5);
            }
        }
        Direction::Left => {
            if tx - 1 >= 0 && map.is_wall((tx - 1) as usize, ty as usize) {
                p.x = p.x.max(tx as f32 + 0.5);
            }
        }
        Direction::Down => {
            if ty + 1 < h && map.is_wall(tx as usize, (ty + 1) as usize) {
                p.y = p.y.min(ty as f32 + 0.5);
            }
        }
        Direction::Up => {
            if ty - 1 >= 0 && map.is_wall(tx as usize, (ty - 1) as usize) {
                p.y = p.y.max(ty as f32 + 0.5);
            }
        }
    }
    p
}