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
        self.pos = clamp_against_wall::<false>(new_pos, self.dir, map);

        self.wrap_tunnel();
    }

    /// Teletransporta o jogador para o outro lado quando ele cruza
    /// a borda do mapa estando na linha do túnel.
    #[inline]
    fn wrap_tunnel(&mut self) {
        let ty = self.pos.y.floor() as i32;
        if ty != TUNNEL_ROW as i32 {
            return;
        }
        if self.pos.x < 0.0 {
            self.pos.x += GRID_WIDTH as f32;
        } else if self.pos.x >= GRID_WIDTH as f32 {
            self.pos.x -= GRID_WIDTH as f32;
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
pub fn clamp_against_wall<const FOR_GHOST: bool>(pos: Vec2, dir: Direction, map: &Map) -> Vec2 {
    let mut p = pos;
    let tx = p.x.floor() as i32;
    let ty = p.y.floor() as i32;
    let w = GRID_WIDTH as i32;
    let h = GRID_HEIGHT as i32;

    // `FOR_GHOST` é constante de compilação → uma única especialização por chamador.
    #[inline(always)]
    fn blocked(map: &Map, x: usize, y: usize, for_ghost: bool) -> bool {
        if for_ghost { map.is_wall_or_tunnel(x, y) } else { map.is_wall(x, y) }
    }

    match dir {
        Direction::Right => {
            if tx + 1 < w && blocked(map, (tx + 1) as usize, ty as usize, FOR_GHOST) {
                p.x = p.x.min(tx as f32 + 0.5);
            }
        }
        Direction::Left => {
            if tx - 1 >= 0 && blocked(map, (tx - 1) as usize, ty as usize, FOR_GHOST) {
                p.x = p.x.max(tx as f32 + 0.5);
            }
        }
        Direction::Down => {
            if ty + 1 < h && blocked(map, tx as usize, (ty + 1) as usize, FOR_GHOST) {
                p.y = p.y.min(ty as f32 + 0.5);
            }
        }
        Direction::Up => {
            if ty - 1 >= 0 && blocked(map, tx as usize, (ty - 1) as usize, FOR_GHOST) {
                p.y = p.y.max(ty as f32 + 0.5);
            }
        }
    }
    p
}