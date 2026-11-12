use macroquad::prelude::*;

use crate::constants::*;
use crate::core::game::{Game, GameState};
use crate::core::ghost::{Ghost, GhostKind, GhostMode};
use crate::core::map::{Map, Tile};
use crate::core::player::{Direction, Player};

pub struct Renderer;

impl Renderer {
    #[inline]
    pub const fn new() -> Self { Self }

    pub fn draw(&self, game: &Game) {
        Self::draw_map(&game.map);

        let warning = game.state() == GameState::Playing
            && game.mode_timer() < MODE_WARNING_DURATION;
        let blink_on = warning && ((get_time() * 4.0) as i32 & 1) == 0;

        for g in game.ghosts() {
            Self::draw_ghost(g, blink_on, game.mode());
        }

        Self::draw_player(&game.player);
        draw_text(game.score_text(), 8.0, 16.0, 22.0, COLOR_SCORE);

        match game.state() {
            GameState::GameOver => Self::draw_game_over(),
            GameState::Win      => Self::draw_win(game.score_text()),
            GameState::Playing  => {}
        }
    }

    fn draw_map(map: &Map) {
        // Acesso ao slice achatado → índice linear, sem `%`/`/` por célula.
        let tiles = map.tiles();
        for y in 0..GRID_HEIGHT {
            let py = y as f32 * TILE_SIZE;
            let row_base = y * GRID_WIDTH;
            for x in 0..GRID_WIDTH {
                let t = tiles[row_base + x];
                let px = x as f32 * TILE_SIZE;
                match t {
                    Tile::Wall => {
                        draw_rectangle(px, py, TILE_SIZE, TILE_SIZE, COLOR_WALL);
                    }
                    Tile::Pellet => {
                        let s = TILE_SIZE * 0.28;
                        let o = (TILE_SIZE - s) * 0.5;
                        draw_rectangle(px + o, py + o, s, s, COLOR_PELLET);
                    }
                    Tile::Tunnel => {
                        draw_rectangle(px, py, TILE_SIZE, TILE_SIZE, COLOR_TUNNEL);
                    }
                    Tile::Empty => {}
                }
            }
        }
    }

    fn draw_ghost(ghost: &Ghost, blink: bool, mode: GhostMode) {
        let color = if blink { COLOR_GHOST_BLINK } else { ghost_color(ghost.kind, mode) };
        let s = GHOST_RADIUS * 2.0;
        draw_rectangle(
            ghost.pos.x * TILE_SIZE - GHOST_RADIUS,
            ghost.pos.y * TILE_SIZE - GHOST_RADIUS,
            s, s, color,
        );
    }

    fn draw_player(player: &Player) {
        let cx = player.pos.x * TILE_SIZE;
        let cy = player.pos.y * TILE_SIZE;

        let t = (get_time() as f32 * 8.0).sin().abs();
        let half_mouth = 0.05 + t * 0.35;

        let base = match player.dir {
            Direction::Right => 0.0,
            Direction::Down  => std::f32::consts::FRAC_PI_2,
            Direction::Left  => std::f32::consts::PI,
            Direction::Up    => -std::f32::consts::FRAC_PI_2,
        };

        const SEGMENTS: u32 = 20;
        let start = base + half_mouth;
        let sweep = std::f32::consts::TAU - 2.0 * half_mouth;
        let step = sweep / SEGMENTS as f32;

        let center = vec2(cx, cy);
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

    fn draw_game_over() {
        let (w, h) = (screen_width(), screen_height());
        draw_rectangle(0.0, 0.0, w, h, COLOR_OVERLAY);
        draw_text("GAME OVER", w * 0.5 - 110.0, h * 0.5 - 10.0, 44.0, COLOR_GAME_OVER);
        draw_text("Press SPACE", w * 0.5 - 78.0, h * 0.5 + 40.0, 24.0, WHITE);
    }

    fn draw_win(score_text: &str) {
        let (w, h) = (screen_width(), screen_height());
        draw_rectangle(0.0, 0.0, w, h, COLOR_OVERLAY);
        draw_text("YOU WIN!", w * 0.5 - 100.0, h * 0.5 - 10.0, 44.0, COLOR_WIN);
        draw_text(score_text, w * 0.5 - 70.0, h * 0.5 + 30.0, 24.0, WHITE);
        draw_text("Press SPACE", w * 0.5 - 78.0, h * 0.5 + 70.0, 24.0, WHITE);
    }
}

#[inline(always)]
fn ghost_color(kind: GhostKind, mode: GhostMode) -> Color {
    match (kind, mode) {
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