use macroquad::prelude::*;

// Dimensões base do jogo
pub const GRID_WIDTH: usize = 31;
pub const GRID_HEIGHT: usize = 33;
pub const TILE_SIZE: f32 = 18.0;

// Tamanho da janela em pixels
pub const WINDOW_WIDTH: i32 = (GRID_WIDTH as f32 * TILE_SIZE) as i32;
pub const WINDOW_HEIGHT: i32 = (GRID_HEIGHT as f32 * TILE_SIZE) as i32;

// Cores
pub const COLOR_BG: Color = BLACK;
pub const COLOR_WALL: Color = BLUE;
pub const COLOR_PELLET: Color = LIGHTGRAY;
pub const COLOR_PACMAN: Color = YELLOW;
pub const COLOR_GHOST: Color = RED;