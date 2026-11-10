use macroquad::prelude::*;

// Dimensões base do jogo
pub const GRID_WIDTH: usize = 31;
pub const GRID_HEIGHT: usize = 33;
pub const TILE_SIZE: f32 = 18.0;

// Tamanho da janela em pixels
pub const WINDOW_WIDTH: i32 = (GRID_WIDTH as f32 * TILE_SIZE) as i32;
pub const WINDOW_HEIGHT: i32 = (GRID_HEIGHT as f32 * TILE_SIZE) as i32;

// Controle do jogador
pub const PLAYER_SPEED: f32 = 6.0;
pub const PLAYER_RADIUS: f32 = TILE_SIZE * 0.42;
pub const TURN_TOLERANCE: f32 = 0.15;

// Pontuação 
pub const PELLET_SCORE: u32 = 10;
pub const COLOR_SCORE: Color = WHITE;
pub const COLOR_HUD_BG : Color = Color::new(0.0, 0.0, 1.0, 0.85);

// Cores
pub const COLOR_BG: Color = BLACK;
pub const COLOR_WALL: Color = BLUE;
pub const COLOR_PELLET: Color = LIGHTGRAY;
pub const COLOR_PACMAN: Color = YELLOW;
pub const COLOR_GHOST: Color = RED;

// Fantasmas
pub const GHOST_COUNT: usize = 4;
pub const GHOST_SPEED: f32 = PLAYER_SPEED * 0.85;
pub const GHOST_RADIUS: f32 = TILE_SIZE * 0.40;
pub const SCATTER_DURATION: f32 = 7.0;
pub const CHASE_DURATION: f32 = 20.0;
pub const CATCH_DISTANCE_SQ: f32 = 0.25; // 0.5 tiles ao quadrado

// Cores dos fantasmas
pub const COLOR_GHOST_BLINKY: Color = RED;
pub const COLOR_GHOST_PINKY:  Color = Color::new(1.00, 0.72, 0.85, 1.0);
pub const COLOR_GHOST_INKY:   Color = Color::new(0.00, 1.00, 1.00, 1.0);
pub const COLOR_GHOST_CLYDE:  Color = Color::new(1.00, 0.60, 0.00, 1.0);
pub const COLOR_GAME_OVER:    Color = Color::new(1.0, 0.2, 0.2, 1.0);
pub const COLOR_OVERLAY:      Color = Color::new(0.0, 0.0, 0.0, 0.75);