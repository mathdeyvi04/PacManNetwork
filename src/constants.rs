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
pub const COLOR_WIN: Color = Color::new(0.20, 1.00, 0.20, 1.0);

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
pub const CATCH_DISTANCE_SQ: f32 = 0.25;
/// Quantos estados `(tile, dir)` memorizar por fantasma.
pub const HIST_SIZE: usize = 8;
/// Quantos tiles de "fuga aleatória" após detectar um loop.
pub const DETOUR_LEN: u8 = 4;

pub const COLOR_GAME_OVER:    Color = Color::new(1.0, 0.2, 0.2, 1.0);
pub const COLOR_OVERLAY:      Color = Color::new(0.0, 0.0, 0.0, 0.75);

// ---------------------------------------------------------------------------
// Painel de visualização da rede neural (abaixo do mapa)
// ---------------------------------------------------------------------------
pub const NEURAL_PANEL_HEIGHT: f32 = 260.0;
pub const WINDOW_HEIGHT_TOTAL: i32 = WINDOW_HEIGHT + NEURAL_PANEL_HEIGHT as i32;

// Cores base (usadas como fallback / identidade)
// --- CHASE: cores agressivas (alta saturação / brilho) ---
pub const COLOR_CHASE_BLINKY: Color = Color::new(1.00, 0.00, 0.00, 1.0); // vermelho puro
pub const COLOR_CHASE_PINKY:  Color = Color::new(1.00, 0.30, 0.65, 1.0); // rosa quente
pub const COLOR_CHASE_INKY:   Color = Color::new(0.00, 0.90, 1.00, 1.0); // ciano elétrico
pub const COLOR_CHASE_CLYDE:  Color = Color::new(1.00, 0.45, 0.00, 1.0); // laranja intenso

// --- SCATTER: cores tranquilas (dessaturadas / claras) ---
pub const COLOR_SCATTER_BLINKY: Color = Color::new(1.00, 0.55, 0.55, 1.0); // vermelho suave
pub const COLOR_SCATTER_PINKY:  Color = Color::new(1.00, 0.80, 0.90, 1.0); // rosa pastel
pub const COLOR_SCATTER_INKY:   Color = Color::new(0.60, 1.00, 1.00, 1.0); // ciano claro
pub const COLOR_SCATTER_CLYDE:  Color = Color::new(1.00, 0.80, 0.55, 1.0); // laranja claro

// Blinking dos fantasmas antes da troca Scatter ↔ Chase
pub const MODE_WARNING_DURATION: f32 = 5.0;
pub const COLOR_GHOST_BLINK: Color = WHITE;

// Túnel: linha central do grid
pub const TUNNEL_ROW: usize = GRID_HEIGHT / 2;    
pub const COLOR_TUNNEL: Color = Color::new(0.15, 0.15, 0.30, 1.0);