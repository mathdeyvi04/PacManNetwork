mod constants;
mod game;
mod ghost;
mod map;
mod player;

use constants::*;
use game::Game;
use macroquad::prelude::*;

fn window_conf() -> Conf {
    Conf {
        window_title: "Pac-Man".to_owned(),
        window_width: WINDOW_WIDTH,
        window_height: WINDOW_HEIGHT,
        window_resizable: false,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut game = Game::new();

    loop {
        let dt = get_frame_time().min(0.05);
        clear_background(COLOR_BG);

        game.update(dt);

        // Restart: SPACE após Game Over gera novo mapa e zera score.
        if game.is_game_over() && is_key_pressed(KeyCode::Space) {
            game = Game::new();
        }

        game.draw();
        next_frame().await;
    }
}