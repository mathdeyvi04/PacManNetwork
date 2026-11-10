mod constants;
mod controller;
mod game;
mod ghost;
mod map;
mod player;
mod renderer;

use constants::*;
use controller::{Action, Controller, KeyboardController};
use game::Game;
use renderer::Renderer;

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
    let mut controller = KeyboardController::new();   // trocar aqui para NeuralController
    let renderer = Renderer::new();

    loop {
        let dt = get_frame_time().min(0.05);
        clear_background(COLOR_BG);

        let action = controller.poll();

        if game.is_finished() && action == Action::Restart {
            game = Game::new();
        } else {
            game.update(dt, action);
        }

        renderer.draw(&game);
        next_frame().await;
    }
}