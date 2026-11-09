mod constants;
mod game;
mod ghost;
mod map;
mod player;

use constants::*;
use macroquad::prelude::*;
use map::Map;

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
    // Gerado UMA vez. O mapa é imutável por enquanto;
    // quando o Pac-Man comer pastilhas, isso vira mutável.
    let map = Map::generate();

    loop {
        clear_background(COLOR_BG);
        map.draw();

        draw_text(
            &format!("FPS: {}", get_fps()),
            8.0,
            14.0,
            18.0,
            WHITE,
        );

        next_frame().await;
    }
}