mod constants;
mod controller;
mod game;
mod ghost;
mod map;
mod neural;
mod player;
mod renderer;
mod neural_viz;
mod rng;

use constants::*;
use controller::{Action, Controller, KeyboardController};
use game::Game;
use neural::NeuralController;
use renderer::Renderer;
use neural_viz::NeuralViz;

use macroquad::prelude::*;


fn is_neural_mode() -> bool {
    matches!(
        std::env::args().nth(1).as_deref(),
        Some("nn") | Some("neural")
    )
}

fn window_conf() -> Conf {
    Conf {
        window_title: "Pac-Man".to_owned(),
        window_width: WINDOW_WIDTH,
        window_height: if is_neural_mode() { WINDOW_HEIGHT_TOTAL } else { WINDOW_HEIGHT },
        window_resizable: false,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

enum Mode {
    Human,
    Neural { weights: Option<String> },
}

fn parse_cli() -> Mode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() { return Mode::Human; }

    match args[0].as_str() {
        "human" | "play" => Mode::Human,
        "nn" | "neural" => Mode::Neural { weights: args.get(1).cloned() },
        "gen-weights" | "gen" => {
            let path = match args.get(1) {
                Some(p) => p,
                None => {
                    eprintln!("Uso: gen-weights <arquivo.bin>");
                    std::process::exit(2);
                }
            };
            let seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(1);
            let net = neural::NeuralNet::random(seed);
            match net.save(path) {
                Ok(()) => {
                    println!("Pesos salvos em: {}", path);
                    std::process::exit(0);
                }
                Err(e) => {
                    eprintln!("Erro ao salvar: {}", e);
                    std::process::exit(1);
                }
            }
        }
        "--help" | "-h" | "help" => {
            println!("Uso: pacman [comando]");
            println!();
            println!("  cargo run --release --bin train -- [geracoes] [threads] [seed]");
            std::process::exit(0);
        }
        outro => {
            eprintln!("Modo desconhecido: {}. Use --help.", outro);
            std::process::exit(2);
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch estático (sem dyn no hot loop)
// ---------------------------------------------------------------------------

enum AnyController {
    Keyboard(KeyboardController),
    Neural(NeuralController),
}

impl AnyController {
    #[inline]
    pub fn as_neural(&self) -> Option<&NeuralController> {
        match self { Self::Neural(c) => Some(c), _ => None }
    }
}

impl Controller for AnyController {
    #[inline]
    fn poll(&mut self, game: &Game) -> Action {
        match self {
            Self::Keyboard(c) => c.poll(game),
            Self::Neural(c)   => c.poll(game),
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[macroquad::main(window_conf)]
async fn main() {
    let mode = parse_cli();

    let mut game = Game::new();
    let mut controller = match mode {
        Mode::Human => AnyController::Keyboard(KeyboardController::new()),
        Mode::Neural { weights } => {
            let c = match weights {
                Some(p) => NeuralController::load(&p)
                    .unwrap_or_else(|e| {
                        eprintln!("Falha ao carregar '{}': {}", p, e);
                        std::process::exit(1);
                    }),
                None => NeuralController::random(),
            };
            AnyController::Neural(c)
        }
    };

    let renderer = Renderer::new();

    loop {
        let dt = get_frame_time().min(0.05);
        clear_background(COLOR_BG);

        let action = controller.poll(&game);

        if game.is_finished() && action == Action::Restart {
            game = Game::new();
        } else {
            game.update(dt, action);
        }

        renderer.draw(&game);

        // Painel neural abaixo do mapa (só no modo NN).
        if let Some(nn) = controller.as_neural() {
            NeuralViz::draw(nn, WINDOW_HEIGHT as f32);
        }

        next_frame().await;
    }
}