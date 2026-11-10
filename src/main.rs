mod constants;
mod controller;
mod game;
mod ghost;
mod map;
mod neural;
mod player;
mod renderer;

use constants::*;
use controller::{Action, Controller, KeyboardController};
use game::Game;
use neural::NeuralController;
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
            println!("  (sem argumento)          → jogar com o teclado");
            println!("  human | play             → idem");
            println!("  nn                       → rede neural (pesos aleatórios)");
            println!("  nn <arquivo.bin>         → rede neural (pesos carregados)");
            println!("  gen-weights <arquivo.bin>→ gera pesos aleatórios e salva");
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
        next_frame().await;
    }
}