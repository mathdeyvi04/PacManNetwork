mod constants;
mod core;
mod ai;
mod ui;

use macroquad::prelude::*;

use constants::*;
use core::controller::{Action, Controller, KeyboardController};
use core::game::Game;
use core::map::Map;
use ai::neural::NeuralNet;
use ai::neural::NeuralController;
use ui::renderer::Renderer;
use ui::neural_viz::NeuralViz;


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
// Carregamento de template de mapa (compartilhado por humano e neural)
// ---------------------------------------------------------------------------

/// Carrega o template de mapa a partir de um caminho opcional.
///
/// * `Some(path)` → carrega o mapa; erro de I/O é **fatal** (`exit(1)`) —
///   se o usuário pediu um mapa específico, não faz sentido silenciosamente
///   trocá-lo por um aleatório e esconder o problema.
/// * `None` → `None` (mapa será gerado aleatoriamente a cada partida).
///
/// O `Map` retornado é mantido **intocado** durante toda a sessão — ele é o
/// "template" que produz uma partida virgem a cada restart.
fn load_template(map: Option<&str>) -> Option<Map> {
    match map {
        Some(p) => match Map::load(p) {
            Ok(m) => {
                println!("Mapa carregado de '{}' ({} pellets)", p, m.pellets_remaining());
                Some(m)
            }
            Err(e) => {
                eprintln!("Falha ao carregar mapa '{}': {}", p, e);
                std::process::exit(1);
            }
        },
        None => None,
    }
}

/// Constrói uma partida nova a partir de um template.
///
/// * `Some(template)` → mesma topologia, pellets restaurados via `fresh()`.
/// * `None` → mapa aleatório novo (`Game::new()`).
#[inline]
fn new_game(template: &Option<Map>) -> Game {
    match template {
        Some(m) => Game::with_map(m.fresh()),
        None => Game::new(),
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

enum Mode {
    Human {
        map: Option<String>,
    },
    Neural {
        weights: Option<String>,
        map: Option<String>,
    },
}

fn parse_cli() -> Mode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() { return Mode::Human { map: None }; }

    match args[0].as_str() {
        "human" | "play" => Mode::Human {
            map: args.get(1).cloned(),
        },
        "nn" | "neural" => Mode::Neural {
            weights: args.get(1).cloned(),
            map:     args.get(2).cloned(),
        },
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
            let net = NeuralNet::random(seed);
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
            println!("  (sem argumento)           → jogar com o teclado (mapa aleatório)");
            println!("  human | play              → idem");
            println!("  human <mapa.bin>          → jogar em um mapa específico");
            println!("  nn                        → rede neural (pesos e mapa aleatórios)");
            println!("  nn <pesos.bin>            → rede neural (pesos carregados, mapa aleatório)");
            println!("  nn <pesos.bin> <mapa.bin> → rede neural + mapa específico (par de treino)");
            println!("  gen-weights <arquivo.bin> → gera pesos aleatórios e salva");
            println!();
            println!("Em partidas com mapa específico, o restart preserva o mapa (pellets repostos).");
            println!("Sem mapa, cada restart gera um novo labirinto aleatório.");
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

    // 1. Constrói o controller conforme o modo, e extrai o caminho do mapa
    //    (comum aos dois modos). O controller é independente do mapa.
    let (mut controller, map_path) = match mode {
        Mode::Human { map } => (
            AnyController::Keyboard(KeyboardController::new()),
            map,
        ),
        Mode::Neural { weights, map } => {
            let ctrl = match weights {
                Some(p) => NeuralController::load(&p).unwrap_or_else(|e| {
                    eprintln!("Falha ao carregar '{}': {}", p, e);
                    std::process::exit(1);
                }),
                None => NeuralController::random(),
            };
            (AnyController::Neural(ctrl), map)
        }
    };

    // 2. Template do mapa (Some = fixo, None = aleatório a cada restart).
    //    O template é preservado intocado durante toda a sessão.
    let template = load_template(map_path.as_deref());

    // 3. Primeira partida.
    let mut game = new_game(&template);

    let renderer = Renderer::new();

    loop {
        let dt = get_frame_time().min(0.05);
        clear_background(COLOR_BG);

        let action = controller.poll(&game);

        if game.is_finished() && action == Action::Restart {
            // `new_game` decide entre:
            //   • template presente → mesmo mapa, pellets repostos (`fresh`);
            //   • template ausente  → novo labirinto aleatório (`Game::new`).
            game = new_game(&template);
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