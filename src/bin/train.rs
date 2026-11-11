
#[path = "../constants.rs"]
mod constants;

#[path = "../controller.rs"]
mod controller;

#[path = "../game.rs"]
mod game;

#[path = "../ghost.rs"]
mod ghost;

#[path = "../map.rs"]
mod map;

#[path = "../neural.rs"]
mod neural;

#[path = "../player.rs"]
mod player;

#[path = "../rng.rs"]
mod rng;

use rayon::prelude::*;
use constants::*;

use std::cmp::Ordering;
use std::time::Instant;

use game::{Game, GameState};
use rng::Rng;
use map::Map;

use player::Direction;

use constants::{GRID_WIDTH, GHOST_COUNT};

use neural::{
    encode,
    action_from_output,
    NeuralNet,
    INPUT_SIZE,
    HIDDEN_SIZE,
    OUTPUT_SIZE,
    infer_action,
};

use rng::fresh_seed;

// ============================================================================
// CONFIGURAÇÕES DO ALGORITMO GENÉTICO
// ============================================================================

/// Quantidade de indivíduos por geração.
const POPULATION_SIZE: usize = 100;

/// Número padrão de gerações.
const DEFAULT_GENERATIONS: usize = 200;

/// Frequência de atualização da simulação.
const FIXED_DT: f32 = 1.0 / 60.0;

/// Limite de sobrevivência de cada episódio.
///
/// Evita que uma rede que nunca morra mantenha a geração
/// executando indefinidamente.
const MAX_EPISODE_SECONDS: f32 = 120.0;

const MAX_EPISODE_STEPS: u64 =
(MAX_EPISODE_SECONDS / FIXED_DT) as u64;

// ============================================================================
// FITNESS MULTIOBJETIVO
// ============================================================================

const SCORE_WEIGHT: f64 = 0.60;
const SURVIVAL_WEIGHT: f64 = 0.25;
const COVERAGE_WEIGHT: f64 = 0.15;

// Divisão espacial do mapa.
const REGION_COLS: usize = 4;
const REGION_ROWS: usize = 4;

const REGION_COUNT: usize =
    REGION_COLS * REGION_ROWS;

// ============================================================================
// SELEÇÃO GENÉTICA
// ============================================================================

const ELITE_COUNT: usize = 2;

const PARENT_POOL_SIZE: usize = 20;

const TOURNAMENT_SIZE: usize = 3;

const RANDOM_IMMIGRANTS: usize = 5;

// ============================================================================
// MUTAÇÃO ADAPTATIVA
// ============================================================================

const MUTATION_RATE: f32 = 0.02;

const MUTATION_STD: f32 = 0.08;

const MUTATION_STD_MIN: f32 = 0.02;

const MUTATION_STD_MAX: f32 = 0.20;

// ============================================================================
// VALIDAÇÃO
// ============================================================================

const VALIDATION_EVERY: usize = 5;

const VALIDATION_TOP: usize = 10;

const VALIDATION_MAP_SEEDS: [u64; 3] = [
    0x1A2B_3C4D_5E6F_7788,
    0x243F_6A88_85A3_08D3,
    0xB7E1_5162_8AED_2A6B,
];

// ---------------------------------------------------------------------------
// CHECKPOINT
// ---------------------------------------------------------------------------

const BEST_WEIGHTS_PATH: &str = "best_pacman.bin";

// Outra parada

#[derive(Clone, Copy, PartialEq, Eq)]
struct ObservationKey {
    player_tile: u16,
    player_direction: u8,

    ghosts: [u16; GHOST_COUNT],

    pellets_remaining: u16,
}

impl ObservationKey {
    fn from_game(game: &Game) -> Self {
        let (px, py) = game.player.tile();

        let player_tile =
            (py * GRID_WIDTH + px) as u16;

        let player_direction = match game.player.dir {
            Direction::Up    => 0,
            Direction::Down  => 1,
            Direction::Left  => 2,
            Direction::Right => 3,
        };

        let ghosts = std::array::from_fn(|i| {
            let (x, y) = game.ghosts()[i].tile();

            (y * GRID_WIDTH + x) as u16
        });

        Self {
            player_tile,
            player_direction,
            ghosts,
            pellets_remaining:
                game.map.pellets_remaining() as u16,
        }
    }
}

// ============================================================================
// INDIVÍDUO
// ============================================================================

/// Um indivíduo possui:
///
/// - Uma rede neural exclusiva.
/// - Um jogo independente.
/// - Buffers próprios para inferência.
/// - Seu próprio cooldown de inversão.
/// - Um contador de tempo vivido.
///
/// Não há alocações dentro do loop de inferência.

struct Individual {
    net: NeuralNet,
    game: Game,

    input: [f32; INPUT_SIZE],
    hidden: [f32; HIDDEN_SIZE],
    output: [f32; OUTPUT_SIZE],

    reverse_cooldown: u8,

    steps: u64,
    visited_regions: u16,
    regions_visited: u8,
    
    last_observation: Option<ObservationKey>,
}

impl Individual {
    fn new(
        net: NeuralNet,
        map: Map,
        ghost_seed: u64,
    ) -> Self {
    
        let game = Game::new_with_map(map, ghost_seed);
    
        let mut individual = Self {
            net,
            game,
    
            input: [0.0; INPUT_SIZE],
            hidden: [0.0; HIDDEN_SIZE],
            output: [0.0; OUTPUT_SIZE],
    
            reverse_cooldown: 0,
    
            steps: 0,
    
            visited_regions: 0,
            regions_visited: 0,
    
            last_observation: None,
        };
    
        // Registra a região inicial.
        individual.mark_region();
    
        individual
    }

    /// Marca a região atual do Pac-Man.
    ///
    /// Cada bit representa uma região do mapa.
    fn mark_region(&mut self) {
        let (x, y) = self.game.player.tile();
    
        if x >= GRID_WIDTH || y >= GRID_HEIGHT {
            return;
        }
    
        let sx = (x * REGION_COLS / GRID_WIDTH)
            .min(REGION_COLS - 1);
    
        let sy = (y * REGION_ROWS / GRID_HEIGHT)
            .min(REGION_ROWS - 1);
    
        let region = sy * REGION_COLS + sx;
    
        let bit = 1u16 << region;
    
        if self.visited_regions & bit == 0 {
            self.visited_regions |= bit;
    
            self.regions_visited += 1;
        }
    }
    
    #[inline(always)]
    fn coverage_fraction(&self) -> f64 {
        self.regions_visited as f64
            / REGION_COUNT as f64
    }

    /// Executa o episódio completo deste indivíduo.
    ///
    /// Não existe renderização e não existe espera por frames.
    /// A simulação avança diretamente usando o timestep fixo.
    fn simulate(&mut self) {
        while !self.game.is_finished()
            && self.steps < MAX_EPISODE_STEPS
        {
            let key = ObservationKey::from_game(&self.game);
    
            // Só executa a rede quando a observação realmente mudou.
            if self.last_observation != Some(key) {
    
                self.input = encode(&self.game);
    
                self.net.forward_into(
                    &self.input,
                    &mut self.hidden,
                    &mut self.output,
                );
    
                self.last_observation = Some(key);
            }
    
            // A seleção da ação continua sendo processada a cada frame.
            // Isso preserva corretamente o cooldown anti-jitter.
            let action = action_from_output(
                &self.game,
                &mut self.reverse_cooldown,
                &self.output,
            );
    
            // A física continua atualizada em todos os frames.
            self.game.update(FIXED_DT, action);
            self.mark_region();
    
            self.steps += 1;
        }
    }

    #[inline(always)]
    fn alive_seconds(&self) -> f64 {
        self.steps as f64 * FIXED_DT as f64
    }

    #[inline(always)]
    fn score(&self) -> u32 {
        self.game.score
    }

    /// Função de avaliação genética.
    ///
    /// Edite SCORE_WEIGHT e SURVIVAL_WEIGHT no início do arquivo.
    #[inline(always)]
    fn fitness(&self) -> f64 {
    
        // Fração dos pellets coletados.
        let maximum_score =
            self.game.map.total_pellets() as f64
            * constants::PELLET_SCORE as f64;
    
        let score_ratio = (
            self.score() as f64 / maximum_score.max(1.0)
        ).clamp(0.0, 1.0);
    
        // Fração do tempo máximo de sobrevivência.
        let survival_ratio = (
            self.alive_seconds() / MAX_EPISODE_SECONDS as f64
        ).clamp(0.0, 1.0);
    
        // Fração das regiões visitadas.
        let coverage_ratio = self.coverage_fraction();
    
        // Fitness final entre 0 e 100.
        100.0 * (
            SCORE_WEIGHT * score_ratio
            + SURVIVAL_WEIGHT * survival_ratio
            + COVERAGE_WEIGHT * coverage_ratio
        )
    }
}

// ============================================================================
// PARALELIZAÇÃO
// ============================================================================

fn run_population(
    population: &mut [Individual],
    pool: &rayon::ThreadPool,
) {
    pool.install(|| {
        population
            .par_iter_mut()
            .for_each(|individual| {
                individual.simulate();
            });
    });
}

// ============================================================================
// GERAÇÃO DE SEEDS
// ============================================================================

/// Misturador determinístico de seeds (SplitMix64).
///
/// Permite gerar seeds independentes a partir de uma seed principal.

fn derive_seed(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);

    x = (x ^ (x >> 30))
        .wrapping_mul(0xBF58_476D_1CE4_E5B9);

    x = (x ^ (x >> 27))
        .wrapping_mul(0x94D0_49BB_1331_11EB);

    x ^ (x >> 31)
}

// ============================================================================
// ARGUMENTOS
// ============================================================================

fn parse_arg<T: std::str::FromStr>(
    args: &[String],
    index: usize,
    default: T,
    name: &str,
) -> Result<T, String> {
    match args.get(index) {
        Some(value) => value
            .parse::<T>()
            .map_err(|_| format!("Argumento inválido: {}", name)),

        None => Ok(default),
    }
}

fn print_help() {
    println!("Pac-Man Neural Genetic Training");
    println!();
    println!("Uso:");
    println!("  cargo run --release --bin train");
    println!();
    println!("Argumentos:");
    println!("  [geracoes] [threads] [seed]");
    println!();
    println!("Exemplo:");
    println!("  cargo run --release --bin train -- 500 8 42");
}

// ============================================================================
// TREINAMENTO
// ============================================================================

fn run() -> Result<(), String> {
    let args: Vec<String> =
        std::env::args().skip(1).collect();

    if args.first().map_or(false, |s| {
        s == "--help" || s == "-h"
    }) {
        print_help();
        return Ok(());
    }

    let available_threads =
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

    let generations = parse_arg(
        &args,
        0,
        DEFAULT_GENERATIONS,
        "geracoes",
    )?;

    let workers = parse_arg(
        &args,
        1,
        available_threads,
        "threads",
    )?;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|e| {
            format!("Falha ao criar pool de threads: {}", e)
        })?;

    let base_seed = parse_arg(
        &args,
        2,
        fresh_seed(),
        "seed",
    )?;

    if generations == 0 {
        return Err("O número de gerações precisa ser maior que zero.".into());
    }

    if workers == 0 {
        return Err("O número de threads precisa ser maior que zero.".into());
    }

    // ------------------------------------------------------------------------
    // POPULAÇÃO INICIAL
    // ------------------------------------------------------------------------

    let mut networks: Vec<NeuralNet> =
        Vec::with_capacity(POPULATION_SIZE);

    for i in 0..POPULATION_SIZE {
        let seed = derive_seed(base_seed ^ i as u64);

        networks.push(NeuralNet::random(seed));
    }

    let training_start = Instant::now();

    println!("==================================================");
    println!(" PAC-MAN - TREINAMENTO GENÉTICO");
    println!("==================================================");
    println!("População:       {}", POPULATION_SIZE);
    println!("Gerações:        {}", generations);
    println!("Threads:         {}", workers);
    println!("Seed:            {}", base_seed);
    println!("Limite episódio: {:.1}s", MAX_EPISODE_SECONDS);
    println!("Mutação:         {:.2}%", MUTATION_RATE * 100.0);
    println!("Desvio mutação:  {:.3}", MUTATION_STD);
    println!("Fitness score:   {:.2}", SCORE_WEIGHT);
    println!("Fitness tempo:   {:.2}", SURVIVAL_WEIGHT);
    println!("==================================================");

    let validation_maps: Vec<(Map, u64)> =
        VALIDATION_MAP_SEEDS
            .iter()
            .map(|&seed| {
    
                let map = Map::generate_seeded(seed);
    
                let ghost_seed = derive_seed(
                    seed ^ 0xA076_1D64_78BD_642F
                );
    
                (map, ghost_seed)
            })
            .collect();

    let mut best_validation_fitness =
        f64::NEG_INFINITY;
    
    let mut best_validation_net: Option<NeuralNet> = None;
    
    let mut best_validation =
        EvalSummary::default();
    
    let mut validation_stagnation = 0usize;
    
    let mut mutation_std = MUTATION_STD;

    // ========================================================================
    // LOOP EVOLUTIVO
    // ========================================================================
    
    for generation in 0..generations {
        let generation_start = Instant::now();

        // Cada geração utiliza um novo mapa.
        //
        // Entretanto, todos os 100 indivíduos desta geração
        // recebem exatamente o mesmo mapa inicial.

        let generation_seed = derive_seed(
            base_seed
                ^ (generation as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15),
        );

        let map_seed = derive_seed(
            generation_seed ^ 0xA076_1D64_78BD_642F,
        );

        let ghost_seed = derive_seed(
            generation_seed ^ 0xE703_7ED1_A0B4_28DB,
        );

        let map_template = Map::generate_seeded(map_seed);

        // --------------------------------------------------------------------
        // CRIAÇÃO DOS 100 JOGOS
        // --------------------------------------------------------------------

        let mut population: Vec<Individual> =
            Vec::with_capacity(POPULATION_SIZE);

        for net in networks.drain(..) {
            population.push(
                Individual::new(
                    net,
                    map_template.clone(),
                    ghost_seed,
                )
            );
        }

        // --------------------------------------------------------------------
        // EXECUÇÃO PARALELA
        // --------------------------------------------------------------------

        run_population(&mut population, &pool);

        // Neste momento, todos os indivíduos terminaram:
        //
        // GameOver = morreu.
        // Win      = coletou todos os pellets.
        // Playing  = atingiu o limite de tempo.

        // --------------------------------------------------------------------
        // AVALIAÇÃO
        // --------------------------------------------------------------------

        let total_fitness: f64 = population.iter().map(|i| i.fitness()).sum();
        let total_time:   f64 = population.iter().map(|i| i.alive_seconds()).sum();
        let total_score:  u64 = population.iter().map(|i| i.score() as u64).sum();
    
        let count = POPULATION_SIZE as f64;
        let average_fitness = total_fitness / count;
        let average_time    = total_time / count;
        let average_score   = total_score as f64 / count;
    
        // NOVO: desvio padrão e mínimo — mostram diversidade e pior caso.
        let variance: f64 = population
            .iter()
            .map(|i| { let d = i.fitness() - average_fitness; d * d })
            .sum::<f64>()
            / count;
        let fitness_std = variance.sqrt();
        let min_fitness = population
            .iter()
            .map(|i| i.fitness())
            .fold(f64::INFINITY, f64::min);
    
        // Melhor da geração atual.
        let mut ranked: Vec<usize> =
            (0..population.len()).collect();
        
        ranked.sort_unstable_by(|&a, &b| {
            population[b]
                .fitness()
                .partial_cmp(&population[a].fitness())
                .unwrap_or(Ordering::Equal)
        });
        
        let best_index = ranked[0];
        
        let best = &population[best_index];
        let best_fitness = best.fitness();
        let best_time    = best.alive_seconds();
        let best_score   = best.score();
    
        // Contagens de estado (inalterado).
        let deaths   = population.iter().filter(|i| i.game.state() == GameState::GameOver).count();
        let wins     = population.iter().filter(|i| i.game.state() == GameState::Win).count();
        let timeouts = population.iter().filter(|i| i.game.state() == GameState::Playing).count();
    
        let generation_time    = generation_start.elapsed().as_secs_f64();
        let total_time_elapsed = training_start.elapsed().as_secs_f64();

        let should_validate =
            generation == 0
            || (generation + 1) % VALIDATION_EVERY == 0
            || generation + 1 == generations;
        
        let mut validation_new_record = false;
        
        if should_validate {
        
            let candidates: Vec<usize> = ranked
                .iter()
                .take(VALIDATION_TOP)
                .copied()
                .collect();
        
            // Os candidatos são validados paralelamente.
            // Cada um joga nos mesmos três mapas de referência.
        
            let validation_results: Vec<(usize, EvalSummary)> =
                pool.install(|| {
        
                    candidates.par_iter()
                        .map(|&index| {
        
                            let stats = validate_network(
                                &population[index].net,
                                &validation_maps,
                            );
        
                            (index, stats)
                        })
                        .collect()
                });
        
            for (index, stats) in validation_results {
        
                if stats.fitness > best_validation_fitness {
        
                    best_validation_fitness = stats.fitness;
        
                    best_validation = stats;
        
                    best_validation_net = Some(
                        population[index].net.clone()
                    );
        
                    validation_new_record = true;
                }
            }
        
            // Adaptação da intensidade das mutações.
            if validation_new_record {
        
                validation_stagnation = 0;
        
                mutation_std = (
                    mutation_std * 0.90
                ).max(MUTATION_STD_MIN);
        
            } else {
        
                validation_stagnation += 1;
        
                if validation_stagnation >= 2 {
        
                    mutation_std = (
                        mutation_std * 1.25
                    ).min(MUTATION_STD_MAX);
        
                    validation_stagnation = 0;
                }
            }
        
            // Salva exclusivamente quando existe um novo recorde
            // validado nos três mapas fixos.
        
            if validation_new_record {
        
                best_validation_net
                    .as_ref()
                    .unwrap()
                    .save(BEST_WEIGHTS_PATH)
                    .map_err(|e| {
                        format!("Erro ao salvar checkpoint: {}", e)
                    })?;
            }
        }
    
        // --------------------------------------------------------------------
        // LOG — NOVO formato com histórico + diversidade
        // --------------------------------------------------------------------
        println!(
            "[train gen {:04}/{:04}] \
                fit_avg={:10.2} fit_std={:8.2} fit_min={:10.2} | \
                time_avg={:7.2}s score_avg={:8.1} | \
                best_fit={:10.2} best_time={:7.2}s best_score={:7} | \
                dead={:3} wins={:3} timeout={:3} | \
                gen_time={:7.2}s total={:9.2}s",
            generation + 1,
            generations,
            average_fitness,
            fitness_std,
            min_fitness,
            average_time,
            average_score,
            best_fitness,
            best_time,
            best_score,
            deaths,
            wins,
            timeouts,
            generation_time,
            total_time_elapsed,
        );
    
        let mut selection_rng = Rng::new(
            derive_seed(
                generation_seed ^ 0xA076_1D64_78BD_642F
            )
        );
        
        let mut next_networks: Vec<NeuralNet> =
            Vec::with_capacity(POPULATION_SIZE);
        
        // ---------------------------------------------------------------------------
        // 1. ELITISMO
        // ---------------------------------------------------------------------------
        
        // Os dois melhores indivíduos passam diretamente,
        // sem sofrer alterações.
        
        let mut next_networks: Vec<NeuralNet> =
            Vec::with_capacity(POPULATION_SIZE);
        
        // Primeiro elite: campeão validado historicamente.
        if let Some(champion) = best_validation_net.as_ref() {
            next_networks.push(champion.clone());
        }
        
        // Segundo elite: melhor indivíduo da geração atual,
        // desde que não seja uma duplicata do campeão.
        for &index in &ranked {
            if next_networks.len() >= ELITE_COUNT {
                break;
            }
        }
        
        // ---------------------------------------------------------------------------
        // 2. DESCENDENTES
        // ---------------------------------------------------------------------------
        
        // 93 descendentes provenientes de diferentes pais.
        
        while next_networks.len()
            < POPULATION_SIZE - RANDOM_IMMIGRANTS
        {
            let parent = select_parent(
                &population,
                &ranked,
                &mut selection_rng,
            );
        
            let index = next_networks.len() as u64;
        
            let seed = derive_seed(
                generation_seed
                    ^ index.wrapping_mul(0xD1B5_4A32_D192_ED03)
            );
        
            let offspring = parent.mutated(
                seed,
                MUTATION_RATE,
                mutation_std,
            );
        
            next_networks.push(offspring);
        }
        
        // ---------------------------------------------------------------------------
        // 3. IMIGRANTES
        // ---------------------------------------------------------------------------
        
        // Cinco redes completamente novas.
        
        while next_networks.len() < POPULATION_SIZE {
            let index = next_networks.len() as u64;
        
            let seed = derive_seed(
                generation_seed
                    ^ 0xE703_7ED1_A0B4_28DB
                    ^ index
            );
        
            next_networks.push(
                NeuralNet::random(seed)
            );
        }
        
        // ---------------------------------------------------------------------------
        // NOVA POPULAÇÃO
        // ---------------------------------------------------------------------------
        
        networks = next_networks;
    }

    println!("==================================================");
    println!(" TREINAMENTO FINALIZADO");
    println!("==================================================");
    println!("Pesos finais: {}", BEST_WEIGHTS_PATH);
    println!(
        "Tempo total: {:.2}s",
        training_start.elapsed().as_secs_f64()
    );
    println!("==================================================");

    Ok(())
}

/// Seleciona um pai por torneio.
///
/// Os participantes são escolhidos entre os 20 melhores
/// indivíduos da geração.
///
/// O processo é assexuado: apenas um indivíduo fornece
/// os pesos para cada descendente.
fn select_parent<'a>(
    population: &'a [Individual],
    ranked: &[usize],
    rng: &mut Rng,
) -> &'a NeuralNet {

    let pool_size =
        ranked.len().min(PARENT_POOL_SIZE);

    let mut best_index =
        ranked[rng.range_usize(pool_size)];

    for _ in 1..TOURNAMENT_SIZE {
        let candidate =
            ranked[rng.range_usize(pool_size)];

        if population[candidate].fitness()
            > population[best_index].fitness()
        {
            best_index = candidate;
        }
    }

    &population[best_index].net
}

#[derive(Clone, Copy, Default)]
struct EvalSummary {
    fitness: f64,
    score: f64,
    time: f64,
    coverage: f64,
}

fn validate_network(
    net: &NeuralNet,
    maps: &[(Map, u64)],
) -> EvalSummary {

    let mut result = EvalSummary::default();

    for (map, ghost_seed) in maps {

        let mut individual = Individual::new(
            net.clone(),
            map.clone(),
            *ghost_seed,
        );

        individual.simulate();

        result.fitness += individual.fitness();

        result.score += individual.score() as f64;

        result.time += individual.alive_seconds();

        result.coverage += individual.coverage_fraction();
    }

    let count = maps.len().max(1) as f64;

    result.fitness /= count;
    result.score /= count;
    result.time /= count;
    result.coverage /= count;

    result
}









fn main() {
    if let Err(error) = run() {
        eprintln!("Erro no treinamento: {}", error);
        std::process::exit(1);
    }
}
