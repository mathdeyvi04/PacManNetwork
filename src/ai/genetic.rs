//! Módulo do Algoritmo Genético (AG).
//!
//! Contém as constantes de configuração do processo evolutivo, a entidade
//! `Agent`, a criação da população inicial, a ordenação por fitness e a
//! **evolução sexuada com elitismo + torneio + mutação uniforme + imigrantes híbridos**.
//!
//! Telemetria: cada `run_episode` devolve um `EpisodeMetrics` que `evaluate_multi`
//! consolida em médias por agente (`avg_pellets_pct`, `death_rate`,
//! `timeout_rate`, `win_rate`) — usadas por `train.rs` para log e por
//! decisões futuras de seleção.

use crate::core::game::{Game, GameState};
use crate::core::map::Map;
use crate::core::controller::Controller;
use crate::ai::neural::NeuralController;
use crate::ai::neural::NeuralNet;
use crate::core::rng::Rng;

// ---------- POPULAÇÃO ----------
pub const POPULATION_SIZE: usize = 100;
pub const MAX_STEPS: usize = 12000;
pub const DT: f32 = 1.0 / 60.0;
pub const N_GENERATIONS: usize = 5000;

// ---------- ELITISMO: reduzido para 2 ----------
// Reduzir de 4 para 2 diminui a pressão de clonagem do mesmo genoma estagnado,
// abrindo mais espaço para diversidade na descendência.
pub const ELITE_SIZE: usize = 2;

// ---------- MUTAÇÃO BASE: mais forte ----------
// Aumentado para permitir saltos maiores em vales de fitness.
pub const MUTATION_RATE: f32 = 0.20;
pub const MUTATION_STRENGTH: f32 = 0.45;

// ---------- MUTAÇÃO ADAPTATIVA ----------
pub const STAGNATION_THRESHOLD: u64 = 8;
pub const STAGNATION_BOOST_RATE: f32 = 2.5;
pub const STAGNATION_BOOST_STRENGTH: f32 = 2.0;

// ---------- IMIGRANTES HÍBRIDOS (ALEATÓRIOS + CAMPEÃO MUTADO) ----------
// Intervalo mais curto (10 gerações) e maior fração (15%) para quebrar platôs rapidamente.
pub const IMMIGRATION_INTERVAL: u64 = 10;
pub const IMMIGRATION_FRACTION: f32 = 0.15;

// ---------- CROSSOVER E TORNEIO ----------
pub const CROSSOVER_RATE: f32 = 0.45;
pub const TOURNAMENT_K: usize = 6;

// ---------- FITNESS ----------
pub const FITNESS_DISTINCT_TILE_REWARD: f32 = 3.0;
pub const FITNESS_TIME_EFFICIENCY_BONUS: f32 = 0.6;
pub const FITNESS_TIMEOUT_PENALTY: f32 = -250.0;
pub const FITNESS_DEATH_PENALTY:   f32 = -250.0;
pub const FITNESS_COMPLETION_BONUS: f32 = 3500.0;
pub const FITNESS_WIN_BONUS:        f32 = 2500.0;

pub const EVAL_SEEDS: [u64; 5] = [
    0xDEAD_BEEF_CAFE_F00D,
    0x0F0F_0F0F_0F0F_0F0F,
    0x5555_5555_5555_5555,
    0xAAAA_AAAA_AAAA_AAAA,
    0x1234_5678_9ABC_DEF0,
];

use crate::constants::{GRID_HEIGHT, GRID_WIDTH};

// ---------------------------------------------------------------------------
// Tiles Visitados — bitmap compacto para contagem de tiles distintos
// ---------------------------------------------------------------------------

const VISITED_WORDS: usize = (GRID_WIDTH * GRID_HEIGHT + 63) / 64;

struct VisitedTiles {
    bits:  [u64; VISITED_WORDS],
    count: u32,
}

impl VisitedTiles {
    #[inline]
    fn new() -> Self {
        Self { bits: [0; VISITED_WORDS], count: 0 }
    }

    #[inline]
    fn visit(&mut self, x: usize, y: usize) -> bool {
        let idx  = y * GRID_WIDTH + x;
        let word = idx / 64;
        let bit  = idx % 64;
        let mask = 1u64 << bit;
        if self.bits[word] & mask != 0 {
            false
        } else {
            self.bits[word] |= mask;
            self.count += 1;
            true
        }
    }

    #[inline]
    fn count(&self) -> u32 { self.count }
}

impl Default for VisitedTiles {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Métricas de um Episódio
// ---------------------------------------------------------------------------

/// Resultado de **uma** partida (um seed). Consumido por `evaluate_multi`,
/// que agrega em médias por agente. `deaths`, `timeouts`, `wins` são
/// flags 0/1 — somar entre seeds dá a contagem de episódios em cada
/// estado terminal.
#[derive(Default, Clone, Copy)]
pub struct EpisodeMetrics {
    pub fitness: f32,
    pub pellets_eaten: u32,
    pub total_pellets: u32,
    pub deaths: u32,
    pub timeouts: u32,
    pub wins: u32,
}

// ---------------------------------------------------------------------------
// Entidade (Um Agente da População)
// ---------------------------------------------------------------------------

pub struct Agent {
    pub game: Game,
    pub controller: NeuralController,
    pub fitness: f32,
    pub avg_score: u32,
    // Novos campos para telemetria
    pub avg_pellets_pct: f32,
    pub death_rate: f32,
    pub timeout_rate: f32,
    pub win_rate: f32,
}

impl Agent {
    pub fn new_random(base_map: &Map, net_seed: u64) -> Self {
        Self {
            game: Game::with_map_and_seed(base_map.clone(), net_seed),
            controller: NeuralController::random_with_seed(net_seed),
            fitness: 0.0,
            avg_score: 0,
            avg_pellets_pct: 0.0,
            death_rate: 0.0,
            timeout_rate: 0.0,
            win_rate: 0.0,
        }
    }

    /// Avalia a rede contra K partidas e consolida em médias:
    /// `fitness`, `avg_score`, `avg_pellets_pct`, `death_rate`,
    /// `timeout_rate`, `win_rate`.
    ///
    /// **Otimização:** só a primeira seed re-aloca o `Game` (clone do
    /// template + varredura do mapa). As seguintes usam `reset_in_place`.
    pub fn evaluate_multi(&mut self, base_map: &Map, seeds: &[u64]) {
        debug_assert!(!seeds.is_empty(), "seeds não pode ser vazio");

        let mut fitness_sum = 0.0f32;
        let mut score_sum = 0u64;
        let mut pellets_pct_sum = 0.0f32;
        let mut deaths_sum = 0u32;
        let mut timeouts_sum = 0u32;
        let mut wins_sum = 0u32;

        for (idx, &seed) in seeds.iter().enumerate() {
            if idx == 0 {
                self.game = Game::with_map_and_seed(base_map.clone(), seed);
            } else {
                self.game.reset_in_place(seed);
            }

            let m = Self::run_episode(&mut self.controller, &mut self.game);
            fitness_sum += m.fitness;
            score_sum += self.game.score as u64;
            pellets_pct_sum += if m.total_pellets > 0 {
                (m.pellets_eaten as f32 / m.total_pellets as f32) * 100.0
            } else {
                100.0
            };
            deaths_sum += m.deaths;
            timeouts_sum += m.timeouts;
            wins_sum += m.wins;
        }

        let k = seeds.len() as f32;
        self.fitness = fitness_sum / k;
        self.avg_score = (score_sum as f32 / k).round() as u32;
        self.avg_pellets_pct = pellets_pct_sum / k;
        self.death_rate = (deaths_sum as f32 / k) * 100.0;
        self.timeout_rate = (timeouts_sum as f32 / k) * 100.0;
        self.win_rate = (wins_sum as f32 / k) * 100.0;
    }

    /// Roda **uma partida completa** e devolve as métricas individuais.
    ///
    /// **Por que é função associada (não método `&mut self`):** precisa
    /// emprestar `&mut self.controller` e `&mut self.game` **ao mesmo
    /// tempo** — campos disjuntos de `Agent`, permitido pelo borrow checker.
    fn run_episode(ctrl: &mut NeuralController, game: &mut Game) -> EpisodeMetrics {
        let initial_pellets = game.map.pellets_remaining();
        let mut steps = 0usize;

        let mut visited = VisitedTiles::new();
        let (sx, sy) = game.player.tile();
        visited.visit(sx, sy);

        while !game.is_finished() && steps < MAX_STEPS {
            let action = ctrl.poll(game);
            game.update(DT, action);

            let (tx, ty) = game.player.tile();
            visited.visit(tx, ty);
            steps += 1;
        }

        let eaten = initial_pellets.saturating_sub(game.map.pellets_remaining());
        let completion = if initial_pellets == 0 {
            1.0
        } else {
            eaten as f32 / initial_pellets as f32
        };

        let (terminal, deaths, timeouts, wins) = match game.state() {
            GameState::GameOver => (FITNESS_DEATH_PENALTY, 1, 0, 0),
            GameState::Win      => (FITNESS_WIN_BONUS, 0, 0, 1),
            GameState::Playing  => (FITNESS_TIMEOUT_PENALTY, 0, 1, 0),
        };

        let efficiency_bonus = if eaten > 0 && steps > 0 {
            (eaten as f32 / steps as f32) * FITNESS_TIME_EFFICIENCY_BONUS * 1000.0
        } else {
            0.0
        };

        let fitness = game.score as f32
            + completion * FITNESS_COMPLETION_BONUS
            + visited.count() as f32 * FITNESS_DISTINCT_TILE_REWARD
            + efficiency_bonus
            + terminal;

        EpisodeMetrics {
            fitness,
            pellets_eaten: eaten as u32,
            total_pellets: initial_pellets as u32,
            deaths,
            timeouts,
            wins,
        }
    }
}

pub fn init_population(size: usize, base_map: &Map, base_seed: u64) -> Vec<Agent> {
    (0..size)
        .map(|i| Agent::new_random(base_map, base_seed.wrapping_add(i as u64)))
        .collect()
}

// ---------------------------------------------------------------------------
// Seleção do Top-K (elite)
// ---------------------------------------------------------------------------

#[inline]
fn cmp_fitness_desc(a: &Agent, b: &Agent) -> std::cmp::Ordering {
    b.fitness
        .partial_cmp(&a.fitness)
        .unwrap_or(std::cmp::Ordering::Equal)
}

pub fn select_top_k_by_fitness(population: &mut [Agent], k: usize) {
    debug_assert!(k > 0, "k deve ser >= 1");
    debug_assert!(k <= population.len(), "k maior que a população");

    if k < population.len() {
        population.select_nth_unstable_by(k - 1, cmp_fitness_desc);
    }
    population[..k].sort_by(cmp_fitness_desc);
}

// ---------------------------------------------------------------------------
// Hiperparâmetros efetivos (fonte única de verdade)
// ---------------------------------------------------------------------------

/// Retorna `(mutation_rate, mutation_strength)` **efetivos** para a
/// geração, considerando o boost adaptativo por estagnação.
///
/// `evolve_population` e `train.rs` devem chamar **esta** função para
/// garantir que o log/relatório coincida com o que foi realmente
/// aplicado — a lógica não está duplicada.
#[inline]
pub fn current_mutation_params(generations_since_improvement: u64) -> (f32, f32) {
    if generations_since_improvement >= STAGNATION_THRESHOLD {
        (
            (MUTATION_RATE * STAGNATION_BOOST_RATE).min(0.7),
            (MUTATION_STRENGTH * STAGNATION_BOOST_STRENGTH).min(1.0),
        )
    } else {
        (MUTATION_RATE, MUTATION_STRENGTH)
    }
}

// ---------------------------------------------------------------------------
// Evolução — sexuada com torneio + elitismo
// ---------------------------------------------------------------------------

#[inline]
fn tournament_select(snapshot: &[(f32, NeuralNet)], k: usize, rng: &mut Rng) -> usize {
    let mut best = rng.range_usize(snapshot.len());
    for _ in 1..k {
        let idx = rng.range_usize(snapshot.len());
        if snapshot[idx].0 > snapshot[best].0 {
            best = idx;
        }
    }
    best
}

pub fn evolve_population(
    population: &mut [Agent],
    generation: u64,
    generations_since_improvement: u64,
) {
    let n = population.len();
    debug_assert!(n > ELITE_SIZE);

    let (mut_rate, mut_strength) = current_mutation_params(generations_since_improvement);

    let snapshot: Vec<(f32, NeuralNet)> = population
        .iter()
        .map(|a| (a.fitness, a.controller.net().clone()))
        .collect();

    let gen_seed = generation
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0xDEAD_BEEF_CAFE_F00D);
    let mut rng = Rng::new(gen_seed);

    // Reprodução sexuada para os descendentes comuns
    for agent in population.iter_mut().skip(ELITE_SIZE) {
        let pa = tournament_select(&snapshot, TOURNAMENT_K, &mut rng);
        let pb = tournament_select(&snapshot, TOURNAMENT_K, &mut rng);

        let mut child_net = if rng.unit() < CROSSOVER_RATE {
            snapshot[pa].1.crossover(&snapshot[pb].1, &mut rng)
        } else {
            snapshot[pa].1.clone()
        };

        let mut_seed = rng.next();
        child_net.mutate(mut_rate, mut_strength, mut_seed);
        agent.controller = NeuralController::new(child_net);
    }

    // ─── IMIGRANTES HÍBRIDOS (A cada IMMIGRATION_INTERVAL gerações) ───
    if generation > 0 && generation % IMMIGRATION_INTERVAL == 0 {
        let n_immigrants = ((n as f32) * IMMIGRATION_FRACTION) as usize;
        let start = n.saturating_sub(n_immigrants);

        let half_immigrants = n_immigrants / 2;

        // 1. Metade totalmente aleatória (quebra drástica de platô)
        for i in start..(start + half_immigrants) {
            let fresh_seed = gen_seed.wrapping_add(i as u64).wrapping_mul(31);
            population[i].controller = NeuralController::random_with_seed(fresh_seed);
            population[i].fitness = 0.0;
            population[i].avg_score = 0;
            population[i].avg_pellets_pct = 0.0;
            population[i].death_rate = 0.0;
            population[i].timeout_rate = 0.0;
            population[i].win_rate = 0.0;
        }

        // 2. Metade com mutação agressiva em cima do campeão atual
        let champion = &snapshot[0].1;
        for i in (start + half_immigrants)..n {
            let mut child = champion.clone();
            let seed = rng.next();
            child.mutate(0.40, 0.65, seed); // Perturbação forte no entorno do líder
            population[i].controller = NeuralController::new(child);
            population[i].fitness = 0.0;
            population[i].avg_score = 0;
            population[i].avg_pellets_pct = 0.0;
            population[i].death_rate = 0.0;
            population[i].timeout_rate = 0.0;
            population[i].win_rate = 0.0;
        }
    }
}