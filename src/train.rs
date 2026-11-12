mod constants;
mod core;
mod ai;
mod utils;

use constants::*;
use core::game::{Game, GameState};
use core::map::Map;
use core::controller::Controller;
use ai::neural::NeuralController;
use utils::logger::GenerationStats;

use rayon::prelude::*;

// ... (resto do código train.rs continua igual, sem os #[path])

// ---------------------------------------------------------------------------
// 1. Configuração do Treinamento
// ---------------------------------------------------------------------------

const POPULATION_SIZE: usize = 100;
const MAX_STEPS: usize = 3000;     
const DT: f32 = 1.0 / 60.0;        

const N_GENERATIONS: usize = 50;   
const ELITE_SIZE: usize = 5;       
const MUTATION_RATE: f32 = 0.10;   
const MUTATION_STRENGTH: f32 = 0.2;

const FITNESS_DEATH_PENALTY: f32 = -500.0;
const FITNESS_WIN_BONUS: f32 = 2000.0;
const FITNESS_SURVIVAL_MULT: f32 = 0.05;

// ---------------------------------------------------------------------------
// 2. Entidade (Um Agente da População)
// ---------------------------------------------------------------------------

pub struct Agent {
    pub game: Game,
    pub controller: NeuralController,
    pub steps: usize,
    pub fitness: f32,
    pub finished: bool,
}

impl Agent {
    fn new_random(base_map: &Map) -> Self {
        Self {
            game: Game::with_map(base_map.clone()),
            controller: NeuralController::random(),
            steps: 0,
            fitness: 0.0,
            finished: false,
        }
    }

    fn step(&mut self) {
        if self.finished { return; }

        let action = self.controller.poll(&self.game);
        self.game.update(DT, action);
        self.steps += 1;

        if self.game.is_finished() || self.steps >= MAX_STEPS {
            self.finished = true;
            self.evaluate();
        }
    }

    fn evaluate(&mut self) {
        self.fitness = self.game.score as f32 + (self.steps as f32 * FITNESS_SURVIVAL_MULT);

        if self.game.state() == GameState::GameOver {
            self.fitness += FITNESS_DEATH_PENALTY;
        } else if self.game.state() == GameState::Win {
            self.fitness += FITNESS_WIN_BONUS;
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Inicialização e Simulação
// ---------------------------------------------------------------------------

fn init_population(size: usize, base_map: &Map) -> Vec<Agent> {
    (0..size).map(|_| Agent::new_random(base_map)).collect()
}

fn simulate_generation(population: &mut [Agent]) {
    population.par_iter_mut().for_each(|agent| {
        while !agent.finished {
            agent.step();
        }
    });
}

fn sort_by_fitness(population: &mut [Agent]) {
    population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));
}

// ---------------------------------------------------------------------------
// 4. Loop Principal
// ---------------------------------------------------------------------------

fn main() {
    println!("=== Iniciando Treinamento Evolutivo ===");
    println!("Gerações: {:3} | População: {:3} | Limite: {:4} frames\n", 
             N_GENERATIONS, POPULATION_SIZE, MAX_STEPS);

    // Chama o cabeçalho do módulo de logs
    GenerationStats::print_header();

    let mut base_map = Map::generate();
    let mut population = init_population(POPULATION_SIZE, &base_map);

    for generation in 1..=N_GENERATIONS {
        // 1. Simular
        simulate_generation(&mut population);

        // 2. Ordenar e Coletar Métricas
        sort_by_fitness(&mut population);

        let best = &population[0];
        let avg_fitness: f32 = population.iter().map(|a| a.fitness).sum::<f32>() / POPULATION_SIZE as f32;
        
        // 3. Instanciar e Exibir Log via Módulo Dedicado
        let stats = GenerationStats {
            generation,
            best_fitness: best.fitness,
            avg_fitness,
            best_score: best.game.score,
        };
        stats.print_row();
        
        // Salva a melhor rede esporadicamente
        if generation == N_GENERATIONS || generation % 10 == 0 {
            let path = format!("best_gen_{:03}.bin", generation);
            let _ = best.controller.net().save(&path);
        }

        // 4. Evolução (Placeholder)
        if generation < N_GENERATIONS {
            base_map = Map::generate(); 
            population = init_population(POPULATION_SIZE, &base_map);
        }
    }

    println!("\nTreinamento concluído!");
}