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
use std::time::Instant; // NOVO: Para medir o tempo

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

// Simulação Multithread (Padrão)
fn simulate_generation_parallel(population: &mut [Agent]) {
    population.par_iter_mut().for_each(|agent| {
        while !agent.finished {
            agent.step();
        }
    });
}

// Simulação Singlethread (Para comparação)
fn simulate_generation_single(population: &mut [Agent]) {
    for agent in population.iter_mut() {
        while !agent.finished {
            agent.step();
        }
    }
}

fn sort_by_fitness(population: &mut [Agent]) {
    population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));
}

// ---------------------------------------------------------------------------
// 4. Loop Principal
// ---------------------------------------------------------------------------

fn main() {
    // Lê os argumentos da linha de comando
    let args: Vec<String> = std::env::args().collect();
    let use_single_thread = args.iter().any(|arg| arg == "--single" || arg == "-s");
    
    let mode_str = if use_single_thread { "Single-Thread (Sequencial)" } else { "Multi-Thread (Rayon)" };

    println!("=== Iniciando Treinamento Evolutivo ===");
    println!("Modo de Execução: {}", mode_str);
    println!("Gerações: {:3} | População: {:3} | Limite: {:4} frames\n", 
             N_GENERATIONS, POPULATION_SIZE, MAX_STEPS);

    GenerationStats::print_header();

    // Marca o início do treinamento total
    let total_start = Instant::now();

    let mut base_map = Map::generate();
    let mut population = init_population(POPULATION_SIZE, &base_map);

    for generation in 1..=N_GENERATIONS {
        // Marca o início da geração
        let gen_start = Instant::now();

        // 1. Simular (Escolhe o método baseado no argumento)
        if use_single_thread {
            simulate_generation_single(&mut population);
        } else {
            simulate_generation_parallel(&mut population);
        }

        // Calcula o tempo apenas desta geração
        let gen_elapsed_secs = gen_start.elapsed().as_secs_f32();

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
            time_secs: gen_elapsed_secs, // Passa o tempo calculado
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

    // Tempo total de treinamento
    let total_elapsed_secs = total_start.elapsed().as_secs_f32();

    println!("----------------------------------------------------------------------");
    println!("Treinamento concluído em {:.2} segundos!", total_elapsed_secs);
}