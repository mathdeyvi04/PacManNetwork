//! Treinamento da rede neural via algoritmos genéticos.
//! Executado de forma "headless" (sem janela gráfica).
//!
//! Separação de responsabilidades:
//! 1. Configuração (Constantes)
//! 2. Entidade (Agent)
//! 3. Inicialização (init_population)
//! 4. Simulação (simulate_generation)
//! 5. Evolução (Próximos passos: select, crossover, mutate)
//! 6. Loop Principal (main)

#[path = "constants.rs"] mod constants;
#[path = "controller.rs"] mod controller;
#[path = "game.rs"] mod game;
#[path = "ghost.rs"] mod ghost;
#[path = "map.rs"] mod map;
#[path = "player.rs"] mod player;
#[path = "neural.rs"] mod neural;

use constants::*;
use game::{Game, GameState};
use neural::NeuralController;
use controller::Controller;

use std::io::{self, Write};

// ---------------------------------------------------------------------------
// 1. Configuração do Treinamento
// ---------------------------------------------------------------------------

// --- Parâmetros da Simulação ---
const POPULATION_SIZE: usize = 100;
const MAX_STEPS: usize = 3000;     // ~50 segundos de jogo por geração
const DT: f32 = 1.0 / 60.0;        // Passo fixo de simulação (60 FPS)

// --- Parâmetros do Algoritmo Genético ---
const N_GENERATIONS: usize = 50;   // Quantas gerações vamos treinar
const ELITE_SIZE: usize = 5;       // Quantos melhores passam intactos para a próxima geração
const MUTATION_RATE: f32 = 0.10;   // Probabilidade de cada peso sofrer mutação
const MUTATION_STRENGTH: f32 = 0.2;// Magnitude máxima da alteração no peso (gaussiana/uniforme)

// --- Recompensas (Fitness) ---
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
    /// Cria um agente com um jogo novo e uma rede neural aleatória.
    fn new_random() -> Self {
        Self {
            game: Game::new(),
            controller: NeuralController::random(),
            steps: 0,
            fitness: 0.0,
            finished: false,
        }
    }

    /// Cria um agente herdando uma rede neural específica (usado para elites/filhos).
    fn from_controller(controller: NeuralController) -> Self {
        Self {
            game: Game::new(),
            controller,
            steps: 0,
            fitness: 0.0,
            finished: false,
        }
    }

    /// Avança um frame da simulação deste agente.
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

    /// Calcula o Fitness com base no resultado da simulação.
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
// 3. Inicialização
// ---------------------------------------------------------------------------

/// Cria a população inicial de agentes com redes neurais aleatórias.
fn init_population(size: usize) -> Vec<Agent> {
    (0..size).map(|_| Agent::new_random()).collect()
}

// ---------------------------------------------------------------------------
// 4. Simulação
// ---------------------------------------------------------------------------

/// Executa a simulação de todos os agentes até que todos terminem.
fn simulate_generation(population: &mut [Agent], generation: usize) {
    let mut active_agents = population.len();
    let mut frame = 0;

    while active_agents > 0 {
        for agent in population.iter_mut() {
            if !agent.finished {
                agent.step();
                if agent.finished {
                    active_agents -= 1;
                }
            }
        }
        
        frame += 1;
        if frame % 60 == 0 {
            print!("\r  Geração {:02} | Frame {:04}/{} | Agentes ativos: {:03}   ", 
                   generation, frame, MAX_STEPS, active_agents);
            let _ = io::stdout().flush();
        }
    }
    println!(); // Pula linha após terminar a geração
}

/// Ordena a população do maior fitness para o menor.
fn sort_by_fitness(population: &mut [Agent]) {
    population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));
}

// ---------------------------------------------------------------------------
// 5. Algoritmo Genético (Stubs para os próximos passos)
// ---------------------------------------------------------------------------

/*
fn select_parents(population: &[Agent]) -> Vec<&NeuralController> {
    // TODO: Torneio ou Roleta
}

fn crossover(parent1: &NeuralController, parent2: &NeuralController) -> NeuralController {
    // TODO: Troca de pesos (Uniform Crossover)
}

fn mutate(net: &mut NeuralController) {
    // TODO: Alterar pesos aleatoriamente baseado em MUTATION_RATE e MUTATION_STRENGTH
}
*/

// ---------------------------------------------------------------------------
// 6. Loop Principal
// ---------------------------------------------------------------------------

fn main() {
    println!("=== Iniciando Treinamento Evolutivo ===");
    println!("Gerações: {} | População: {} | Limite de Frames: {}", N_GENERATIONS, POPULATION_SIZE, MAX_STEPS);

    // Cria a população inicial (Geração 0)
    let mut population = init_population(POPULATION_SIZE);

    // Loop de Gerações
    for generation in 1..=N_GENERATIONS {
        // 1. Simular todos os agentes
        simulate_generation(&mut population, generation);

        // 2. Ordenar por Fitness
        sort_by_fitness(&mut population);

        // 3. Exibir Resultados
        let best = &population[0];
        let avg_fitness: f32 = population.iter().map(|a| a.fitness).sum::<f32>() / POPULATION_SIZE as f32;
        
        println!("=== Resumo da Geração {:02} ===", generation);
        println!("  Melhor Fitness: {:.2} (Score: {})", best.fitness, best.game.score);
        println!("  Fitness Médio:  {:.2}", avg_fitness);
        
        // Salva o melhor da geração atual (opcional, útil para inspecionar)
        if generation == N_GENERATIONS || generation % 10 == 0 {
            let path = format!("best_gen_{:03}.bin", generation);
            let _ = best.controller.net().save(&path);
            println!("  Melhor rede salva em: {}", path);
        }

        // 4. Evolução (Se não for a última geração)
        if generation < N_GENERATIONS {
            // TODO: Implementar o ciclo evolutivo
            // let new_population = evolve_population(&population);
            // population = new_population;
            
            // Por enquanto, recriamos a população aleatoriamente para não travar o loop
            // quando as funções de crossover/mutação forem implementadas.
            population = init_population(POPULATION_SIZE);
        }
        println!();
    }

    println!("Treinamento concluído!");
}