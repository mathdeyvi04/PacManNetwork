//! Módulo do Algoritmo Genético (AG).
//!
//! Contém as **constantes de configuração** do processo evolutivo, a
//! entidade `Agent` (um indivíduo da população), a criação da população
//! inicial, a ordenação por fitness e — por ora — um placeholder de
//! evolução (`evolve_population`).
//!
//! A **simulação** em si (loop de `step()` por agente, em single ou
//! multi-thread) permanece em `train.rs`, pois é uma preocupação de
//! execução, não de genética.

use crate::core::game::{Game, GameState};
use crate::core::map::Map;
use crate::core::controller::Controller;
use crate::ai::neural::NeuralController;

// ---------------------------------------------------------------------------
// Tamanho da População e Simulação
// ---------------------------------------------------------------------------

/// Número de agentes por geração.
pub const POPULATION_SIZE: usize = 100;

/// Limite máximo de frames (`steps`) que um agente pode simular antes
/// de ser considerado "estagnado" e ter seu fitness avaliado.
pub const MAX_STEPS: usize = 3000;

/// Passo de tempo fixo por frame (`1/60 s`). Fixo para determinismo e
/// para que o fitness de diferentes agentes seja comparável.
pub const DT: f32 = 1.0 / 60.0;

// ---------------------------------------------------------------------------
// Parâmetros do Algoritmo Genético
// ---------------------------------------------------------------------------

/// Número total de gerações a evoluir.
pub const N_GENERATIONS: usize = 50;

/// Quantidade de melhores indivíduos preservados intactos a cada geração.
pub const ELITE_SIZE: usize = 5;

/// Probabilidade de mutação por gene (peso/bias) durante a reprodução.
pub const MUTATION_RATE: f32 = 0.10;

/// Magnitude máxima da perturbação aplicada a um gene mutado.
pub const MUTATION_STRENGTH: f32 = 0.2;

// ---------------------------------------------------------------------------
// Função de Fitness — Recompensas e Penalidades
// ---------------------------------------------------------------------------

/// Penalidade aplicada quando o agente termina a simulação em `GameOver`.
pub const FITNESS_DEATH_PENALTY: f32 = -500.0;

/// Bônus aplicado quando o agente termina a simulação em `Win`.
pub const FITNESS_WIN_BONUS: f32 = 2000.0;

/// Multiplicador aplicado ao número de passos sobrevividos.
/// Pequeno, para incentivar a sobrevivência **sem** eclipsar a coleta de pellets.
pub const FITNESS_SURVIVAL_MULT: f32 = 0.05;

// ---------------------------------------------------------------------------
// Entidade (Um Agente da População)
// ---------------------------------------------------------------------------

pub struct Agent {
    pub game: Game,
    pub controller: NeuralController,
    pub steps: usize,
    pub fitness: f32,
    pub finished: bool,
}

impl Agent {
    pub fn new_random(base_map: &Map) -> Self {
        Self {
            game: Game::with_map(base_map.clone()),
            controller: NeuralController::random(),
            steps: 0,
            fitness: 0.0,
            finished: false,
        }
    }

    /// Reseta o agente in-place, reutilizando a struct `Agent` já alocada
    /// no `Vec` — zero realocação de `Vec<Agent>` entre gerações.
    /// Cria um `Game` novo a partir do mapa-base (obrigatório, pois o estado
    /// do jogo é mutável) e zera os contadores de fitness/passos.
    pub fn reset(&mut self, base_map: &Map) {
        self.game = Game::with_map(base_map.clone());
        self.steps = 0;
        self.fitness = 0.0;
        self.finished = false;
    }

    /// Avança a simulação em um passo de tempo.
    /// Público porque o loop de simulação (em `train.rs`) precisa chamar
    /// este método repetidamente até `finished == true`.
    pub fn step(&mut self) {
        if self.finished {
            return;
        }

        let action = self.controller.poll(&self.game);
        self.game.update(DT, action);
        self.steps += 1;

        if self.game.is_finished() || self.steps >= MAX_STEPS {
            self.finished = true;
            self.evaluate();
        }
    }

    /// Calcula e atribui o fitness do agente ao final da simulação.
    /// Uso interno — chamado apenas por `step()` quando a simulação encerra.
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
// Inicialização da População
// ---------------------------------------------------------------------------

pub fn init_population(size: usize, base_map: &Map) -> Vec<Agent> {
    (0..size).map(|_| Agent::new_random(base_map)).collect()
}

// ---------------------------------------------------------------------------
// Ordenação por Fitness
// ---------------------------------------------------------------------------

pub fn sort_by_fitness(population: &mut [Agent]) {
    population.sort_by(|a, b| {
        b.fitness
            .partial_cmp(&a.fitness)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

// ---------------------------------------------------------------------------
// Evolução (placeholder)
// ---------------------------------------------------------------------------

/// Prepara a próxima geração.
///
/// **Placeholder.** Hoje apenas troca cada controller por uma rede
/// aleatória e reseta o agente — efetivamente reiniciando a população.
///
/// Próximos passos (quando o AG real for implementado):
/// 1. Preservar `ELITE_SIZE` melhores (já ordenados por `sort_by_fitness`).
/// 2. Sortear pais por fitness (torneio ou roleta).
/// 3. Crossover entre os pais para gerar filhos.
/// 4. Mutação com `MUTATION_RATE` / `MUTATION_STRENGTH`.
pub fn evolve_population(population: &mut [Agent], base_map: &Map) {
    for agent in population.iter_mut() {
        agent.controller = NeuralController::random();
        agent.reset(base_map);
    }
}