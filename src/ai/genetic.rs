//! Módulo do Algoritmo Genético (AG).
//!
//! Contém as constantes de configuração do processo evolutivo, a entidade
//! `Agent`, a criação da população inicial, a ordenação por fitness e a
//! **evolução assexual com elitismo + mutação uniforme**.
//!
//! A simulação (loop de `step()` por agente, single/multi-thread) fica
//! em `train.rs`, por ser uma preocupação de execução.

use crate::core::game::{Game, GameState};
use crate::core::map::Map;
use crate::core::controller::Controller;
use crate::ai::neural::NeuralController;

// ---------------------------------------------------------------------------
// Tamanho da População e Simulação
// ---------------------------------------------------------------------------

/// Número de agentes por geração.
pub const POPULATION_SIZE: usize = 100;

/// Limite máximo de frames (`steps`) que um agente pode simular.
pub const MAX_STEPS: usize = 3000;

/// Passo de tempo fixo por frame (`1/60 s`).
pub const DT: f32 = 1.0 / 60.0;

// ---------------------------------------------------------------------------
// Parâmetros do Algoritmo Genético
// ---------------------------------------------------------------------------

/// Número total de gerações a evoluir.
pub const N_GENERATIONS: usize = 50;

/// Quantidade de melhores indivíduos preservados intactos a cada geração.
/// O restante da população é preenchido por cópias mutadas do **melhor**
/// indivíduo (população já ordenada por fitness antes da evolução).
pub const ELITE_SIZE: usize = 5;

/// Probabilidade de mutação por gene (peso/bias).
pub const MUTATION_RATE: f32 = 0.10;

/// Magnitude máxima da perturbação uniforme aplicada a um gene mutado.
pub const MUTATION_STRENGTH: f32 = 0.2;

// ---------------------------------------------------------------------------
// Função de Fitness — Recompensas e Penalidades
// ---------------------------------------------------------------------------
//
// Modelo: `fitness = score − steps × STEP_COST − death_penalty + win_bonus`.
//
// STEP_COST é uma **penalidade** (subtraída), não um bônus — inverter o sinal
// do antigo `SURVIVAL_MULT` (que somava `+0.05 × steps`) elimina o incentivo
// para "enrolar" até o timeout. `death_penalty` foi rebaixada de −500 para
// −200 para que o break-even "comer N e morrer" vs "enrolar 3000 steps"
// caia de N≈63 (impossível para agentes aleatórios) para N≈15 (alcançável
// já na geração 1). `win_bonus` foi elevado de 2000 para 5000 para manter
// dominância clara sobre qualquer combinação de score + timeout.

/// Custo por passo de simulação. **Positivo** semanticamente, **subtraído**
/// em `evaluate()` — pune estagnação sem proibir tempo produtivo.
pub const FITNESS_STEP_COST: f32 = 0.02;

/// Penalidade por terminar em `GameOver`. Aplicada como **soma** (negativa).
pub const FITNESS_DEATH_PENALTY: f32 = -200.0;

/// Bônus por vencer (todos os pellets comidos).
pub const FITNESS_WIN_BONUS: f32 = 5000.0;

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

    /// Reseta o agente in-place. **Não** toca no `controller` — é assim que
    /// os elites preservam seus pesos entre gerações.
    pub fn reset(&mut self, base_map: &Map) {
        self.game = Game::with_map(base_map.clone());
        self.steps = 0;
        self.fitness = 0.0;
        self.finished = false;
    }

    /// Avança a simulação em um passo de tempo.
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

    /// Avalia o fitness ao final da simulação.
    ///
    /// `Playing` (timeout sem fim explícito) simplesmente não aplica nem
    /// penalidade nem bônus: `fitness = score − step_cost`. Isso mantém o
    /// gradiente "comer mais rápido > comer devagar > enrolar" uniforme.
    fn evaluate(&mut self) {
        self.fitness = self.game.score as f32 - (self.steps as f32 * FITNESS_STEP_COST);

        match self.game.state() {
            GameState::GameOver => self.fitness += FITNESS_DEATH_PENALTY,
            GameState::Win      => self.fitness += FITNESS_WIN_BONUS,
            GameState::Playing  => {} // timeout: score − step_cost já basta
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

// ---------------------------------------------------------------------------
// Seleção do Top-K (elite)
// ---------------------------------------------------------------------------

/// Comparador de fitness **decrescente** (maior primeiro).
///
/// Extraído como `fn` (não closure) para poder ser reutilizado por
/// `select_nth_unstable_by` e `sort_by` — um item `fn` é `Copy`, um
/// closure que captura não seria.
#[inline]
fn cmp_fitness_desc(a: &Agent, b: &Agent) -> std::cmp::Ordering {
    b.fitness
        .partial_cmp(&a.fitness)
        .unwrap_or(std::cmp::Ordering::Equal)
}

/// Reorganiza `population` de modo que os `k` melhores indivíduos
/// (por fitness decrescente) fiquem em `population[..k]`, ordenados
/// do melhor para o pior. Os demais ocupam `population[k..]` em
/// ordem arbitrária.
///
/// **Por que não `sort_by` completo?** Só precisamos dos top-K — para
/// `population[0]` (clone do melhor), `population[..ELITE_SIZE]` (elite
/// preservado) e para a média (que independe de ordem). Um `sort_by`
/// completo gastaria `O(n log n)` em trabalho inútil.
///
/// Complexidade: `O(n)` amortizado via `select_nth_unstable_by` +
/// `O(k log k)` para ordenar apenas o top-K.
///
/// **Pré-condições:** `1 <= k <= population.len()`.
pub fn select_top_k_by_fitness(population: &mut [Agent], k: usize) {
    debug_assert!(k > 0, "k deve ser >= 1");
    debug_assert!(k <= population.len(), "k maior que a população");

    if k < population.len() {
        // Após esta chamada: `population[k-1]` é o k-ésimo maior; tudo
        // antes dele tem fitness >= ; tudo depois tem fitness <= .
        // Isso garante que o top-K está exatamente em `population[..k]`.
        population.select_nth_unstable_by(k - 1, cmp_fitness_desc);
    }
    // Ordena só o subconjunto top-K (k pequeno — O(k log k) desprezível).
    population[..k].sort_by(cmp_fitness_desc);
}

// ---------------------------------------------------------------------------
// Evolução — assexual com elitismo + mutação uniforme
// ---------------------------------------------------------------------------

/// Prepara a próxima geração.
///
/// **Pré-condição:** `population[..ELITE_SIZE]` deve conter os `ELITE_SIZE`
/// melhores indivíduos em ordem decrescente de fitness (i.e.,
/// `population[0]` é o melhor). `train.rs` garante isso chamando
/// `select_top_k_by_fitness(&mut population, ELITE_SIZE)` antes.
///
/// Os demais elementos (`population[ELITE_SIZE..]`) **não** precisam estar
/// ordenados — serão sobrescritos pelas cópias mutadas do melhor.
///
/// Estratégia (assexual, com elitismo):
/// 1. Preserva os `ELITE_SIZE` melhores **intactos** (rede e tudo).
/// 2. Cada um dos restantes recebe uma **cópia mutada** do melhor
///    (índice 0), com `MUTATION_RATE` por gene e perturbação uniforme
///    em `[-MUTATION_STRENGTH, +MUTATION_STRENGTH)`.
/// 3. Todos os agentes (elites inclusive) são resetados contra `base_map`.
pub fn evolve_population(population: &mut [Agent], base_map: &Map) {
    let n = population.len();
    debug_assert!(n > ELITE_SIZE, "população menor ou igual a ELITE_SIZE");

    // Snapshot do melhor indivíduo **antes** de qualquer modificação.
    let best_net = population[0].controller.net().clone();

    let base_seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E37_79B9_7F4A_7C15);

    // 1. Elites: reset do estado do jogo, rede intacta.
    for agent in population.iter_mut().take(ELITE_SIZE) {
        agent.reset(base_map);
    }

    // 2. Não-elites: cópia mutada do melhor, com seed distinta por filho.
    for (i, agent) in population.iter_mut().enumerate().skip(ELITE_SIZE) {
        let mut child_net = best_net.clone();
        // Offset no seed garante streams de RNG independentes por filho.
        let seed = base_seed.wrapping_add(
            (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        );
        child_net.mutate(MUTATION_RATE, MUTATION_STRENGTH, seed);

        agent.controller = NeuralController::new(child_net);
        agent.reset(base_map);
    }
}