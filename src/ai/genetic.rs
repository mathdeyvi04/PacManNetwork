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
use crate::ai::neural::NeuralNet;
use crate::core::rng::Rng;

// ---------------------------------------------------------------------------
// Tamanho da População e Simulação
// ---------------------------------------------------------------------------

pub const POPULATION_SIZE: usize = 100;

/// Limite máximo de frames. Aumentado de 3000→8000 para dar espaço
/// a agentes que consigam sobreviver bem explorando (não apenas aos
/// que morrem cedo); partidas curtas continuam curtas.
pub const MAX_STEPS: usize = 8000;

pub const DT: f32 = 1.0 / 60.0;

// ---------------------------------------------------------------------------
// Parâmetros do Algoritmo Genético (sexuado + torneio + elitismo)
// ---------------------------------------------------------------------------

/// Gerações. Aumentado de 50→200: com o bug de congelamento corrigido,
/// a curva tende a subir de forma consistente e precisa de orçamento
/// para mostrar todo o progresso possível.
pub const N_GENERATIONS: usize = 1000;

/// Elites preservados intactos. Aumentado de 5→8: pool um pouco maior
/// de top performers (diversidade na base da população seguinte).
pub const ELITE_SIZE: usize = 8;

/// Taxa de mutação por gene. Reduzida de 0.10→0.05: agora a diversidade
/// principal vem do crossover; mutação passa a ser **refinamento**.
pub const MUTATION_RATE: f32 = 0.05;

/// Magnitude da perturbação. Reduzida de 0.2→0.15: ajuste mais fino
/// (menos "salto no escuro", mais "polir o que já funciona").
pub const MUTATION_STRENGTH: f32 = 0.15;

/// Fração de filhos gerados por **crossover** (resto: clone + mutação).
/// 0.7 = 70% sexuado, 30% assexual — combina exploração (sexo) com
/// continuidade (mutação pura do pai).
pub const CROSSOVER_RATE: f32 = 0.7;

/// Tamanho do torneio para escolher cada pai. k=3 → pressão seletiva
/// moderada (o melhor de 3 vence, mas o 2º e o 3º ainda têm chance).
pub const TOURNAMENT_K: usize = 3;

// Cada agente é avaliado contra K partidas com estas K sementes FIXAS,
// e o fitness reportado é a MÉDIA. Sem isso, um agente pode parecer
// ótimo por ter recebido uma trajetória sortuda de fantasmas, e o elite
// sortudo fica preso no topo por centenas de gerações.
//
// Escolhidas por serem valores "aleatórios" com boa distribuição de
// bits — qualquer conjunto de constantes serve, desde que fixo.
//
// Para aumentar robustez, aumente o array. Custo: treino K× mais lento.
pub const EVAL_SEEDS: [u64; 5] = [
    0xDEAD_BEEF_CAFE_F00D,
    0x0F0F_0F0F_0F0F_0F0F,
    0x5555_5555_5555_5555,
    0xAAAA_AAAA_AAAA_AAAA,
    0x1234_5678_9ABC_DEF0,
];

// ---------------------------------------------------------------------------
// Função de Fitness
// ---------------------------------------------------------------------------
//
// fitness = score − steps×STEP_COST + penalty_estado_final
//
// penalty_estado_final:
//   • GameOver (morreu)     → DEATH_PENALTY
//   • Playing  (timeout)    → TIMEOUT_PENALTY   ← NOVO
//   • Win      (venceu)     → WIN_BONUS
//
// O TIMEOUT_PENALTY é o que **elimina o "survivalist trap"**: sem ele,
// um agente que come 60 pellets e fica parado até o timeout (fitness
// 600−100=500) seria **melhor** que um agente que come 60 pellets e
// morre tentando explorar (600−22−200=378). Com −500 no timeout, morrer
// explorando passa a valer 378 e "escondido" passa a valer 500−500=0.
// Força o comportamento correto: melhor tentar e perder do que travar.

pub const FITNESS_STEP_COST: f32 = 0.02;
pub const FITNESS_DEATH_PENALTY: f32 = -200.0;
pub const FITNESS_TIMEOUT_PENALTY: f32 = -500.0;
pub const FITNESS_WIN_BONUS: f32 = 5000.0;

// ---------------------------------------------------------------------------
// Entidade (Um Agente da População)
// ---------------------------------------------------------------------------

pub struct Agent {
    pub game: Game,
    pub controller: NeuralController,
    // pub steps: usize,     // ← deletar (nunca lido)
    pub fitness: f32,
    pub avg_score: u32,
    // pub finished: bool,    // ← deletar (nunca lido)
}

impl Agent {
    /// Cria um agente com pesos aleatórios (`net_seed`) e um jogo vazio.
    /// O jogo inicial é descartado em `evaluate_multi` — só existe para
    /// satisfazer o tipo.
    pub fn new_random(base_map: &Map, net_seed: u64) -> Self {
        Self {
            game: Game::with_map_and_seed(base_map.clone(), net_seed),
            controller: NeuralController::random_with_seed(net_seed),
            fitness: 0.0,
            avg_score: 0,
        }
    }

    /// Avalia a rede contra K partidas (sementes fornecidas) e atribui
    /// a **média de fitness** em `self.fitness` e a **média de score**
    /// em `self.avg_score`.
    ///
    /// **Otimização:** a primeira seed usa `with_map_and_seed` (uma
    /// alocação de Game + varredura do mapa para achar spawns). As
    /// seeds seguintes usam `Game::reset_in_place`, que restaura o
    /// estado sem re-alocar e sem re-varrer o mapa.
    ///
    /// **Pré-condição:** `base_map` é um template virgem (todo `Empty`
    /// é uma pastilha comida por alguma partida anterior do template).
    /// O `with_map_and_seed` clona, então o template permanece intocado.
    pub fn evaluate_multi(&mut self, base_map: &Map, seeds: &[u64]) {
        debug_assert!(!seeds.is_empty(), "seeds não pode ser vazio");

        let mut fitness_sum = 0.0f32;
        let mut score_sum = 0u64;

        // ─── Primeira seed: inicialização completa (1 clone do template) ──
        self.game = Game::with_map_and_seed(base_map.clone(), seeds[0]);
        fitness_sum += Self::run_episode(&mut self.controller, &mut self.game);
        score_sum += self.game.score as u64;

        // ─── Seeds seguintes: reset in-place (zero alocação) ──
        for &seed in &seeds[1..] {
            self.game.reset_in_place(seed);
            fitness_sum += Self::run_episode(&mut self.controller, &mut self.game);
            score_sum += self.game.score as u64;
        }

        let k = seeds.len() as f32;
        self.fitness = fitness_sum / k;
        self.avg_score = (score_sum as f32 / k).round() as u32;
    }

    /// Roda **uma partida completa** com o `controller` e `game` dados,
    /// até o jogo encerrar (`GameOver`/`Win`) ou `MAX_STEPS` ser atingido.
    ///
    /// **Retorna** o fitness da partida **individual** (não a média).
    /// A agregação em média fica em `evaluate_multi`.
    ///
    /// **Por que é função associada (não método `&mut self`):** precisamos
    /// emprestar `&mut self.controller` e `&mut self.game` **ao mesmo
    /// tempo**. Como são campos disjuntos de `Agent`, Rust permite. Se
    /// fosse `fn run_episode(&mut self)`, o `self` inteiro estaria
    /// emprestado uma vez e o compilador reclamaria.
    fn run_episode(ctrl: &mut NeuralController, game: &mut Game) -> f32 {
        let mut steps = 0usize;

        while !game.is_finished() && steps < MAX_STEPS {
            // `game: &mut Game` → coerção automática para `&Game` no `poll`.
            let action = ctrl.poll(game);
            game.update(DT, action);
            steps += 1;
        }

        // Mesma fórmula de antes — apenas com `steps` local em vez de
        // `self.steps` (que não é mais usado por ninguém).
        game.score as f32
            - steps as f32 * FITNESS_STEP_COST
            + match game.state() {
                GameState::GameOver => FITNESS_DEATH_PENALTY,
                GameState::Win      => FITNESS_WIN_BONUS,
                GameState::Playing  => FITNESS_TIMEOUT_PENALTY,
            }
    }
}

// ---------------------------------------------------------------------------
// Inicialização da População
// ---------------------------------------------------------------------------

pub fn init_population(size: usize, base_map: &Map, base_seed: u64) -> Vec<Agent> {
    (0..size)
        .map(|i| Agent::new_random(base_map, base_seed.wrapping_add(i as u64)))
        .collect()
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

// ---------------------------------------------------------------------------
// Evolução — sexuada com torneio + elitismo
// ---------------------------------------------------------------------------

/// Seleciona um índice via **torneio de tamanho k** sobre o snapshot.
/// Retorna o índice do melhor entre `k` sorteados uniformemente.
///
/// Não depende da ordenação da população — pode varrer o vetor inteiro.
/// É isso que permite que **elites e não-elites** contribuam como pais.
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

/// Prepara a próxima geração (sexuado com torneio + elitismo).
///
/// **Pré-condição:** `population[..ELITE_SIZE]` contém os top-K em ordem
/// decrescente de fitness. `train.rs` garante via `select_top_k_by_fitness`.
///
/// **`generation`** entra na derivação da RNG por geração. Sem isso, filhos
/// seriam idênticos quando o elite não mudasse.
///
/// **Não toca em `game`, `fitness` nem `finished`.** A próxima simulação
/// (`evaluate_multi`) reinicia tudo. Só os `controller`s (pesos) são
/// modificados — elites preservam; filhos recombinam + mutam.
pub fn evolve_population(population: &mut [Agent], generation: u64) {
    let n = population.len();
    debug_assert!(n > ELITE_SIZE);

    // Snapshot: fitness + net de cada indivíduo, antes de qualquer escrita.
    let snapshot: Vec<(f32, NeuralNet)> = population
        .iter()
        .map(|a| (a.fitness, a.controller.net().clone()))
        .collect();

    let gen_seed = generation
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0xDEAD_BEEF_CAFE_F00D);
    let mut rng = Rng::new(gen_seed);

    // 1. Elites: `population[..ELITE_SIZE]` já contém as melhores redes
    //    no `controller`. Nada a fazer aqui — apenas NÃO sobrescrevê-las.

    // 2. Filhos: crossover + mutação.
    for agent in population.iter_mut().skip(ELITE_SIZE) {
        let pa = tournament_select(&snapshot, TOURNAMENT_K, &mut rng);
        let pb = tournament_select(&snapshot, TOURNAMENT_K, &mut rng);

        let mut child_net = if rng.unit() < CROSSOVER_RATE {
            snapshot[pa].1.crossover(&snapshot[pb].1, &mut rng)
        } else {
            snapshot[pa].1.clone()
        };

        let mut_seed = rng.next();
        child_net.mutate(MUTATION_RATE, MUTATION_STRENGTH, mut_seed);

        agent.controller = NeuralController::new(child_net);
    }
}