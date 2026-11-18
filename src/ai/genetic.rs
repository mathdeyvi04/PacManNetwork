//! Módulo do Algoritmo Genético (AG).
//!
//! Contém as constantes de configuração do processo evolutivo, a entidade
//! `Agent`, a criação da população inicial, a ordenação por fitness e a
//! **evolução sexuada com elitismo + torneio + mutação uniforme**.
//!
//! A simulação (loop de `step()` por agente, single/multi-thread) fica
//! em `train.rs`, por ser uma preocupação de execução.

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

// ---------- ELITISMO: menos rígido ----------
// 8 elites congelados bloqueiam exploração. Reduzir para 3 libera
// mais slots para descendentes reais (não clones do mesmo pool).
pub const ELITE_SIZE: usize = 4;

// ---------- MUTAÇÃO BASE: mais forte ----------
// 0.05/0.15 nunca atravessa vales de fitness. 0.15/0.35 é o mínimo
// para um MLP sair de um ótimo local raso.
pub const MUTATION_RATE: f32 = 0.15;
pub const MUTATION_STRENGTH: f32 = 0.35;

// ---------- MUTAÇÃO ADAPTATIVA ----------
// Se não melhorar por N gerações, intensifica mutação temporariamente.
pub const STAGNATION_THRESHOLD: u64 = 10;
pub const STAGNATION_BOOST_RATE: f32 = 2.5;      // 0.15 → 0.375
pub const STAGNATION_BOOST_STRENGTH: f32 = 2.0;  // 0.35 → 0.70

// ---------- IMIGRANTES ALEATÓRIOS ----------
// A cada N gerações, substitui os piores X% por redes aleatórias puras.
// Isso injeta DNA que não existe na população (exploração genuína).
pub const IMMIGRATION_INTERVAL: u64 = 15;
pub const IMMIGRATION_FRACTION: f32 = 0.10;

// ---------- CROSSOVER: manter, mas adicionar ruído ----------
pub const CROSSOVER_RATE: f32 = 0.4;
pub const TOURNAMENT_K: usize = 6;

// ---------- FITNESS: exploração via tiles distintos ----------
//
// Substitui FITNESS_MOVEMENT_REWARD. Uma rede que fica parada, oscila
// entre dois tiles, ou empurra parede sem sair do lugar vê `count`
// congelado em 1–3. A única forma de subir o contador é entrar em tiles
// NOVOS — exatamente o comportamento que queremos premiar.
pub const FITNESS_DISTINCT_TILE_REWARD: f32 = 2.5;

// ---------- FITNESS: eficiência temporal ----------
//
// Coeficiente do bônus de pellets-por-passo. Escalado por 1000× em
// `run_episode` para que 0.5 produza um fator efetivo de 500× — mesma
// ordem de grandeza do score base (10 pts/pellet × dezenas de pellets).
// Aumente para enfatizar velocidade, diminua para deixar o agente
// explorar mais devagar.
pub const FITNESS_TIME_EFFICIENCY_BONUS: f32 = 0.5;

// ---------- FITNESS: inatividade ----------
//
// Penalidade por frame sem descobrir um tile novo **após** um limiar.
// Reservada: `run_episode` não a aplica ainda (o bitmap `VisitedTiles`
// já satura `count` em vaivém; esta penalidade explícita reforçaria o
// sinal). Fica exposta para experimentos — aplicar no loop principal
// quando `consecutive_stagnant > limiar` (ex.: 120 frames ≈ 2 s @ 60 fps).
pub const FITNESS_STAGNATION_PENALTY: f32 = -0.1;

// Mantidos da Rota A:
pub const FITNESS_TIMEOUT_PENALTY: f32 = -200.0;
pub const FITNESS_DEATH_PENALTY:   f32 = -200.0;
pub const FITNESS_COMPLETION_BONUS: f32 = 3000.0;
pub const FITNESS_WIN_BONUS:        f32 = 2000.0;

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

use crate::constants::{GRID_HEIGHT, GRID_WIDTH};

// ---------------------------------------------------------------------------
// Tiles Visitados — bitmap compacto para contagem de tiles distintos
// ---------------------------------------------------------------------------

/// Quantas palavras de 64 bits cabem no grid.
/// `31 × 33 = 1023 tiles` → `ceil(1023 / 64) = 16` palavras = **128 bytes**.
const VISITED_WORDS: usize = (GRID_WIDTH * GRID_HEIGHT + 63) / 64;

/// Bitmap de tiles já pisados pelo Pac-Man durante **um** episódio.
///
/// Conta cada tile **diferente** uma única vez. Re-entrar num tile já
/// visitado é um no-op — é essa propriedade que penaliza vaivém: uma
/// rede que oscila entre dois tiles fica presa em `count == 2`, enquanto
/// uma que explora o mapa chega a 300–500.
///
/// Instanciado localmente em `run_episode`. Cada chamada começa zerada:
/// não persiste entre seeds, o que é correto (cada seed é uma trajetória
/// independente; somar tiles entre seeds seria contar exploração dupla).
struct VisitedTiles {
    bits:  [u64; VISITED_WORDS],
    count: u32,
}

impl VisitedTiles {
    #[inline]
    fn new() -> Self {
        Self { bits: [0; VISITED_WORDS], count: 0 }
    }

    /// Marca `(x, y)` como visitado. Retorna `true` se era inédito.
    ///
    /// **Pré-condição:** `x < GRID_WIDTH && y < GRID_HEIGHT`. Satisfeita
    /// por construção — `game.player.tile()` nunca devolve coordenadas
    /// fora do grid (o `Player::update` faz clamp via `Map::is_wall`).
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
        let initial_pellets = game.map.pellets_remaining();
        let mut steps = 0usize;

        // ── Contador de tiles DISTINTOS ──
        // Marca o spawn primeiro (Pac-Man "está" ali). Cada frame depois disso
        // tenta marcar o tile atual: se for inédito, `count` incrementa; se
        // for revisita, é no-op. É essa "deduplicação" que diferencia uma
        // rede exploradora de uma rede que oscila entre dois tiles.
        let mut visited = VisitedTiles::new();
        {
            let (sx, sy) = game.player.tile();
            visited.visit(sx, sy);
        }

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

        let terminal = match game.state() {
            GameState::GameOver => FITNESS_DEATH_PENALTY,
            GameState::Win      => FITNESS_WIN_BONUS,
            GameState::Playing  => FITNESS_TIMEOUT_PENALTY,
        };

        // ── Eficiência temporal ──
        // Pellets por passo, escalado para a mesma ordem de grandeza do
        // score base. `FITNESS_TIME_EFFICIENCY_BONUS = 0.5` → fator 500×.
        // Penaliza agentes que gastam 10k passos para comer 5 pellets
        // (eaten/steps ≈ 0.0005 → bônus desprezível); premia quem limpa
        // rápido (eaten/steps ≈ 0.2 → bônus ≈ 100).
        //
        // Guard `steps > 0` é defensivo: `with_map_and_seed` /
        // `reset_in_place` sempre produzem um jogo em `Playing`, então o
        // loop executa ao menos uma iteração na prática; mas se
        // `MAX_STEPS == 0` fosse configurado, evitamos divisão por zero.
        let efficiency_bonus = if eaten > 0 && steps > 0 {
            (eaten as f32 / steps as f32) * FITNESS_TIME_EFFICIENCY_BONUS * 1000.0
        } else {
            0.0
        };

        // ── Fórmula final aprimorada ──
        // Score + completude + exploração + eficiência + terminal.
        game.score as f32
            + completion * FITNESS_COMPLETION_BONUS
            + visited.count() as f32 * FITNESS_DISTINCT_TILE_REWARD
            + efficiency_bonus
            + terminal
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
pub fn evolve_population(
    population: &mut [Agent],
    generation: u64,
    generations_since_improvement: u64,
) {
    let n = population.len();
    debug_assert!(n > ELITE_SIZE);

    // Mutação adaptativa: escala quando estagnado.
    let (mut_rate, mut_strength) = if generations_since_improvement >= STAGNATION_THRESHOLD {
        (
            (MUTATION_RATE * STAGNATION_BOOST_RATE).min(0.6),
            (MUTATION_STRENGTH * STAGNATION_BOOST_STRENGTH).min(1.0),
        )
    } else {
        (MUTATION_RATE, MUTATION_STRENGTH)
    };

    let snapshot: Vec<(f32, NeuralNet)> = population
        .iter()
        .map(|a| (a.fitness, a.controller.net().clone()))
        .collect();

    let gen_seed = generation
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0xDEAD_BEEF_CAFE_F00D);
    let mut rng = Rng::new(gen_seed);

    // Elites em [0..ELITE_SIZE] são preservados intactos.

    // Descendentes a partir de [ELITE_SIZE..n).
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

    // Imigrantes aleatórios puros quase nunca sobrevivem; o que funciona é
    // explorar a VIZINHANÇA do campeão com perturbação agressiva.
    if generation > 0 && generation % IMMIGRATION_INTERVAL == 0 {
        let n_immigrants = ((n as f32) * IMMIGRATION_FRACTION) as usize;
        let start = n.saturating_sub(n_immigrants);

        // snapshot[0] é o elite (campeão, ainda não modificado).
        let champion = &snapshot[0].1;
        for i in start..n {
            let mut child = champion.clone();
            let seed = rng.next();
            child.mutate(0.35, 0.55, seed);  // agressivo, mas entorno do campeão
            population[i].controller = NeuralController::new(child);
        }
    }
}