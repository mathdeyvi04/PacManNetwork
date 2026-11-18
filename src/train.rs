#[macro_use]
mod utils; // ← NOVO: traz macros (logln!) para este módulo

mod constants;
mod core;
mod ai;
// mod utils; ← REMOVIDA: foi movida para cima com #[macro_use]

use core::map::Map;
use ai::genetic::*;
use ai::neural::NeuralNet;
use utils::logger::{GenerationStats, Logger}; // ← NOVO: importa Logger

use std::sync::{Arc, Mutex};
use std::io::Write;
use std::collections::VecDeque;

use rayon::prelude::*;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Constantes de telemetria
// ---------------------------------------------------------------------------

/// Janela da média móvel de fitness (em gerações).
const MOVING_AVG_WINDOW: usize = 50;

// ---------------------------------------------------------------------------
// Simulação
// ---------------------------------------------------------------------------

fn simulate_generation_parallel(population: &mut [Agent], base_map: &Map, seeds: &[u64]) {
    population.par_iter_mut().for_each(|agent| {
        agent.evaluate_multi(base_map, seeds);
    });
}

fn simulate_generation_single(population: &mut [Agent], base_map: &Map, seeds: &[u64]) {
    for agent in population.iter_mut() {
        agent.evaluate_multi(base_map, seeds);
    }
}

// ---------------------------------------------------------------------------
// Diversidade Genética
// ---------------------------------------------------------------------------

/// Desvio padrão médio dos pesos sinápticos em relação ao **centroide**
/// da população (genoma médio).
///
/// **Complexidade:** `O(N · W)` — duas passadas sobre a população
/// (`N` indivíduos × `W` pesos cada). Evita o custo `O(N² · W)` de
/// comparar todos os pares.
///
/// **Interpretação:**
/// - `≈ 0.0` → convergência prematura (elite clonado, falta de diversidade);
/// - `≈ 0.1..0.3` → regime saudável (mutação vs. seleção equilibrados);
/// - `> 0.5` → população dispersa (exploração alta, seleção fraca).
///
/// **Custo de memória:** `W · sizeof(f32)` ≈ `6.8 kB` para a arquitetura
/// 22→64→4 (`W = 1732`). Alocado **uma vez por geração**, descartado ao
/// fim da função.
fn compute_genomic_diversity(population: &[Agent]) -> f32 {
    if population.len() <= 1 {
        return 0.0;
    }
    // `PARAM_COUNT` é `const` — resolvido em compilação, zero runtime.
    // Não depende de existir população nem de acessar `population[0]`.
    let w_len = NeuralNet::PARAM_COUNT;
    if w_len == 0 {
        return 0.0;
    }

    let n = population.len() as f32;
    let inv_n = 1.0 / n;
    let mut mean_weights = vec![0.0f32; w_len];

    // Passada 1: centroide. O visitor preenche o índice — sem contador manual.
    for agent in population {
        agent.controller.net().for_each_param(|i, w| {
            mean_weights[i] += w;
        });
    }
    for mw in &mut mean_weights {
        *mw *= inv_n;
    }

    // Passada 2: variância acumulada em torno do centroide.
    let mut total_variance = 0.0f32;
    for agent in population {
        agent.controller.net().for_each_param(|i, w| {
            let diff = w - mean_weights[i];
            total_variance += diff * diff;
        });
    }

    (total_variance * inv_n / w_len as f32).sqrt()
}

// ---------------------------------------------------------------------------
// Recorde global
// ---------------------------------------------------------------------------

/// Snapshot do melhor indivíduo visto em **toda** a evolução.
///
/// Existe porque o elite `top-K` pode perder o campeão de uma geração
/// anterior — seja por competição com mutações posteriores, seja por
/// variação de fitness causada pelo RNG global dos fantasmas (rayon
/// intercala `gen_range` de forma não-determinística entre threads).
#[derive(Clone)]
struct BestEver {
    fitness: f32,
    generation: usize,
    score: u32,
    net: NeuralNet,
}

// ---------------------------------------------------------------------------
// Loop principal
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let use_single_thread = args.iter().any(|arg| arg == "--single" || arg == "-s");

    let map_path = args.iter()
        .position(|a| a == "--map")
        .and_then(|i| args.get(i + 1).cloned());

    // ── Logger fixo em "train_historic.txt" ──
    // Sempre recriado do início a cada execução (o arquivo contém apenas o
    // último treinamento realizado) e sem exigir caminho via argumento.
    let mut logger = match Logger::new(None) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Falha ao criar arquivo de log 'train_historic.txt': {}", e);
            std::process::exit(1);
        }
    };

    let base_map = match map_path.as_deref() {
        Some(path) => match Map::load(path) {
            Ok(m) => {
                logln!(logger, "Mapa carregado de '{}' ({} pellets)\n", path, m.pellets_remaining());
                m
            }
            Err(e) => {
                // Erros graves (Aviso) redirecionados ao log para manter o terminal limpo.
                logln!(logger, "Aviso: falha ao carregar '{}': {}. Gerando um novo mapa.", path, e);
                let m = Map::generate();
                let _ = m.save("map.bin");
                logln!(logger, "Mapa novo gerado e salvo em 'map.bin' ({} pellets)\n", m.pellets_remaining());
                m
            }
        },
        None => {
            let m = Map::generate();
            let _ = m.save("map.bin");
            logln!(logger, "Mapa novo gerado e salvo em 'map.bin' ({} pellets)\n", m.pellets_remaining());
            m
        }
    };

    let mode_str = if use_single_thread {
        "Single-Thread (Sequencial)"
    } else {
        "Multi-Thread (Rayon)"
    };

    logln!(logger, "=== Iniciando Treinamento Evolutivo ===");
    logln!(logger, "Modo de Execução: {}", mode_str);
    logln!(
        logger,
        "Gerações: {:3} | População: {:3} | Limite: {:4} frames\n",
        N_GENERATIONS, POPULATION_SIZE, MAX_STEPS
    );

    let _ = GenerationStats::print_header(&mut logger);
    let _ = logger.flush(); // Força a escrita imediata do cabeçalho no arquivo

    let total_start = Instant::now();

    // --seed <u64> → treinamento reprodutível (mesma seed = mesma trajetória).
    // Sem --seed, usa relógio do sistema (cada run é única).
    let base_seed = args.iter()
        .position(|a| a == "--seed")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x9E37_79B9_7F4A_7C15)
        });

    let mut population = init_population(POPULATION_SIZE, &base_map, base_seed);
    let mut best_ever: Option<BestEver> = None;

    // -----------------------------------------------------------------------
    // Snapshot compartilhado com o handler de Ctrl+C.
    // -----------------------------------------------------------------------
    let best_snapshot: Arc<Mutex<Option<BestEver>>> = Arc::new(Mutex::new(None));

    // -----------------------------------------------------------------------
    // Handler de SIGINT (Ctrl+C).
    // -----------------------------------------------------------------------
    {
        let snapshot = Arc::clone(&best_snapshot);
        if let Err(e) = ctrlc::set_handler(move || {
            eprintln!("\n[!] Ctrl+C recebido — salvando a melhor rede conhecida...");

            match snapshot.lock() {
                Ok(guard) => match guard.as_ref() {
                    Some(best) => {
                        let filename = format!(
                            "best_interrupted_gen_{:03}.bin",
                            best.generation
                        );
                        match best.net.save(&filename) {
                            Ok(()) => eprintln!(
                                "[!] Salvo em '{}' (geração {}, fitness {:+.2}, score {})",
                                filename, best.generation, best.fitness, best.score,
                            ),
                            Err(err) => eprintln!(
                                "[!] Erro ao salvar '{}': {}", filename, err
                            ),
                        }
                    }
                    None => eprintln!(
                        "[!] Nenhuma geração completada ainda — nada a salvar."
                    ),
                },
                Err(_) => eprintln!(
                    "[!] Mutex envenenado — saindo sem salvar."
                ),
            }

            std::process::exit(130);
        }) {
            eprintln!("[!] Aviso: não foi possível instalar o handler de Ctrl+C: {}", e);
            eprintln!("    O treinamento continuará, mas Ctrl+C encerrará sem salvar.");
        }
    }

    let mut gens_since_improvement: u64 = 0;
    let mut prev_best_ever = f32::NEG_INFINITY;

    // Buffer circular para a média móvel de fitness. `VecDeque` mantém
    // `push_back`/`pop_front` em O(1) e, no caso comum de janela cheia,
    // evita realocações. `with_capacity(MOVING_AVG_WINDOW)` pré-aloca
    // exatamente o tamanho máximo — zero alocações depois do warm-up.
    let mut avg_history: VecDeque<f32> = VecDeque::with_capacity(MOVING_AVG_WINDOW);

    for generation in 1..=N_GENERATIONS {
        let gen_start = Instant::now();

        if use_single_thread {
            simulate_generation_single(&mut population, &base_map, &EVAL_SEEDS);
        } else {
            simulate_generation_parallel(&mut population, &base_map, &EVAL_SEEDS);
        }

        let gen_elapsed_secs = gen_start.elapsed().as_secs_f32();

        // Ordenação PARCIAL: só precisamos dos top-ELITE_SIZE em ordem.
        // Os demais em `population[ELITE_SIZE..]` ficam em ordem qualquer —
        // serão sobrescritos por `evolve_population`.
        select_top_k_by_fitness(&mut population, ELITE_SIZE);

        let best = &population[0];
        let avg = population.iter().map(|a| a.fitness).sum::<f32>() / POPULATION_SIZE as f32;
        let std_dev = (population.iter().map(|a| (a.fitness - avg).powi(2)).sum::<f32>()
            / POPULATION_SIZE as f32).sqrt();

        // ── Média Móvel de 50 gerações ────────────────────────────────────
        // Janela deslizante: quando cheia, remove o mais antigo antes de
        // inserir o novo. `avg_history.len()` converge para `MOVING_AVG_WINDOW`
        // após as primeiras 50 gerações, então o divisor estabiliza.
        if avg_history.len() == MOVING_AVG_WINDOW {
            avg_history.pop_front();
        }
        avg_history.push_back(avg);
        let moving_avg_50 = avg_history.iter().sum::<f32>() / avg_history.len() as f32;

        // ── Diversidade Genética ──────────────────────────────────────────
        // Mede quão espalhados os pesos estão em torno do centroide.
        // Queda brusca → convergência prematura (aumentar mutação/imigração).
        let diversity = compute_genomic_diversity(&population);

        // ── Causas de término (médias da população) ───────────────────────
        // `death_rate`/`timeout_rate` por agente já estão em % (0..100);
        // a média da população reflete a fração de episódios em cada estado.
        let pop_death_pct = population.iter().map(|a| a.death_rate).sum::<f32>()
            / POPULATION_SIZE as f32;
        let pop_timeout_pct = population.iter().map(|a| a.timeout_rate).sum::<f32>()
            / POPULATION_SIZE as f32;

        // ── Progresso na tarefa (% de pellets) ────────────────────────────
        // `avg` mostra o desempenho típico; `max` mostra o quão perto o
        // melhor indivíduo chegou da vitória. Ambos são médias entre as
        // 5 seeds de avaliação, não valores de uma partida isolada.
        let avg_pellets_pct = population.iter().map(|a| a.avg_pellets_pct).sum::<f32>()
            / POPULATION_SIZE as f32;
        let max_pellets_pct = population.iter().map(|a| a.avg_pellets_pct).fold(0.0f32, f32::max);

        // Atualiza o recorde global se o campeão desta geração o superou.
        if best_ever.as_ref().map_or(true, |b| best.fitness > b.fitness) {
            best_ever = Some(BestEver {
                fitness: best.fitness,
                generation,
                score: best.avg_score,
                net: best.controller.net().clone(),
            });

            // Espelha no snapshot compartilhado para o handler de Ctrl+C.
            // Bloqueia por nanossegundos; contenção com o handler é
            // teórica (ele só dispara uma vez).
            if let Ok(mut guard) = best_snapshot.lock() {
                *guard = best_ever.clone();
            }
        }
        let record = best_ever.as_ref().unwrap();

        // Atualiza contador de estagnação
        if record.fitness > prev_best_ever + 0.01 {
            prev_best_ever = record.fitness;
            gens_since_improvement = 0;
        } else {
            gens_since_improvement += 1;
        }

        // ── Hiperparâmetros efetivos (fonte única) ────────────────────────
        // `current_mutation_params` encapsula o boost adaptativo. Chamada
        // aqui apenas para LOG — `evolve_population` chama a mesma função
        // internamente, garantindo que o valor reportado seja o aplicado.
        let (mut_rate, mut_strength) = current_mutation_params(gens_since_improvement);

        let stats = GenerationStats {
            generation,
            best_fitness: best.fitness,
            avg_fitness: avg,
            moving_avg_50,
            std_dev,
            diversity,
            best_score: best.avg_score,
            avg_pellets_pct,
            max_pellets_pct,
            death_pct: pop_death_pct,
            timeout_pct: pop_timeout_pct,
            mut_rate,
            mut_strength,
            best_ever_fitness: record.fitness,
            best_ever_generation: record.generation,
            best_ever_score: record.score,
            time_secs: gen_elapsed_secs,
        };
        let _ = stats.print_row(&mut logger);

        if generation % 10 == 0 {
            let _ = logger.flush();
        }

        if generation < N_GENERATIONS {
            evolve_population(&mut population, generation as u64, gens_since_improvement);
        }
    }

    let total_elapsed_secs = total_start.elapsed().as_secs_f32();

    // ---------------------------------------------------------------------
    // Salvamento ÚNICO da melhor rede de todo o treinamento
    // ---------------------------------------------------------------------
    let record = best_ever.expect("população vazia — impossível com N_GENERATIONS >= 1");
    let filename = format!("best_gen_{:03}.bin", record.generation);

    logln!(logger, "\n----------------------------------------------------------------------");
    logln!(logger, "Treinamento concluído em {:.2} segundos!", total_elapsed_secs);
    logln!(
        logger,
        "Melhor rede: geração {:02} | fitness {:+.2} | score {}",
        record.generation, record.fitness, record.score
    );

    match record.net.save(&filename) {
        Ok(()) => {
            logln!(logger, "Salva em '{}' (use com `nn {} map.bin`)", filename, filename);
        }
        Err(e) => {
            logln!(logger, "Erro ao salvar '{}': {}", filename, e);
        }
    }
    let _ = logger.flush(); // Garante que as últimas linhas sejam escritas
}