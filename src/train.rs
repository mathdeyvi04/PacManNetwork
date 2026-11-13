mod constants;
mod core;
mod ai;
mod utils;

use core::map::Map;
use ai::genetic::*;
use ai::neural::NeuralNet;
use utils::logger::GenerationStats;

use rayon::prelude::*;
use std::time::Instant;

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
// Recorde global
// ---------------------------------------------------------------------------

/// Snapshot do melhor indivíduo visto em **toda** a evolução.
///
/// Existe porque o elite `top-K` pode perder o campeão de uma geração
/// anterior — seja por competição com mutações posteriores, seja por
/// variação de fitness causada pelo RNG global dos fantasmas (rayon
/// intercala `gen_range` de forma não-determinística entre threads).
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

    let base_map = match map_path.as_deref() {
        Some(path) => match Map::load(path) {
            Ok(m) => {
                println!("Mapa carregado de '{}' ({} pellets)\n", path, m.pellets_remaining());
                m
            }
            Err(e) => {
                eprintln!(
                    "Aviso: falha ao carregar '{}': {}. Gerando um novo mapa.",
                    path, e
                );
                let m = Map::generate();
                let _ = m.save("map.bin");
                println!("Mapa novo gerado e salvo em 'map.bin' ({} pellets)\n", m.pellets_remaining());
                m
            }
        },
        None => {
            let m = Map::generate();
            let _ = m.save("map.bin");
            println!("Mapa novo gerado e salvo em 'map.bin' ({} pellets)\n", m.pellets_remaining());
            m
        }
    };

    let mode_str = if use_single_thread {
        "Single-Thread (Sequencial)"
    } else {
        "Multi-Thread (Rayon)"
    };

    println!("=== Iniciando Treinamento Evolutivo ===");
    println!("Modo de Execução: {}", mode_str);
    println!(
        "Gerações: {:3} | População: {:3} | Limite: {:4} frames\n",
        N_GENERATIONS, POPULATION_SIZE, MAX_STEPS
    );

    GenerationStats::print_header();

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

    let mut gens_since_improvement: u64 = 0;
    let mut prev_best_ever = f32::NEG_INFINITY;

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

        // Atualiza o recorde global se o campeão desta geração o superou.
        if best_ever.as_ref().map_or(true, |b| best.fitness > b.fitness) {
            best_ever = Some(BestEver {
                fitness: best.fitness,
                generation,
                score: best.avg_score,           
                net: best.controller.net().clone(),
            });
        }
        let record = best_ever.as_ref().unwrap();

        // Atualiza contador de estagnação
        let record = best_ever.as_ref().unwrap();
        if record.fitness > prev_best_ever + 0.01 {
            prev_best_ever = record.fitness;
            gens_since_improvement = 0;
        } else {
            gens_since_improvement += 1;
        }

        let stats = GenerationStats {
            generation,
            best_fitness: best.fitness,
            avg_fitness: avg,
            std_dev,
            best_score: best.avg_score,
            best_ever_fitness: record.fitness,
            best_ever_generation: record.generation,
            best_ever_score: record.score,
            time_secs: gen_elapsed_secs,
        };
        stats.print_row();

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

    println!("\n----------------------------------------------------------------------");
    println!("Treinamento concluído em {:.2} segundos!", total_elapsed_secs);
    println!(
        "Melhor rede: geração {:02} | fitness {:+.2} | score {}",
        record.generation, record.fitness, record.score
    );

    match record.net.save(&filename) {
        Ok(()) => println!("Salva em '{}' (use com `nn {} map.bin`)", filename, filename),
        Err(e) => eprintln!("Erro ao salvar '{}': {}", filename, e),
    }
}