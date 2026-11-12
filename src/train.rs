mod constants;
mod core;
mod ai;
mod utils;

use core::map::Map;
use ai::genetic::*;
use utils::logger::GenerationStats;

use rayon::prelude::*;
use std::time::Instant;

// ---------------------------------------------------------------------------
// 1. Simulação
// ---------------------------------------------------------------------------
//
// Apenas orquestra `Agent::step()` em loop até o agente encerrar.
// Toda a lógica do indivíduo (fitness, reset, etc.) vive em `ai::genetic`.

fn simulate_generation_parallel(population: &mut [Agent]) {
    population.par_iter_mut().for_each(|agent| {
        while !agent.finished {
            agent.step();
        }
    });
}

fn simulate_generation_single(population: &mut [Agent]) {
    for agent in population.iter_mut() {
        while !agent.finished {
            agent.step();
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Loop Principal
// ---------------------------------------------------------------------------

fn main() {
    // Lê os argumentos da linha de comando
    let args: Vec<String> = std::env::args().collect();
    let use_single_thread = args.iter().any(|arg| arg == "--single" || arg == "-s");

    // `--map <path>` carrega um mapa pré-existente; caso contrário, gera um
    // novo e salva em `map.bin` para reuso futuro.
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

    // Marca o início do treinamento total
    let total_start = Instant::now();

    let mut population = init_population(POPULATION_SIZE, &base_map);

    for generation in 1..=N_GENERATIONS {
        // Marca o início da geração
        let gen_start = Instant::now();

        // 1. Simular (escolhe o método baseado no argumento)
        if use_single_thread {
            simulate_generation_single(&mut population);
        } else {
            simulate_generation_parallel(&mut population);
        }

        // Tempo apenas desta geração
        let gen_elapsed_secs = gen_start.elapsed().as_secs_f32();

        // 2. Ordenar por fitness e coletar métricas
        sort_by_fitness(&mut population);

        let best = &population[0];
        let avg_fitness: f32 = population.iter().map(|a| a.fitness).sum::<f32>()
            / POPULATION_SIZE as f32;

        // 3. Log via módulo dedicado
        let stats = GenerationStats {
            generation,
            best_fitness: best.fitness,
            avg_fitness,
            best_score: best.game.score,
            time_secs: gen_elapsed_secs,
        };
        stats.print_row();

        // Salva a melhor rede esporadicamente
        if generation == N_GENERATIONS || generation % 10 == 0 {
            let path = format!("best_gen_{:03}.bin", generation);
            let _ = best.controller.net().save(&path);
        }

        // 4. Evolução da próxima geração (delegado ao módulo AG)
        //
        // O mapa é FIXO: `base_map` nunca é regenerado. Todos os elite,
        // filhos e mutações serão avaliados no mesmo ambiente, o que:
        //   • torna `best_fitness` comparável entre gerações;
        //   • permite que o elitismo de fato preserve "quem aprendeu este mapa";
        //   • dá sentido a continuar um treinamento pausado (`--map map.bin`).
        if generation < N_GENERATIONS {
            evolve_population(&mut population, &base_map);
        }
    }

    // Tempo total de treinamento
    let total_elapsed_secs = total_start.elapsed().as_secs_f32();

    println!("----------------------------------------------------------------------");
    println!("Treinamento concluído em {:.2} segundos!", total_elapsed_secs);
    println!("Mapa usado neste treinamento: 'map.bin' (reutilize com --map map.bin).");
}