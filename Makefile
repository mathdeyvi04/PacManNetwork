# Humano jogando normalmente
game:
	cargo run --release --bin PacManNetwork

# Humano jogando com mapa carregado
game_with_map:
	cargo run --release --bin PacManNetwork -- human map.bin

# Rede aleatoria, mapa aleatório
nn_full_random:
	cargo run --release --bin PacManNetwork -- nn

# Rede treinada, mapa aleatório
nn_map_random:
	cargo run --release --bin PacManNetwork -- nn best_gen_080.bin

# Rede treinada sobre o mapa dado
nn:
	cargo run --release --bin PacManNetwork -- nn best_gen_832.bin map.bin

test:
	cargo run --release --bin train

test_single:
	cargo run --release --bin train -- --single No newline at end of file