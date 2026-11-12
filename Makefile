# Humano jogando normalmente
game:
	cargo run --bin PacManNetwork

# Humano jogando com mapa carregado
game_with_map:
	cargo run --bin PacManNetwork -- human map.bin

# Rede aleatoria, mapa aleatório
nn_full_random:
	cargo run --bin PacManNetwork -- nn

# Rede treinada, mapa aleatório
nn_map_random:
	cargo run -- bin PacManNetwork -- nn pesos.bin

# Rede treinada sobre o mapa dado
nn:
	cargo run -- bin PacManNetwork -- nn pesos.bin map.bin

test:
	cargo run --bin train

test_single:
	cargo run --bin train -- --single