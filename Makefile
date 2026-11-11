game:
	cargo run --bin PacManNetwork

test:
	cargo run --bin train

nn:
	cargo run --bin PacManNetwork -- nn best_pacman.bin