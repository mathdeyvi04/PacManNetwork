game:
	cargo run --bin PacManNetwork

nn:
	cargo run --bin PacManNetwork nn

test:
	cargo run --bin train

test_single:
	cargo run --bin train -- --single