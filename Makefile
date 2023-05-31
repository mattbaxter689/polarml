format:
	cargo fmt

smartcore:
	cargo run -q -- smart

linfa: 
	cargo run -q -- linfa

describe:
	cargo run -q -- describe

info:
	cargo run -q -- info

help:
	cargo run -q -- -help

check:
	cargo check

fix:
	cargo fix
