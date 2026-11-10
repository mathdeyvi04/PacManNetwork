use macroquad::prelude::*;
use crate::player::Direction;

/// Comando atômico produzido por um Controller e consumido pela Engine.
/// `Copy` + 2 bytes → passado por valor, sem alocação em nenhum caminho.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Action {
    None,
    Move(Direction),
    Restart,
}

/// Contrato mínimo: 1 polling por frame. Sem alocação, sem `dyn` no hot loop.
pub trait Controller {
    fn poll(&mut self) -> Action;
}

/// Zero-sized: `new()` é `const`, o `&self` de `poll` não gera código.
pub struct KeyboardController;

impl KeyboardController {
    #[inline]
    pub const fn new() -> Self { Self }
}

impl Controller for KeyboardController {
    #[inline]
    fn poll(&mut self) -> Action {
        if is_key_pressed(KeyCode::Right) || is_key_pressed(KeyCode::D) {
            return Action::Move(Direction::Right);
        }
        if is_key_pressed(KeyCode::Left) || is_key_pressed(KeyCode::A) {
            return Action::Move(Direction::Left);
        }
        if is_key_pressed(KeyCode::Up) || is_key_pressed(KeyCode::W) {
            return Action::Move(Direction::Up);
        }
        if is_key_pressed(KeyCode::Down) || is_key_pressed(KeyCode::S) {
            return Action::Move(Direction::Down);
        }
        if is_key_pressed(KeyCode::Space) {
            return Action::Restart;
        }
        Action::None
    }
}