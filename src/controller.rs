use macroquad::prelude::*;
use crate::game::Game;
use crate::player::Direction;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Action {
    None,
    Move(Direction),
    Restart,
}

pub trait Controller {
    fn poll(&mut self, game: &Game) -> Action;
}

pub struct KeyboardController;

impl KeyboardController {
    #[inline]
    pub const fn new() -> Self { Self }
}

impl Controller for KeyboardController {
    #[inline]
    fn poll(&mut self, _game: &Game) -> Action {
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