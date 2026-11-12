//! Painel de visualização da rede neural, desenhado abaixo do mapa.
//!
//! Três colunas centradas no painel:
//!   • INPUT  — visão 7×7 (49) + estado 8 (direção/ghost/pellet).
//!   • HIDDEN — 64 neurônios em grade 8×8.
//!   • OUTPUT — 4 neurônios (U/D/L/R), argmax destacado.
//!
//! Sem linhas de conexão — só ativações. Performance e legibilidade.

use macroquad::prelude::*;

use crate::constants::{NEURAL_PANEL_HEIGHT, WINDOW_WIDTH};
use crate::neural::{NeuralController, HIDDEN_SIZE, OUTPUT_SIZE};

// Layout — alinhado e centrado no painel de 260 px.
const CONTENT_Y: f32 = 40.0;

const VISION_X:    f32 = 80.0;
const VISION_CELL: f32 = 16.0;
const STATE_X:     f32 = VISION_X + 7.0 * VISION_CELL + 16.0; // 208
const STATE_W:     f32 = 20.0;
const STATE_STEP:  f32 = 24.0;

const HIDDEN_X:    f32 = STATE_X + STATE_W + 40.0; // 268
const HIDDEN_CELL: f32 = 18.0;

const OUTPUT_X:    f32 = HIDDEN_X + 8.0 * HIDDEN_CELL + 40.0; // 452
const OUTPUT_W:    f32 = 24.0;
const OUTPUT_STEP: f32 = 44.0;

pub struct NeuralViz;

impl NeuralViz {
    /// Desenha o painel. `y0` = topo do painel (normalmente `WINDOW_HEIGHT`).
    pub fn draw(ctrl: &NeuralController, y0: f32) {
        draw_rectangle(
            0.0, y0, WINDOW_WIDTH as f32, NEURAL_PANEL_HEIGHT,
            Color::new(0.04, 0.04, 0.08, 1.0),
        );
        draw_line(
            0.0, y0, WINDOW_WIDTH as f32, y0, 2.0,
            Color::new(0.30, 0.30, 0.55, 1.0),
        );

        let in_max = ctrl.last_input.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-3);
        let hid_max = ctrl.last_hidden.iter().cloned().fold(0.0f32, f32::max).max(1e-3);
        let (out_min, out_max) = ctrl.last_output.iter().fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(lo, hi), &v| (lo.min(v), hi.max(v)),
        );
        let out_range = (out_max - out_min).max(1e-3);

        draw_input_vision(ctrl, y0, in_max);
        draw_input_state(ctrl, y0, in_max);
        draw_hidden(ctrl, y0, hid_max);
        draw_output(ctrl, y0, out_min, out_range);

        let hdr = Color::new(0.75, 0.75, 0.85, 1.0);
        draw_text("INPUT (57)",  114.0, y0 + 24.0, 16.0, hdr);
        draw_text("HIDDEN (64)", 293.0, y0 + 24.0, 16.0, hdr);
        draw_text("OUTPUT (4)",  424.0, y0 + 24.0, 16.0, hdr);
    }
}

// ---------------------------------------------------------------------------
// Camadas
// ---------------------------------------------------------------------------

fn draw_input_vision(ctrl: &NeuralController, y0: f32, in_max: f32) {
    for i in 0..49 {
        let t = (ctrl.last_input[i] / in_max).clamp(-1.0, 1.0);
        let (row, col) = (i / 7, i % 7);
        let x = VISION_X + col as f32 * VISION_CELL;
        let y = y0 + CONTENT_Y + row as f32 * VISION_CELL;
        draw_rectangle(x, y, VISION_CELL - 1.0, VISION_CELL - 1.0, heat_signed(t));
    }
}

fn draw_input_state(ctrl: &NeuralController, y0: f32, in_max: f32) {
    for rel in 0..8 {
        let t = (ctrl.last_input[49 + rel] / in_max).clamp(-1.0, 1.0);
        let x = STATE_X;
        let y = y0 + CONTENT_Y + rel as f32 * STATE_STEP;
        draw_rectangle(x, y, STATE_W, STATE_W, heat_signed(t));
    }
}

fn draw_hidden(ctrl: &NeuralController, y0: f32, hid_max: f32) {
    for j in 0..HIDDEN_SIZE {
        let t = (ctrl.last_hidden[j] / hid_max).clamp(0.0, 1.0);
        let (row, col) = (j / 8, j % 8);
        let x = HIDDEN_X + col as f32 * HIDDEN_CELL;
        let y = y0 + CONTENT_Y + row as f32 * HIDDEN_CELL;
        draw_rectangle(x, y, HIDDEN_CELL - 1.0, HIDDEN_CELL - 1.0, heat_positive(t));
    }
}

fn draw_output(ctrl: &NeuralController, y0: f32, out_min: f32, out_range: f32) {
    let mut argmax = 0usize;
    for k in 1..OUTPUT_SIZE {
        if ctrl.last_output[k] > ctrl.last_output[argmax] { argmax = k; }
    }
    const LABELS: [&str; 4] = ["U", "D", "L", "R"];
    for k in 0..OUTPUT_SIZE {
        let a = ctrl.last_output[k];
        let t = ((a - out_min) / out_range).clamp(0.0, 1.0);
        let c = Color::new(0.55 + t * 0.45, 0.55 + t * 0.45, 0.0, 1.0);
        let x = OUTPUT_X;
        let y = y0 + CONTENT_Y + k as f32 * OUTPUT_STEP;
        draw_rectangle(x, y, OUTPUT_W, OUTPUT_W, c);
        if k == argmax {
            draw_rectangle_lines(x - 2.0, y - 2.0, OUTPUT_W + 4.0, OUTPUT_W + 4.0, 2.0, WHITE);
        }
        draw_text(LABELS[k], x + OUTPUT_W + 8.0, y + OUTPUT_W - 4.0, 20.0, WHITE);
        let s = format!("{:.2}", a);
        draw_text(&s, x + OUTPUT_W + 8.0, y + OUTPUT_W + 14.0, 12.0, Color::new(0.6, 0.6, 0.7, 1.0));
    }
}

// ---------------------------------------------------------------------------
// Cores
// ---------------------------------------------------------------------------

#[inline(always)]
fn heat_signed(t: f32) -> Color {
    if t >= 0.0 {
        Color::new(0.08 + t * 0.92, 0.08 * (1.0 - t), 0.08 * (1.0 - t), 1.0)
    } else {
        let u = -t;
        Color::new(0.08 * (1.0 - u), 0.08 * (1.0 - u), 0.08 + u * 0.92, 1.0)
    }
}

#[inline(always)]
fn heat_positive(t: f32) -> Color {
    Color::new(0.05 + t * 0.25, 0.10 + t * 0.85, 0.05 + t * 0.25, 1.0)
}