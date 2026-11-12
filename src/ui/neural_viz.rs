//! Painel de visualização da rede neural, desenhado abaixo do mapa.
//!
//! Layout: 4 SENSORES (D-Pad) → 16 HIDDEN (4×4) → 4 OUTPUT (D-Pad)
//! Textos de valor renderizados dentro dos quadrados com contraste dinâmico.

use macroquad::prelude::*;

use crate::constants::{NEURAL_PANEL_HEIGHT, WINDOW_WIDTH};
use crate::ai::neural::{NeuralController, HIDDEN_SIZE, OUTPUT_SIZE};

// ---------------------------------------------------------------------------
// Constantes de Layout
// ---------------------------------------------------------------------------

const CONTENT_Y: f32 = 90.0; // Centro vertical do painel

// Tamanhos dos quadrados
const DPAD_CELL: f32 = 38.0;
const DPAD_GAP: f32 = 8.0;
const HIDDEN_CELL: f32 = 24.0;

// Centros X das três seções
const INPUT_CX: f32 = 100.0;           // Centro da cruz de INPUT
const HIDDEN_X: f32 = 231.0;           // Topo-esquerda do grid HIDDEN
const OUTPUT_CX: f32 = WINDOW_WIDTH as f32 - 100.0; // Centro da cruz de OUTPUT

pub struct NeuralViz;

impl NeuralViz {
    pub fn draw(ctrl: &NeuralController, y0: f32) {
        // Fundo do painel
        draw_rectangle(
            0.0, y0, WINDOW_WIDTH as f32, NEURAL_PANEL_HEIGHT,
            Color::new(0.04, 0.04, 0.08, 1.0),
        );
        draw_line(
            0.0, y0, WINDOW_WIDTH as f32, y0, 2.0,
            Color::new(0.30, 0.30, 0.55, 1.0),
        );

        // Títulos
        let hdr = Color::new(0.75, 0.75, 0.85, 1.0);
        draw_text("SENSORS (4)", INPUT_CX - 45.0, y0 + 24.0, 16.0, hdr);
        draw_text("HIDDEN (16)", HIDDEN_X + 4.0, y0 + 24.0, 16.0, hdr);
        draw_text("OUTPUT (4)", OUTPUT_CX - 40.0, y0 + 24.0, 16.0, hdr);

        draw_input_dpad(ctrl, y0 + 10.0);
        draw_hidden(ctrl, y0 + 10.0);
        draw_output_dpad(ctrl, y0 + 10.0);
    }
}

// ---------------------------------------------------------------------------
// Helpers de Renderização
// ---------------------------------------------------------------------------

/// Desenha uma caixa com texto centralizado dentro.
/// Adapta a cor do texto (preto/branco) baseado no brilho da cor de fundo.
fn draw_square_with_text(x: f32, y: f32, size: f32, color: Color, text: &str, font_size: u16) {
    draw_rectangle(x, y, size, size, color);
    draw_rectangle_lines(x, y, size, size, 1.0, Color::new(1.0, 1.0, 1.0, 0.2));

    // Contraste dinâmico
    let brightness = (color.r + color.g + color.b) / 3.0;
    let text_color = if brightness > 0.5 { BLACK } else { WHITE };

    // Centralização do texto
    let dims = measure_text(text, None, font_size, 1.0);
    let tx = x + (size - dims.width) / 2.0;
    let ty = y + (size + dims.height) / 2.0 - 2.0; // -2 ajuste fino de baseline

    draw_text(text, tx, ty, font_size as f32, text_color);
}

/// Desenha os 4 valores em formato de D-Pad (Cruzeta).
/// A ordem do array de valores deve ser [Up, Down, Left, Right].
fn draw_d_pad(cx: f32, cy: f32, values: &[f32], is_output: bool) {
    let c = DPAD_CELL;
    let g = DPAD_GAP;
    let half = c / 2.0;

    // Offsets a partir do centro (cx, cy)
    let positions = [
        (cx - half, cy - half - g - c), // 0: U (Cima)
        (cx - half, cy - half + g + c), // 1: D (Baixo)
        (cx - half - g - c, cy - half), // 2: L (Esquerda)
        (cx - half + g + c, cy - half), // 3: R (Direita)
    ];

    for i in 0..4 {
        let (x, y) = positions[i];
        let val = values[i];

        let color = if is_output {
            // Output usa escala de amarelo/dourado
            let t = ((val + 2.0) / 4.0).clamp(0.0, 1.0); // Assumindo saída entre -2 e +2
            Color::new(0.2 + t * 0.8, 0.2 + t * 0.8, 0.0, 1.0)
        } else {
            // Input usa escala azul/vermelho
            heat_signed(val)
        };

        let text = if is_output {
            format!("{:.1}", val)
        } else {
            format!("{:+.1}", val)
        };

        draw_square_with_text(x, y, c, color, &text, 14);
    }
}

// ---------------------------------------------------------------------------
// Camadas
// ---------------------------------------------------------------------------

fn draw_input_dpad(ctrl: &NeuralController, y0: f32) {
    draw_d_pad(INPUT_CX, y0 + CONTENT_Y, &ctrl.last_input, false);
}

fn draw_hidden(ctrl: &NeuralController, y0: f32) {
    let hid_max = ctrl.last_hidden.iter().cloned().fold(0.0f32, f32::max).max(1e-3);
    
    for j in 0..HIDDEN_SIZE {
        let t = (ctrl.last_hidden[j] / hid_max).clamp(0.0, 1.0);
        let (row, col) = (j / 4, j % 4); // grade 4×4
        
        let x = HIDDEN_X + col as f32 * HIDDEN_CELL;
        let y = (y0 + CONTENT_Y) - (HIDDEN_CELL * 2.0) + row as f32 * HIDDEN_CELL;
        
        draw_rectangle(
            x, y, HIDDEN_CELL - 1.0, HIDDEN_CELL - 1.0,
            heat_positive(t),
        );
    }
}

fn draw_output_dpad(ctrl: &NeuralController, y0: f32) {
    // Descobre o vencedor (argmax)
    let mut argmax = 0usize;
    for k in 1..OUTPUT_SIZE {
        if ctrl.last_output[k] > ctrl.last_output[argmax] { argmax = k; }
    }

    // Desenha a cruzeta de output
    draw_d_pad(OUTPUT_CX, y0 + CONTENT_Y, &ctrl.last_output, true);

    // Desenha um destaque branco no vencedor (argmax)
    let c = DPAD_CELL;
    let g = DPAD_GAP;
    let half = c / 2.0;
    let positions = [
        (OUTPUT_CX - half, y0 + CONTENT_Y - half - g - c),
        (OUTPUT_CX - half, y0 + CONTENT_Y - half + g + c),
        (OUTPUT_CX - half - g - c, y0 + CONTENT_Y - half),
        (OUTPUT_CX - half + g + c, y0 + CONTENT_Y - half),
    ];
    
    let (ax, ay) = positions[argmax];
    draw_rectangle_lines(ax - 2.0, ay - 2.0, c + 4.0, c + 4.0, 2.0, WHITE);
}

// ---------------------------------------------------------------------------
// Cores (Heatmaps)
// ---------------------------------------------------------------------------

#[inline(always)]
fn heat_signed(t: f32) -> Color {
    if t >= 0.0 {
        // +1 = vermelho (pellet), 0 = cinza escuro
        Color::new(0.3 + t * 0.7, 0.1 * (1.0 - t), 0.1 * (1.0 - t), 1.0)
    } else {
        // -1 = azul (perigo), 0 = cinza escuro
        let u = -t;
        Color::new(0.1 * (1.0 - u), 0.1 * (1.0 - u), 0.3 + u * 0.7, 1.0)
    }
}

#[inline(always)]
fn heat_positive(t: f32) -> Color {
    // Verde neon para ativação dos neurônios ocultos
    Color::new(0.05 + t * 0.25, 0.10 + t * 0.85, 0.05 + t * 0.25, 1.0)
}