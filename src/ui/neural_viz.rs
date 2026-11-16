//! Painel de visualização da rede neural, desenhado abaixo do mapa.
//!
//! Layout:
//!   INPUT  — 8 sensores como D-Pad **aninhado**
//!            • anel externo: evento por direção (fantasma/parede/túnel/pellet/nada)
//!            • anel interno: distância normalizada ao evento (0.0 ..= 0.5)
//!   HIDDEN — 32 neurônios em grade 8 × 4
//!   OUTPUT — 4 ações em D-Pad
//!
//! O aninhamento preserva a leitura espacial: olhando para o quadrado
//! "esquerda", o externo diz o que há lá e o interno diz a que distância.
//! Diferente de intercalar os 8 em uma única fileira, isso mantém a
//! associação evento↔distância imediata, sem precisar de rótulos extras.

use macroquad::prelude::*;

use crate::constants::{NEURAL_PANEL_HEIGHT, WINDOW_WIDTH};
use crate::ai::neural::{NeuralController, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE};

// ---------------------------------------------------------------------------
// Constantes de Layout
// ---------------------------------------------------------------------------

const CONTENT_Y: f32 = 90.0;

// D-Pad externo (eventos) — mesmas dimensões do layout antigo.
const DPAD_CELL: f32 = 38.0;
const DPAD_GAP:  f32 = 8.0;

// D-Pad interno (distâncias) — cabe no vão central do D-Pad externo:
//   vão = 2 * (half + gap) = 2 * (19 + 8) = 54
//   inner = 2 * (INNER_CELL/2 + INNER_GAP + INNER_CELL) = 2 * (8 + 1 + 16) = 50
// Sobram 2 px de folga em cada lado.
const INNER_CELL: f32 = 16.0;
const INNER_GAP:  f32 = 1.0;

// Camada oculta: grade 8 × 4 = 32 neurônios.
const HIDDEN_COLS: usize = 8;
const HIDDEN_ROWS: usize = 4;
const HIDDEN_CELL: f32  = 18.0;

// Garantias em tempo de compilação — falham se neural.rs mudar sem
// a visualização ser atualizada junto.
const _: () = assert!(INPUT_SIZE  == 8, "viz de input assumindo 8 sensores");
const _: () = assert!(HIDDEN_SIZE == HIDDEN_COLS * HIDDEN_ROWS, "grade ≠ HIDDEN_SIZE");

// Centros X das três seções.
const INPUT_CX:  f32 = 100.0;
const HIDDEN_X:  f32 = 231.0;
const OUTPUT_CX: f32 = WINDOW_WIDTH as f32 - 100.0;

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

        // Títulos (centralizados por measure_text — resiliente a mudar o texto).
        let hdr = Color::new(0.75, 0.75, 0.85, 1.0);
        draw_text_centered("SENSORS (8)", INPUT_CX, y0 + 24.0, 16.0, hdr);
        draw_text_centered(
            "HIDDEN (32)",
            HIDDEN_X + HIDDEN_COLS as f32 * HIDDEN_CELL * 0.5,
            y0 + 24.0, 16.0, hdr,
        );
        draw_text_centered("OUTPUT (4)", OUTPUT_CX, y0 + 24.0, 16.0, hdr);

        draw_input_nested(ctrl, y0 + 10.0);
        draw_hidden(ctrl, y0 + 10.0);
        draw_output_dpad(ctrl, y0 + 10.0);
    }
}

// ---------------------------------------------------------------------------
// Helpers de Renderização
// ---------------------------------------------------------------------------

fn draw_text_centered(text: &str, cx: f32, y: f32, size: f32, color: Color) {
    let dims = measure_text(text, None, size as u16, 1.0);
    draw_text(text, cx - dims.width * 0.5, y, size, color);
}

/// Desenha uma caixa com texto centralizado dentro.
/// Adapta a cor do texto (preto/branco) ao brilho da cor de fundo.
fn draw_square_with_text(x: f32, y: f32, size: f32, color: Color, text: &str, font_size: u16) {
    draw_rectangle(x, y, size, size, color);
    draw_rectangle_lines(x, y, size, size, 1.0, Color::new(1.0, 1.0, 1.0, 0.2));

    let brightness = (color.r + color.g + color.b) / 3.0;
    let text_color = if brightness > 0.5 { BLACK } else { WHITE };

    let dims = measure_text(text, None, font_size as u16, 1.0);
    let tx = x + (size - dims.width) * 0.5;
    let ty = y + (size + dims.height) * 0.5 - 2.0;
    draw_text(text, tx, ty, font_size as f32, text_color);
}

// ---------------------------------------------------------------------------
// Camada de Entrada — D-Pad aninhado (evento externo, distância interna)
// ---------------------------------------------------------------------------

fn draw_input_nested(ctrl: &NeuralController, y0: f32) {
    let cx = INPUT_CX;
    let cy = y0 + CONTENT_Y;

    // ---- Anel externo: eventos (índices 0..4 de last_input) ----
    // Ordem esperada em last_input: [Up, Down, Left, Right].
    let c = DPAD_CELL;
    let g = DPAD_GAP;
    let half = c / 2.0;

    let outer_positions = [
        (cx - half, cy - half - g - c), // Up
        (cx - half, cy - half + g + c), // Down
        (cx - half - g - c, cy - half), // Left
        (cx - half + g + c, cy - half), // Right
    ];

    for i in 0..4 {
        let (x, y) = outer_positions[i];
        let val = ctrl.last_input[i];
        let color = heat_signed(val);
        let text = format!("{:+.1}", val);
        draw_square_with_text(x, y, c, color, &text, 14);
    }

    // ---- Anel interno: distâncias (índices 4..8 de last_input) ----
    // Mesma ordem. Faixa: (0.0, 0.5]. Reamostra-se t=2·val p/ heatmap (0..1).
    let ic = INNER_CELL;
    let ig = INNER_GAP;
    let ihalf = ic / 2.0;

    let inner_positions = [
        (cx - ihalf, cy - ihalf - ig - ic), // Up
        (cx - ihalf, cy - ihalf + ig + ic), // Down
        (cx - ihalf - ig - ic, cy - ihalf), // Left
        (cx - ihalf + ig + ic, cy - ihalf), // Right
    ];

    for i in 0..4 {
        let (x, y) = inner_positions[i];
        let val = ctrl.last_input[4 + i];
        let t = (val * 2.0).clamp(0.0, 1.0);
        let color = heat_distance(t);
        let text = format!("{:.1}", val); // "0.5", "0.3", "0.0"
        draw_square_with_text(x, y, ic, color, &text, 9);
    }

    // Legenda curta: deixa explícito qual anel é qual. Posicionada
    // abaixo do D-Pad, ainda dentro do painel de 180 px (compatível
    // com a altura antiga) e com folga confortável em 240 px.
    let cap = Color::new(0.55, 0.55, 0.65, 1.0);
    draw_text_centered("outer: event",    INPUT_CX, y0 + 168.0, 10.0, cap);
    draw_text_centered("inner: distance", INPUT_CX, y0 + 180.0, 10.0, cap);
}

// ---------------------------------------------------------------------------
// Camada Oculta — grade 8 × 4 (32 neurônios)
// ---------------------------------------------------------------------------

fn draw_hidden(ctrl: &NeuralController, y0: f32) {
    let hid_max = ctrl.last_hidden.iter().cloned().fold(0.0f32, f32::max).max(1e-3);

    let grid_h = HIDDEN_ROWS as f32 * HIDDEN_CELL;
    let origin_y = (y0 + CONTENT_Y) - grid_h * 0.5;

    for j in 0..HIDDEN_SIZE {
        let row = j / HIDDEN_COLS;
        let col = j % HIDDEN_COLS;
        let t = (ctrl.last_hidden[j] / hid_max).clamp(0.0, 1.0);

        let x = HIDDEN_X + col as f32 * HIDDEN_CELL;
        let y = origin_y + row as f32 * HIDDEN_CELL;

        draw_rectangle(
            x, y, HIDDEN_CELL - 1.0, HIDDEN_CELL - 1.0,
            heat_positive(t),
        );
    }
}

// ---------------------------------------------------------------------------
// Camada de Saída — D-Pad simples (inalterado)
// ---------------------------------------------------------------------------

fn draw_output_dpad(ctrl: &NeuralController, y0: f32) {
    let cx = OUTPUT_CX;
    let cy = y0 + CONTENT_Y;
    let c = DPAD_CELL;
    let g = DPAD_GAP;
    let half = c / 2.0;

    let positions = [
        (cx - half, cy - half - g - c),
        (cx - half, cy - half + g + c),
        (cx - half - g - c, cy - half),
        (cx - half + g + c, cy - half),
    ];

    // Descobre o vencedor (argmax) para destacar.
    let mut argmax = 0usize;
    for k in 1..OUTPUT_SIZE {
        if ctrl.last_output[k] > ctrl.last_output[argmax] { argmax = k; }
    }

    for i in 0..OUTPUT_SIZE {
        let (x, y) = positions[i];
        let val = ctrl.last_output[i];
        // Saídas brutas (ReLU, sem sigmoid): mapeia [-2, +2] → [0, 1].
        let t = ((val + 2.0) / 4.0).clamp(0.0, 1.0);
        let color = Color::new(0.2 + t * 0.8, 0.2 + t * 0.8, 0.0, 1.0);
        draw_square_with_text(x, y, c, color, &format!("{:.1}", val), 14);
    }

    // Destaque branco no vencedor.
    let (ax, ay) = positions[argmax];
    draw_rectangle_lines(ax - 2.0, ay - 2.0, c + 4.0, c + 4.0, 2.0, WHITE);
}

// ---------------------------------------------------------------------------
// Cores (Heatmaps)
// ---------------------------------------------------------------------------

/// Evento com sinal: negativo → azul (perigo), positivo → vermelho (pellet).
#[inline(always)]
fn heat_signed(t: f32) -> Color {
    if t >= 0.0 {
        Color::new(0.3 + t * 0.7, 0.1 * (1.0 - t), 0.1 * (1.0 - t), 1.0)
    } else {
        let u = -t;
        Color::new(0.1 * (1.0 - u), 0.1 * (1.0 - u), 0.3 + u * 0.7, 1.0)
    }
}

/// Ativação dos ocultos: verde-neon. Mantido igual.
#[inline(always)]
fn heat_positive(t: f32) -> Color {
    Color::new(0.05 + t * 0.25, 0.10 + t * 0.85, 0.05 + t * 0.25, 1.0)
}

/// Distância: escuro (longe) → teal brilhante (perto).
///
/// A paleta teal/ciano foi escolhida para ser visualmente **distinta**:
///   - do vermelho/azul do `heat_signed`  (eventos)
///   - do verde-neon do `heat_positive`   (ocultos)
///   - do dourado da camada de saída
/// Assim o anel interno nunca é confundido com um anel externo desbotado.
#[inline(always)]
fn heat_distance(t: f32) -> Color {
    Color::new(
        0.02 + t * 0.10,   // R: ~0.02 → 0.12
        0.15 + t * 0.60,   // G:  0.15 → 0.75
        0.20 + t * 0.70,   // B:  0.20 → 0.90
        1.0,
    )
}