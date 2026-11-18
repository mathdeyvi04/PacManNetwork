//! Painel de visualização da rede neural, desenhado abaixo do mapa.
//!
//! Layout atualizado para a arquitetura expandida (22 inputs → 64 hidden → 4 output):
//!
//!   INPUTS DE SENSORIAL (22 total):
//!     • D-Pad Aninhado (8 sensores): Eventos externos e distâncias.
//!     • Grade de Fantasmas (8 sensores): Distâncias e ângulos de Blinky, Pinky, Inky, Clyde.
//!     • Gradiente de Pellets (2 sensores): Atração para clusters de pellets (X e Y).
//!     • Memória Recorrente (4 neurônios): Estado de contexto persistido entre frames.
//!
//!   HIDDEN — Omitido/Sacrificado visualmente para priorizar a clareza e o espaço dos inputs.
//!   OUTPUT — 4 ações em D-Pad (direções).

use macroquad::prelude::*;

use crate::constants::{NEURAL_PANEL_HEIGHT, WINDOW_WIDTH};
use crate::ai::neural::{NeuralController, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE};

// ---------------------------------------------------------------------------
// Constantes de Layout
// ---------------------------------------------------------------------------

const CONTENT_Y: f32 = 90.0;

// ---- Raycast D-Pad (bloco 1 do input: 0..8) -------------------------------
const DPAD_CELL: f32 = 32.0;
const DPAD_GAP:  f32 = 6.0;

const INNER_CELL: f32 = 14.0;
const INNER_GAP:  f32 = 1.0;

// ---- Ghost block (bloco 2 do input: 8..16) --------------------------------
const GHOST_COLS: usize = 4;       
const GHOST_ROWS: usize = 2;       
const GHOST_CELL: f32   = 20.0;
const GHOST_GAP:  f32   = 2.0;
const GHOST_BLOCK_W: f32 =
    GHOST_COLS as f32 * GHOST_CELL + (GHOST_COLS - 1) as f32 * GHOST_GAP;
const GHOST_BLOCK_H: f32 =
    GHOST_ROWS as f32 * GHOST_CELL + (GHOST_ROWS - 1) as f32 * GHOST_GAP;

// ---- Pellets & Memory block (blocos 3 e 4 do input: 16..22) --------------
const SMALL_CELL: f32 = 18.0;

// Garantias em tempo de compilação
const _: () = assert!(INPUT_SIZE  == 22, "viz de input assumindo 22 sensores");
const _: () = assert!(HIDDEN_SIZE == 64, "viz assume HIDDEN_SIZE == 64");

// Centros X das seções reorganizadas para acomodar os novos inputs com clareza.
const INPUT_RAYS_CX: f32 = 65.0;
const GHOST_LX:      f32 = 135.0;
const GHOST_CX:      f32 = GHOST_LX + GHOST_BLOCK_W * 0.5;
const EXTRA_CX:      f32 = 245.0; // Bloco unificado para Gradiente de Pellets + Memória
const OUTPUT_CX:     f32 = WINDOW_WIDTH as f32 - 80.0;

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

        // Títulos das seções
        let hdr = Color::new(0.75, 0.75, 0.85, 1.0);
        draw_text_centered("RAYS & GHOSTS (16)", 150.0, y0 + 22.0, 14.0, hdr);
        draw_text_centered("CLUSTERS & MEMORY (6)", EXTRA_CX, y0 + 22.0, 14.0, hdr);
        draw_text_centered("OUTPUT (4)", OUTPUT_CX, y0 + 22.0, 14.0, hdr);

        draw_input_nested(ctrl, y0 + 10.0);
        draw_input_ghosts(ctrl, y0 + 10.0);
        draw_extra_sensors(ctrl, y0 + 10.0); // Gradiente de Pellets + Memória
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

fn draw_square_with_text(x: f32, y: f32, size: f32, color: Color, text: &str, font_size: u16) {
    draw_rectangle(x, y, size, size, color);
    draw_rectangle_lines(x, y, size, size, 1.0, Color::new(1.0, 1.0, 1.0, 0.2));

    let brightness = (color.r + color.g + color.b) / 3.0;
    let text_color = if brightness > 0.5 { BLACK } else { WHITE };

    let dims = measure_text(text, None, font_size, 1.0);
    let tx = x + (size - dims.width) * 0.5;
    let ty = y + (size + dims.height) * 0.5 - 2.0;
    draw_text(text, tx, ty, font_size as f32, text_color);
}

// ---------------------------------------------------------------------------
// Camada de Entrada — D-Pad aninhado (0..8)
// ---------------------------------------------------------------------------

fn draw_input_nested(ctrl: &NeuralController, y0: f32) {
    let cx = INPUT_RAYS_CX;
    let cy = y0 + CONTENT_Y;
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
        draw_square_with_text(x, y, c, color, &format!("{:+.1}", val), 11);
    }

    let ic = INNER_CELL;
    let ig = INNER_GAP;
    let ihalf = ic / 2.0;

    let inner_positions = [
        (cx - ihalf, cy - ihalf - ig - ic),
        (cx - ihalf, cy - ihalf + ig + ic),
        (cx - ihalf - ig - ic, cy - ihalf),
        (cx - ihalf + ig + ic, cy - ihalf),
    ];

    for i in 0..4 {
        let (x, y) = inner_positions[i];
        let val = ctrl.last_input[4 + i];
        let t = (val * 2.0).clamp(0.0, 1.0);
        let color = heat_distance(t);
        draw_square_with_text(x, y, ic, color, &format!("{:.1}", val), 8);
    }

    let cap = Color::new(0.55, 0.55, 0.65, 1.0);
    draw_text_centered("rays: event/dist", INPUT_RAYS_CX, y0 + 172.0, 9.0, cap);
}

// ---------------------------------------------------------------------------
// Camada de Entrada — Fantasmas (8..16)
// ---------------------------------------------------------------------------

fn draw_input_ghosts(ctrl: &NeuralController, y0: f32) {
    let cy = y0 + CONTENT_Y;
    let cx = GHOST_CX;
    let ox = cx - GHOST_BLOCK_W * 0.5;
    let oy = cy - GHOST_BLOCK_H * 0.5;

    let hdr = Color::new(0.65, 0.65, 0.75, 1.0);
    let names = ["B", "P", "I", "C"];
    for i in 0..GHOST_COLS {
        let col_cx = ox + i as f32 * (GHOST_CELL + GHOST_GAP) + GHOST_CELL * 0.5;
        draw_text_centered(names[i], col_cx, oy - 6.0, 9.0, hdr);
    }

    // Linha 0: Distâncias (8..12)
    for i in 0..GHOST_COLS {
        let x = ox + i as f32 * (GHOST_CELL + GHOST_GAP);
        let val = ctrl.last_input[8 + i];
        let t = (val * 2.0).clamp(0.0, 1.0);
        draw_rectangle(x, oy, GHOST_CELL, GHOST_CELL, heat_distance(t));
        draw_rectangle_lines(x, oy, GHOST_CELL, GHOST_CELL, 1.0, Color::new(1.0, 1.0, 1.0, 0.15));
    }

    // Linha 1: Ângulos (12..16)
    let y_ang = oy + GHOST_CELL + GHOST_GAP;
    for i in 0..GHOST_COLS {
        let x = ox + i as f32 * (GHOST_CELL + GHOST_GAP);
        let val = ctrl.last_input[12 + i];
        draw_rectangle(x, y_ang, GHOST_CELL, GHOST_CELL, heat_signed(val));
        draw_rectangle_lines(x, y_ang, GHOST_CELL, GHOST_CELL, 1.0, Color::new(1.0, 1.0, 1.0, 0.15));
    }

    let cap = Color::new(0.55, 0.55, 0.65, 1.0);
    draw_text_centered("ghosts: dist/angle", GHOST_CX, y0 + 172.0, 9.0, cap);
}

// ---------------------------------------------------------------------------
// Novos Sensores: Gradiente de Pellets (16..18) e Memória Recorrente (18..22)
// ---------------------------------------------------------------------------

fn draw_extra_sensors(ctrl: &NeuralController, y0: f32) {
    let cy = y0 + CONTENT_Y;
    let sc = SMALL_CELL;
    let gap = 4.0;

    // -- Bloco 1: Gradiente de Pellets (2 células lado a lado: X e Y) --
    let p_ox = EXTRA_CX - (2.0 * sc + gap) * 0.5;
    let p_oy = cy - sc * 0.5 - 12.0;

    draw_text_centered("Pellet Vector", EXTRA_CX, p_oy - 10.0, 9.0, Color::new(0.65, 0.65, 0.75, 1.0));

    let pellet_labels = ["X", "Y"];
    for i in 0..2 {
        let x = p_ox + i as f32 * (sc + gap);
        let val = ctrl.last_input[16 + i];
        let color = heat_signed(val);
        draw_square_with_text(x, p_oy, sc, color, pellet_labels[i], 10);
    }

    // -- Bloco 2: Memória Recorrente (4 células em linha: Contexto 0..3) --
    let m_ox = EXTRA_CX - (4.0 * sc + 3.0 * gap) * 0.5;
    let m_oy = cy + sc * 0.5 + 8.0;

    draw_text_centered("Memory Context", EXTRA_CX, m_oy - 4.0, 9.0, Color::new(0.65, 0.65, 0.75, 1.0));

    for i in 0..4 {
        let x = m_ox + i as f32 * (sc + gap);
        let val = ctrl.last_input[18 + i];
        let t = ((val + 1.0) * 0.5).clamp(0.0, 1.0);
        let color = Color::new(0.1 + t * 0.4, 0.1, 0.2 + t * 0.8, 1.0); // Tons de roxo/magenta
        draw_square_with_text(x, m_oy, sc, color, &format!("{}", i), 9);
    }

    let cap = Color::new(0.55, 0.55, 0.65, 1.0);
    draw_text_centered("clusters & memory state", EXTRA_CX, y0 + 172.0, 9.0, cap);
}

// ---------------------------------------------------------------------------
// Camada de Saída — D-Pad de Ações (4 saídas)
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

    let mut argmax = 0usize;
    for k in 1..OUTPUT_SIZE {
        if ctrl.last_output[k] > ctrl.last_output[argmax] { argmax = k; }
    }

    for i in 0..OUTPUT_SIZE {
        let (x, y) = positions[i];
        let val = ctrl.last_output[i];
        let t = ((val + 2.0) / 4.0).clamp(0.0, 1.0);
        let color = Color::new(0.2 + t * 0.8, 0.2 + t * 0.8, 0.0, 1.0);
        draw_square_with_text(x, y, c, color, &format!("{:.1}", val), 11);
    }

    let (ax, ay) = positions[argmax];
    draw_rectangle_lines(ax - 2.0, ay - 2.0, c + 4.0, c + 4.0, 2.0, WHITE);

    let cap = Color::new(0.55, 0.55, 0.65, 1.0);
    draw_text_centered("actions (argmax)", OUTPUT_CX, y0 + 172.0, 9.0, cap);
}

// ---------------------------------------------------------------------------
// Cores (Heatmaps)
// ---------------------------------------------------------------------------

#[inline(always)]
fn heat_signed(t: f32) -> Color {
    if t >= 0.0 {
        Color::new(0.3 + t * 0.7, 0.1 * (1.0 - t), 0.1 * (1.0 - t), 1.0)
    } else {
        let u = -t;
        Color::new(0.1 * (1.0 - u), 0.1 * (1.0 - u), 0.3 + u * 0.7, 1.0)
    }
}

#[inline(always)]
fn heat_distance(t: f32) -> Color {
    Color::new(
        0.02 + t * 0.10,
        0.15 + t * 0.60,
        0.20 + t * 0.70,
        1.0,
    )
}