//! Módulo dedicado à formatação e exibição de logs do treinamento.

pub struct GenerationStats {
    pub generation: usize,
    pub best_fitness: f32,
    pub avg_fitness: f32,
    pub std_dev: f32,
    pub best_score: u32,
    /// Melhor fitness já visto em **toda a evolução** até esta geração.
    pub best_ever_fitness: f32,
    /// Geração em que o `best_ever_fitness` foi obtido.
    pub best_ever_generation: usize,
    /// Score associado ao `best_ever_fitness` (pellets × `PELLET_SCORE`).
    pub best_ever_score: u32,
    pub time_secs: f32,
}

// Larguras centralizadas em constantes nomeadas. Cabeçalho e linha usam
// **as mesmas** larguras — impossível desalinhar se as colunas novas
// forem adicionadas nos dois lugares.
//
// Ordem das colunas:
//   Gen | Best Fit | Avg Fit | StdDev | Score | Best-Ever | @Gen | Time(s)
const W_GEN:        usize = 3;
const W_FIT:        usize = 9;   // Best Fit, Avg Fit, Best-Ever
const W_STD:        usize = 8;   // StdDev
const W_SCORE:      usize = 7;   // Score
const W_GEN_REF:    usize = 4;   // @Gen (ex.: "G42")
const W_TIME:       usize = 8;   // Time(s)

impl GenerationStats {
    pub fn print_header() {
        let header = format!(
            "{:<wg$} | {:>wf$} | {:>wf$} | {:>ws$} | {:>wsc$} | {:>wf$} | {:>wr$} | {:>wt$}",
            "Gen", "Best Fit", "Avg Fit", "StdDev", "Score",
            "Best-Ever", "@Gen", "Time(s)",
            wg = W_GEN, wf = W_FIT, ws = W_STD, wsc = W_SCORE,
            wr = W_GEN_REF, wt = W_TIME,
        );
        println!("{}", header);
        println!("{}", "-".repeat(header.chars().count()));
    }

    pub fn print_row(&self) {
        // Formata `generation` já como string ("01") — evita misturar
        // `Display` de usize com um `String` temporário.
        let gen_str = format!("{:02}", self.generation);
        let gen_ref = format!("G{:02}", self.best_ever_generation);

        println!(
            "{:<wg$} | {:>+wf$.2} | {:>+wf$.2} | {:>ws$.2} | {:>wsc$} | {:>+wf$.2} | {:>wr$} | {:>wt$.2}",
            gen_str,
            self.best_fitness,
            self.avg_fitness,
            self.std_dev,
            self.best_score,
            self.best_ever_fitness,
            gen_ref,
            self.time_secs,
            wg = W_GEN, wf = W_FIT, ws = W_STD, wsc = W_SCORE,
            wr = W_GEN_REF, wt = W_TIME,
        );
    }
}