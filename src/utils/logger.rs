//! Módulo dedicado à formatação e exibição de logs do treinamento.

pub struct GenerationStats {
    pub generation: usize,
    pub best_fitness: f32,
    pub avg_fitness: f32,
    pub best_score: u32,
    /// Melhor fitness já visto em **toda a evolução** até esta geração.
    pub best_ever_fitness: f32,
    /// Geração em que o `best_ever_fitness` foi obtido.
    pub best_ever_generation: usize,
    /// Score associado ao `best_ever_fitness` (pellets × `PELLET_SCORE`).
    pub best_ever_score: u32,
    pub time_secs: f32,
}

impl GenerationStats {
    pub fn print_header() {
        let header = format!(
            "{:<3} | {:>9} | {:>9} | {:>7} | {:>9} | {:>4} | {:>8}",
            "Gen", "Best Fit", "Avg Fit", "Score", "Best-Ever", "@Gen", "Time(s)"
        );
        println!("{}", header);
        println!("{}", "-".repeat(header.chars().count()));
    }

    pub fn print_row(&self) {
        println!(
            "{:<3} | {:>+9.2} | {:>+9.2} | {:>7} | {:>+9.2} | {:>4} | {:>8.2}",
            format!("{:02}", self.generation),
            self.best_fitness,
            self.avg_fitness,
            self.best_score,
            self.best_ever_fitness,
            format!("G{:02}", self.best_ever_generation),
            self.time_secs
        );
    }
}