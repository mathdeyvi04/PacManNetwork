//! Módulo dedicado à formatação e exibição de logs do treinamento.

pub struct GenerationStats {
    pub generation: usize,
    pub best_fitness: f32,
    pub avg_fitness: f32,
    pub best_score: u32,
    pub time_secs: f32, // NOVO: Tempo gasto na geração
}

impl GenerationStats {
    /// Imprime o cabeçalho da tabela.
    pub fn print_header() {
        let header = format!(
            "{:<3} | {:>9} | {:>9} | {:>7} | {:>8}", 
            "Gen", "Best Fit", "Avg Fit", "Score", "Time(s)"
        );
        println!("{}", header);
        println!("{}", "-".repeat(header.chars().count()));
    }

    /// Imprime uma linha formatada com os dados da geração atual.
    pub fn print_row(&self) {
        println!(
            "{:<3} | {:>+9.2} | {:>+9.2} | {:>7} | {:>8.2}",
            format!("{:02}", self.generation),
            self.best_fitness,
            self.avg_fitness,
            self.best_score,
            self.time_secs
        );
    }
}