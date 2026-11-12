//! Módulo dedicado à formatação e exibição de logs do treinamento.
//!
//! Centralizar a apresentação aqui permite adicionar novas métricas
//! sem precisar alterar a lógica do loop principal de treinamento.

pub struct GenerationStats {
    pub generation: usize,
    pub best_fitness: f32,
    pub avg_fitness: f32,
    pub best_score: u32,
    
    // ---------------------------------------------------------
    // FUTURO: Adicione novos campos de métricas aqui.
    // Exemplo:
    // pub avg_steps: f32,
    // pub death_rate: f32,
    // ---------------------------------------------------------
}

impl GenerationStats {
    /// Imprime o cabeçalho da tabela.
    pub fn print_header() {
        let header = format!(
            "{:<3} | {:>9} | {:>9} | {:>7}", 
            "Gen", "Best Fit", "Avg Fit", "Score"
        );
        println!("{}", header);
        // Usa .chars().count() para garantir o alinhamento correto com caracteres UTF-8
        println!("{}", "-".repeat(header.chars().count()));
    }

    /// Imprime uma linha formatada com os dados da geração atual.
    pub fn print_row(&self) {
        println!(
            "{:<3} | {:>+9.2} | {:>+9.2} | {:>7}",
            format!("{:02}", self.generation),
            self.best_fitness,
            self.avg_fitness,
            self.best_score
        );
    }
}