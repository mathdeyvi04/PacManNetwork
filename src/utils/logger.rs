//! Módulo dedicado à formatação e exibição de logs do treinamento.

use std::fs::File;
use std::io::{self, BufWriter, Write};

// ---------------------------------------------------------------------------
// Abstração de Log
// ---------------------------------------------------------------------------

/// Nome fixo do arquivo de log. Cada execução do treinamento **recria** este
/// arquivo (via `File::create`, que trunca o conteúdo anterior), garantindo
/// que ele reflita sempre o último treinamento realizado.
pub const DEFAULT_LOG_PATH: &str = "train_historic.txt";

pub struct Logger {
    out: BufWriter<Box<dyn Write>>,
}

impl Logger {
    /// Cria (ou trunca) o arquivo de log.
    ///
    /// - `Some(p)` → usa o caminho informado (útil para testes ou uso
    ///   programático).
    /// - `None`    → usa sempre `DEFAULT_LOG_PATH` (`train_historic.txt`).
    ///
    /// Em ambos os casos o arquivo é **recriado do início**, de modo que
    /// cada execução sobrescreve a anterior.
    pub fn new(file_path: Option<&str>) -> io::Result<Self> {
        let path = file_path.unwrap_or(DEFAULT_LOG_PATH);
        let out: Box<dyn Write> = Box::new(File::create(path)?);
        Ok(Self {
            out: BufWriter::new(out),
        })
    }
}

impl Write for Logger {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.out.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.out.flush()
    }
}

/// Macro para facilitar o uso do `Logger` sem precisar de `let _ = writeln!(...)` o tempo todo.
#[macro_export]
macro_rules! logln {
    ($logger:expr, $($arg:tt)*) => {
        {
            use std::io::Write;
            let _ = writeln!($logger, $($arg)*);
        }
    };
}

// ---------------------------------------------------------------------------
// Estatísticas de Geração
// ---------------------------------------------------------------------------

pub struct GenerationStats {
    pub generation: usize,
    pub best_fitness: f32,
    pub avg_fitness: f32,
    pub std_dev: f32,
    pub best_score: u32,
    pub best_ever_fitness: f32,
    pub best_ever_generation: usize,
    pub best_ever_score: u32,
    pub time_secs: f32,
}

const W_GEN:        usize = 4;
const W_FIT:        usize = 9;
const W_STD:        usize = 8;
const W_SCORE:      usize = 7;
const W_GEN_REF:    usize = 4;
const W_TIME:       usize = 8;

impl GenerationStats {
    pub fn print_header<W: Write>(w: &mut W) -> io::Result<()> {
        let header = format!(
            "{:<wg$} | {:>wf$} | {:>wf$} | {:>ws$} | {:>wsc$} | {:>wf$} | {:>wr$} | {:>wt$}",
            "Gen", "Best Fit", "Avg Fit", "StdDev", "Score",
            "Best-Ever", "@Gen", "Time(s)",
            wg = W_GEN, wf = W_FIT, ws = W_STD, wsc = W_SCORE,
            wr = W_GEN_REF, wt = W_TIME,
        );
        writeln!(w, "{}", header)?;
        writeln!(w, "{}", "-".repeat(header.chars().count()))?;
        Ok(())
    }

    pub fn print_row<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let gen_str = format!("{:02}", self.generation);
        let gen_ref = format!("G{:02}", self.best_ever_generation);

        writeln!(
            w,
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
        )
    }
}