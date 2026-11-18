//! Logger de treinamento: grava o histórico de gerações em arquivo texto.
//!
//! `Logger` é um `Write` fino sobre um `File`, criado uma única vez no
//! início de `train.rs` (`Logger::new(None)` → `"train_historic.txt"`) e
//! consumido por dois canais:
//!
//!   1. A macro `logln!` (**definida neste arquivo**, exportada via
//!      `#[macro_export]`), para mensagens de texto livre (cabeçalho,
//!      avisos, linha final de resumo).
//!   2. `GenerationStats::{print_header, print_row}`, para a tabela de
//!      gerações com colunas de largura fixa.
//!
//! Layout tabular: todas as colunas cabem em ~140 caracteres. O separador
//! `-----` sob o cabeçalho é dimensionado por `header.chars().count()` —
//! resiliente a mudar qualquer rótulo ou largura de coluna sem dessincronizar.

use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Macro logln!
// ---------------------------------------------------------------------------

/// Escreve uma linha formatada no `Logger` (ou qualquer `Write`),
/// equivalente a `writeln!(logger, fmt, args...)` mas engolindo erros
/// de I/O — falhas de log nunca devem abortar o treinamento.
///
/// **Uso:** `logln!(logger, "texto {} {}\n", a, b);`
///
/// **Por que `writeln!` e não `write!`:** a convenção de chamada em
/// `train.rs` deixa o `\n` explícito **quando quer uma linha em branco
/// extra** (separadores entre blocos) e o omite quando basta uma linha.
/// `writeln!` fornece a quebra padrão; o `\n` explícito no format
/// vira uma segunda quebra — exatamente o comportamento esperado.
///
/// **Por que `#[macro_export]`:** coloca o macro no crate root, tornando-o
/// visível como `logln!` (sem `use`) em qualquer módulo do crate. Como
/// `train.rs` já tem `#[macro_use] mod utils;`, a exportação é redundante
/// mas inofensiva — garante compatibilidade se o `#[macro_use]` for
/// removido no futuro.
///
/// **Por que `::std::io::Write as _`:** `writeln!` precisa do trait `Write`
/// em escopo. O `as _` traz o trait sem nome (evita conflito com eventuais
/// `use std::io::Write` já presentes no local de expansão) e `::std` força
/// a resolução para o `std`, mesmo que o chamador tenha um módulo local
/// `std` (patológico, mas defensivo).
#[macro_export]
macro_rules! logln {
    ($logger:expr, $($arg:tt)*) => {{
        #[allow(unused_imports)]
        use ::std::io::Write as _;
        let _ = ::std::writeln!($logger, $($arg)*);
    }};
}

// ---------------------------------------------------------------------------
// Logger
// ---------------------------------------------------------------------------

pub struct Logger {
    file: File,
    path: PathBuf,
}

impl Logger {
    /// Cria (ou trunca) o arquivo de log.
    ///
    /// `None` → `"train_historic.txt"` no diretório corrente. Cada
    /// execução **recria** o arquivo do zero — o log representa apenas
    /// o último treinamento, não um histórico acumulado entre runs.
    pub fn new(path: Option<&str>) -> io::Result<Self> {
        let path = path.unwrap_or("train_historic.txt");
        let file = File::create(path)?;
        Ok(Self { file, path: PathBuf::from(path) })
    }

    #[inline]
    pub fn path(&self) -> &Path { &self.path }
}

impl Write for Logger {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.file.write(buf)
    }
    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

// ---------------------------------------------------------------------------
// GenerationStats — uma linha por geração
// ---------------------------------------------------------------------------

/// Snapshot tabular de uma geração. Preenchido em `train.rs` após
/// `select_top_k_by_fitness` e impresso com `print_row`.
///
/// Campos `*_pct` estão em **percentual (0..100)**, não fração — a
/// conversão acontece em `evaluate_multi`/`run_episode` (genetic.rs),
/// não aqui.
pub struct GenerationStats {
    pub generation: usize,
    pub best_fitness: f32,
    pub avg_fitness: f32,
    pub moving_avg_50: f32,
    pub std_dev: f32,
    pub diversity: f32,
    pub best_score: u32,
    pub avg_pellets_pct: f32,
    pub max_pellets_pct: f32,
    pub death_pct: f32,
    pub timeout_pct: f32,
    pub mut_rate: f32,
    pub mut_strength: f32,
    pub best_ever_fitness: f32,
    pub best_ever_generation: usize,
    pub best_ever_score: u32,
    pub time_secs: f32,
}

// Larguras fixas das colunas. Constantes (não literais espalhados) para
// que mudar uma largura exija editar **um** lugar e o cabeçalho + linhas
// permaneçam sincronizados por construção.
const W_GEN:     usize = 4;
const W_FIT:     usize = 9;
const W_STD:     usize = 8;
const W_DIV:     usize = 7;
const W_SCORE:   usize = 6;
const W_PEL:     usize = 11; // "45.2%/82.1%"
const W_TERM:    usize = 11; // "D:80% T:20%"
const W_MUT:     usize = 11; // "0.50 / 0.90"
const W_GEN_REF: usize = 5;
const W_TIME:    usize = 7;

impl GenerationStats {
    pub fn print_header<W: Write>(w: &mut W) -> io::Result<()> {
        let header = format!(
            "{:<wg$} | {:>wf$} | {:>wf$} | {:>wf$} | {:>ws$} | {:>wd$} | {:>wsc$} | {:>wp$} | {:>wtm$} | {:>wm$} | {:>wf$} | {:>wr$} | {:>wt$}",
            "Gen", "Best Fit", "Avg Fit", "MA50 Fit", "StdDev", "DivIdx",
            "Score", "Pellets A/M", "End D/T", "Mut R/S",
            "Best-Ever", "@Gen", "Time(s)",
            wg = W_GEN, wf = W_FIT, ws = W_STD, wd = W_DIV,
            wsc = W_SCORE, wp = W_PEL, wtm = W_TERM, wm = W_MUT,
            wr = W_GEN_REF, wt = W_TIME,
        );
        writeln!(w, "{}", header)?;
        writeln!(w, "{}", "-".repeat(header.chars().count()))?;
        Ok(())
    }

    pub fn print_row<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let gen_str = format!("{:02}", self.generation);
        let gen_ref = format!("G{:02}", self.best_ever_generation);
        let pellets_str = format!("{:>4.1}/{:>4.1}%", self.avg_pellets_pct, self.max_pellets_pct);
        let term_str = format!("{:>3.0}%/{:>3.0}%", self.death_pct, self.timeout_pct);
        let mut_str = format!("{:.2} / {:.2}", self.mut_rate, self.mut_strength);

        writeln!(
            w,
            "{:<wg$} | {:>+wf$.2} | {:>+wf$.2} | {:>+wf$.2} | {:>ws$.2} | {:>wd$.4} | {:>wsc$} | {:>wp$} | {:>wtm$} | {:>wm$} | {:>+wf$.2} | {:>wr$} | {:>wt$.2}",
            gen_str,
            self.best_fitness,
            self.avg_fitness,
            self.moving_avg_50,
            self.std_dev,
            self.diversity,
            self.best_score,
            pellets_str,
            term_str,
            mut_str,
            self.best_ever_fitness,
            gen_ref,
            self.time_secs,
            wg = W_GEN, wf = W_FIT, ws = W_STD, wd = W_DIV,
            wsc = W_SCORE, wp = W_PEL, wtm = W_TERM, wm = W_MUT,
            wr = W_GEN_REF, wt = W_TIME,
        )
    }
}