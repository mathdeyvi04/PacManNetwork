//! Gravação da tela em vídeo, uma partida por arquivo.
//!
//! Estratégia: `macroquad::texture::get_screen_data()` lê o framebuffer
//! já renderizado como RGBA8. Enviamos cada frame cru para um subprocesso
//! `ffmpeg` (`-f rawvideo -pix_fmt rgba -i -`) que faz o encode H.264 e
//! escreve o `.mp4`.
//!
//! Escolhas:
//!   • zero dependências Rust novas — só `std::process`
//!   • compressão toda fora do processo do jogo
//!   • funciona com qualquer tamanho de janela (descoberto do 1º frame)
//!
//! Custo: `get_screen_data()` faz `glReadPixels` síncrono (~1–5 ms/frame).
//! Se atrapalhar no seu hardware, veja `push_frame_skipped` abaixo.
//!
//! Pré-requisito: `ffmpeg` no PATH. No Windows, `winget install ffmpeg`
//! (ou baixe de ffmpeg.org e adicione ao PATH).

use std::io::Write;
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};

use macroquad::prelude::get_screen_data;

pub struct MatchRecorder {
    ffmpeg: Option<Child>,
    stdin: Option<ChildStdin>,
    path: PathBuf,
    fps: u32,
    /// Se > 1, captura só 1 a cada N frames (vídeo fica com fps/N).
    capture_every: u32,
    frame_count: u32,
    finished: bool, // ← NOVO
}

impl MatchRecorder {
    pub fn new(path: PathBuf, fps: u32) -> Self {
        Self {
            ffmpeg: None,
            stdin: None,
            path,
            fps,
            capture_every: 1,
            frame_count: 0,
            finished: false, // ← NOVO
        }
    }

    /// Configura captura "pulada" para reduzir custo. `n = 1` → todos os
    /// frames. `n = 2` → metade, e o vídeo fica com `fps / n` quadros por
    /// segundo (gameplay não muda; só a taxa aparente no .mp4).
    pub fn with_capture_every(mut self, n: u32) -> Self {
        self.capture_every = n.max(1);
        self
    }

    /// Captura o frame atual e envia ao ffmpeg. Chame **depois** de todos
    /// os `draw_*` e **antes** de `next_frame().await`.
    pub fn push_frame(&mut self) -> std::io::Result<()> {
        self.frame_count = self.frame_count.wrapping_add(1);
        if self.frame_count % self.capture_every != 0 {
            return Ok(());
        }

        let img = get_screen_data();
        if self.stdin.is_none() {
            self.spawn_encoder(img.width as u32, img.height as u32)?;
        }
        if let Some(stdin) = self.stdin.as_mut() {
            stdin.write_all(&img.bytes)?;
        }
        Ok(())
    }

    fn spawn_encoder(&mut self, w: u32, h: u32) -> std::io::Result<()> {
        let effective_fps = (self.fps / self.capture_every).max(1);

        let mut child = Command::new("ffmpeg")
            .args([
                "-y",
                "-loglevel", "error",
                "-f", "rawvideo",
                "-pix_fmt", "rgba",
                "-s", &format!("{}x{}", w, h),
                "-r", &effective_fps.to_string(),
                "-i", "-",
                "-vf", "vflip", // ← CORRIGIDO: Apenas vflip, sem hflip
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "20",
                "-pix_fmt", "yuv420p",
            ])
            .arg(&self.path)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    std::io::Error::new(
                        e.kind(),
                        "ffmpeg não encontrado no PATH — instale-o para gravar vídeos",
                    )
                } else {
                    e
                }
            })?;

        self.stdin = child.stdin.take();
        self.ffmpeg = Some(child);
        Ok(())
    }

    /// Fecha o pipe (EOF → ffmpeg termina o encode) e aguarda a escrita.
    pub fn finish(mut self) -> std::io::Result<()> {
        self.finished = true; // ← NOVO: Marca como finalizado com sucesso
        drop(self.stdin.take());
        if let Some(mut child) = self.ffmpeg.take() {
            child.wait()?;
        }
        Ok(())
    }
}

impl Drop for MatchRecorder {
    fn drop(&mut self) {
        // Fecha o pipe (EOF). O ffmpeg finalizará o arquivo.
        drop(self.stdin.take());
        if let Some(mut child) = self.ffmpeg.take() {
            // Aguarda o ffmpeg fechar o arquivo (essencial no Windows para poder deletá-lo)
            let _ = child.wait();
        }

        // Se o finish() não foi chamado, é porque o usuário fechou a janela no meio. Apaga o lixo.
        if !self.finished {
            let _ = std::fs::remove_file(&self.path);
            println!("[rec] janela fechada durante a partida — vídeo temporário descartado.");
        }
    }
}