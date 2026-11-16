//! Gravação da tela em vídeo assíncrona.
//! Os frames são lidos da GPU na main thread, mas o encode via ffmpeg
//! ocorre inteiramente em uma thread separada para não impactar o jogo.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use macroquad::prelude::get_screen_data;

pub struct MatchRecorder {
    tx: Option<Sender<(u32, u32, Vec<u8>)>>,
    handle: Option<JoinHandle<()>>,
    finished: Arc<AtomicBool>,
}

impl MatchRecorder {
    pub fn new(path: PathBuf, fps: u32) -> Self {
        // Força a inferência do tipo do canal para evitar erros de [u8] vs Vec<u8>
        let (tx, rx) = channel::<(u32, u32, Vec<u8>)>();
        let finished = Arc::new(AtomicBool::new(false));
        let finished_clone = Arc::clone(&finished);
        
        let crf = "28"; 
        let preset = "ultrafast";
        let fps_str = fps.to_string();

        // Thread dedicada para I/O do ffmpeg
        let handle = thread::spawn(move || {
            let mut child: Option<Child> = None;
            let mut stdin: Option<std::process::ChildStdin> = None;

            // Recebe os frames da main thread
            while let Ok((w, h, bytes)) = rx.recv() {
                if stdin.is_none() {
                    let mut c = Command::new("ffmpeg")
                        .args([
                            "-y", "-loglevel", "error",
                            "-f", "rawvideo",
                            "-pix_fmt", "rgba",
                            "-s", &format!("{}x{}", w, h),
                            "-r", &fps_str,
                            "-i", "-",
                            "-vf", "vflip",
                            "-c:v", "libx264",
                            "-preset", preset,
                            "-crf", crf,
                            "-pix_fmt", "yuv420p",
                        ])
                        .arg(&path)
                        .stdin(Stdio::piped())
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .spawn()
                        .expect("ffmpeg não encontrado no PATH");

                    stdin = c.stdin.take();
                    child = Some(c);
                }

                if let Some(stdin) = stdin.as_mut() {
                    // Se o ffmpeg travar/morrer, ignora e continua
                    let _ = stdin.write_all(&bytes);
                }
            }

            // Fim da gravação: fecha o pipe e espera o ffmpeg finalizar
            drop(stdin);
            if let Some(mut c) = child {
                let _ = c.wait();
            }

            // Se o usuário fechou a janela sem chamar `finish()`, apaga o lixo.
            if !finished_clone.load(Ordering::SeqCst) {
                let _ = std::fs::remove_file(&path);
            }
        });

        Self {
            tx: Some(tx),
            handle: Some(handle),
            finished,
        }
    }

    /// Envia o frame para a thread de gravação de forma não-bloqueante.
    pub fn push_frame(&mut self) {
        let img = get_screen_data();
        if let Some(tx) = &self.tx {
            // Força a conversão para Vec<u8> alocado, caso img.bytes seja uma slice
            let bytes: Vec<u8> = img.bytes.to_vec();
            
            // Transfere a posse dos bytes (move) para a thread
            let _ = tx.send((img.width as u32, img.height as u32, bytes));
        }
    }

    /// Finaliza a gravação com sucesso.
    pub fn finish(mut self) {
        self.finished.store(true, Ordering::SeqCst);
        drop(self.tx.take()); // Fecha o canal, sinalizando EOF para a thread
        if let Some(handle) = self.handle.take() {
            let _ = handle.join(); // Aguarda o ffmpeg terminar de salvar
        }
    }
}

impl Drop for MatchRecorder {
    fn drop(&mut self) {
        // Se não foi finalizado, o drop do canal acionará o aviso de deleção na thread
        drop(self.tx.take());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}