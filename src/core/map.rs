use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

use macroquad::rand::{gen_range, srand};

use crate::constants::*;

// ---------------------------------------------------------------------------
// Formato de arquivo de mapa
// ---------------------------------------------------------------------------
//
// MAGIC (8) | VERSION (u32 LE) | GRID_WIDTH (u32 LE) | GRID_HEIGHT (u32 LE)
//         | tiles (GRID_WIDTH * GRID_HEIGHT bytes, 1 byte por Tile)
//
// O arquivo representa o **template** do mapa, não um snapshot de estado:
//   byte 0 = Wall   |  byte 1 = Walkable (vira Pellet no load)
//   byte 2 = legado (Empty, aceito no load → Pellet) |  byte 3 = Tunnel
//
// `pellets_remaining` NÃO é serializado — é recomputado no `load` a partir
// dos tiles, garantindo consistência. Todo mapa carregado nasce virgem.

const MAP_MAGIC: &[u8; 8] = b"PACMAP01";
const MAP_VERSION: u32 = 1;

/// Conteúdo de uma célula do grid.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tile {
    Wall,
    Pellet,
    Empty,
    Tunnel,
}

impl Tile {
    /// Discriminante `u8` (`#[repr(u8)]`) — usado pela (de)serialização.
    #[inline(always)]
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Converte o byte de volta para `Tile`. Retorna `None` se inválido.
    #[inline]
    pub fn from_u8(v: u8) -> Option<Tile> {
        match v {
            0 => Some(Tile::Wall),
            1 => Some(Tile::Pellet),
            2 => Some(Tile::Empty),
            3 => Some(Tile::Tunnel),
            _ => None,
        }
    }
}

/// Labirinto em grid de tiles.
#[derive(Clone)]
pub struct Map {
    /// Grid achatado: `tiles[y * GRID_WIDTH + x]`.
    /// `Box<[Tile]>` — tamanho exato, sem `capacity` sobrando;
    /// `clone()` aloca exatamente `len * size_of::<Tile>()` bytes.
    tiles: Box<[Tile]>,
    /// Pontos Restando
    pellets_remaining: usize,
}

impl Map {

    /// Converte coordenadas `(x, y)` no índice flat do vetor `tiles`.
    #[inline(always)]
    fn idx(x: usize, y: usize) -> usize {
        y * GRID_WIDTH + x
    }

    /// Gera um labirinto novo com pastilhas preenchidas.
    pub fn generate() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        srand(seed);

        // Box<[Tile]> — sem overhead de capacidade.
        let mut tiles: Box<[Tile]> =
            vec![Tile::Wall; GRID_WIDTH * GRID_HEIGHT].into_boxed_slice();

        // Recursive Backtracker sobre as células ímpares do grid.
        let max_cells = ((GRID_WIDTH - 1) / 2) * ((GRID_HEIGHT - 1) / 2);
        let mut stack: Vec<usize> = Vec::with_capacity(max_cells);

        stack.push(Self::idx(1, 1));
        tiles[Self::idx(1, 1)] = Tile::Empty;

        const DIRS: [(i32, i32); 4] = [(0, -2), (0, 2), (-2, 0), (2, 0)];
        let w_i = GRID_WIDTH as i32;
        let h_i = GRID_HEIGHT as i32;

        while let Some(&i) = stack.last() {
            let x = i % GRID_WIDTH;
            let y = i / GRID_WIDTH;

            // Buffer fixo (máx. 4 vizinhos) na stack, sem heap.
            let mut neighbors: [(usize, usize, i32, i32); 4] = [(0, 0, 0, 0); 4];
            let mut n_count = 0usize;

            for &(dx, dy) in &DIRS {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx > 0
                    && nx < w_i - 1
                    && ny > 0
                    && ny < h_i - 1
                    && tiles[Self::idx(nx as usize, ny as usize)] == Tile::Wall
                {
                    neighbors[n_count] = (nx as usize, ny as usize, dx, dy);
                    n_count += 1;
                }
            }

            if n_count == 0 {
                stack.pop();
            } else {
                let (nx, ny, dx, dy) = neighbors[gen_range(0, n_count)];

                // Escava o muro entre a célula atual e a vizinha escolhida.
                tiles[Self::idx((x as i32 + dx / 2) as usize,
                                (y as i32 + dy / 2) as usize)] = Tile::Empty;
                tiles[Self::idx(nx, ny)] = Tile::Empty;

                stack.push(Self::idx(nx, ny));
            }
        }

        // Abre ~35% dos muros que separam dois corredores, criando loops.
        for y in 1..GRID_HEIGHT - 1 {
            for x in 1..GRID_WIDTH - 1 {
                let i = Self::idx(x, y);
                if tiles[i] != Tile::Wall {
                    continue;
                }
                let mut cn = 0u8;
                if tiles[Self::idx(x, y - 1)] != Tile::Wall { cn += 1; }
                if tiles[Self::idx(x, y + 1)] != Tile::Wall { cn += 1; }
                if tiles[Self::idx(x - 1, y)] != Tile::Wall { cn += 1; }
                if tiles[Self::idx(x + 1, y)] != Tile::Wall { cn += 1; }
                if cn >= 2 && gen_range(0, 100) < 35 {
                    tiles[i] = Tile::Empty;
                }
            }
        }

        tiles[Self::idx(0, TUNNEL_ROW)] = Tile::Tunnel;
        tiles[Self::idx(GRID_WIDTH - 1, TUNNEL_ROW)] = Tile::Tunnel;
        tiles[Self::idx(1, TUNNEL_ROW)] = Tile::Empty;
        tiles[Self::idx(GRID_WIDTH - 2, TUNNEL_ROW)] = Tile::Empty;

        // (fill de pellets já existente abaixo — converte Empty→Pellet)
        let mut pellets_remaining = 0usize;
        for t in tiles.iter_mut() {
            if *t == Tile::Empty {
                *t = Tile::Pellet;
                pellets_remaining += 1;
            }
        }

        Map { tiles, pellets_remaining }
    }

    #[inline(always)]
    pub fn is_wall(&self, x: usize, y: usize) -> bool {
        self.tiles[Self::idx(x, y)] == Tile::Wall
    }

    /// Acesso read-only ao grid achatado, usado pelo Renderer.
    /// Slice emprestado — zero cópia, zero alocação.
    #[inline(always)]
    pub fn tiles(&self) -> &[Tile] {
        &self.tiles
    }

    #[inline(always)]
    pub fn is_wall_or_tunnel(&self, x: usize, y: usize) -> bool {
        matches!(self.tiles[Self::idx(x, y)], Tile::Wall | Tile::Tunnel)
    }

    #[inline(always)]
    pub fn tile_at(&self, x: usize, y: usize) -> Tile {
        self.tiles[Self::idx(x, y)]
    }

    /// Consome a pastilha do tile, se houver. Retorna `true` se comeu.
    pub fn eat_pellet(&mut self, x: usize, y: usize) -> bool {
        let i = Self::idx(x, y);
        if self.tiles[i] == Tile::Pellet {
            self.tiles[i] = Tile::Empty;
            self.pellets_remaining = self.pellets_remaining.saturating_sub(1);
            true
        } else {
            false
        }
    }

    /// Getter para obtermos o total de pontos restando
    #[inline(always)]
    pub fn pellets_remaining(&self) -> usize {
        self.pellets_remaining
    }

    /// Retorna uma cópia do mapa com a **mesma topologia** (paredes, túneis,
    /// corredores), mas com todas as pastilhas restauradas.
    ///
    /// Útil para reiniciar uma partida sem tocar em disco: o `Game` pode
    /// ser recriado a partir de `game.map.fresh()`.
    ///
    /// O invariante "todo tile `Empty` é uma pastilha comida" vale porque
    /// `generate()` converte **todos** os `Empty` iniciais em `Pellet`.
    #[inline]
    pub fn fresh(&self) -> Self {
        let mut tiles: Box<[Tile]> = self.tiles.clone();
        let mut pellets_remaining = 0usize;
        for t in tiles.iter_mut() {
            if *t == Tile::Empty {
                *t = Tile::Pellet;
            }
            if *t == Tile::Pellet {
                pellets_remaining += 1;
            }
        }
        Map { tiles, pellets_remaining }
    }

    // -----------------------------------------------------------------------
    // Persistência
    // -----------------------------------------------------------------------

    /// Serializa o mapa em disco (formato binário próprio — ver topo do módulo).
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut f = File::create(path)?;
        f.write_all(MAP_MAGIC)?;
        f.write_all(&MAP_VERSION.to_le_bytes())?;
        f.write_all(&(GRID_WIDTH  as u32).to_le_bytes())?;
        f.write_all(&(GRID_HEIGHT as u32).to_le_bytes())?;

        // O arquivo representa o *template* do mapa, não um snapshot de estado.
        // Qualquer tile caminhável não-túnel (Pellet ou Empty) é gravado como
        // "caminhável com pastilha" (byte 1) — pois ao carregar queremos
        // sempre um mapa virgem. Pastilhas consumidas não são persistidas.
        let buf: Vec<u8> = self.tiles.iter().map(|t| match t {
            Tile::Wall                 => 0u8,
            Tile::Pellet | Tile::Empty => 1u8,
            Tile::Tunnel               => 3u8,
        }).collect();
        f.write_all(&buf)?;
        Ok(())
    }

    /// Lê um mapa previamente salvo por `save`.
    ///
    /// Valida magic, versão e dimensões — rejeita arquivos incompatíveis com
    /// a build atual (mudou `GRID_WIDTH`/`GRID_HEIGHT`). `pellets_remaining`
    /// é recomputado a partir dos tiles (não confia em estado externo).
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let mut f = File::open(path)?;

        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != MAP_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "magic inválido"));
        }

        let version = read_u32(&mut f)?;
        if version != MAP_VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "versão incompatível"));
        }

        let w = read_u32(&mut f)? as usize;
        let h = read_u32(&mut f)? as usize;
        if w != GRID_WIDTH || h != GRID_HEIGHT {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "dimensões incompatíveis com a build atual",
            ));
        }

        let total = w * h;
        let mut tiles: Vec<Tile> = Vec::with_capacity(total);
        let mut pellets_remaining = 0usize;
        let mut byte = [0u8; 1];

        for _ in 0..total {
            f.read_exact(&mut byte)?;
            // Compatibilidade com saves antigos: byte 2 era `Tile::Empty` (pastilha
            // comida). Hoje `save` nunca escreve 2, mas aceitamos por robustez —
            // de qualquer forma, um mapa carregado é SEMPRE virgem.
            let t = match byte[0] {
                0        => Tile::Wall,
                1 | 2    => Tile::Pellet,
                3        => Tile::Tunnel,
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "byte de tile inválido",
                    ))
                }
            };
            if t == Tile::Pellet {
                pellets_remaining += 1;
            }
            tiles.push(t);
        }

        Ok(Map {
            tiles: tiles.into_boxed_slice(),
            pellets_remaining,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn read_u32(f: &mut File) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}