use macroquad::rand::{gen_range, srand};

use crate::constants::*;

/// Conteúdo de uma célula do grid.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tile {
    Wall,
    Pellet,
    Empty,
    Tunnel,
}

/// Labirinto em grid de tiles.
#[derive(Clone)]
pub struct Map {
    /// Grid achatado: `tiles[y * GRID_WIDTH + x]`.
    tiles: Vec<Tile>,
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

        let mut tiles = vec![Tile::Wall; GRID_WIDTH * GRID_HEIGHT];

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
}