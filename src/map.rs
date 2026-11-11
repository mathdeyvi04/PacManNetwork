use crate::constants::*;
use crate::rng::{fresh_seed, Rng};

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
    tiles: Vec<Tile>,

    /// Quantidade atual de pellets.
    pellets_remaining: usize,

    /// Quantidade original de pellets.
    total_pellets: usize,

    /// Somente posições dos pellets ainda existentes.
    pellet_positions: Vec<u16>,

    /// Índice inverso: posição no mapa -> posição em pellet_positions.
    ///
    /// u16::MAX significa que não existe pellet naquela célula.
    pellet_slots: Vec<u16>,
}

impl Map {
    /// Converte coordenadas `(x, y)` no índice flat do vetor `tiles`.
    #[inline(always)]
    fn idx(x: usize, y: usize) -> usize {
        y * GRID_WIDTH + x
    }

    /// Gera um mapa aleatório.
    pub fn generate() -> Self {
        Self::generate_seeded(fresh_seed())
    }
    
    /// Gera um mapa utilizando uma seed específica.
    pub fn generate_seeded(seed: u64) -> Self {
        let mut rng = Rng::new(seed);

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
                let (nx, ny, dx, dy) = neighbors[rng.range_usize(n_count)];

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
                if cn >= 2 && rng.range_usize(100) < 35 {
                    tiles[i] = Tile::Empty;
                }
            }
        }

        tiles[Self::idx(0, TUNNEL_ROW)] = Tile::Tunnel;
        tiles[Self::idx(GRID_WIDTH - 1, TUNNEL_ROW)] = Tile::Tunnel;
        tiles[Self::idx(1, TUNNEL_ROW)] = Tile::Empty;
        tiles[Self::idx(GRID_WIDTH - 2, TUNNEL_ROW)] = Tile::Empty;
        
        // (fill de pellets já existente abaixo — converte Empty→Pellet)
        let mut pellet_positions =
            Vec::with_capacity(tiles.len());
        
        let mut pellet_slots =
            vec![u16::MAX; tiles.len()];
        
        for (index, tile) in tiles.iter_mut().enumerate() {
            if *tile == Tile::Empty {
                *tile = Tile::Pellet;
        
                pellet_slots[index] = pellet_positions.len() as u16;
        
                pellet_positions.push(index as u16);
            }
        }
        
        let total_pellets = pellet_positions.len();
        
        let pellets_remaining = total_pellets;
        
        Map {
            tiles,
            pellets_remaining,
            total_pellets,
            pellet_positions,
            pellet_slots,
        }
    }

    #[inline(always)]
    pub fn total_pellets(&self) -> usize {
        self.total_pellets
    }

    #[inline(always)]
    pub fn pellet_positions(&self) -> &[u16] {
        &self.pellet_positions
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
    
        if self.tiles[i] != Tile::Pellet {
            return false;
        }
    
        self.tiles[i] = Tile::Empty;
    
        self.pellets_remaining -= 1;
    
        let slot = self.pellet_slots[i] as usize;
    
        debug_assert!(slot < self.pellet_positions.len());
    
        // Último elemento que poderá ser movido para preencher
        // o espaço deixado pelo pellet removido.
        let moved_index = *self.pellet_positions.last().unwrap() as usize;
    
        self.pellet_positions.swap_remove(slot);
    
        // Invalida o índice do pellet consumido.
        self.pellet_slots[i] = u16::MAX;
    
        // Atualiza a posição do elemento que foi deslocado.
        if slot < self.pellet_positions.len() {
            self.pellet_slots[moved_index] = slot as u16;
        }
    
        true
    }

    /// Getter para obtermos o total de pontos restando
    #[inline(always)]
    pub fn pellets_remaining(&self) -> usize {
        self.pellets_remaining
    }
}