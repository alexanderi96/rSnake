//! Snake simulation logic
//!
//! Contains SnakeInstance, GameState, and state computation functions.

use rand::Rng;
use std::collections::{HashSet, VecDeque};

use bevy::prelude::{Color, Resource};
use uuid::Uuid;

pub use super::grid::{
    Direction, GridDimensions, GridMap, Position, BASE_STATE_SIZE, BLOCK_SIZE, RAY_DIRECTIONS,
};

use crate::config::Hyperparameters;
use crate::plugins::map_elites::individual::GenomeColor;
use crate::plugins::terrain;

/// Food sequence length
pub const FOOD_SEQ_LEN: usize = 1000;

// ============================================================================
// Snake Instance
// ============================================================================

#[derive(Clone)]
pub struct SnakeInstance {
    pub id: usize,
    pub uuid: uuid::Uuid,
    pub snake: VecDeque<Position>,
    pub body_set: HashSet<Position>,
    pub food: Position,
    pub direction: Direction,
    pub is_game_over: bool,
    pub steps_without_food: u32,
    pub score: u32,
    pub color: Color,
    pub previous_state: [f32; BASE_STATE_SIZE],
    pub frames_survived: u32,
    pub visited_cells: std::collections::HashSet<(i32, i32)>,
    pub path_directness_sum: f32,
    pub food_real_distance: u32,
    pub obstacle_adjacency_sum: f32,
    pub path_progress_sum: f32, // accumula progress ratio per-frame
}

/// Seed condiviso per la generazione corrente.
#[derive(Resource, Clone)]
pub struct GenerationSeed {
    pub spawn_pos: Position,
    pub spawn_dir: Direction,
    pub food_sequence: Vec<Position>, // pre-generata, lunga FOOD_SEQ_LEN
    pub terrain: Vec<bool>,
}

impl GenerationSeed {
    pub fn new_for_grid_with_config(grid: &GridDimensions, config: &Hyperparameters) -> Self {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};

        let seed: u64 = rand::thread_rng().gen();
        let mut rng = SmallRng::seed_from_u64(seed);

        let margin = 4i32;
        let spawn_pos = Position {
            x: rng.gen_range(margin..(grid.width - margin)),
            y: rng.gen_range(margin..(grid.height - margin)),
        };
        let spawn_dir = match rng.gen_range(0..4u8) {
            0 => Direction::Up,
            1 => Direction::Down,
            2 => Direction::Left,
            _ => Direction::Right,
        };

        let terrain = terrain::generate(
            seed,
            grid.width,
            grid.height,
            config.terrain_fill_rate,
            config.terrain_blob_scale,
            config.terrain_smooth_passes,
            spawn_pos,
            config.terrain_spawn_clearance,
        );

        let food_sequence = build_food_sequence(&mut rng, grid, &terrain);

        Self {
            spawn_pos,
            spawn_dir,
            food_sequence,
            terrain,
        }
    }

    pub fn new_for_grid(grid: &GridDimensions) -> Self {
        Self::new_for_grid_with_config(grid, &Hyperparameters::default())
    }

    pub fn food_at(&self, index: usize) -> Position {
        self.food_sequence[index % FOOD_SEQ_LEN]
    }

    /// Returns the first free position starting from `start_index` in the sequence,
    /// escludendo celle occupate dal corpo del serpente e dal terrain.
    /// If zone parameters are provided (radius < INFINITY), only positions within
    /// the zone circle are considered first.
    /// Deterministica: stessi argomenti → stesso risultato.
    pub fn food_at_free(
        &self,
        start_index: usize,
        body_set: &std::collections::HashSet<Position>,
        terrain: &[bool],
        width: i32,
        zone_center: Option<(f32, f32)>,
        zone_radius: f32,
    ) -> Position {
        // If zone_radius is INFINITY, use all positions (current behavior)
        let use_zone = zone_radius.is_finite() && zone_radius > 0.0;

        if use_zone {
            // Collect candidate positions within the zone
            let (center_x, center_y) = zone_center.unwrap_or((0.0, 0.0));
            let radius_sq = zone_radius * zone_radius;

            // First pass: collect all candidates within zone that are free
            let mut candidates: Vec<Position> = Vec::new();
            for i in 0..FOOD_SEQ_LEN {
                let pos = self.food_sequence[(start_index + i) % FOOD_SEQ_LEN];
                let idx = (pos.y * width + pos.x) as usize;

                // Check if within zone circle
                let dx = pos.x as f32 - center_x;
                let dy = pos.y as f32 - center_y;
                let dist_sq = dx * dx + dy * dy;

                if dist_sq <= radius_sq
                    && !body_set.contains(&pos)
                    && idx < terrain.len()
                    && !terrain[idx]
                {
                    candidates.push(pos);
                }
            }

            // If we have candidates in zone, pick uniformly at random
            if !candidates.is_empty() {
                let mut rng = rand::thread_rng();
                return candidates[rng.gen_range(0..candidates.len())];
            }

            // No candidates in zone - log warning and fall back to full-map random
            eprintln!(
                "[WARNING] No food positions found in zone (center: {:.1}, {:.1}, radius: {:.1}), \
                falling back to full map",
                center_x, center_y, zone_radius
            );
        }

        // Fallback: original behavior - search from start_index through entire sequence
        for i in 0..FOOD_SEQ_LEN {
            let pos = self.food_sequence[(start_index + i) % FOOD_SEQ_LEN];
            let idx = (pos.y * width + pos.x) as usize;
            if !body_set.contains(&pos) && idx < terrain.len() && !terrain[idx] {
                return pos;
            }
        }
        // Fallback: tutta la sequenza è occupata (impossibile con serpenti normali)
        self.food_sequence[start_index % FOOD_SEQ_LEN]
    }
}

fn build_food_sequence(
    rng: &mut impl rand::RngCore,
    grid: &GridDimensions,
    terrain: &[bool],
) -> Vec<Position> {
    use rand::Rng;
    let mut seq = Vec::with_capacity(FOOD_SEQ_LEN);
    let mut attempts = 0usize;
    while seq.len() < FOOD_SEQ_LEN && attempts < FOOD_SEQ_LEN * 20 {
        attempts += 1;
        let pos = Position {
            x: rng.gen_range(1..(grid.width - 1)),
            y: rng.gen_range(1..(grid.height - 1)),
        };
        let idx = (pos.y * grid.width + pos.x) as usize;
        if idx < terrain.len() && !terrain[idx] {
            seq.push(pos);
        }
    }
    let centre = Position {
        x: grid.width / 2,
        y: grid.height / 2,
    };
    while seq.len() < FOOD_SEQ_LEN {
        seq.push(centre);
    }
    seq
}

impl SnakeInstance {
    pub fn calculate_behavioral_color(
        courage: f32,
        agility: f32,
        fitness: f32,
        best_fitness: f32,
    ) -> Color {
        let r = courage;
        let g = agility;
        let b = if best_fitness > 0.0 {
            (fitness / best_fitness).clamp(0.0, 1.0)
        } else {
            0.5
        };

        Color::rgb(r, g, b)
    }

    pub fn get_random_spawn_data(grid: &GridDimensions) -> (Position, Direction) {
        let mut rng = rand::thread_rng();
        let margin = 3;

        let x = rng.gen_range(margin..(grid.width - margin));
        let y = rng.gen_range(margin..(grid.height - margin));

        let direction = match rng.gen_range(0..4) {
            0 => Direction::Up,
            1 => Direction::Down,
            2 => Direction::Left,
            3 => Direction::Right,
            _ => unreachable!(),
        };

        (Position { x, y }, direction)
    }

    pub fn new(id: usize, grid: &GridDimensions, total_snakes: usize) -> Self {
        Self::new_with_color(id, grid, total_snakes, None, 0.5, 0.5, 0.0, 1.0)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_color(
        id: usize,
        grid: &GridDimensions,
        _total_snakes: usize,
        _genetic_color: Option<GenomeColor>,
        parent_courage: f32,
        parent_agility: f32,
        parent_fitness: f32,
        best_fitness: f32,
    ) -> Self {
        let (spawn_pos, spawn_dir) = Self::get_random_spawn_data(grid);

        let mut snake = VecDeque::new();
        snake.push_back(spawn_pos);

        let mut body_set = HashSet::with_capacity(32);
        body_set.insert(spawn_pos);

        let color = Self::calculate_behavioral_color(
            parent_courage,
            parent_agility,
            parent_fitness,
            best_fitness,
        );

        let food = Position {
            x: (spawn_pos.x + 5) % grid.width,
            y: (spawn_pos.y + 5) % grid.height,
        };
        let food_real_distance =
            ((food.x - spawn_pos.x).abs() + (food.y - spawn_pos.y).abs()) as u32;

        Self {
            id,
            uuid: Uuid::now_v7(),
            snake,
            body_set,
            food,
            direction: spawn_dir,
            is_game_over: false,
            steps_without_food: 0,
            score: 0,
            color,
            previous_state: [0.0; BASE_STATE_SIZE],
            frames_survived: 0,
            visited_cells: std::collections::HashSet::with_capacity(64),
            path_directness_sum: 0.0,
            food_real_distance,
            obstacle_adjacency_sum: 0.0,
            path_progress_sum: 0.0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn reset_with_seed(
        &mut self,
        grid: &GridDimensions,
        _total_snakes: usize,
        seed: &GenerationSeed,
        parent_courage: f32,
        parent_agility: f32,
        parent_fitness: f32,
        best_fitness: f32,
    ) {
        self.uuid = Uuid::now_v7();
        self.snake.clear();
        self.snake.push_back(seed.spawn_pos);
        self.body_set.clear();
        self.body_set.shrink_to_fit();
        self.body_set.insert(seed.spawn_pos);
        self.direction = seed.spawn_dir;
        self.is_game_over = false;
        self.steps_without_food = 0;
        self.score = 0;
        self.food = seed.food_at(0);
        self.color = Self::calculate_behavioral_color(
            parent_courage,
            parent_agility,
            parent_fitness,
            best_fitness,
        );
        self.previous_state = [0.0; BASE_STATE_SIZE];
        self.frames_survived = 0;
        self.visited_cells.clear();
        self.visited_cells.shrink_to_fit();
        self.path_directness_sum = 0.0;
        self.path_progress_sum = 0.0;

        // Calcola distanza BFS reale per il primo cibo
        let tail = self.snake.back().copied();
        self.food_real_distance = bfs_distance(
            self.snake[0],
            self.food,
            &self.body_set,
            tail,
            &seed.terrain,
            grid.width,
            grid.height,
        )
        .unwrap_or(0);

        // Se già alla partenza il cibo è irraggiungibile, prova altri slot
        if self.food_real_distance == 0 && self.snake[0] != self.food {
            for i in 1..FOOD_SEQ_LEN {
                let candidate = seed.food_at(i);
                if let Some(d) = bfs_distance(
                    self.snake[0],
                    candidate,
                    &self.body_set,
                    tail,
                    &seed.terrain,
                    grid.width,
                    grid.height,
                ) {
                    self.food = candidate;
                    self.food_real_distance = d;
                    break;
                }
            }
        }

        self.obstacle_adjacency_sum = 0.0;
        self.path_progress_sum = 0.0;
    }

    /// D1: how directly the snake reaches food [0,1]
    /// score > 0: average BFS-efficiency per apple eaten
    /// score = 0: average per-frame progress toward current food (fallback)
    pub fn path_efficiency(&self) -> f32 {
        if self.score > 0 {
            (self.path_directness_sum / self.score as f32).clamp(0.0, 1.0)
        } else if self.frames_survived > 0 {
            (self.path_progress_sum / self.frames_survived as f32).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// D2: mean proximity to walls and own body per frame [0,1]
    /// Uses the raycast obstacle sensors (current_17[0..8]) accumulated per frame.
    /// 0.0 = always in open space | 1.0 = always surrounded by obstacles
    pub fn danger_affinity(&self) -> f32 {
        if self.frames_survived == 0 {
            return 0.0;
        }
        (self.obstacle_adjacency_sum / self.frames_survived as f32).clamp(0.0, 1.0)
    }

    /// D3: unique cells visited per frame [0,1]
    /// Max theoretical = 1.0 (snake always moves to a new cell).
    /// Low = circles in a small area | High = explores broadly
    pub fn spatial_spread(&self) -> f32 {
        if self.frames_survived == 0 {
            return 0.0;
        }
        (self.visited_cells.len() as f32 / self.frames_survived as f32).clamp(0.0, 1.0)
    }

    /// Funzione di Fitness con pesi dinamici
    /// Obiettivo: premiare serpenti efficienti E serpenti che fanno zigzag strategico
    /// La fitness è garantita monotona crescente con lo score
    pub fn fitness(&self, _grid: &GridDimensions) -> f32 {
        if self.frames_survived == 0 {
            return 0.0;
        }

        let frames = self.frames_survived as f32;
        let score = self.score as f32;

        // ── 1. Navigazione ─────────────────────────────────────────────
        // Attiva sempre, anche a score=0. Guida il serpente verso il cibo.
        let longevity = 1.0 - (-frames / 80.0).exp();
        let avg_ratio = (self.path_progress_sum / frames).clamp(0.0, 1.0);
        let nav = avg_ratio * longevity * 300.0;

        // ── 2. Score grezzo ─────────────────────────────────────────────
        // powf(1.2): super-lineare, garantisce che ogni mela aggiuntiva
        // valga sempre *più* della precedente. Non satura mai.
        let raw_score = score.powf(1.2) * 250.0;

        // ── 3. Efficienza cumulata ──────────────────────────────────────
        // path_directness_sum = Σ(eff_i) per ogni mela mangiata.
        // Nessuna divisione per score → nessun NaN a score=0, nessun if.
        // Cresce con ogni mela (quantità) e premia traiettorie dirette (qualità).
        // Con score alto il suo peso relativo diminuisce naturalmente perché
        // raw_score cresce più velocemente — senza decay esplicito.
        let eff_bonus = self.path_directness_sum * 500.0;

        // ── 4. Sopravvivenza ────────────────────────────────────────────
        // ln(score+1): cresce senza saturare, vale 0 a score=0 (corretto:
        // sopravvivere a lungo senza mangiare nulla non è un risultato).
        let surv = frames.sqrt() * (score + 1.0).ln() * 60.0;

        // ── 5. Esplorazione ─────────────────────────────────────────────
        // Peso basso: tiebreaker per MAP-Elites, non obiettivo primario.
        // sqrt(score+1): diventa rilevante solo con serpenti di media lunghezza.
        let expl = self.spatial_spread() * (score + 1.0).sqrt() * 80.0;

        nav + raw_score + eff_bonus + surv + expl
    }
}

// ============================================================================
// Game State
// ============================================================================

#[derive(Resource, Clone)]
pub struct GameState {
    pub high_score: u32,
    pub total_iterations: u32,
    pub snakes: Vec<SnakeInstance>,
    /// Pending population size to apply at next generation boundary (batch mode)
    pub pending_population_size: Option<usize>,
}

impl GameState {
    pub fn new_with_behavioral_colors(
        grid: &GridDimensions,
        snake_count: usize,
        behaviors: Option<Vec<(f32, f32, f32, f32)>>,
    ) -> Self {
        let snakes = if let Some(behaviors) = behaviors {
            (0..snake_count)
                .map(|id| {
                    let (courage, agility, fitness, best) =
                        behaviors.get(id).copied().unwrap_or((0.5, 0.5, 0.0, 1.0));
                    SnakeInstance::new_with_color(
                        id,
                        grid,
                        snake_count,
                        None,
                        courage,
                        agility,
                        fitness,
                        best,
                    )
                })
                .collect()
        } else {
            (0..snake_count)
                .map(|id| SnakeInstance::new(id, grid, snake_count))
                .collect()
        };
        Self {
            high_score: 0,
            total_iterations: 0,
            snakes,
            pending_population_size: None,
        }
    }

    pub fn alive_count(&self) -> usize {
        self.snakes.iter().filter(|s| !s.is_game_over).count()
    }

    /// Grow or shrink the snake population to `new_size`.
    /// Use existing snake initialization logic for new individuals.
    pub fn resize_population_to(&mut self, new_size: usize, hyperparams: &Hyperparameters) {
        let current = self.snakes.len();
        if new_size > current {
            self.grow_population_to(new_size, hyperparams);
        } else if new_size < current {
            self.shrink_population_to(new_size);
        }
    }

    /// Append new snakes up to `new_size` using default/random brain initialization.
    /// Mirror the initialization done in `new_with_behavioral_colors`.
    pub fn grow_population_to(&mut self, new_size: usize, _hyperparams: &Hyperparameters) {
        let current = self.snakes.len();
        if new_size <= current {
            return;
        }

        for id in current..new_size {
            // Use new_with_color with default values (same as new())
            let new_snake = SnakeInstance::new_with_color(
                id,
                // We need grid dimensions - but GameState doesn't store it
                // Use a default, it will be overridden on reset_with_seed
                &GridDimensions {
                    width: 40,
                    height: 30,
                },
                new_size,
                None,
                0.5, // parent_courage
                0.5, // parent_agility
                0.0, // parent_fitness
                1.0, // best_fitness
            );
            self.snakes.push(new_snake);
        }
    }

    /// Remove snakes down to `new_size`.
    /// Prefer removing dead snakes first; then remove from the tail of the Vec.
    pub fn shrink_population_to(&mut self, new_size: usize) {
        let current = self.snakes.len();
        if new_size >= current {
            return;
        }

        // First, remove dead snakes (prefer those at the end since they're likely older)
        let mut removed = 0;
        let target_remove = current - new_size;

        // Collect indices of dead snakes in reverse order (from end)
        let dead_indices: Vec<usize> = self
            .snakes
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_game_over)
            .map(|(i, _)| i)
            .collect();

        // Remove dead snakes from the end first to minimize vec reallocation
        for idx in dead_indices.iter().rev() {
            if removed >= target_remove {
                break;
            }
            // Remove from the end of the Vec
            let last_idx = self.snakes.len() - 1;
            if *idx == last_idx {
                self.snakes.pop();
                removed += 1;
            }
        }

        // If we still need to remove more, remove from the tail
        while self.snakes.len() > new_size {
            self.snakes.pop();
        }

        // Re-index snake IDs after shrink
        for (i, snake) in self.snakes.iter_mut().enumerate() {
            snake.id = i;
        }
    }
}

// ============================================================================
// State Computation (Neural Network Input)
// ============================================================================

/// BFS Pathfinding
/// Ritorna la distanza BFS (numero di passi) dalla testa al target,
/// oppure None se non esiste un percorso libero.
/// Ostacoli: terrain walls + body del serpente (esclusa la coda,
/// che si sposterà al prossimo step).
pub fn bfs_distance(
    head: Position,
    target: Position,
    body_set: &HashSet<Position>,
    tail: Option<Position>, // esclusa dagli ostacoli (si sposterà)
    terrain: &[bool],
    width: i32,
    height: i32,
) -> Option<u32> {
    if head == target {
        return Some(0);
    }

    let mut visited = vec![false; (width * height) as usize];
    let mut queue = std::collections::VecDeque::new();

    let start_idx = (head.y * width + head.x) as usize;
    visited[start_idx] = true;
    queue.push_back((head, 0u32));

    while let Some((pos, dist)) = queue.pop_front() {
        for (dx, dy) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
            let nx = pos.x + dx;
            let ny = pos.y + dy;
            if nx < 0 || nx >= width || ny < 0 || ny >= height {
                continue;
            }
            let nidx = (ny * width + nx) as usize;
            if visited[nidx] {
                continue;
            }
            if terrain[nidx] {
                continue;
            }
            let npos = Position { x: nx, y: ny };
            // Corpo del serpente è ostacolo, tranne la coda (che si libererà al prossimo step)
            if body_set.contains(&npos) && Some(npos) != tail {
                continue;
            }
            if npos == target {
                return Some(dist + 1);
            }
            visited[nidx] = true;
            queue.push_back((npos, dist + 1));
        }
    }
    None
}

/// Compute the 17-dimensional state vector for the neural network.
/// This is the core sensory input used by the brain to make decisions.
pub fn get_current_17_state(
    snake: &SnakeInstance,
    grid_map: &GridMap,
    grid: &GridDimensions,
    snake_vs_snake: bool,
) -> [f32; BASE_STATE_SIZE] {
    let decay = 0.15_f32; // unico decay per tutto il sistema sensoriale
    let mut current_state = [0.0f32; BASE_STATE_SIZE];
    let head = snake.snake[0];

    let dir_offset: usize = match snake.direction {
        Direction::Up => 0,
        Direction::Right => 2,
        Direction::Down => 4,
        Direction::Left => 6,
    };

    // [0-7] Ostacoli: decay aggiustato, adiacente = 1.0
    for i in 0..8 {
        let ray_idx = (i + dir_offset) % 8;
        let (dx, dy) = RAY_DIRECTIONS[ray_idx];
        let mut curr_x = head.x;
        let mut curr_y = head.y;

        loop {
            curr_x += dx;
            curr_y += dy;

            let hit_wall =
                curr_x < 0 || curr_x >= grid.width || curr_y < 0 || curr_y >= grid.height;

            let hit_obstacle = !hit_wall
                && if snake_vs_snake {
                    grid_map.is_collision_with_self(curr_x, curr_y)
                } else {
                    let tidx = (curr_y * grid_map.width + curr_x) as usize;
                    grid_map.terrain[tidx]
                        || snake.body_set.contains(&Position {
                            x: curr_x,
                            y: curr_y,
                        })
                };

            if hit_wall || hit_obstacle {
                let contact_x = curr_x.clamp(0, grid.width - 1);
                let contact_y = curr_y.clamp(0, grid.height - 1);
                let diff_x = (contact_x - head.x) as f32;
                let diff_y = (contact_y - head.y) as f32;
                let dist = (diff_x * diff_x + diff_y * diff_y).sqrt();
                // dist aggiustata: adiacente cardinale (1.0) → 0.0 → exp(0) = 1.0
                let adjusted = (dist - 1.0).max(0.0);
                current_state[i] = (-decay * adjusted).exp();
                break;
            }
        }
    }

    // [8-15] Direzione cibo: dot product, nessun decay (già normalizzato [-1,1])
    let target_dx = (snake.food.x - head.x) as f32;
    let target_dy = (snake.food.y - head.y) as f32;
    let target_dist = (target_dx * target_dx + target_dy * target_dy).sqrt();
    let target_vec = if target_dist > 0.0 {
        (target_dx / target_dist, target_dy / target_dist)
    } else {
        (0.0, 0.0)
    };

    for i in 0..8 {
        let ray_idx = (i + dir_offset) % 8;
        let (dx, dy) = RAY_DIRECTIONS[ray_idx];
        let dir_len = ((dx * dx + dy * dy) as f32).sqrt();
        let dir_vec = (dx as f32 / dir_len, dy as f32 / dir_len);
        current_state[8 + i] = target_vec.0 * dir_vec.0 + target_vec.1 * dir_vec.1;
    }

    // [16] Prossimità cibo: stesso decay aggiustato degli ostacoli
    let adjusted_food = (target_dist - 1.0).max(0.0);
    current_state[16] = (-decay * adjusted_food).exp();

    current_state
}

/// Calculate grid dimensions based on window size
pub fn calculate_grid_dimensions(window_width: f32, window_height: f32) -> (i32, i32) {
    let width = (window_width / BLOCK_SIZE).floor() as i32;
    let height = (window_height / BLOCK_SIZE).floor() as i32;
    (width.max(10), height.max(10))
}
