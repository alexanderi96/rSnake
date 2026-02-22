use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::OnceLock;
use std::time::Instant;

use uuid::Uuid;

pub const BLOCK_SIZE: f32 = 20.0;
/// Base state size for a single frame (8 obstacle + 8 target direction + 1 distance)
pub const BASE_STATE_SIZE: usize = 17;
/// Total state size with frame stacking (current frame + previous frame)
pub const STATE_SIZE: usize = BASE_STATE_SIZE * 2; // 34

/// Numero di thread/core del sistema - inizializzato una sola volta
static NUM_PARALLEL_THREADS: OnceLock<usize> = OnceLock::new();

/// Inizializza e restituisce il numero di thread paralleli disponibili
pub fn init_parallel_threads() -> usize {
    let threads = rayon::current_num_threads();
    NUM_PARALLEL_THREADS.get_or_init(|| threads);
    threads
}

/// Configurazione parallelism - snake_count separato dai core CPU
#[derive(Resource, Clone)]
#[allow(dead_code)]
pub struct ParallelConfig {
    pub cpu_cores: usize,   // Numero di core CPU (solo per logging)
    pub snake_count: usize, // Numero di serpenti (dal config)
}

impl ParallelConfig {
    pub fn new(snake_count: usize) -> Self {
        let cpu_cores = init_parallel_threads();
        println!("CPU cores: {}, Population size: {}", cpu_cores, snake_count);
        Self {
            cpu_cores,
            snake_count,
        }
    }
}

/// Configurazione rendering - toggle per accelerare training
#[derive(Resource)]
pub struct RenderConfig {
    pub enabled: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// ECS Components
#[derive(Component)]
pub struct SnakeSegment;

#[derive(Component)]
pub struct Food;

#[derive(Component)]
#[allow(dead_code)]
pub struct SnakeId(pub usize);

#[derive(Component, Clone, Copy, PartialEq, Debug, Default)]
pub struct Position {
    pub x: i32,
    pub y: i32,
}

#[derive(Resource, Default)]
pub struct GameConfig {}

/// Risorse per il caching delle mesh
#[derive(Resource)]
pub struct MeshCache {
    pub segment_mesh: Handle<Mesh>,
    pub food_mesh: Handle<Mesh>,
    pub food_material: Handle<ColorMaterial>,
    pub head_material: Handle<ColorMaterial>,
}

/// Object pool for snake segment entities
#[derive(Resource, Default)]
pub struct SegmentPool {
    pub pools: Vec<Vec<Entity>>,
    pub active_counts: Vec<usize>,
}

impl SegmentPool {
    pub fn new(snake_count: usize) -> Self {
        Self {
            pools: vec![Vec::new(); snake_count],
            active_counts: vec![0; snake_count],
        }
    }

    pub fn get_or_spawn(
        &mut self,
        commands: &mut Commands,
        snake_id: usize,
        segment_idx: usize,
        mesh: Handle<Mesh>,
        material: Handle<ColorMaterial>,
        transform: Transform,
    ) -> Entity {
        if snake_id >= self.pools.len() {
            self.pools.resize_with(snake_id + 1, Vec::new);
            self.active_counts.resize(snake_id + 1, 0);
        }

        let pool = &mut self.pools[snake_id];

        if segment_idx < pool.len() {
            let entity = pool[segment_idx];
            commands
                .entity(entity)
                .insert((mesh, material, transform, Visibility::Visible));
            entity
        } else {
            let entity = commands
                .spawn((
                    MaterialMesh2dBundle {
                        mesh: mesh.into(),
                        material,
                        transform,
                        ..default()
                    },
                    SnakeSegment,
                    SnakeId(snake_id),
                ))
                .id();
            pool.push(entity);
            entity
        }
    }

    pub fn hide_excess(&mut self, commands: &mut Commands, snake_id: usize, needed_count: usize) {
        if snake_id >= self.pools.len() {
            return;
        }

        let pool = &self.pools[snake_id];
        for i in needed_count..pool.len() {
            commands.entity(pool[i]).insert(Visibility::Hidden);
        }

        if snake_id < self.active_counts.len() {
            self.active_counts[snake_id] = needed_count;
        }
    }

    pub fn set_active_count(&mut self, snake_id: usize, count: usize) {
        if snake_id >= self.active_counts.len() {
            self.active_counts.resize(snake_id + 1, 0);
        }
        self.active_counts[snake_id] = count;
    }
}

#[derive(Resource)]
pub struct CollisionSettings {
    pub snake_vs_snake: bool,
}

impl Default for CollisionSettings {
    fn default() -> Self {
        Self {
            snake_vs_snake: false,
        }
    }
}

/// GridMap for O(1) collision detection
#[derive(Resource)]
pub struct GridMap {
    pub width: i32,
    pub height: i32,
    pub data: Vec<u8>,
}

impl GridMap {
    pub fn new(width: i32, height: i32) -> Self {
        let size = (width * height) as usize;
        Self {
            width,
            height,
            data: vec![0; size],
        }
    }

    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    pub fn set(&mut self, x: i32, y: i32, value: u8) {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return;
        }
        let idx = (y * self.width + x) as usize;
        self.data[idx] = value;
    }

    pub fn is_collision(&self, x: i32, y: i32, self_snake_id: usize) -> bool {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return true;
        }
        let idx = (y * self.width + x) as usize;
        let cell = self.data[idx];
        cell != 0 && cell != (self_snake_id + 1) as u8
    }

    /// Controlla solo muri (usato quando snake_vs_snake = false)
    pub fn is_wall_collision(&self, x: i32, y: i32) -> bool {
        x < 0 || x >= self.width || y < 0 || y >= self.height
    }
}

#[derive(Resource)]
pub struct GridDimensions {
    pub width: i32,
    pub height: i32,
}

#[derive(Resource)]
pub struct TrainingStats {
    pub fps: f32,
    pub last_fps_update: Instant,
    pub frame_count: u32,
}

#[derive(Resource)]
pub struct AppStartTime(pub Instant);

impl Default for AppStartTime {
    fn default() -> Self {
        Self(Instant::now())
    }
}

#[derive(Resource, Default)]
pub struct GlobalTrainingHistory {
    pub records: Vec<GenerationRecord>,
    pub accumulated_time_secs: u64,
}

#[derive(Resource, Serialize, Deserialize, Debug, Default, Clone)]
pub struct GameStats {
    pub total_generations: u32,
    pub high_score: u32,
    pub total_games_played: u64,
    pub total_food_eaten: u64,
    pub best_score_per_snake: Vec<u32>,
    pub last_saved: Option<String>,
}

// Use GenerationRecord from evolution module
pub use crate::evolution::GenerationRecord;

#[derive(Resource, Serialize, Deserialize, Debug, Default)]
pub struct TrainingSession {
    pub total_time_secs: u64,
    pub records: Vec<GenerationRecord>,
}

#[derive(Clone)]
pub struct SnakeInstance {
    pub id: usize,
    /// UUIDv7 per identificazione univoca dell'individuo
    pub uuid: uuid::Uuid,
    pub snake: VecDeque<Position>,
    pub food: Position,
    pub direction: Direction,
    pub is_game_over: bool,
    pub steps_without_food: u32,
    pub score: u32,
    pub color: Color,
    /// Previous frame state for temporal frame stacking [f32; 17]
    pub previous_state: [f32; BASE_STATE_SIZE],
    /// Frames survived (for fitness calculation)
    pub frames_survived: u32,
    /// Unique grid cells visited by the snake head (for exploration ratio)
    pub visited_cells: std::collections::HashSet<(i32, i32)>,
    /// Number of turns made (for agility descriptor)
    pub turn_count: u32,
    /// Previous action (for turn detection)
    pub previous_action: crate::brain::Action,
    /// Sum of frames spent to reach each apple (for per-apple efficiency)
    pub food_time_sum: u64,
}

impl SnakeInstance {
    /// Calcola il colore comportamentale basato su:
    /// - R (Rosso): Courage (più vicino ai muri = più rosso)
    /// - G (Verde): Agility (più curve = più verde)
    /// - B (Blu): Fitness relativa (fitness / best_fitness_globale)
    pub fn calculate_behavioral_color(
        courage: f64,
        agility: f64,
        fitness: f64,
        best_fitness: f64,
    ) -> Color {
        let r = courage as f32; // 0.0 - 1.0
        let g = agility as f32; // 0.0 - 1.0
                                // Fitness relativa: se best_fitness è 0, usa 0.5 come default
        let b = if best_fitness > 0.0 {
            (fitness / best_fitness).clamp(0.0, 1.0) as f32
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

    /// Create a new snake with default behavioral color (for backwards compatibility)
    pub fn new(id: usize, grid: &GridDimensions, total_snakes: usize) -> Self {
        // Valori di default: courage=0.5, agility=0.5, fitness=0, best_fitness=1
        // Questo produce un colore neutro (0.5, 0.5, 0.0)
        Self::new_with_color(id, grid, total_snakes, None, 0.5, 0.5, 0.0, 1.0)
    }

    /// Create a new snake with behavioral color based on parent's traits
    /// Il colore comportamentale viene calcolato da courage/agility/fitness del genitore
    pub fn new_with_color(
        id: usize,
        grid: &GridDimensions,
        _total_snakes: usize,
        _genetic_color: Option<crate::brain::GenomeColor>,
        parent_courage: f64,
        parent_agility: f64,
        parent_fitness: f64,
        best_fitness: f64,
    ) -> Self {
        let (spawn_pos, spawn_dir) = Self::get_random_spawn_data(grid);

        let mut snake = VecDeque::new();
        snake.push_back(spawn_pos);

        // Usa colore comportamentale basato sui tratti del genitore
        let color = Self::calculate_behavioral_color(
            parent_courage,
            parent_agility,
            parent_fitness,
            best_fitness,
        );

        Self {
            id,
            uuid: Uuid::now_v7(),
            snake,
            food: Position {
                x: (spawn_pos.x + 5) % grid.width,
                y: (spawn_pos.y + 5) % grid.height,
            },
            direction: spawn_dir,
            is_game_over: false,
            steps_without_food: 0,
            score: 0,
            color,
            previous_state: [0.0; BASE_STATE_SIZE],
            frames_survived: 0,
            visited_cells: std::collections::HashSet::new(),
            turn_count: 0,
            previous_action: crate::brain::Action::Straight,
            food_time_sum: 0,
        }
    }

    pub fn reset(&mut self, grid: &GridDimensions, total_snakes: usize) {
        // Default behavioral values (neutral color)
        self.reset_with_behavioral_color(grid, total_snakes, 0.5, 0.5, 0.0, 1.0);
    }

    /// Reset snake with behavioral color based on parent's traits
    pub fn reset_with_behavioral_color(
        &mut self,
        grid: &GridDimensions,
        _total_snakes: usize,
        parent_courage: f64,
        parent_agility: f64,
        parent_fitness: f64,
        best_fitness: f64,
    ) {
        let (spawn_pos, spawn_dir) = Self::get_random_spawn_data(grid);

        self.uuid = Uuid::now_v7(); // Nuovo UUID per ogni vita
        self.snake.clear();
        self.snake.push_back(spawn_pos);
        self.direction = spawn_dir;
        self.is_game_over = false;
        self.steps_without_food = 0;
        self.score = 0;
        self.food = Position {
            x: (spawn_pos.x + 5) % grid.width,
            y: (spawn_pos.y + 5) % grid.height,
        };
        // Calcola colore comportamentale basato sui tratti del genitore
        self.color = Self::calculate_behavioral_color(
            parent_courage,
            parent_agility,
            parent_fitness,
            best_fitness,
        );
        self.previous_state = [0.0; BASE_STATE_SIZE];
        self.frames_survived = 0;
        self.visited_cells.clear();
        self.turn_count = 0;
        self.previous_action = crate::brain::Action::Straight;
        self.food_time_sum = 0;
    }

    /// Exploration ratio: fraction of grid cells visited by the snake head
    /// 0.0 = stayed in one spot, 1.0 = visited every cell
    pub fn exploration_ratio(&self, grid: &GridDimensions) -> f64 {
        let total_cells = (grid.width * grid.height) as f64;
        (self.visited_cells.len() as f64 / total_cells).clamp(0.0, 1.0)
    }

    /// Calculate agility descriptor (turn frequency)
    /// 0.0 = always straight, 1.0 = turns every frame
    pub fn agility(&self) -> f64 {
        if self.frames_survived == 0 {
            return 0.0;
        }
        (self.turn_count as f64 / self.frames_survived as f64).clamp(0.0, 1.0)
    }

    /// Calculate fitness based on per-apple efficiency with quadratic penalty
    /// Quadratic penalty makes circling much more costly than linear
    pub fn fitness(&self, _grid: &GridDimensions) -> f64 {
        if self.score == 0 {
            return 0.0;
        }

        // Timeout base per serpente di lunghezza 1 (60 + 8*1 = 68)
        // Usiamo questo come riferimento per normalizzare l'efficienza
        let baseline_timeout = 68.0_f64;

        let avg_frames = self.food_time_sum as f64 / self.score as f64;

        // Efficiency: 1.0 = mangiato immediatamente, 0.0 = mangiato all'ultimo momento
        // Clamp a 0 per evitare valori negativi se avg_frames > baseline
        let efficiency = (1.0 - avg_frames / baseline_timeout).clamp(0.0, 1.0);

        // Penalità QUADRATICA: il circling crolla molto più velocemente
        // efficiency=0.85 (10 frames) → reward=724
        // efficiency=0.50 (34 frames) → reward=250
        // efficiency=0.15 (58 frames) → reward=22
        let per_apple_reward = efficiency * efficiency * 1000.0;

        // Scala con numero di mele mangiate
        (per_apple_reward * self.score as f64).max(0.0)
    }
}

#[derive(Resource)]
pub struct GameState {
    pub high_score: u32,
    pub total_iterations: u32,
    pub snakes: Vec<SnakeInstance>,
}

impl GameState {
    #[allow(dead_code)]
    pub fn new(grid: &GridDimensions, snake_count: usize) -> Self {
        Self::new_with_behavioral_colors(grid, snake_count, None)
    }

    /// Create GameState with behavioral colors based on parent's courage/agility/fitness
    /// Each tuple: (parent_courage, parent_agility, parent_fitness, best_fitness)
    pub fn new_with_behavioral_colors(
        grid: &GridDimensions,
        snake_count: usize,
        behaviors: Option<Vec<(f64, f64, f64, f64)>>,
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
        }
    }

    /// Count alive snakes
    pub fn alive_count(&self) -> usize {
        self.snakes.iter().filter(|s| !s.is_game_over).count()
    }
}

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum Direction {
    #[default]
    Right,
    Up,
    Down,
    Left,
}

impl Direction {
    pub fn as_vec(&self) -> (i32, i32) {
        match self {
            Direction::Up => (0, 1),
            Direction::Down => (0, -1),
            Direction::Left => (-1, 0),
            Direction::Right => (1, 0),
        }
    }

    pub fn turn_right(&self) -> Self {
        match self {
            Direction::Up => Direction::Right,
            Direction::Right => Direction::Down,
            Direction::Down => Direction::Left,
            Direction::Left => Direction::Up,
        }
    }

    pub fn turn_left(&self) -> Self {
        match self {
            Direction::Up => Direction::Left,
            Direction::Left => Direction::Down,
            Direction::Down => Direction::Right,
            Direction::Right => Direction::Up,
        }
    }
}

/// 8 compass directions (N, NE, E, SE, S, SW, W, NW)
/// These are normalized direction vectors for raycasting
const RAY_DIRECTIONS: [(i32, i32); 8] = [
    (0, 1),   // N
    (1, 1),   // NE
    (1, 0),   // E
    (1, -1),  // SE
    (0, -1),  // S
    (-1, -1), // SW
    (-1, 0),  // W
    (-1, 1),  // NW
];

/// Calculate current 17 sensors in pure read-only mode.
/// This function is thread-safe and can be used with par_iter().
///
/// SENSORS ARE EGOCENTRIC (First-Person):
/// - Sensor 0 (N) is always where the snake is looking (Forward)
/// - Sensor 2 (E) is always to the snake's Right
/// - Sensor 4 (S) is always Behind the snake
/// - Sensor 6 (W) is always to the snake's Left
///
/// Sensors 0-7: Obstacle raycasting (8 directions) - exponential decay based on Euclidean distance
/// Sensors 8-15: Target direction dot products (8 directions) - value = max(0.0, dot_product)
/// Sensor 16: Target distance - exponential decay based on Euclidean distance
pub fn get_current_17_state(
    snake: &SnakeInstance,
    grid_map: &GridMap,
    grid: &GridDimensions,
) -> [f32; BASE_STATE_SIZE] {
    let decay_rate = 0.1_f32;
    let mut current_state = [0.0f32; BASE_STATE_SIZE];
    let head = snake.snake[0];

    // Direction offset: N=0, E=2, S=4, W=6 (indices in RAY_DIRECTIONS)
    let dir_offset: usize = match snake.direction {
        Direction::Up => 0,    // N
        Direction::Right => 2, // E
        Direction::Down => 4,  // S
        Direction::Left => 6,  // W
    };

    // === SENSORS 0-7: OBSTACLE RAYCASTING ===
    // Cast rays in 8 egocentric directions with exponential decay
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
            let hit_obstacle = !hit_wall && grid_map.is_collision(curr_x, curr_y, snake.id);

            if hit_wall || hit_obstacle {
                // Calcola distanza fino all'ultimo punto VALIDO (prima dell'ostacolo)
                // per i muri, il punto di contatto è sul bordo della griglia
                let contact_x = curr_x.clamp(0, grid.width - 1);
                let contact_y = curr_y.clamp(0, grid.height - 1);
                let diff_x = (contact_x - head.x) as f32;
                let diff_y = (contact_y - head.y) as f32;
                let euclidean_dist = (diff_x * diff_x + diff_y * diff_y).sqrt();
                current_state[i] = (-decay_rate * euclidean_dist).exp();
                break;
            }
        }
    }

    // === SENSORS 8-15: TARGET DIRECTION DOT PRODUCTS ===
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

        let dot_product = target_vec.0 * dir_vec.0 + target_vec.1 * dir_vec.1;
        current_state[8 + i] = dot_product.max(0.0);
    }

    // === SENSOR 16: TARGET DISTANCE ===
    let target_euclidean_dist = target_dist; // Already calculated as Euclidean
    current_state[16] = (-decay_rate * target_euclidean_dist).exp();

    current_state
}

pub fn spawn_food(snake: &SnakeInstance, grid: &GridDimensions) -> Position {
    let mut rng = rand::thread_rng();
    loop {
        let x = rng.gen_range(0..grid.width);
        let y = rng.gen_range(0..grid.height);
        let pos = Position { x, y };
        if !snake.snake.contains(&pos) {
            return pos;
        }
    }
}

pub fn calculate_grid_dimensions(window_width: f32, window_height: f32) -> (i32, i32) {
    let ui_padding = 60.0;
    let available_height = window_height - ui_padding;

    let width = (window_width / BLOCK_SIZE).floor() as i32;
    let height = (available_height / BLOCK_SIZE).floor() as i32;

    (width.max(10), height.max(10))
}

impl GameStats {
    pub fn new(num_snakes: usize) -> Self {
        Self {
            total_generations: 0,
            high_score: 0,
            total_games_played: 0,
            total_food_eaten: 0,
            best_score_per_snake: vec![0; num_snakes],
            last_saved: None,
        }
    }
}

// --- GESTIONE CARTELLE DI SALVATAGGIO ---

pub fn get_or_create_run_dir() -> std::path::PathBuf {
    let runs_dir = std::path::PathBuf::from("runs");
    std::fs::create_dir_all(&runs_dir).ok();

    if let Ok(entries) = std::fs::read_dir(&runs_dir) {
        let mut dirs: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .map(|e| e.path())
            .collect();

        if !dirs.is_empty() {
            // UUIDv7 are lexicographically sortable, so we can just sort
            dirs.sort_by(|a, b| b.cmp(a));
            return dirs[0].clone();
        }
    }

    // Generate new UUIDv7 (time-ordered, collision-free)
    let uuid = Uuid::now_v7();
    let new_dir = runs_dir.join(uuid.to_string());
    std::fs::create_dir_all(&new_dir).ok();
    let sessions_dir = new_dir.join("sessions");
    std::fs::create_dir_all(&sessions_dir).ok();

    println!("📂 Created new run: {}", uuid);
    new_dir
}

// In snake.rs

pub fn session_path(filename: &str) -> std::path::PathBuf {
    let sessions_dir = get_or_create_run_dir().join("sessions");
    std::fs::create_dir_all(&sessions_dir).ok();
    sessions_dir.join(filename)
}

pub fn new_session_path() -> std::path::PathBuf {
    // Generate UUIDv7 for session file
    let uuid = Uuid::now_v7();
    session_path(format!("{}.json", uuid).as_str())
}

/// Load global training history from previous sessions
/// Returns (history, max_generation_found)
pub fn load_global_history() -> (GlobalTrainingHistory, u32) {
    let sessions_dir = get_or_create_run_dir().join("sessions");
    let mut global_history = GlobalTrainingHistory::default();
    let mut max_gen = 0;

    if let Ok(entries) = std::fs::read_dir(sessions_dir) {
        let mut paths: Vec<_> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "json"))
            .collect();

        paths.sort();

        for path in paths {
            if let Ok(content) = std::fs::read_to_string(&path) {
                if let Ok(session) = serde_json::from_str::<TrainingSession>(&content) {
                    global_history.accumulated_time_secs += session.total_time_secs;

                    for record in session.records {
                        if record.generation > max_gen {
                            max_gen = record.generation;
                        }
                        global_history.records.push(record);
                    }
                }
            }
        }
    }

    global_history.records.sort_by_key(|r| r.generation);

    println!(
        "✅ Global History Loaded: {} records, {}s total, max gen: {}",
        global_history.records.len(),
        global_history.accumulated_time_secs,
        max_gen
    );

    (global_history, max_gen)
}

/// Save training session (history + stats) to file
pub fn save_training_session(
    session_path: &std::path::Path,
    history: &GlobalTrainingHistory,
    _game_stats: &GameStats,
    session_duration_secs: u64,
) -> std::io::Result<()> {
    let session = TrainingSession {
        total_time_secs: session_duration_secs,
        records: history.records.clone(),
    };

    let json = serde_json::to_string_pretty(&session)?;
    std::fs::write(session_path, json)?;

    println!("💾 Session saved to: {}", session_path.display());
    Ok(())
}

use bevy::sprite::MaterialMesh2dBundle;
use rand::Rng;
