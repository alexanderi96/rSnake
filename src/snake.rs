use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::OnceLock;
use std::time::Instant;

use crate::map_elites::MapElitesArchive;
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

/// Configurazione parallelism - numero serpenti = core CPU
#[derive(Resource, Clone)]
pub struct ParallelConfig {
    pub snake_count: usize,
}

impl ParallelConfig {
    pub fn new() -> Self {
        let snake_count = init_parallel_threads();
        println!(
            "CPU cores: {}, Parallel snakes: {}",
            snake_count, snake_count
        );
        Self { snake_count }
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

    pub fn is_collision_no_snakes(&self, x: i32, y: i32) -> bool {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return true;
        }
        false
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
    /// Sum of wall distances (for courage descriptor)
    pub wall_distance_sum: f64,
    /// Number of turns made (for agility descriptor)
    pub turn_count: u32,
    /// Previous action (for turn detection)
    pub previous_action: crate::brain::Action,
    /// Frame when food was last eaten (for efficiency bonus)
    pub last_food_frame: u32,
    /// Frame when simulation started
    pub start_frame: u32,
}

impl SnakeInstance {
    /// Assegna un colore basato sull'ID del serpente:
    /// - Agente 0: sempre verde
    /// - Altri agenti: gradiente da Rosso a Blu
    fn assign_color(id: usize, total_snakes: usize) -> Color {
        if id == 0 {
            return Color::rgb(0.0, 1.0, 0.0); // Agente 0 sempre Verde
        }
        if total_snakes <= 2 {
            return Color::rgb(1.0, 0.2, 0.2); // Fallback se pochi agenti
        }

        // Gradiente da Rosso (start) a Blu (end)
        let start = Color::rgb(1.0, 0.2, 0.2);
        let end = Color::rgb(0.2, 0.2, 1.0);

        // Normalizza l'indice tra 0.0 e 1.0 per gli agenti da 1 a N-1
        let t = (id - 1) as f32 / (total_snakes - 2) as f32;

        // Interpolazione lineare RGB manuale
        Color::rgb(
            start.r() + (end.r() - start.r()) * t,
            start.g() + (end.g() - start.g()) * t,
            start.b() + (end.b() - start.b()) * t,
        )
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
        Self::new_with_color(id, grid, total_snakes, None)
    }

    /// Create a new snake with a specific genetic color (or None for random ID-based color)
    pub fn new_with_color(
        id: usize,
        grid: &GridDimensions,
        total_snakes: usize,
        genetic_color: Option<crate::brain::GenomeColor>,
    ) -> Self {
        let (spawn_pos, spawn_dir) = Self::get_random_spawn_data(grid);

        let mut snake = VecDeque::new();
        snake.push_back(spawn_pos);

        // Use genetic color if provided, otherwise fallback to ID-based color
        let color = genetic_color
            .map(|c| c.to_bevy_color())
            .unwrap_or_else(|| Self::assign_color(id, total_snakes));

        Self {
            id,
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
            wall_distance_sum: 0.0,
            turn_count: 0,
            previous_action: crate::brain::Action::Straight,
            last_food_frame: 0,
            start_frame: 0,
        }
    }

    pub fn reset(&mut self, grid: &GridDimensions, total_snakes: usize) {
        self.reset_with_color(grid, total_snakes, None);
    }

    /// Reset snake with a specific genetic color (or None to keep existing color)
    pub fn reset_with_color(
        &mut self,
        grid: &GridDimensions,
        _total_snakes: usize,
        genetic_color: Option<crate::brain::GenomeColor>,
    ) {
        let (spawn_pos, spawn_dir) = Self::get_random_spawn_data(grid);

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
        // Use genetic color if provided, otherwise keep existing color
        if let Some(c) = genetic_color {
            self.color = c.to_bevy_color();
        }
        self.previous_state = [0.0; BASE_STATE_SIZE];
        self.frames_survived = 0;
        self.wall_distance_sum = 0.0;
        self.turn_count = 0;
        self.previous_action = crate::brain::Action::Straight;
        self.last_food_frame = 0;
        self.start_frame = 0;
    }

    /// Calculate courage descriptor (average distance from walls)
    pub fn courage(&self) -> f64 {
        if self.frames_survived == 0 {
            0.5
        } else {
            (self.wall_distance_sum / self.frames_survived as f64).clamp(0.0, 1.0)
        }
    }

    /// Calculate agility descriptor (turn frequency)
    pub fn agility(&self) -> f64 {
        if self.frames_survived == 0 {
            0.5
        } else {
            (self.turn_count as f64 / self.frames_survived as f64).clamp(0.0, 1.0)
        }
    }

    /// Calculate par time (ideal time) to reach food based on grid dimensions and snake length
    /// Returns the ideal number of frames to reach the food
    pub fn calculate_par_time(&self, grid: &GridDimensions) -> u32 {
        let grid_size = (grid.width + grid.height) as f64;
        // Base par time on grid diagonal-ish measure
        let base_par = (grid_size * 0.5) as u32;
        // Adjust based on current snake length (longer snakes = slightly more time allowed)
        let length_adjustment = (self.snake.len() as f64 * 0.1) as u32;
        base_par + length_adjustment
    }

    /// Calculate efficiency bonus: reward for reaching food faster than par time
    fn efficiency_bonus(&self, grid: &GridDimensions) -> f64 {
        if self.frames_survived == 0 || self.score == 0 {
            return 0.0;
        }

        let par_time = self.calculate_par_time(grid);

        // If snake reached food faster than par time, award bonus
        // We estimate food eaten frequency based on score
        // Approximate frames per apple = total frames / score
        let frames_per_apple = if self.score > 0 {
            self.frames_survived / self.score
        } else {
            u32::MAX
        };

        if frames_per_apple < par_time {
            // Bonus proportional to time saved
            let time_saved = par_time as f64 - frames_per_apple as f64;
            // Scale bonus: more for greater efficiency
            let bonus = time_saved * 2.0;
            bonus.min(500.0) // Cap the bonus to avoid runaway values
        } else {
            0.0
        }
    }

    /// Calculate improved fitness function
    /// Combines:
    /// - Food reward (high weight)
    /// - Survival frames (base reward)
    /// - Death penalty (negative)
    /// - Minimum constant reward (encourage early performance)
    /// - Efficiency bonus (reward for reaching food quickly)
    pub fn fitness(&self, grid: &GridDimensions) -> f64 {
        // Constants for fitness calculation
        const FOOD_REWARD: f64 = 1000.0; // High reward per apple
        const SURVIVAL_REWARD: f64 = 1.0; // Base reward per frame survived
        const DEATH_PENALTY: f64 = -100.0; // Penalty when snake dies
        const MIN_REWARD: f64 = 0.1; // Minimum constant reward per frame

        // Base components
        let food_reward = (self.score as f64) * FOOD_REWARD;
        let survival_reward = (self.frames_survived as f64) * SURVIVAL_REWARD;

        // Death penalty (only if dead)
        let death_penalty = if self.is_game_over {
            DEATH_PENALTY
        } else {
            0.0
        };

        // Minimum constant reward to encourage early performance
        let min_reward = (self.frames_survived as f64) * MIN_REWARD;

        // Efficiency bonus for reaching food quickly
        let efficiency_bonus = self.efficiency_bonus(grid);

        // Total fitness
        food_reward + survival_reward + death_penalty + min_reward + efficiency_bonus
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
        Self::new_with_colors(grid, snake_count, None)
    }

    /// Create GameState with optional genetic colors
    pub fn new_with_colors(
        grid: &GridDimensions,
        snake_count: usize,
        colors: Option<Vec<crate::brain::GenomeColor>>,
    ) -> Self {
        let snakes = if let Some(colors) = colors {
            (0..snake_count)
                .map(|id| {
                    SnakeInstance::new_with_color(id, grid, snake_count, colors.get(id).copied())
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
/// Sensors 0-7: Obstacle raycasting (8 directions) - value = 1.0 / distance
/// Sensors 8-15: Target direction dot products (8 directions) - value = max(0.0, dot_product)
/// Sensor 16: Target distance - value = 1.0 / absolute_distance
pub fn get_current_17_state(
    snake: &SnakeInstance,
    grid_map: &GridMap,
    grid: &GridDimensions,
) -> [f32; BASE_STATE_SIZE] {
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
    // Cast rays in 8 egocentric directions
    for i in 0..8 {
        let ray_idx = (i + dir_offset) % 8;
        let (dx, dy) = RAY_DIRECTIONS[ray_idx];

        let mut curr_x = head.x;
        let mut curr_y = head.y;
        let mut distance = 1;

        loop {
            curr_x += dx;
            curr_y += dy;

            if curr_x < 0
                || curr_x >= grid.width
                || curr_y < 0
                || curr_y >= grid.height
                || grid_map.is_collision(curr_x, curr_y, snake.id)
            {
                current_state[i] = 1.0 / (distance as f32);
                break;
            }

            distance += 1;

            if distance > grid.width.max(grid.height) {
                current_state[i] = 0.0;
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
    let max_possible_dist = ((grid.width * grid.width + grid.height * grid.height) as f32).sqrt();

    current_state[16] = if target_dist > 0.0 {
        1.0 - (target_dist / max_possible_dist) // Valore lineare da 1.0 (vicinissimo) a 0.0 (lontanissimo)
    } else {
        1.0
    };

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

/// Find the most recent session file in the sessions directory
#[allow(dead_code)]
pub fn find_latest_session() -> Option<std::path::PathBuf> {
    let sessions_dir = get_or_create_run_dir().join("sessions");

    if let Ok(entries) = std::fs::read_dir(sessions_dir) {
        let mut paths: Vec<_> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "json"))
            .collect();

        if paths.is_empty() {
            return None;
        }

        // Sort by filename (which contains timestamp)
        paths.sort();
        paths.last().cloned()
    } else {
        None
    }
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

/// Load GameStats from the latest session file
#[allow(dead_code)]
pub fn load_game_stats() -> Option<GameStats> {
    let latest_session = find_latest_session()?;

    if let Ok(content) = std::fs::read_to_string(&latest_session) {
        if let Ok(session) = serde_json::from_str::<TrainingSession>(&content) {
            // Convert session records to GameStats
            let mut stats = GameStats::new(10); // Default snake count

            for record in &session.records {
                stats.total_generations = record.generation.max(stats.total_generations);
                // Use best_fitness as proxy for high_score
                stats.high_score = stats.high_score.max(record.best_fitness as u32);
            }

            stats.last_saved = Some(latest_session.to_string_lossy().to_string());

            println!("✅ GameStats loaded from: {:?}", latest_session);
            return Some(stats);
        }
    }

    None
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

/// Try to load latest archive, history, and stats for resuming training
#[allow(dead_code)]
pub fn try_resume_training() -> Option<(MapElitesArchive, GlobalTrainingHistory, GameStats)> {
    let run_dir = get_or_create_run_dir();

    // Try to load archive
    let archive_path = run_dir.join("archive.json");
    let archive = if archive_path.exists() {
        match MapElitesArchive::load(archive_path.to_str().unwrap_or("archive.json")) {
            Ok(a) => {
                println!("📂 Resuming with archive: {} elites", a.filled_cells());
                Some(a)
            }
            Err(e) => {
                eprintln!("⚠️ Failed to load archive: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Load history
    let (history, _max_gen) = load_global_history();

    // Load stats
    let stats = load_game_stats().unwrap_or_else(|| GameStats::new(10));

    Some((
        archive.unwrap_or_else(MapElitesArchive::default),
        history,
        stats,
    ))
}

use bevy::sprite::MaterialMesh2dBundle;
use rand::Rng;
