use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

pub const BLOCK_SIZE: f32 = 20.0;
pub const MEMORY_SIZE: usize = 100_00;
pub const BATCH_SIZE: usize = 256;
pub const TARGET_UPDATE_FREQ: usize = 100;
pub const STATE_SIZE: usize = 8;
pub const TRAIN_INTERVAL: usize = 5;
pub const AUTO_SAVE_INTERVAL: u32 = 100; // Auto-save brain every N generations

/// Configurazione parallelism - numero serpenti = core CPU
#[derive(Resource, Clone)]
pub struct ParallelConfig {
    pub snake_count: usize,
}

impl ParallelConfig {
    pub fn new() -> Self {
        let snake_count = rayon::current_num_threads();
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
    pub steps_per_frame: usize,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            steps_per_frame: 100,
        }
    }
}

/// ECS Components
#[derive(Component)]
pub struct SnakeSegment;

#[derive(Component)]
pub struct Food;

#[derive(Component)]
pub struct SnakeId(pub usize);

#[derive(Component, Clone, Copy, PartialEq, Debug, Default)]
pub struct Position {
    pub x: i32,
    pub y: i32,
}

#[derive(Resource)]
pub struct GameConfig {
    pub speed_timer: Timer,
    pub session_path: std::path::PathBuf,
}

#[derive(Resource)]
pub struct WindowSettings {
    pub is_fullscreen: bool,
}

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

    pub fn get(&self, x: i32, y: i32) -> u8 {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return 1;
        }
        let idx = (y * self.width + x) as usize;
        self.data[idx]
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
    pub total_training_time: Duration,
    pub last_update: Instant,
    pub parallel_threads: usize,
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
    /// Number of records in the current session (from the RL thread)
    pub current_session_records: usize,
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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GameHistoryEntry {
    pub timestamp: String,
    pub generation: u32,
    pub scores: Vec<u32>,
    pub high_score: u32,
    pub total_score: u32,
    pub alive_snakes: usize,
}

#[derive(Resource, Serialize, Deserialize, Debug, Default, Clone)]
pub struct GameHistory {
    pub entries: Vec<GameHistoryEntry>,
    pub max_entries: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GenerationRecord {
    pub gen: u32,
    pub timestamp: u64,
    pub avg_score: f32,
    pub max_score: u32,
    pub min_score: u32,
    pub avg_loss: f32,
    pub epsilon: f32,
}

#[derive(Resource, Serialize, Deserialize, Debug, Default)]
pub struct TrainingSession {
    pub total_time_secs: u64,
    pub records: Vec<GenerationRecord>,
}

impl TrainingSession {
    pub fn new() -> Self {
        Self {
            total_time_secs: 0,
            records: Vec::new(),
        }
    }

    pub fn add_record(&mut self, record: GenerationRecord) {
        self.records.push(record);
    }

    pub fn compress_history(&mut self) {
        const MAX_DETAILED_ENTRIES: usize = 200;
        const COMPRESSION_RATIO: usize = 4;

        if self.records.len() <= MAX_DETAILED_ENTRIES * 2 {
            return;
        }

        let split_idx = self.records.len() - MAX_DETAILED_ENTRIES;
        let recent_records = self.records.split_off(split_idx);
        let old_records = std::mem::take(&mut self.records);

        let mut compressed = Vec::new();
        for chunk in old_records.chunks(COMPRESSION_RATIO) {
            if chunk.is_empty() {
                continue;
            }

            let first = &chunk[0];
            let count = chunk.len() as f32;

            let avg_gen = chunk.iter().map(|r| r.gen).sum::<u32>() / chunk.len() as u32;
            let avg_score = chunk.iter().map(|r| r.avg_score).sum::<f32>() / count;
            let max_score = chunk.iter().map(|r| r.max_score).max().unwrap_or(0);
            let avg_loss = chunk.iter().map(|r| r.avg_loss).sum::<f32>() / count;
            let avg_epsilon = chunk.iter().map(|r| r.epsilon).sum::<f32>() / count;

            compressed.push(GenerationRecord {
                gen: avg_gen,
                timestamp: first.timestamp,
                avg_score,
                max_score,
                min_score: 0,
                avg_loss,
                epsilon: avg_epsilon,
            });
        }

        self.records = compressed;
        self.records.extend(recent_records);
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let session: TrainingSession = serde_json::from_str(&json)?;
        Ok(session)
    }

    pub fn last_generation(&self) -> Option<u32> {
        self.records.last().map(|r| r.gen)
    }
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
    pub epsilon: f32,
}

impl SnakeInstance {
    pub fn generate_random_color() -> Color {
        let mut rng = rand::thread_rng();
        Color::rgb(rng.gen(), rng.gen(), rng.gen())
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
        let (spawn_pos, spawn_dir) = Self::get_random_spawn_data(grid);

        let mut snake = VecDeque::new();
        snake.push_back(spawn_pos);

        // Calcolo epsilon con distribuzione esponenziale (Ape-X style)
        let min_eps = 0.01_f32;
        let max_eps = 0.8_f32;
        let epsilon = if total_snakes <= 1 {
            min_eps
        } else {
            let progress = id as f32 / (total_snakes - 1) as f32;
            min_eps * (max_eps / min_eps).powf(progress)
        };

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
            color: Self::generate_random_color(),
            epsilon,
        }
    }

    pub fn reset(&mut self, grid: &GridDimensions) {
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
        self.color = Self::generate_random_color();
    }
}

#[derive(Resource)]
pub struct GameState {
    pub high_score: u32,
    pub total_iterations: u32,
    pub snakes: Vec<SnakeInstance>,
}

impl GameState {
    pub fn new(grid: &GridDimensions, snake_count: usize) -> Self {
        let snakes = (0..snake_count)
            .map(|id| SnakeInstance::new(id, grid, snake_count))
            .collect();
        Self {
            high_score: 0,
            total_iterations: 0,
            snakes,
        }
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

/// Generate relative coordinate offsets based on current heading
pub fn get_egocentric_directions(current_dir: Direction) -> [(i32, i32); 4] {
    let (fx, fy) = current_dir.as_vec(); // Forward
    let (rx, ry) = current_dir.turn_right().as_vec(); // Right

    [
        (fx, fy),   // 0: Forward
        (rx, ry),   // 1: Right
        (-fx, -fy), // 2: Back
        (-rx, -ry), // 3: Left
    ]
}

/// Get 8-dimensional egocentric state for DQN
/// [obstacle_front, obstacle_right, obstacle_back, obstacle_left,
///  food_front, food_right, food_back, food_left]
pub fn get_state_egocentric(
    snake: &SnakeInstance,
    grid_map: &GridMap,
    grid: &GridDimensions,
) -> [f32; STATE_SIZE] {
    let mut state = [0.0f32; STATE_SIZE];
    let head = snake.snake[0];
    let ego_dirs = get_egocentric_directions(snake.direction);

    // OPTIMIZATION: Rimossa allocazione HashSet - Ora usa grid_map.get() per lookup O(1)
    // Il GridMap contiene già la posizione di tutti i segmenti dei serpenti

    // Obstacle sensors (indices 0-3)
    for (i, (dx, dy)) in ego_dirs.iter().enumerate() {
        let mut dist_val = 0.0;
        let mut curr_x = head.x;
        let mut curr_y = head.y;

        for d in 1..=15 {
            curr_x += dx;
            curr_y += dy;
            // Usa grid_map.get() per O(1) lookup senza allocazioni heap
            if curr_x < 0
                || curr_x >= grid.width
                || curr_y < 0
                || curr_y >= grid.height
                || grid_map.is_collision(curr_x, curr_y, snake.id)
            {
                dist_val = 1.0 / (d as f32);
                break;
            }
        }
        state[i] = dist_val;
    }

    // Food sensors (indices 4-7) - Relative Radar System
    // Transform food position from world coordinates to snake-relative coordinates
    let world_dx = (snake.food.x - head.x) as f32;
    let world_dy = (snake.food.y - head.y) as f32;

    // Get the snake's forward and right vectors in world space
    let (forward_x, forward_y) = ego_dirs[0]; // Forward direction in world coords
    let (right_x, right_y) = ego_dirs[1]; // Right direction in world coords

    // Rotate world vector to snake-relative coordinate system
    // Relative X: projection onto Right axis (positive = right, negative = left)
    // Relative Y: projection onto Forward axis (positive = forward, negative = back)
    let relative_x = world_dx * (right_x as f32) + world_dy * (right_y as f32);
    let relative_y = world_dx * (forward_x as f32) + world_dy * (forward_y as f32);

    state[4] = if relative_y > 0.0 { 1.0 } else { 0.0 }; // Cibo Davanti
    state[5] = if relative_x > 0.0 { 1.0 } else { 0.0 }; // Cibo a Destra
    state[6] = if relative_y < 0.0 { 1.0 } else { 0.0 }; // Cibo Dietro
    state[7] = if relative_x < 0.0 { 1.0 } else { 0.0 }; // Cibo a Sinistra

    state
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

impl GameHistory {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
        }
    }
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

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let stats: GameStats = serde_json::from_str(&json)?;
        Ok(stats)
    }

    pub fn update(&mut self, game: &GameState) {
        self.total_generations = game.total_iterations;
        self.high_score = self.high_score.max(game.high_score);

        for (i, snake) in game.snakes.iter().enumerate() {
            if i < self.best_score_per_snake.len() {
                self.best_score_per_snake[i] = self.best_score_per_snake[i].max(snake.score);
            }
        }

        let now = std::time::SystemTime::now();
        let datetime: chrono::DateTime<chrono::Local> = now.into();
        self.last_saved = Some(datetime.format("%Y-%m-%d %H:%M:%S").to_string());
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
            dirs.sort_by(|a, b| b.cmp(a));
            return dirs[0].clone();
        }
    }

    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
    let new_dir = runs_dir.join(&timestamp);
    std::fs::create_dir_all(&new_dir).ok();
    let sessions_dir = new_dir.join("sessions");
    std::fs::create_dir_all(&sessions_dir).ok();

    println!("📂 Created new run: {}", timestamp);
    new_dir
}

// In snake.rs
pub fn brain_path() -> std::path::PathBuf {
    get_or_create_run_dir().join("brain.bin")
}

pub fn session_path(filename: &str) -> std::path::PathBuf {
    let sessions_dir = get_or_create_run_dir().join("sessions");
    std::fs::create_dir_all(&sessions_dir).ok();
    sessions_dir.join(filename)
}

pub fn new_session_path() -> std::path::PathBuf {
    let timestamp = chrono::Local::now().format("%Y-%m-%d-%H%M%S").to_string();
    session_path(format!("{}.json", timestamp).as_str())
}

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
                        if record.gen > max_gen {
                            max_gen = record.gen;
                        }
                        global_history.records.push(record);
                    }
                }
            }
        }
    }

    global_history.records.sort_by_key(|r| r.gen);

    println!(
        "✅ Global History Loaded: {} records, {}s total",
        global_history.records.len(),
        global_history.accumulated_time_secs
    );

    (global_history, max_gen)
}

pub struct SnakePlugin;

impl Plugin for SnakePlugin {
    fn build(&self, _app: &mut App) {
        // Snake plugin systems are added in main.rs
    }
}

use bevy::sprite::MaterialMesh2dBundle;
use rand::Rng;
