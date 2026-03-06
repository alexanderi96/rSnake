use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};
use std::io::Write;
use std::path::PathBuf;
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

/// Risorsa per tenere traccia della directory della Run attuale
#[derive(Resource, Clone, Debug)]
pub struct RunDirectory(pub PathBuf);

/// Seed condiviso per la generazione corrente.
#[derive(Resource, Clone)]
pub struct GenerationSeed {
    #[allow(dead_code)]
    pub seed: u64,
    pub spawn_pos: Position,
    pub spawn_dir: Direction,
    pub food_sequence: Vec<Position>, // pre-generata, lunga FOOD_SEQ_LEN
    pub terrain: Vec<bool>,
}

pub const FOOD_SEQ_LEN: usize = 1000;

impl GenerationSeed {
    pub fn new_for_grid_with_config(
        grid: &GridDimensions,
        config: &crate::config::Hyperparameters,
    ) -> Self {
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

        let terrain = crate::terrain::generate(
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
            seed,
            spawn_pos,
            spawn_dir,
            food_sequence,
            terrain,
        }
    }

    pub fn new_for_grid(grid: &GridDimensions) -> Self {
        Self::new_for_grid_with_config(grid, &crate::config::Hyperparameters::default())
    }

    pub fn food_at(&self, index: usize) -> Position {
        self.food_sequence[index % FOOD_SEQ_LEN]
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

/// Configurazione rendering
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
pub struct Food;

#[derive(Component)]
#[allow(dead_code)]
pub struct SnakeId(pub usize);

#[derive(Component, Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
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
    /// Static terrain walls generated from seed (true = wall)
    pub terrain: Vec<bool>,
}

impl GridMap {
    pub fn new(width: i32, height: i32) -> Self {
        let size = (width * height) as usize;
        Self {
            width,
            height,
            data: vec![0; size],
            terrain: vec![false; size],
        }
    }

    pub fn clear(&mut self) {
        self.data.fill(0);
        // NOTE: terrain is NOT cleared here — it persists for the whole generation
    }

    /// Copy terrain from a generated terrain slice into the map
    pub fn apply_terrain(&mut self, terrain: &[bool]) {
        debug_assert_eq!(terrain.len(), self.terrain.len());
        self.terrain.copy_from_slice(terrain);
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
        if self.terrain[idx] {
            return true;
        }
        let cell = self.data[idx];
        cell != 0 && cell != (self_snake_id + 1) as u8
    }

    /// Check collision against terrain walls only (no snake bodies)
    pub fn is_wall_collision(&self, x: i32, y: i32) -> bool {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return true;
        }
        self.terrain[(y * self.width + x) as usize]
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

/// Global training history with separated read-only history and current session
/// This prevents the "Matrioska" effect where each session saved all previous data
#[derive(Resource, Default)]
pub struct GlobalTrainingHistory {
    /// Historical records loaded from previous sessions (read-only, from .json/.json.gz files)
    pub history_log: Vec<GenerationRecord>,
    /// Current session records (write-only, saved to new file)
    pub current_session: Vec<GenerationRecord>,
    /// Total accumulated time from all previous sessions
    pub accumulated_time_secs: u64,
    /// All-time high score across all sessions
    pub all_time_high_score: u32,
}

impl GlobalTrainingHistory {
    /// Get all records combined (for UI display)
    pub fn all_records(&self) -> impl Iterator<Item = &GenerationRecord> {
        self.history_log.iter().chain(self.current_session.iter())
    }

    /// Add a record to the current session
    pub fn push(&mut self, record: GenerationRecord) {
        self.current_session.push(record);
    }
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

pub use crate::evolution::GenerationRecord;

#[derive(Resource, Serialize, Deserialize, Debug, Default)]
pub struct TrainingSession {
    pub total_time_secs: u64,
    pub records: Vec<GenerationRecord>,
}

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
    pub turn_count: u32,
    pub previous_action: crate::brain::Action,
    pub food_time_sum: u64,
    pub path_directness_sum: f32,
    pub food_spawn_distance: u32,
    pub body_pressure_sum: f32,
    pub timeout_budget_sum: u64,
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

    pub fn new_with_color(
        id: usize,
        grid: &GridDimensions,
        _total_snakes: usize,
        _genetic_color: Option<crate::brain::GenomeColor>,
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
        let food_spawn_distance =
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
            turn_count: 0,
            previous_action: crate::brain::Action::Straight,
            food_time_sum: 0,
            path_directness_sum: 0.0,
            food_spawn_distance,
            body_pressure_sum: 0.0,
            timeout_budget_sum: 0,
        }
    }

    pub fn reset_with_seed(
        &mut self,
        _grid: &GridDimensions,
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
        self.turn_count = 0;
        self.previous_action = crate::brain::Action::Straight;
        self.food_time_sum = 0;
        self.path_directness_sum = 0.0;
        self.food_spawn_distance = ((self.food.x - seed.spawn_pos.x).abs()
            + (self.food.y - seed.spawn_pos.y).abs()) as u32;
        self.body_pressure_sum = 0.0;
        self.timeout_budget_sum = 0;
    }

    pub fn path_directness(&self) -> f32 {
        if self.score == 0 {
            return 0.0;
        }
        (self.path_directness_sum / self.score as f32).clamp(0.0, 1.0)
    }

    pub fn body_avoidance(&self) -> f32 {
        if self.frames_survived == 0 {
            return 0.0;
        }
        (self.body_pressure_sum / self.frames_survived as f32).clamp(0.0, 1.0)
    }

    /// Funzione di Fitness Bilanciata (Lineare + Bonus Efficienza)
    pub fn fitness(&self, _grid: &GridDimensions) -> f32 {
        // Se non ha mangiato nulla, piccolo premio di sopravvivenza
        if self.score == 0 {
            return (self.frames_survived as f32 * 0.1).min(10.0);
        }

        let base_reward = self.score as f32 * 1000.0;

        // Efficienza: quanto velocemente ha raggiunto il cibo
        let efficiency = if self.timeout_budget_sum > 0 {
            (1.0 - self.food_time_sum as f32 / self.timeout_budget_sum as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Bonus per efficienza (maggiore se ha mangiato più mele)
        let efficiency_bonus = (self.score as f32) * efficiency * 500.0;

        // Premio extra per sopravvivenza (logaritmico per evitare dominanza)
        let survival_reward = if self.score > 0 {
            (self.frames_survived as f32).ln().max(0.0) * 5.0
        } else {
            (self.frames_survived as f32 * 0.1).min(5.0)
        };

        base_reward + efficiency_bonus + survival_reward
    }
}

#[derive(Resource)]
pub struct GameState {
    pub high_score: u32,
    pub total_iterations: u32,
    pub snakes: Vec<SnakeInstance>,
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
        }
    }

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

pub fn get_current_17_state(
    snake: &SnakeInstance,
    grid_map: &GridMap,
    grid: &GridDimensions,
) -> [f32; BASE_STATE_SIZE] {
    let decay_rate = 0.1_f32;
    let mut current_state = [0.0f32; BASE_STATE_SIZE];
    let head = snake.snake[0];

    let dir_offset: usize = match snake.direction {
        Direction::Up => 0,
        Direction::Right => 2,
        Direction::Down => 4,
        Direction::Left => 6,
    };

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
                let contact_x = curr_x.clamp(0, grid.width - 1);
                let contact_y = curr_y.clamp(0, grid.height - 1);
                let diff_x = (contact_x - head.x) as f32;
                let diff_y = (contact_y - head.y) as f32;
                let euclidean_dist = (diff_x * diff_x + diff_y * diff_y).sqrt();

                if euclidean_dist <= 1.0 {
                    current_state[i] = 1.0;
                } else {
                    current_state[i] = (-decay_rate * euclidean_dist).exp();
                }
                break;
            }
        }
    }

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

    let target_euclidean_dist = target_dist;
    current_state[16] = (-decay_rate * target_euclidean_dist).exp();

    current_state
}

pub fn calculate_grid_dimensions(window_width: f32, window_height: f32) -> (i32, i32) {
    let width = (window_width / BLOCK_SIZE).floor() as i32;
    let height = (window_height / BLOCK_SIZE).floor() as i32;
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

// --- GESTIONE CARTELLE E FILE ---

/// Crea o recupera la directory di run.
/// Se `force_new` è true, crea sempre una nuova directory con timestamp/UUID.
/// Se false, tenta di recuperare l'ultima creata.
pub fn get_or_create_run_dir(force_new: bool) -> PathBuf {
    let runs_dir = PathBuf::from("runs");
    std::fs::create_dir_all(&runs_dir).ok();

    if !force_new {
        if let Ok(entries) = std::fs::read_dir(&runs_dir) {
            let mut dirs: Vec<_> = entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .map(|e| e.path())
                .collect();

            if !dirs.is_empty() {
                // Ordina per UUIDv7 (timestamp decrescente)
                dirs.sort_by(|a, b| b.cmp(a));
                println!("Resume last run: {}", dirs[0].display());
                return dirs[0].clone();
            }
        }
    }

    // Crea nuova se richiesto o se non ne esistono
    let uuid = Uuid::now_v7();
    let new_dir = runs_dir.join(uuid.to_string());
    std::fs::create_dir_all(&new_dir).ok();
    let sessions_dir = new_dir.join("sessions");
    std::fs::create_dir_all(&sessions_dir).ok();

    println!("✨ Created NEW run: {}", uuid);
    new_dir
}

pub fn new_session_path(run_dir: &std::path::Path) -> PathBuf {
    let uuid = Uuid::now_v7();
    run_dir.join("sessions").join(format!("{}.json.gz", uuid))
}

pub fn load_global_history(run_dir: &std::path::Path) -> (GlobalTrainingHistory, u32) {
    let sessions_dir = run_dir.join("sessions");
    let mut global_history = GlobalTrainingHistory::default();
    let mut max_gen = 0;

    if let Ok(entries) = std::fs::read_dir(sessions_dir) {
        let mut paths: Vec<_> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                // Accept both .json and .json.gz files
                p.extension()
                    .map_or(false, |ext| ext == "json" || ext == "gz")
            })
            .collect();

        paths.sort();

        for path in paths {
            // Try to load as gzip first, then as plain JSON
            let session_result = load_json_gz::<TrainingSession>(&path);

            if let Ok(session) = session_result {
                global_history.accumulated_time_secs += session.total_time_secs;

                for record in session.records {
                    if record.generation > max_gen {
                        max_gen = record.generation;
                    }
                    global_history.history_log.push(record);
                }
            }
        }
    }

    global_history.history_log.sort_by_key(|r| r.generation);

    global_history.all_time_high_score = global_history
        .history_log
        .iter()
        .map(|r| r.generation_high_score)
        .max()
        .unwrap_or(0);

    (global_history, max_gen)
}

pub fn save_training_session(
    session_path: &std::path::Path,
    history: &GlobalTrainingHistory,
    _game_stats: &GameStats,
    session_duration_secs: u64,
) -> std::io::Result<()> {
    // Save only current session records, not the entire history
    let session = TrainingSession {
        total_time_secs: session_duration_secs,
        records: history.current_session.clone(),
    };

    // Save as gzip-compressed JSON
    save_json_gz(session_path, &session)?;

    println!("💾 Session saved to: {}", session_path.display());
    Ok(())
}

// ============================================================================
// GZIP COMPRESSION HELPERS
// ============================================================================

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;

/// Save JSON to a gzip-compressed file (.json.gz)
pub fn save_json_gz<T: Serialize>(path: &std::path::Path, data: &T) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let encoder = GzEncoder::new(file, Compression::default());
    let mut writer = std::io::BufWriter::new(encoder);
    serde_json::to_writer(&mut writer, data)?;
    writer.flush()?;
    Ok(())
}

/// Load JSON from a gzip-compressed file (.json.gz) or uncompressed (.json)
/// Automatically detects format based on file extension
pub fn load_json_gz<T: for<'de> Deserialize<'de>>(path: &std::path::Path) -> std::io::Result<T> {
    let content = std::fs::read(path)?;

    // Detect gzip via magic bytes (0x1f 0x8b), ignoring extension
    let is_gzip = content.len() >= 2 && content[0] == 0x1f && content[1] == 0x8b;

    if is_gzip {
        let decoder = GzDecoder::new(&content[..]);
        let reader = std::io::BufReader::new(decoder);
        return Ok(serde_json::from_reader(reader)?);
    }

    // Otherwise treat as plain JSON
    Ok(serde_json::from_slice(&content)?)
}

/// Check if a path has gzip compression
pub fn is_gzipped(path: &std::path::Path) -> bool {
    path.extension().map_or(false, |ext| ext == "gz")
}

/// Sanitize old run data: deduplicate records and compress
/// This fixes the "Matrioska" effect where each session saved all previous data
/// Called at startup if old .json files are detected
pub fn sanitize_run_data(run_dir: &std::path::Path) -> std::io::Result<u64> {
    let sessions_dir = run_dir.join("sessions");
    if !sessions_dir.exists() {
        return Ok(0);
    }

    // Find all .json files (non-gzipped)
    let mut old_files: Vec<std::path::PathBuf> = Vec::new();
    let mut compressed_files: Vec<std::path::PathBuf> = Vec::new();
    let mut migrated_exists = false;

    if let Ok(entries) = std::fs::read_dir(&sessions_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

            if ext == "json" {
                old_files.push(path.clone());
            } else if ext == "gz" {
                compressed_files.push(path.clone());
            }
            if path
                .file_name()
                .map_or(false, |n| n == "history_migrated.json.gz")
            {
                migrated_exists = true;
            }
        }
    }

    // If already migrated or no old files, skip
    if migrated_exists || old_files.is_empty() {
        return Ok(0);
    }

    println!(
        "🔧 Sanitizing run data: found {} old .json files",
        old_files.len()
    );

    // Calculate total size before
    let mut total_size_before: u64 = 0;
    for f in &old_files {
        total_size_before += f.metadata().map(|m| m.len()).unwrap_or(0);
    }

    // Load all records from all old files
    let mut all_records: Vec<GenerationRecord> = Vec::new();
    for path in &old_files {
        if let Ok(content) = std::fs::read_to_string(path) {
            if let Ok(session) = serde_json::from_str::<TrainingSession>(&content) {
                all_records.extend(session.records);
            }
        }
    }

    // Deduplicate: keep only the latest record for each generation
    all_records.sort_by_key(|r| r.generation);
    all_records.dedup_by_key(|r| r.generation);
    all_records.sort_by_key(|r| r.generation);

    // Create backup directory
    let backup_dir = sessions_dir.join("backup");
    if !backup_dir.exists() {
        std::fs::create_dir_all(&backup_dir)?;
    }

    // Move old files to backup
    for old_file in &old_files {
        if let Some(name) = old_file.file_name() {
            let backup_path = backup_dir.join(name);
            std::fs::rename(old_file, backup_path)?;
        }
    }

    // Calculate total size after (backup files)
    let mut total_size_after: u64 = 0;
    if let Ok(entries) = std::fs::read_dir(&backup_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            total_size_after += entry.path().metadata().map(|m| m.len()).unwrap_or(0);
        }
    }

    // Save consolidated history as gzip
    let migrated_path = sessions_dir.join("history_migrated.json.gz");
    save_json_gz(&migrated_path, &all_records)?;

    // Calculate new file size
    let new_size = migrated_path.metadata().map(|m| m.len()).unwrap_or(0);
    total_size_after += new_size;

    let saved = if total_size_before > total_size_after {
        total_size_before - total_size_after
    } else {
        0
    };

    println!(
        "✅ Sanitization complete: {} records consolidated, saved {} bytes",
        all_records.len(),
        saved
    );

    Ok(saved)
}

use rand::Rng;
