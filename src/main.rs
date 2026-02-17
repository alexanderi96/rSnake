use bevy::app::AppExit;
use bevy::prelude::*;
use bevy::sprite::MaterialMesh2dBundle;
use bevy::text::TextStyle;
use bevy::window::WindowMode;
use chrono;
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use std::path::Path;

use std::time::{Duration, Instant};

// --- CONFIGURAZIONE ---
const BLOCK_SIZE: f32 = 20.0;
// PARALLEL_SNAKES: impostato dinamicamente da ParallelConfig in base ai core CPU
const MEMORY_SIZE: usize = 5000;
const BATCH_SIZE: usize = 256;
const LEARNING_RATE: f32 = 0.0005;
const HIDDEN_NODES: usize = 128;
const TARGET_UPDATE_FREQ: usize = 100;
const STATE_SIZE: usize = 16;
const TRAIN_INTERVAL: usize = 100; // Train every N global iterations for continuous learning

/// Configurazione parallelism - numero serpenti = core CPU
#[derive(Resource, Clone)]
struct ParallelConfig {
    snake_count: usize,
}

impl ParallelConfig {
    fn new() -> Self {
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
struct RenderConfig {
    enabled: bool,
    steps_per_frame: usize, // Quanti step logici fare per frame video (es. 50 o 100)
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            steps_per_frame: 100, // Fallback/Reference value
        }
    }
}

// --- COMPONENTI ECS ---
#[derive(Component)]
struct SnakeSegment;

#[derive(Component)]
struct Food;

#[derive(Component)]
struct SnakeId(usize);

#[derive(Component, Clone, Copy, PartialEq, Debug, Default)]
struct Position {
    x: i32,
    y: i32,
}

#[derive(Resource)]
struct GameConfig {
    speed_timer: Timer,
    session_path: std::path::PathBuf,
}

#[derive(Resource)]
struct WindowSettings {
    is_fullscreen: bool,
}

/// Risorse per il caching delle mesh (create una sola volta)
#[derive(Resource)]
struct MeshCache {
    segment_mesh: Handle<Mesh>,
    food_mesh: Handle<Mesh>,
    food_material: Handle<ColorMaterial>,
    head_material: Handle<ColorMaterial>,
}

/// Object pool for snake segment entities to avoid despawning/spawning churn
#[derive(Resource, Default)]
struct SegmentPool {
    /// Pool of available entities per snake (indexed by snake_id)
    /// Each snake has its own pool of segment entities
    pools: Vec<Vec<Entity>>,
    /// Tracks how many segments are currently active per snake
    active_counts: Vec<usize>,
}

impl SegmentPool {
    fn new(snake_count: usize) -> Self {
        Self {
            pools: vec![Vec::new(); snake_count],
            active_counts: vec![0; snake_count],
        }
    }

    /// Get or create a segment entity for a snake at a specific segment index
    fn get_or_spawn(
        &mut self,
        commands: &mut Commands,
        snake_id: usize,
        segment_idx: usize,
        mesh: Handle<Mesh>,
        material: Handle<ColorMaterial>,
        transform: Transform,
    ) -> Entity {
        // Ensure pool exists for this snake
        if snake_id >= self.pools.len() {
            self.pools.resize_with(snake_id + 1, Vec::new);
            self.active_counts.resize(snake_id + 1, 0);
        }

        let pool = &mut self.pools[snake_id];

        if segment_idx < pool.len() {
            // Reuse existing entity - update transform and material only
            // Don't re-insert the full bundle to avoid duplicate Visibility component
            let entity = pool[segment_idx];
            commands
                .entity(entity)
                .insert((mesh, material, transform, Visibility::Visible));
            entity
        } else {
            // Spawn new entity
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

    /// Hide excess segments that are no longer needed
    fn hide_excess(&mut self, commands: &mut Commands, snake_id: usize, needed_count: usize) {
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

    /// Update active count for a snake
    fn set_active_count(&mut self, snake_id: usize, count: usize) {
        if snake_id >= self.active_counts.len() {
            self.active_counts.resize(snake_id + 1, 0);
        }
        self.active_counts[snake_id] = count;
    }
}

#[derive(Resource)]
struct CollisionSettings {
    snake_vs_snake: bool,
}

impl Default for CollisionSettings {
    fn default() -> Self {
        Self {
            snake_vs_snake: false,
        }
    }
}

/// GridMap for O(1) collision detection
/// Uses a flattened Vec<u8> where index = y * width + x
/// 0 = empty, 1+ = snake id + 1 (to distinguish from empty)
#[derive(Resource)]
struct GridMap {
    width: i32,
    height: i32,
    data: Vec<u8>,
}

impl GridMap {
    fn new(width: i32, height: i32) -> Self {
        let size = (width * height) as usize;
        Self {
            width,
            height,
            data: vec![0; size],
        }
    }

    fn clear(&mut self) {
        self.data.fill(0);
    }

    fn get(&self, x: i32, y: i32) -> u8 {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return 1; // Treat out of bounds as collision
        }
        let idx = (y * self.width + x) as usize;
        self.data[idx]
    }

    fn set(&mut self, x: i32, y: i32, value: u8) {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return;
        }
        let idx = (y * self.width + x) as usize;
        self.data[idx] = value;
    }

    /// Check collision at position in O(1) - includes collisions with other snakes
    fn is_collision(&self, x: i32, y: i32, self_snake_id: usize) -> bool {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return true; // Wall collision
        }
        let idx = (y * self.width + x) as usize;
        let cell = self.data[idx];
        // Collision if cell is occupied by another snake (different id)
        cell != 0 && cell != (self_snake_id + 1) as u8
    }

    /// Check collision at position in O(1) - wall only, ignores other snakes
    fn is_collision_no_snakes(&self, x: i32, y: i32) -> bool {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return true; // Wall collision only
        }
        false
    }
}

#[derive(Resource)]
struct GridDimensions {
    width: i32,
    height: i32,
}

#[derive(Resource)]
struct TrainingStats {
    total_training_time: Duration,
    last_update: Instant,
    parallel_threads: usize,
    fps: f32,
    last_fps_update: Instant,
    frame_count: u32,
}

/// App start time for calculating current session duration
#[derive(Resource)]
struct AppStartTime(Instant);

impl Default for AppStartTime {
    fn default() -> Self {
        Self(Instant::now())
    }
}

/// Holds ALL historical data (past sessions + current session).
/// Used ONLY for the Graph UI and Total Time display.
#[derive(Resource, Default)]
struct GlobalTrainingHistory {
    pub records: Vec<GenerationRecord>,
    pub accumulated_time_secs: u64, // Sum of all previous sessions' time
}

#[derive(Resource, Serialize, Deserialize, Debug, Default, Clone)]
struct GameStats {
    total_generations: u32,
    high_score: u32,
    total_games_played: u64,
    total_food_eaten: u64,
    // REMOVED: total_training_time_secs - calculated at runtime
    // Total time = GlobalTrainingHistory.accumulated_time_secs + (now - app_start_time)
    best_score_per_snake: Vec<u32>,
    last_saved: Option<String>,
}

/// Singola entry nello storico partite
#[derive(Serialize, Deserialize, Debug, Clone)]
struct GameHistoryEntry {
    timestamp: String,
    generation: u32,
    scores: Vec<u32>,
    high_score: u32,
    total_score: u32,
    alive_snakes: usize,
}

/// Storico completo delle partite
#[derive(Resource, Serialize, Deserialize, Debug, Default, Clone)]
struct GameHistory {
    entries: Vec<GameHistoryEntry>,
    max_entries: usize,
}

// =============================================================================
// NUOVO SISTEMA DI STATISTICHE UNIFICATO
// =============================================================================

/// Record singolo di una generazione con metriche AI
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GenerationRecord {
    pub gen: u32,
    pub timestamp: u64, // Unix timestamp

    // Performance di Gioco
    pub avg_score: f32,
    pub max_score: u32,
    pub min_score: u32,

    // Metriche AI (Nuove)
    pub avg_loss: f32,
    pub epsilon: f32,
}

/// Sessione di training completa con storico compresso
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

    /// Aggiunge un nuovo record di generazione
    pub fn add_record(&mut self, record: GenerationRecord) {
        self.records.push(record);
    }

    /// Comprime lo storico per evitare file JSON troppo grandi
    /// Mantiene i record recenti dettagliati, comprime quelli vecchi
    pub fn compress_history(&mut self) {
        const MAX_DETAILED_ENTRIES: usize = 200;
        const COMPRESSION_RATIO: usize = 4; // Unisci 4 vecchi record in 1

        if self.records.len() <= MAX_DETAILED_ENTRIES * 2 {
            return; // Non c'è bisogno di comprimere
        }

        let split_idx = self.records.len() - MAX_DETAILED_ENTRIES;

        // Separa i vecchi dai nuovi
        let recent_records = self.records.split_off(split_idx);
        let old_records = std::mem::take(&mut self.records);

        // Comprimi i vecchi
        let mut compressed = Vec::new();
        for chunk in old_records.chunks(COMPRESSION_RATIO) {
            if chunk.is_empty() {
                continue;
            }

            let first = &chunk[0];
            let count = chunk.len() as f32;

            // Calcola medie
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
                min_score: 0, // Non ha senso preservare min nei record compressi
                avg_loss,
                epsilon: avg_epsilon,
            });
        }

        // Unisci: Vecchi Compressi + Recenti Dettagliati
        let total_old = old_records.len() + recent_records.len();
        self.records = compressed;
        self.records.extend(recent_records);
        println!(
            "📊 Storico compresso: {} entry (da {} originali)",
            self.records.len(),
            total_old
        );
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self)?;
        fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = fs::read_to_string(path)?;
        let session: TrainingSession = serde_json::from_str(&json)?;
        println!(
            "📊 Sessione training caricata! {} records",
            session.records.len()
        );
        Ok(session)
    }

    /// Ottiene l'ultima generazione registrata, se esiste
    pub fn last_generation(&self) -> Option<u32> {
        self.records.last().map(|r| r.gen)
    }
}

// =============================================================================
// GESTIONE CARTELLE DI SALVATAGGIO
// =============================================================================

/// Restituisce o crea la cartella runs/{uuid}/ per il run corrente
/// Usa l'ultima cartella esistente, oppure crea un nuovo UUID
fn get_or_create_run_dir() -> std::path::PathBuf {
    use std::fs;

    let runs_dir = std::path::PathBuf::from("runs");
    fs::create_dir_all(&runs_dir).ok();

    // Trova l'ultima cartella (run) esistente
    if let Ok(entries) = fs::read_dir(&runs_dir) {
        let mut dirs: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .map(|e| e.path())
            .collect();

        if !dirs.is_empty() {
            dirs.sort_by(|a, b| b.cmp(a)); // Ordina decrescente (più recente prima)
            println!("📂 Usando run esistente: {}", dirs[0].display());
            return dirs[0].clone();
        }
    }

    // Crea nuovo run con UUID
    let uuid = uuid::Uuid::new_v4().to_string();
    let new_dir = runs_dir.join(&uuid);
    fs::create_dir_all(&new_dir).ok();

    // Crea anche la sottocartella sessions/
    let sessions_dir = new_dir.join("sessions");
    fs::create_dir_all(&sessions_dir).ok();

    println!("📂 Creato nuovo run: {}", uuid);
    new_dir
}

/// Restituisce il percorso del brain.json (runs/{uuid}/brain.json)
fn brain_path() -> std::path::PathBuf {
    get_or_create_run_dir().join("brain.json")
}

/// Restituisce il percorso per un file di sessione (runs/{uuid}/sessions/{nome})
fn session_path(filename: &str) -> std::path::PathBuf {
    let sessions_dir = get_or_create_run_dir().join("sessions");
    std::fs::create_dir_all(&sessions_dir).ok();
    sessions_dir.join(filename)
}

/// Restituisce il percorso per una nuova sessione con nome basato su timestamp
fn new_session_path() -> std::path::PathBuf {
    let timestamp = chrono::Local::now().format("%Y-%m-%d-%H%M%S").to_string();
    session_path(format!("{}.json", timestamp).as_str())
}

/// Scans the sessions/ directory and aggregates all past data for the UI.
fn load_global_history() -> (GlobalTrainingHistory, u32) {
    let sessions_dir = get_or_create_run_dir().join("sessions");
    let mut global_history = GlobalTrainingHistory::default();
    let mut max_gen = 0;

    println!("📂 Scanning sessions in: {}", sessions_dir.display());

    if let Ok(entries) = std::fs::read_dir(sessions_dir) {
        let mut paths: Vec<_> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "json"))
            .collect();

        // Sort by filename (timestamp) to ensure chronological order in the graph
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

    // Ensure records are sorted by generation
    global_history.records.sort_by_key(|r| r.gen);

    println!(
        "✅ Global History Loaded: {} records, {}s total",
        global_history.records.len(),
        global_history.accumulated_time_secs
    );

    (global_history, max_gen)
}

// =============================================================================

impl GameHistory {
    fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
        }
    }

    fn add_entry(&mut self, generation: u32, scores: &[u32]) {
        let now = std::time::SystemTime::now();
        let datetime: chrono::DateTime<chrono::Local> = now.into();

        let high_score = scores.iter().copied().max().unwrap_or(0);
        let total_score: u32 = scores.iter().sum();
        let alive_count = scores.iter().filter(|&&s| s > 0).count();

        let entry = GameHistoryEntry {
            timestamp: datetime.format("%Y-%m-%d %H:%M:%S").to_string(),
            generation,
            scores: scores.to_vec(),
            high_score,
            total_score,
            alive_snakes: alive_count,
        };

        self.entries.push(entry);

        // Mantieni solo le ultime N partite
        if self.entries.len() > self.max_entries {
            self.entries.remove(0);
        }
    }

    fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self)?;
        fs::write(path, json)?;
        Ok(())
    }

    fn load(path: &str) -> std::io::Result<Self> {
        let json = fs::read_to_string(path)?;
        let history: GameHistory = serde_json::from_str(&json)?;
        Ok(history)
    }
}

impl GameStats {
    fn new(num_snakes: usize) -> Self {
        Self {
            total_generations: 0,
            high_score: 0,
            total_games_played: 0,
            total_food_eaten: 0,
            best_score_per_snake: vec![0; num_snakes],
            last_saved: None,
        }
    }

    fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self)?;
        fs::write(path, json)?;
        println!("Statistiche salvate in: {}", path);
        Ok(())
    }

    fn load(path: &str) -> std::io::Result<Self> {
        let json = fs::read_to_string(path)?;
        let stats: GameStats = serde_json::from_str(&json)?;
        println!("Statistiche caricate da: {}", path);
        Ok(stats)
    }

    fn update(&mut self, game: &GameState) {
        self.total_generations = game.total_iterations;
        self.high_score = game.high_score.max(self.high_score);

        // Aggiorna best score per ogni snake
        for (i, snake) in game.snakes.iter().enumerate() {
            if i < self.best_score_per_snake.len() {
                self.best_score_per_snake[i] = self.best_score_per_snake[i].max(snake.score);
            }
        }

        // Aggiungi timestamp
        let now = std::time::SystemTime::now();
        let datetime: chrono::DateTime<chrono::Local> = now.into();
        self.last_saved = Some(datetime.format("%Y-%m-%d %H:%M:%S").to_string());
    }
}

#[derive(Clone)]
struct SnakeInstance {
    id: usize,
    snake: VecDeque<Position>,
    food: Position,
    direction: Direction,
    is_game_over: bool,
    steps_without_food: u32,
    score: u32,
    color: Color,
}

impl SnakeInstance {
    fn generate_random_color() -> Color {
        let mut rng = rand::thread_rng();
        Color::rgb(rng.gen(), rng.gen(), rng.gen())
    }

    fn get_random_spawn_data(grid: &GridDimensions) -> (Position, Direction) {
        let mut rng = rand::thread_rng();
        let margin = 3; // Margine dai bordi

        // Genera posizione casuale all'interno della griglia con margine
        let x = rng.gen_range(margin..(grid.width - margin));
        let y = rng.gen_range(margin..(grid.height - margin));

        // Scegli una direzione casuale
        let direction = match rng.gen_range(0..4) {
            0 => Direction::Up,
            1 => Direction::Down,
            2 => Direction::Left,
            3 => Direction::Right,
            _ => unreachable!(),
        };

        (Position { x, y }, direction)
    }

    fn new(id: usize, grid: &GridDimensions) -> Self {
        let (spawn_pos, spawn_dir) = Self::get_random_spawn_data(grid);

        let mut snake = VecDeque::new();
        snake.push_back(spawn_pos);

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
        }
    }

    fn reset(&mut self, grid: &GridDimensions) {
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
        // Genera nuovo colore casuale per la nuova generazione
        self.color = Self::generate_random_color();
    }
}

#[derive(Resource)]
struct GameState {
    high_score: u32,
    total_iterations: u32,
    snakes: Vec<SnakeInstance>,
}

impl GameState {
    fn new(grid: &GridDimensions, snake_count: usize) -> Self {
        let snakes = (0..snake_count)
            .map(|id| SnakeInstance::new(id, grid))
            .collect();
        Self {
            high_score: 0,
            total_iterations: 0,
            snakes,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug, Default)]
enum Direction {
    #[default]
    Right,
    Up,
    Down,
    Left,
}

impl Direction {
    fn as_vec(&self) -> (i32, i32) {
        match self {
            Direction::Up => (0, 1),
            Direction::Down => (0, -1),
            Direction::Left => (-1, 0),
            Direction::Right => (1, 0),
        }
    }

    fn turn_right(&self) -> Self {
        match self {
            Direction::Up => Direction::Right,
            Direction::Right => Direction::Down,
            Direction::Down => Direction::Left,
            Direction::Left => Direction::Up,
        }
    }

    fn turn_left(&self) -> Self {
        match self {
            Direction::Up => Direction::Left,
            Direction::Left => Direction::Down,
            Direction::Down => Direction::Right,
            Direction::Right => Direction::Up,
        }
    }
}

/// Generate relative coordinate offsets based on current heading
/// Returns 8 directions: Forward, Forward-Right, Right, Back-Right, Back, Back-Left, Left, Forward-Left
fn get_egocentric_directions(current_dir: Direction) -> [(i32, i32); 8] {
    let (fx, fy) = current_dir.as_vec(); // Forward vector
    let (rx, ry) = current_dir.turn_right().as_vec(); // Right vector

    [
        (fx, fy),             // 0: Forward
        (fx + rx, fy + ry),   // 1: Forward-Right
        (rx, ry),             // 2: Right
        (-fx + rx, -fy + ry), // 3: Back-Right
        (-fx, -fy),           // 4: Back
        (-fx - rx, -fy - ry), // 5: Back-Left
        (-rx, -ry),           // 6: Left
        (fx - rx, fy - ry),   // 7: Forward-Left
    ]
}

// --- MICRO-DQN (IMPLEMENTAZIONE MANUALE) ---
#[derive(Clone, Serialize, Deserialize, Debug)]
struct Layer {
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
}

impl Layer {
    fn new(inputs: usize, outputs: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / inputs as f32).sqrt();
        let weights = (0..inputs)
            .map(|_| (0..outputs).map(|_| rng.gen_range(-scale..scale)).collect())
            .collect();
        let biases = vec![0.0; outputs];
        Self { weights, biases }
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = self.biases.clone();
        for (i, row) in self.weights.iter().enumerate() {
            for (j, w) in row.iter().enumerate() {
                output[j] += input[i] * w;
            }
        }
        output
    }

    fn copy_from(&mut self, other: &Layer) {
        self.weights = other.weights.clone();
        self.biases = other.biases.clone();
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct DqnBrainData {
    l1: Layer,
    l2: Layer,
    epsilon: f32,
    iterations: u32,
}

#[derive(Resource)]
struct DqnBrain {
    l1: Layer,
    l2: Layer,
    l1_target: Layer,
    l2_target: Layer,
    memory: VecDeque<(Vec<f32>, usize, f32, Vec<f32>, bool)>,
    epsilon: f32,
    loss: f32,
    target_update_counter: usize,
    iterations: u32,
}

impl DqnBrain {
    fn new() -> Self {
        let l1 = Layer::new(STATE_SIZE, HIDDEN_NODES);
        let l2 = Layer::new(HIDDEN_NODES, 3);
        Self {
            l1: l1.clone(),
            l2: l2.clone(),
            l1_target: l1,
            l2_target: l2,
            memory: VecDeque::with_capacity(MEMORY_SIZE),
            epsilon: 1.0,
            loss: 0.0,
            target_update_counter: 0,
            iterations: 0,
        }
    }

    fn from_data(data: DqnBrainData) -> Self {
        Self {
            l1: data.l1.clone(),
            l2: data.l2.clone(),
            l1_target: data.l1.clone(),
            l2_target: data.l2.clone(),
            memory: VecDeque::with_capacity(MEMORY_SIZE),
            epsilon: data.epsilon,
            loss: 0.0,
            target_update_counter: 0,
            iterations: data.iterations,
        }
    }

    fn to_data(&self) -> DqnBrainData {
        DqnBrainData {
            l1: self.l1.clone(),
            l2: self.l2.clone(),
            epsilon: self.epsilon,
            iterations: self.iterations,
        }
    }

    fn save(&self, path: &str) -> std::io::Result<()> {
        let data = self.to_data();
        let json = serde_json::to_string_pretty(&data)?;
        fs::write(path, json)?;
        println!("Modello salvato in: {}", path);
        Ok(())
    }

    fn load(path: &str) -> std::io::Result<Self> {
        let json = fs::read_to_string(path)?;
        let data: DqnBrainData = serde_json::from_str(&json)?;
        println!(
            "Modello caricato da: {} (iterazioni: {})",
            path, data.iterations
        );
        Ok(Self::from_data(data))
    }

    fn update_target_network(&mut self) {
        self.l1_target.copy_from(&self.l1);
        self.l2_target.copy_from(&self.l2);
    }

    /// Forward pass ottimizzato che lavora con slice (accetta array fissi)
    fn forward(&self, state: &[f32]) -> Vec<f32> {
        let h1_raw = self.l1.forward(state);
        let h1: Vec<f32> = h1_raw.iter().map(|&x| x.max(0.0)).collect();
        self.l2.forward(&h1)
    }

    /// Versione super veloce per inference con array fissi (evita allocazioni)
    fn forward_array(&self, state: &[f32; STATE_SIZE]) -> [f32; 3] {
        let mut h1 = [0.0f32; HIDDEN_NODES];
        for j in 0..HIDDEN_NODES {
            let mut sum = self.l1.biases[j];
            for i in 0..STATE_SIZE {
                sum += self.l1.weights[i][j] * state[i];
            }
            h1[j] = sum.max(0.0); // ReLU
        }

        let mut output = [0.0f32; 3];
        for j in 0..3 {
            let mut sum = self.l2.biases[j];
            for i in 0..HIDDEN_NODES {
                sum += self.l2.weights[i][j] * h1[i];
            }
            output[j] = sum;
        }
        output
    }

    /// Ottimizzato: converte array in Vec solo per la memoria replay
    fn remember_array(
        &mut self,
        state: [f32; STATE_SIZE],
        action: usize,
        reward: f32,
        next_state: [f32; STATE_SIZE],
        done: bool,
    ) {
        if self.memory.len() >= MEMORY_SIZE {
            self.memory.pop_front();
        }
        // Conversione necessaria per compatibilità con replay buffer
        self.memory
            .push_back((state.to_vec(), action, reward, next_state.to_vec(), done));
    }

    fn train(&mut self) {
        if self.memory.len() < BATCH_SIZE {
            return;
        }

        let mut rng = rand::thread_rng();
        let indices: Vec<usize> = (0..self.memory.len()).choose_multiple(&mut rng, BATCH_SIZE);

        let gamma = 0.95;

        let l1 = self.l1.clone();
        let l2 = self.l2.clone();
        let l1_target = self.l1_target.clone();
        let l2_target = self.l2_target.clone();

        let samples: Vec<_> = indices
            .iter()
            .map(|&idx| self.memory[idx].clone())
            .collect();

        let gradients: Vec<_> = samples
            .par_iter()
            .map(|(state, action, reward, next_state, done)| {
                let h1_raw = l1.forward(state);
                let h1: Vec<f32> = h1_raw.iter().map(|&x| x.max(0.0)).collect();
                let q_values = l2.forward(&h1);

                let target = if *done {
                    *reward
                } else {
                    let next_q_policy = Self::forward_static(&l1, &l2, next_state);
                    let best_action = next_q_policy
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i)
                        .unwrap();
                    let next_q_target = Self::forward_static(&l1_target, &l2_target, next_state);
                    *reward + gamma * next_q_target[best_action]
                };

                let error = target - q_values[*action];
                let clipped_error = error.clamp(-1.0, 1.0);

                (
                    h1_raw,
                    h1,
                    clipped_error,
                    *action,
                    state.clone(),
                    error * error,
                )
            })
            .collect();

        let mut total_loss = 0.0;
        for (h1_raw, h1, clipped_error, action, state, loss_sq) in gradients {
            total_loss += loss_sq;

            for j in 0..3 {
                let diff = if j == action { clipped_error } else { 0.0 };
                self.l2.biases[j] += LEARNING_RATE * diff;
                for k in 0..HIDDEN_NODES {
                    self.l2.weights[k][j] += LEARNING_RATE * diff * h1[k];
                }
            }

            for k in 0..HIDDEN_NODES {
                let weight_contribution = self.l2.weights[k][action] * clipped_error;
                let d_relu = if h1_raw[k] > 0.0 { 1.0 } else { 0.0 };
                let hidden_error = weight_contribution * d_relu;

                self.l1.biases[k] += LEARNING_RATE * hidden_error;
                for i in 0..STATE_SIZE {
                    self.l1.weights[i][k] += LEARNING_RATE * hidden_error * state[i];
                }
            }
        }

        self.loss = total_loss / BATCH_SIZE as f32;

        self.target_update_counter += 1;
        if self.target_update_counter >= TARGET_UPDATE_FREQ {
            self.update_target_network();
            self.target_update_counter = 0;
        }
    }

    fn forward_static(l1: &Layer, l2: &Layer, state: &[f32]) -> Vec<f32> {
        let h1: Vec<f32> = l1.forward(state).iter().map(|&x| x.max(0.0)).collect();
        l2.forward(&h1)
    }

    fn forward_target(&self, state: &[f32]) -> Vec<f32> {
        let h1: Vec<f32> = self
            .l1_target
            .forward(state)
            .iter()
            .map(|&x| x.max(0.0))
            .collect();
        self.l2_target.forward(&h1)
    }
}

// --- COMPONENTI UI ---
#[derive(Component)]
struct StatsText;

#[derive(Component)]
struct LeaderboardText;

#[derive(Component)]
struct CommandsText;

// Sostituisci spawn_stats_ui con questa versione "ASCII safe"
fn spawn_stats_ui(mut commands: Commands, game: Res<GameState>) {
    // CLASSIFICA: Uso [LEADERBOARD] invece della coppa
    let mut leaderboard_sections = vec![TextSection::new(
        "[LEADERBOARD]\n",
        TextStyle {
            font_size: 18.0,
            color: Color::GOLD,
            ..default()
        },
    )];

    for snake in game.snakes.iter() {
        leaderboard_sections.push(TextSection::new(
            "",
            TextStyle {
                font_size: 15.0,
                color: snake.color,
                ..default()
            },
        ));
    }

    commands.spawn((
        TextBundle::from_sections(leaderboard_sections).with_style(Style {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        }),
        LeaderboardText,
    ));

    // STATS: Rimosse emoji snake/skull
    commands.spawn((
        TextBundle::from_sections([
            TextSection::new(
                "H: 0  G: 0  Best: 0\n",
                TextStyle {
                    font_size: 18.0,
                    color: Color::WHITE,
                    ..default()
                },
            ),
            TextSection::new(
                "Time: 00:00:00  Total: 00:00:00  FPS: 0\n",
                TextStyle {
                    font_size: 16.0,
                    color: Color::WHITE,
                    ..default()
                },
            ),
            TextSection::new(
                "Alive:0 Dead:0 | Food: 0  Games: 0", // ASCII puro
                TextStyle {
                    font_size: 14.0,
                    color: Color::WHITE,
                    ..default()
                },
            ),
        ])
        .with_style(Style {
            position_type: PositionType::Absolute,
            bottom: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        }),
        StatsText,
    ));

    // COMANDI - Permanently visible with all hotkeys
    commands.spawn((
        TextBundle::from_section(
            "[R]Render:ON  [G]Graph  [F]Fullscreen  [C]Collision:OFF  [ESC]Exit",
            TextStyle {
                font_size: 14.0,
                color: Color::GRAY,
                ..default()
            },
        )
        .with_style(Style {
            position_type: PositionType::Absolute,
            bottom: Val::Px(10.0),
            right: Val::Px(10.0),
            ..default()
        }),
        CommandsText,
    ));
}

// Sostituisci update_stats_ui per rimuovere le emoji durante l'aggiornamento
fn update_stats_ui(
    mut leaderboard_query: Query<
        &mut Text,
        (
            With<LeaderboardText>,
            Without<StatsText>,
            Without<CommandsText>,
        ),
    >,
    mut stats_query: Query<
        &mut Text,
        (
            With<StatsText>,
            Without<LeaderboardText>,
            Without<CommandsText>,
        ),
    >,
    mut commands_query: Query<
        &mut Text,
        (
            With<CommandsText>,
            Without<LeaderboardText>,
            Without<StatsText>,
        ),
    >,
    game: Res<GameState>,
    stats: Res<TrainingStats>,
    game_stats: Res<GameStats>,
    collision_settings: Res<CollisionSettings>,
    render_config: Res<RenderConfig>,
    app_start_time: Res<AppStartTime>,
    global_history: Res<GlobalTrainingHistory>,
) {
    // FPS Calculation
    let now = Instant::now();
    let _elapsed = now.duration_since(stats.last_update);

    // --- CORRECTED TIMER CALCULATION ---
    // Total Time = Loaded History Time + Current Session Time
    // Current Session Time = Instant::now() - app_start_time
    let current_session_duration = now.duration_since(app_start_time.0);
    let total_training_time =
        Duration::from_secs(global_history.accumulated_time_secs) + current_session_duration;

    let session_secs = current_session_duration.as_secs();
    let session_hours = session_secs / 3600;
    let session_minutes = (session_secs % 3600) / 60;
    let session_seconds = session_secs % 60;

    let total_secs = total_training_time.as_secs();
    let total_hours = total_secs / 3600;
    let total_minutes = (total_secs % 3600) / 60;
    let total_seconds = total_secs % 60;

    let persistent_high = game_stats.high_score.max(game.high_score);
    let alive_count = game.snakes.iter().filter(|s| !s.is_game_over).count();
    let dead_count = game.snakes.len() - alive_count;

    // CLASSIFICA
    let mut snake_data: Vec<(usize, &SnakeInstance)> = game.snakes.iter().enumerate().collect();
    snake_data.sort_by(|a, b| b.1.score.cmp(&a.1.score));

    if let Ok(mut lb_text) = leaderboard_query.get_single_mut() {
        for (rank, (original_idx, snake)) in snake_data.iter().enumerate() {
            let section_idx = rank + 1;
            if section_idx < lb_text.sections.len() {
                // USA CARATTERI ASCII: [OK] o [XX] invece delle emoji
                let status = if snake.is_game_over { "[XX]" } else { "[OK]" };
                lb_text.sections[section_idx].value = format!(
                    "{:2}. S{:02} {}{:3}\n",
                    rank + 1,
                    original_idx + 1,
                    status,
                    snake.score
                );
                lb_text.sections[section_idx].style.color = if snake.is_game_over {
                    Color::GRAY
                } else {
                    snake.color
                };
            }
        }
    }

    // STATS
    if let Ok(mut st_text) = stats_query.get_single_mut() {
        st_text.sections[0].value = format!(
            "H: {:3}  G: {:5}  Best: {:3}\n",
            game.high_score, game.total_iterations, persistent_high
        );
        // FIXED: Total Time = Loaded History Time + Current Session Time
        st_text.sections[1].value = format!(
            "Session: {:02}:{:02}:{:02}  Total: {:02}:{:02}:{:02}  FPS: {:5.1}\n",
            session_hours,
            session_minutes,
            session_seconds,
            total_hours,
            total_minutes,
            total_seconds,
            stats.fps
        );
        // ASCII puro anche qui
        st_text.sections[2].value = format!(
            "Alive:{} Dead:{} | Food: {}  Games: {}",
            alive_count, dead_count, game_stats.total_food_eaten, game_stats.total_games_played
        );
    }

    // COMANDI - Permanently visible with all hotkeys
    if let Ok(mut cmd_text) = commands_query.get_single_mut() {
        let render_status = if render_config.enabled { "ON" } else { "TURBO" };
        let collision_status = if collision_settings.snake_vs_snake {
            "ON"
        } else {
            "OFF"
        };
        cmd_text.sections[0].value = format!(
            "[R]Render:{}  [G]Graph  [F]Fullscreen  [C]Collision:{}  [ESC]Exit",
            render_status, collision_status
        );
    }
}

// --- SISTEMI BEVY ---

fn calculate_grid_dimensions(window_width: f32, window_height: f32) -> (i32, i32) {
    let ui_padding = 60.0;
    let available_height = window_height - ui_padding;

    // Calcola quanti blocchi interi ci stanno esattamente
    // Floor per assicurarci di non superare i bordi
    let width = (window_width / BLOCK_SIZE).floor() as i32;
    let height = (available_height / BLOCK_SIZE).floor() as i32;

    // Minimo 10x10, massimo illimitato
    let width = width.max(10);
    let height = height.max(10);

    (width, height)
}

fn setup(
    mut commands: Commands,
    windows: Query<&Window>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn(Camera2dBundle::default());

    let window = windows.single();
    let (grid_width, grid_height) =
        calculate_grid_dimensions(window.resolution.width(), window.resolution.height());

    // Crea mesh cache (una sola volta!)
    let segment_mesh = meshes.add(Rectangle::new(BLOCK_SIZE - 2.0, BLOCK_SIZE - 2.0));
    let food_mesh = meshes.add(Circle::new(BLOCK_SIZE / 2.0));

    let food_material = materials.add(Color::rgb(1.0, 0.0, 0.0)); // Rosso
    let head_material = materials.add(Color::rgb(1.0, 1.0, 1.0)); // Bianco

    commands.insert_resource(MeshCache {
        segment_mesh,
        food_mesh,
        food_material,
        head_material,
    });

    commands.insert_resource(CollisionSettings::default());
    commands.insert_resource(GraphPanelState::default());
    commands.insert_resource(RenderConfig::default());

    // Initialize GridMap for O(1) collision detection
    commands.insert_resource(GridMap::new(grid_width, grid_height));

    // Initialize SegmentPool for object-pooled rendering
    let parallel_config = ParallelConfig::new();
    commands.insert_resource(SegmentPool::new(parallel_config.snake_count));

    // Initialize AppStartTime for correct timer calculations
    commands.insert_resource(AppStartTime::default());

    // 1. Load Global History (Past) - Aggregates ALL previous sessions for the UI
    let (global_history, last_gen) = load_global_history();
    let accumulated_time = global_history.accumulated_time_secs;

    // 2. Setup Current Session (Fresh for saving new data)
    let session_file = new_session_path();
    let training_session = TrainingSession::new();
    println!("📊 New session file: {}", session_file.display());

    // 3. Load Brain
    let mut brain = if brain_path().exists() {
        match DqnBrain::load(brain_path().to_str().unwrap_or("brain.json")) {
            Ok(b) => {
                println!("🧠 Model loaded!");
                b
            }
            Err(e) => {
                eprintln!("Error loading model: {}, creating new brain", e);
                DqnBrain::new()
            }
        }
    } else {
        println!("No model found, initializing new brain");
        DqnBrain::new()
    };

    // Sync brain iterations with history
    if last_gen > 0 {
        println!("🔄 Sync: Resuming from generation {}", last_gen);
        brain.iterations = last_gen;
    }

    // Crea GameState (uses already-created parallel_config)
    let grid = GridDimensions {
        width: grid_width,
        height: grid_height,
    };
    let mut game_state = GameState::new(&grid, parallel_config.snake_count);
    game_state.total_iterations = last_gen; // Resume generation count

    // 4. Insert Resources
    commands.insert_resource(global_history); // For Graph UI
    commands.insert_resource(training_session); // For Saving
    commands.insert_resource(brain);
    commands.insert_resource(GameConfig {
        speed_timer: Timer::from_seconds(0.001, TimerMode::Repeating),
        session_path: session_file,
    });

    commands.insert_resource(WindowSettings {
        is_fullscreen: false,
    });

    let grid = GridDimensions {
        width: grid_width,
        height: grid_height,
    };

    commands.insert_resource(grid);

    // Inserisci GameState creato precedentemente con i colori casuali
    commands.insert_resource(game_state);

    // Initialize TrainingStats with accumulated time!
    commands.insert_resource(TrainingStats {
        total_training_time: Duration::from_secs(accumulated_time),
        last_update: Instant::now(),
        parallel_threads: rayon::current_num_threads(),
        fps: 0.0,
        last_fps_update: Instant::now(),
        frame_count: 0,
    });

    // GameStats e GameHistory sono ora gestiti tramite TrainingSession
    // Manteniamo le risorse per retrocompatibilità ma le inizializziamo vuote
    commands.insert_resource(parallel_config.clone());
    commands.insert_resource(GameStats::new(parallel_config.snake_count));
    commands.insert_resource(GameHistory::new(0)); // Non usato, tutto in TrainingSession

    // NOTA: spawn_stats_ui deve essere chiamato DOPO che game_state è stato creato
    // ma PRIMA che venga inserito come risorsa (per evitare il borrow checker)
    // In realtà, dato che abbiamo bisogno di accedere ai dati degli snake,
    // creiamo la UI dopo aver inserito la risorsa e la leggiamo nella query
}

fn spawn_food(snake: &SnakeInstance, grid: &GridDimensions) -> Position {
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

/// Struttura temporanea per passare dati tra step paralleli
struct StepResult {
    snake_idx: usize,
    state: [f32; STATE_SIZE],
    action_idx: usize,
    reward: f32,
    next_state: [f32; STATE_SIZE],
    done: bool,
    new_head: Position,
    ate_food: bool,
}

/// Decision result from parallel inference
struct Decision {
    snake_idx: usize,
    action_idx: usize,
    state: [f32; STATE_SIZE],
}

fn game_loop(
    time: Res<Time>,
    mut config: ResMut<GameConfig>,
    mut game: ResMut<GameState>,
    mut brain: ResMut<DqnBrain>,
    mut game_stats: ResMut<GameStats>,
    mut training_session: ResMut<TrainingSession>,
    mut global_history: ResMut<GlobalTrainingHistory>,
    mut stats: ResMut<TrainingStats>,
    grid: Res<GridDimensions>,
    collision_settings: Res<CollisionSettings>,
    parallel_config: Res<ParallelConfig>,
    render_config: Res<RenderConfig>,
    mut grid_map: ResMut<GridMap>,
) {
    // FPS Calculation
    stats.frame_count += 1;
    let now = Instant::now();
    if now.duration_since(stats.last_fps_update).as_secs_f32() >= 1.0 {
        stats.fps =
            stats.frame_count as f32 / now.duration_since(stats.last_fps_update).as_secs_f32();
        stats.last_fps_update = now;
        stats.frame_count = 0;
    }

    // --- DYNAMIC LOOP LOGIC ---

    if render_config.enabled {
        // MODE A: Visual (Render ON) - Respect the Timer
        config.speed_timer.tick(time.delta());
        if !config.speed_timer.finished() {
            return;
        }
        // Run 1 step per visual frame update
        run_simulation_step(
            &mut game,
            &mut brain,
            &mut game_stats,
            &mut training_session,
            &mut global_history,
            &config,
            &grid,
            &collision_settings,
            &parallel_config,
            &mut stats,
            &mut grid_map,
        );
    } else {
        // MODE B: Turbo (Render OFF) - Dynamic Time Budget Loop
        // Target ~30 FPS for UI responsiveness = 33ms per frame
        // Leave ~3ms buffer for system responsiveness
        let ui_target = Duration::from_millis(33);
        let buffer = Duration::from_millis(3);
        let start = Instant::now();

        // Run simulation steps in batches to reduce overhead
        const BATCH_SIZE: usize = 10;

        loop {
            // Run a batch of simulation steps
            for _ in 0..BATCH_SIZE {
                run_simulation_step(
                    &mut game,
                    &mut brain,
                    &mut game_stats,
                    &mut training_session,
                    &mut global_history,
                    &config,
                    &grid,
                    &collision_settings,
                    &parallel_config,
                    &mut stats,
                    &mut grid_map,
                );
            }

            // Break if we are close to the UI deadline
            if start.elapsed() >= ui_target.saturating_sub(buffer) {
                break;
            }
        }

        // Accumulate actual training time
        stats.total_training_time += start.elapsed();
    }
}

// Helper function to encapsulate ONE simulation step
fn run_simulation_step(
    game: &mut GameState,
    brain: &mut DqnBrain,
    game_stats: &mut GameStats,
    training_session: &mut TrainingSession,
    global_history: &mut GlobalTrainingHistory,
    config: &GameConfig,
    grid: &GridDimensions,
    collision_settings: &CollisionSettings,
    parallel_config: &ParallelConfig,
    stats: &mut TrainingStats,
    grid_map: &mut GridMap,
) {
    let mut all_dead = true;

    // --- FASE 1: INFERENZA PARALLELA ---
    let active_snakes: Vec<usize> = game
        .snakes
        .iter()
        .enumerate()
        .filter(|(_, s)| !s.is_game_over)
        .map(|(idx, _)| idx)
        .collect();

    if active_snakes.is_empty() {
        handle_generation_end(
            game,
            brain,
            game_stats,
            training_session,
            global_history,
            config,
            parallel_config,
            stats,
            grid,
        );
        return;
    }

    let decisions: Vec<Decision> = active_snakes
        .par_iter()
        .map(|&snake_idx| {
            let snake = &game.snakes[snake_idx];
            let state = get_state_gridmap(snake, grid_map, grid, collision_settings.snake_vs_snake);

            let mut rng = rand::thread_rng();
            let action_idx = if rng.gen::<f32>() < brain.epsilon {
                rng.gen_range(0..3)
            } else {
                let q_vals = brain.forward_array(&state);
                q_vals
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap()
            };

            Decision {
                snake_idx,
                action_idx,
                state,
            }
        })
        .collect();

    // --- FASE 2: APPLICAZIONE LOGICA E REWARD (SEQUENZIALE) ---
    grid_map.clear();
    for (idx, snake) in game.snakes.iter().enumerate() {
        if !snake.is_game_over {
            for pos in snake.snake.iter() {
                grid_map.set(pos.x, pos.y, (idx + 1) as u8);
            }
        }
    }

    let mut step_results: Vec<StepResult> = Vec::with_capacity(decisions.len());

    for decision in decisions {
        let snake_idx = decision.snake_idx;
        let action_idx = decision.action_idx;
        let state = decision.state;

        all_dead = false;
        let snake_ref = &mut game.snakes[snake_idx];

        // Salviamo la distanza prima del movimento per calcolare la progressione
        let old_dist_sq = ((snake_ref.snake[0].x - snake_ref.food.x).pow(2)
            + (snake_ref.snake[0].y - snake_ref.food.y).pow(2)) as f32;

        // Movimento
        match action_idx {
            1 => snake_ref.direction = snake_ref.direction.turn_right(),
            2 => snake_ref.direction = snake_ref.direction.turn_left(),
            _ => {}
        }

        let (dx, dy) = snake_ref.direction.as_vec();
        let new_head = Position {
            x: snake_ref.snake[0].x + dx,
            y: snake_ref.snake[0].y + dy,
        };

        snake_ref.steps_without_food += 1;
        let mut reward: f32 = 0.0;
        let mut done = false;

        // Collision detection
        let collision = if collision_settings.snake_vs_snake {
            grid_map.is_collision(new_head.x, new_head.y, snake_idx)
        } else {
            grid_map.is_collision_no_snakes(new_head.x, new_head.y)
        } || snake_ref.snake.contains(&new_head);

        if collision {
            reward = -15.0; // Penalità morte aumentata
            done = true;
            snake_ref.is_game_over = true;
        } else if new_head == snake_ref.food {
            // Reward Cibo: Bilanciato tra fisso e bonus lunghezza
            reward = 12.0 + (snake_ref.score as f32 * 0.2);
            snake_ref.snake.push_front(new_head);
            snake_ref.score += 1;
            if snake_ref.score > game.high_score {
                game.high_score = snake_ref.score;
            }
            snake_ref.food = spawn_food(snake_ref, grid);
            snake_ref.steps_without_food = 0;
            grid_map.set(new_head.x, new_head.y, (snake_idx + 1) as u8);
        } else {
            // --- LOGICA DI REWARD PER MOVIMENTO (Il cuore del problema) ---

            // 1. Penalità temporale (Costo di Esistenza)
            // Più è affamato, più la penalità aumenta per spingerlo a rischiare
            let max_steps = (grid.width * grid.height) as f32;
            let hunger_ratio = snake_ref.steps_without_food as f32 / max_steps;
            reward -= 0.01 + (hunger_ratio * 0.05);

            // 2. Penalità per sterzata (Anti-Loop)
            // Girare costa un pochino. Questo favorisce traiettorie dritte verso il cibo.
            if action_idx != 0 {
                reward -= 0.02;
            }

            // 3. Reward di Prossimità (Potential-based)
            let new_dist_sq = ((new_head.x - snake_ref.food.x).pow(2)
                + (new_head.y - snake_ref.food.y).pow(2)) as f32;

            if new_dist_sq < old_dist_sq {
                reward += 0.15; // Si è avvicinato
            } else {
                reward -= 0.20; // Si è allontanato (punizione maggiore del premio)
            }

            // Aggiorna corpo
            let old_tail = snake_ref.snake.back().copied();
            snake_ref.snake.push_front(new_head);
            snake_ref.snake.pop_back();

            // Timeout affamamento
            if snake_ref.steps_without_food > (grid.width * grid.height) as u32 {
                reward = -10.0;
                done = true;
                snake_ref.is_game_over = true;
            }

            grid_map.set(new_head.x, new_head.y, (snake_idx + 1) as u8);
            if let Some(tail) = old_tail {
                grid_map.set(tail.x, tail.y, 0);
            }
        }

        step_results.push(StepResult {
            snake_idx,
            state,
            action_idx,
            reward,
            next_state: get_state_gridmap(
                &game.snakes[snake_idx],
                grid_map,
                grid,
                collision_settings.snake_vs_snake,
            ),
            done,
            new_head,
            ate_food: new_head == game.snakes[snake_idx].food,
        });
    }

    // Salvataggio in memoria e training continuo
    for result in step_results {
        brain.remember_array(
            result.state,
            result.action_idx,
            result.reward,
            result.next_state,
            result.done,
        );
    }

    brain.iterations += 1;
    if brain.iterations % TRAIN_INTERVAL as u32 == 0 {
        brain.train();
    }

    if all_dead {
        handle_generation_end(
            game,
            brain,
            game_stats,
            training_session,
            global_history,
            config,
            parallel_config,
            stats,
            grid,
        );
    }
}

// Handle end of generation - reset snakes, save data, etc.
fn handle_generation_end(
    game: &mut GameState,
    brain: &mut DqnBrain,
    game_stats: &mut GameStats,
    training_session: &mut TrainingSession,
    global_history: &mut GlobalTrainingHistory,
    config: &GameConfig,
    parallel_config: &ParallelConfig,
    stats: &mut TrainingStats,
    grid: &GridDimensions,
) {
    let food_eaten: u32 = game.snakes.iter().map(|s| s.score).sum();
    game_stats.total_food_eaten += food_eaten as u64;

    for (i, snake) in game.snakes.iter().enumerate() {
        if i < game_stats.best_score_per_snake.len() {
            game_stats.best_score_per_snake[i] =
                game_stats.best_score_per_snake[i].max(snake.score);
        }
    }

    let current_scores: Vec<u32> = game.snakes.iter().map(|s| s.score).collect();
    let max_score = current_scores.iter().copied().max().unwrap_or(0);
    let min_score = current_scores.iter().copied().min().unwrap_or(0);
    let avg_score = current_scores.iter().sum::<u32>() as f32 / current_scores.len() as f32;

    let record = GenerationRecord {
        gen: game.total_iterations,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        avg_score,
        max_score,
        min_score,
        avg_loss: brain.loss,
        epsilon: brain.epsilon,
    };

    // 1. Add to GLOBAL HISTORY (For Graph & Continuity)
    global_history.records.push(record.clone());

    // 2. Add to CURRENT SESSION (For File Saving)
    training_session.add_record(record);

    // Calculate ONLY current session duration for the file
    training_session.total_time_secs = stats
        .total_training_time
        .as_secs()
        .saturating_sub(global_history.accumulated_time_secs);

    // Save only the current session file
    if let Err(e) = training_session.save(config.session_path.to_str().unwrap_or("session.json")) {
        eprintln!("Errore salvataggio sessione: {}", e);
    }

    if game.total_iterations % 100 == 0 {
        if let Err(e) = brain.save(brain_path().to_str().unwrap_or("brain.json")) {
            eprintln!("Errore salvataggio modello: {}", e);
        }
    }

    for snake in game.snakes.iter_mut() {
        snake.reset(grid);
    }
    game.total_iterations += 1;
    brain.iterations = game.total_iterations;

    // Train at generation end as well
    brain.train();
    brain.epsilon = (brain.epsilon * 0.995).max(0.01);

    game_stats.total_generations = game.total_iterations;
    game_stats.high_score = game_stats.high_score.max(game.high_score);
    game_stats.total_games_played += parallel_config.snake_count as u64;

    let total_score: u32 = game.snakes.iter().map(|s| s.score).sum();
    let active_count = game.snakes.iter().filter(|s| !s.is_game_over).count();
    println!(
        "Gen: {}, Active: {}/{}, Total Score: {}, High: {}, Eps: {:.3}, Loss: {:.5}",
        game.total_iterations,
        active_count,
        parallel_config.snake_count,
        total_score,
        game.high_score,
        brain.epsilon,
        brain.loss
    );
}

/// Grid-Agnostic Ego-Centric State Function
/// Uses 8-directional raycasting relative to snake's current heading
/// State[0..7]: Obstacle radar (1.0 / distance)
/// State[8..15]: Target radar (1.0 / distance in active sector)
fn get_state_gridmap(
    snake: &SnakeInstance,
    grid_map: &GridMap,
    grid: &GridDimensions,
    check_other_snakes: bool,
) -> [f32; STATE_SIZE] {
    let mut state = [0.0f32; STATE_SIZE];
    let head = snake.snake[0];
    let ego_dirs = get_egocentric_directions(snake.direction);

    // --- 1. RADAR OSTACOLI (0..8) ---
    for (i, (dx, dy)) in ego_dirs.iter().enumerate() {
        let mut dist_val = 0.0;
        let mut curr_x = head.x;
        let mut curr_y = head.y;

        for d in 1..15 {
            // Ridotto raggio per focus locale
            curr_x += dx;
            curr_y += dy;
            if curr_x < 0
                || curr_x >= grid.width
                || curr_y < 0
                || curr_y >= grid.height
                || grid_map.get(curr_x, curr_y) != 0
            {
                dist_val = 1.0 / (d as f32);
                break;
            }
        }
        state[i] = dist_val;
    }

    // --- 2. TARGET RADAR (8..15) ---
    let dx_glob = (snake.food.x - head.x) as f32;
    let dy_glob = (snake.food.y - head.y) as f32;
    let dist_target = (dx_glob.powi(2) + dy_glob.powi(2)).sqrt().max(1.0);

    // Proiezioni egocentriche (Avanti e Destra)
    let (fx, fy) = snake.direction.as_vec();
    let (rx, ry) = snake.direction.turn_right().as_vec();
    let local_f = (dx_glob * fx as f32 + dy_glob * fy as f32) / dist_target;
    let local_r = (dx_glob * rx as f32 + dy_glob * ry as f32) / dist_target;

    // Inseriamo i valori di allineamento direttamente nello stato
    state[8] = local_f; // Quanto il cibo è "davanti" (-1 a 1)
    state[9] = local_r; // Quanto il cibo è "a destra" (-1 a 1)

    // Settore angolare (mappato su 4 canali invece di 8 per densità)
    let angle = local_r.atan2(local_f);
    let sector = (((angle / std::f32::consts::PI) * 2.0 + 4.5) as usize) % 4;
    state[10 + sector] = 1.0;

    // --- 3. MEMORIA DI MOVIMENTO (14..16) ---
    // Aggiungiamo la lunghezza relativa e i passi senza cibo
    state[14] = (snake.snake.len() as f32 / 50.0).min(1.0);
    state[15] = (snake.steps_without_food as f32 / (grid.width * grid.height) as f32).min(1.0);

    state
}

fn render_system(
    mut commands: Commands,
    game: Res<GameState>,
    windows: Query<&Window>,
    mesh_cache: Res<MeshCache>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut segment_pool: ResMut<SegmentPool>,
    render_config: Res<RenderConfig>,
    q_food: Query<Entity, With<Food>>,
) {
    // Skip rendering when disabled (training-only mode)
    if !render_config.enabled {
        return;
    }

    // Despawn all food entities from previous frame
    for e in q_food.iter() {
        commands.entity(e).despawn();
    }

    // Usa la finestra principale (quella senza GraphWindow component)
    let Ok(window) = windows.get_single() else {
        return;
    };

    let ui_padding = 60.0;
    let offset_x = -window.resolution.width() / 2.0 + BLOCK_SIZE / 2.0;
    let offset_y = window.resolution.height() / 2.0 - ui_padding - BLOCK_SIZE / 2.0;

    // Object-pooled rendering: Reuse entities instead of despawning/spawning
    for snake in game.snakes.iter() {
        // Salta snake morti - non renderizzare né snake né cibo
        if snake.is_game_over {
            // Hide all segments for dead snakes
            segment_pool.hide_excess(&mut commands, snake.id, 0);
            continue;
        }

        // Crea materiale con il colore ATTUALE dello snake (cambia ad ogni reset!)
        let body_material = materials.add(snake.color);
        let snake_len = snake.snake.len();

        // Update or spawn segments for active snake positions
        for (i, pos) in snake.snake.iter().enumerate() {
            // Testa = bianca, corpo = colore dello snake
            let material = if i == 0 {
                mesh_cache.head_material.clone()
            } else {
                body_material.clone()
            };

            let transform = Transform::from_xyz(
                offset_x + (pos.x as f32 * BLOCK_SIZE),
                offset_y - (pos.y as f32 * BLOCK_SIZE),
                0.0,
            );

            // Use object pool to get or spawn segment entity
            segment_pool.get_or_spawn(
                &mut commands,
                snake.id,
                i,
                mesh_cache.segment_mesh.clone(),
                material,
                transform,
            );
        }

        // Hide excess segments if snake got shorter
        segment_pool.hide_excess(&mut commands, snake.id, snake_len);
        segment_pool.set_active_count(snake.id, snake_len);

        // Render food for this snake (food uses separate spawning, could be pooled too)
        // For now, we spawn food each frame since there's one per snake
        // This is acceptable as food count = snake count (constant)
        commands.spawn((
            MaterialMesh2dBundle {
                mesh: mesh_cache.food_mesh.clone().into(),
                material: mesh_cache.food_material.clone(),
                transform: Transform::from_xyz(
                    offset_x + (snake.food.x as f32 * BLOCK_SIZE),
                    offset_y - (snake.food.y as f32 * BLOCK_SIZE),
                    0.0,
                ),
                ..default()
            },
            Food,
            SnakeId(snake.id),
        ));
    }
}

fn handle_input(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut app_exit_events: EventWriter<AppExit>,
    brain: ResMut<DqnBrain>,
    mut game_stats: ResMut<GameStats>,
    mut training_session: ResMut<TrainingSession>,
    config: Res<GameConfig>,
    game: Res<GameState>,
    mut window_settings: ResMut<WindowSettings>,
    mut windows: Query<&mut Window>,
    mut collision_settings: ResMut<CollisionSettings>,
    mut render_config: ResMut<RenderConfig>,
    mut graph_state: ResMut<GraphPanelState>,
    app_start_time: Res<AppStartTime>,
    global_history: Res<GlobalTrainingHistory>,
) {
    // [ESC] SALVA E ESCI
    if keyboard_input.just_pressed(KeyCode::Escape) {
        // Aggiorna statistiche finali
        game_stats.update(&game);

        // Salva brain.json (unico per run)
        if let Err(e) = brain.save(brain_path().to_str().unwrap_or("brain.json")) {
            eprintln!("Errore salvataggio modello: {}", e);
        }

        // Salva sessione corrente - calcola solo il tempo di questa sessione
        let current_session_duration = Instant::now().duration_since(app_start_time.0);
        training_session.total_time_secs = current_session_duration.as_secs();
        training_session.compress_history();
        if let Err(e) =
            training_session.save(config.session_path.to_str().unwrap_or("session.json"))
        {
            eprintln!("Errore salvataggio sessione: {}", e);
        }

        // Calcola tempi a runtime
        let current_session_duration = Instant::now().duration_since(app_start_time.0);
        let total_training_time =
            Duration::from_secs(global_history.accumulated_time_secs) + current_session_duration;

        println!("\n=== RIEPILOGO SESSIONE ===");
        println!("Generazioni totali: {}", game_stats.total_generations);
        println!("High Score: {}", game_stats.high_score);
        println!(
            "Tempo sessione corrente: {}s",
            current_session_duration.as_secs()
        );
        println!("Tempo totale (runtime): {}s", total_training_time.as_secs());
        println!("Records in sessione: {}", training_session.records.len());
        println!("Salvato in: {}", get_or_create_run_dir().display());
        println!("========================\n");

        app_exit_events.send(AppExit);
    }

    // [C] COLLISIONI
    if keyboard_input.just_pressed(KeyCode::KeyC) {
        collision_settings.snake_vs_snake = !collision_settings.snake_vs_snake;
        println!(
            "Collisioni snake-vs-snake: {}",
            if collision_settings.snake_vs_snake {
                "ON"
            } else {
                "OFF"
            }
        );
    }

    // [R] Render / Turbo Mode Toggle
    if keyboard_input.just_pressed(KeyCode::KeyR) {
        render_config.enabled = !render_config.enabled;
        println!(
            "Rendering: {}",
            if render_config.enabled {
                "ON (Normal)"
            } else {
                "OFF (Turbo)"
            }
        );
        // Note: We don't change graph_state.visible here. The display logic handles the override.
    }

    // [G] Graph Visibility Toggle (User Preference)
    if keyboard_input.just_pressed(KeyCode::KeyG) {
        graph_state.visible = !graph_state.visible;
        graph_state.needs_redraw = true;
    }

    // [F] Fullscreen Toggle
    if keyboard_input.just_pressed(KeyCode::KeyF) {
        window_settings.is_fullscreen = !window_settings.is_fullscreen;
        let mut window = windows.single_mut();
        window.mode = if window_settings.is_fullscreen {
            WindowMode::Fullscreen
        } else {
            WindowMode::Windowed
        };
        // Ensure graph is visible in fullscreen
        if window_settings.is_fullscreen {
            graph_state.visible = true;
            graph_state.needs_redraw = true;
        }
    }
}

fn on_window_resize(
    mut resize_events: EventReader<bevy::window::WindowResized>,
    mut grid: ResMut<GridDimensions>,
    mut game: ResMut<GameState>,
    mut grid_map: ResMut<GridMap>, // <--- AGGIUNGI QUESTO
    mut graph_state: ResMut<GraphPanelState>,
) {
    for event in resize_events.read() {
        let (new_width, new_height) = calculate_grid_dimensions(event.width, event.height);

        grid.width = new_width;
        grid.height = new_height;

        // FIX CRITICO: Ri-inizializza il GridMap con le nuove dimensioni!
        *grid_map = GridMap::new(new_width, new_height);

        // Forza il reset degli snake sulla nuova griglia
        for snake in game.snakes.iter_mut() {
            snake.reset(&grid);
        }

        graph_state.needs_redraw = true;
        println!(
            "Resized: GridMap re-initialized to {}x{}",
            new_width, new_height
        );
    }
}

// Componenti per il grafico UI
#[derive(Component)]
struct GraphPanel;

#[derive(Component)]
struct GraphPanelHeader;

#[derive(Component)]
struct GraphPanelContent;

#[derive(Component)]
struct GraphCloseButton;

#[derive(Component)]
struct GraphCollapseButton;

#[derive(Component)]
struct GraphResizeHandle;

#[derive(Component)]
struct GraphLine;

#[derive(Resource)]
struct GraphPanelState {
    visible: bool,
    collapsed: bool,
    fullscreen: bool, // Modalità fullscreen per il grafico (rendering disabilitato)
    position: Vec2,
    size: Vec2,
    // Stati interni per il drag/resize
    is_dragging: bool,
    drag_offset: Vec2, // Offset tra cursore e angolo in alto a sx del pannello
    is_resizing: bool,
    resize_start_pos: Vec2,  // Dove era il mouse quando è iniziato il resize
    resize_start_size: Vec2, // Quanto era grande la finestra all'inizio
    needs_redraw: bool,
    last_entry_count: usize,
}

impl Default for GraphPanelState {
    fn default() -> Self {
        Self {
            visible: false,
            collapsed: false,
            fullscreen: false,
            position: Vec2::new(50.0, 50.0),
            size: Vec2::new(600.0, 400.0),
            is_dragging: false,
            drag_offset: Vec2::ZERO,
            is_resizing: false,
            resize_start_pos: Vec2::ZERO,
            resize_start_size: Vec2::ZERO,
            needs_redraw: true,
            last_entry_count: 0,
        }
    }
}

fn spawn_graph_panel(
    mut commands: Commands,
    graph_state: Res<GraphPanelState>,
    existing_panel: Query<Entity, With<GraphPanel>>,
) {
    if !graph_state.visible || existing_panel.iter().next().is_some() {
        return;
    }

    let header_height = 30.0;
    let content_height = if graph_state.collapsed {
        0.0
    } else {
        graph_state.size.y - header_height
    };

    commands
        .spawn((
            NodeBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    left: Val::Px(graph_state.position.x),
                    top: Val::Px(graph_state.position.y),
                    width: Val::Px(graph_state.size.x),
                    height: Val::Px(if graph_state.collapsed {
                        header_height
                    } else {
                        graph_state.size.y
                    }),
                    flex_direction: FlexDirection::Column,
                    ..default()
                },
                background_color: Color::rgba(0.1, 0.1, 0.1, 0.95).into(),
                ..default()
            },
            GraphPanel,
        ))
        .with_children(|parent| {
            // Header
            parent
                .spawn((
                    NodeBundle {
                        style: Style {
                            width: Val::Percent(100.0),
                            height: Val::Px(header_height),
                            flex_direction: FlexDirection::Row,
                            justify_content: JustifyContent::SpaceBetween,
                            align_items: AlignItems::Center,
                            padding: UiRect::horizontal(Val::Px(10.0)),
                            ..default()
                        },
                        background_color: Color::rgb(0.2, 0.2, 0.3).into(),
                        ..default()
                    },
                    GraphPanelHeader,
                ))
                .with_children(|header| {
                    header.spawn(TextBundle::from_section(
                        "Training History",
                        TextStyle {
                            font_size: 16.0,
                            color: Color::WHITE,
                            ..default()
                        },
                    ));

                    header
                        .spawn(NodeBundle {
                            style: Style {
                                flex_direction: FlexDirection::Row,
                                column_gap: Val::Px(5.0),
                                ..default()
                            },
                            ..default()
                        })
                        .with_children(|buttons| {
                            // Bottone Collasa (Usa ^ / v ASCII)
                            buttons
                                .spawn((
                                    ButtonBundle {
                                        style: Style {
                                            width: Val::Px(25.0),
                                            height: Val::Px(25.0),
                                            justify_content: JustifyContent::Center,
                                            align_items: AlignItems::Center,
                                            ..default()
                                        },
                                        background_color: Color::rgba(0.3, 0.3, 0.3, 1.0).into(),
                                        ..default()
                                    },
                                    GraphCollapseButton,
                                ))
                                .with_children(|btn| {
                                    btn.spawn(TextBundle::from_section(
                                        if graph_state.collapsed { "v" } else { "^" },
                                        TextStyle {
                                            font_size: 14.0,
                                            color: Color::WHITE,
                                            ..default()
                                        },
                                    ));
                                });

                            // Bottone Chiudi (Usa X ASCII)
                            buttons
                                .spawn((
                                    ButtonBundle {
                                        style: Style {
                                            width: Val::Px(25.0),
                                            height: Val::Px(25.0),
                                            justify_content: JustifyContent::Center,
                                            align_items: AlignItems::Center,
                                            ..default()
                                        },
                                        background_color: Color::rgba(0.8, 0.2, 0.2, 1.0).into(),
                                        ..default()
                                    },
                                    GraphCloseButton,
                                ))
                                .with_children(|btn| {
                                    btn.spawn(TextBundle::from_section(
                                        "X",
                                        TextStyle {
                                            font_size: 14.0,
                                            color: Color::WHITE,
                                            ..default()
                                        },
                                    ));
                                });
                        });
                });

            // Content
            if !graph_state.collapsed {
                parent.spawn((
                    NodeBundle {
                        style: Style {
                            width: Val::Percent(100.0),
                            height: Val::Px(content_height),
                            // *** QUESTA E' LA FIX CRITICA PER IL GRAFICO ***
                            overflow: Overflow::clip(),
                            // ***********************************************
                            ..default()
                        },
                        background_color: Color::rgba(0.05, 0.05, 0.05, 0.9).into(),
                        ..default()
                    },
                    GraphPanelContent,
                ));

                // Resize Handle
                parent.spawn((
                    NodeBundle {
                        style: Style {
                            position_type: PositionType::Absolute,
                            right: Val::Px(0.0),
                            bottom: Val::Px(0.0),
                            width: Val::Px(20.0),
                            height: Val::Px(20.0),
                            ..default()
                        },
                        background_color: Color::rgba(0.5, 0.5, 0.5, 0.5).into(),
                        ..default()
                    },
                    GraphResizeHandle,
                ));
            }
        });
}

fn toggle_graph_panel(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut graph_state: ResMut<GraphPanelState>,
    windows: Query<&Window>,
) {
    if keyboard_input.just_pressed(KeyCode::KeyG) {
        // Logica toggle: Finestra Normale -> Fullscreen -> Nascosto
        // IMPORTANT: G key only controls graph visibility, NOT rendering state
        // Rendering state is controlled separately by R key
        if !graph_state.visible && !graph_state.fullscreen {
            // Prima pressione: mostra grafico a finestra
            graph_state.visible = true;
            graph_state.fullscreen = false;
        } else if graph_state.visible && !graph_state.fullscreen {
            // Seconda pressione: passa a fullscreen
            graph_state.fullscreen = true;

            // Imposta dimensioni fullscreen dalla finestra
            if let Ok(window) = windows.get_single() {
                graph_state.size = Vec2::new(window.width(), window.height() - 60.0);
                graph_state.position = Vec2::new(0.0, 0.0);
            }
        } else {
            // Terza pressione: nascondi tutto
            graph_state.visible = false;
            graph_state.fullscreen = false;
        }

        // Ridisegna sempre quando il pannello cambia stato
        graph_state.needs_redraw = true;

        println!(
            "Grafico: {}",
            if graph_state.visible {
                if graph_state.fullscreen {
                    "FULLSCREEN"
                } else {
                    "FINESTRA"
                }
            } else {
                "NASCOSTO"
            }
        );
    }
}

fn update_graph_panel_visibility(
    mut commands: Commands,
    mut graph_state: ResMut<GraphPanelState>,
    panel_query: Query<Entity, With<GraphPanel>>,
) {
    let panel_exists = !panel_query.is_empty();

    // VISIBILITY RULE: Only user preference via G key
    // Graph visibility is now independent of rendering state
    let should_be_visible = graph_state.visible;

    if should_be_visible && !panel_exists {
        // OPEN
        graph_state.needs_redraw = true;
        graph_state.last_entry_count = 0;
        let state_res = graph_state.into();
        spawn_graph_panel(commands, state_res, panel_query);
    } else if !should_be_visible && panel_exists {
        // CLOSE
        for entity in panel_query.iter() {
            commands.entity(entity).despawn_recursive();
        }
    }
}

fn handle_graph_panel_interactions(
    mut graph_state: ResMut<GraphPanelState>,
    mouse_button: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    // Query per i componenti interattivi
    header_query: Query<&Interaction, (Changed<Interaction>, With<GraphPanelHeader>)>,
    collapse_query: Query<&Interaction, (Changed<Interaction>, With<GraphCollapseButton>)>,
    close_query: Query<&Interaction, (Changed<Interaction>, With<GraphCloseButton>)>,
    resize_query: Query<&Interaction, (Changed<Interaction>, With<GraphResizeHandle>)>,
) {
    let Ok(window) = windows.get_single() else {
        return;
    };
    let cursor_pos = window.cursor_position().unwrap_or(Vec2::ZERO);

    // --- 1. GESTIONE TRASCINAMENTO (DRAG) ---
    if graph_state.is_dragging {
        // Se stiamo trascinando, continuiamo finché il tasto non viene rilasciato
        if mouse_button.just_released(MouseButton::Left) {
            graph_state.is_dragging = false;
        } else {
            // Aggiorna posizione: Posizione Mouse - Offset Iniziale
            // Clampiamo per non far uscire la finestra dallo schermo
            let new_pos = cursor_pos - graph_state.drag_offset;
            graph_state.position.x = new_pos.x.clamp(0.0, window.width() - 50.0);
            graph_state.position.y = new_pos.y.clamp(0.0, window.height() - 50.0);
        }
        return; // Se trasciniamo, non fare altro
    }

    // --- 2. GESTIONE RIDIMENSIONAMENTO (RESIZE) ---
    if graph_state.is_resizing {
        if mouse_button.just_released(MouseButton::Left) {
            graph_state.is_resizing = false;
            graph_state.needs_redraw = true; // Ridisegna il grafico alla fine
        } else {
            let mouse_delta = cursor_pos - graph_state.resize_start_pos;
            let new_size = graph_state.resize_start_size + mouse_delta;
            // Limiti minimi
            graph_state.size.x = new_size.x.max(300.0);
            graph_state.size.y = new_size.y.max(200.0);
            graph_state.needs_redraw = true; // Ridisegna live
        }
        return; // Se ridimensioniamo, non fare altro
    }

    // --- 3. INIZIO INTERAZIONI (CLICK) ---

    // Header Drag Start
    for interaction in header_query.iter() {
        if *interaction == Interaction::Pressed {
            graph_state.is_dragging = true;
            // Calcola l'offset: dove ho cliccato rispetto all'angolo della finestra
            graph_state.drag_offset = cursor_pos - graph_state.position;
        }
    }

    // Resize Handle Start
    for interaction in resize_query.iter() {
        if *interaction == Interaction::Pressed {
            graph_state.is_resizing = true;
            graph_state.resize_start_pos = cursor_pos;
            graph_state.resize_start_size = graph_state.size;
        }
    }

    // Collapse Button
    for interaction in collapse_query.iter() {
        if *interaction == Interaction::Pressed {
            graph_state.collapsed = !graph_state.collapsed;
            graph_state.needs_redraw = true;
        }
    }

    // Close Button
    for interaction in close_query.iter() {
        if *interaction == Interaction::Pressed {
            graph_state.visible = false;
        }
    }
}

// Sistema per aggiornare visivamente il testo del bottone collapse
fn update_collapse_button_text(
    graph_state: Res<GraphPanelState>,
    button_query: Query<&Children, With<GraphCollapseButton>>,
    mut text_query: Query<&mut Text>,
) {
    if graph_state.is_changed() {
        for children in button_query.iter() {
            for &child in children.iter() {
                if let Ok(mut text) = text_query.get_mut(child) {
                    // Se collassato mostra "v" (espandi), altrimenti "^" (collassa)
                    // Usa caratteri ASCII semplici
                    text.sections[0].value = if graph_state.collapsed {
                        "v".to_string()
                    } else {
                        "^".to_string()
                    };
                }
            }
        }
    }
}

fn draw_graph_in_panel(
    mut commands: Commands,
    mut graph_state: ResMut<GraphPanelState>,
    global_history: Res<GlobalTrainingHistory>,
    content_query: Query<Entity, With<GraphPanelContent>>,
    children_query: Query<&Children>,
) {
    // Graph visibility is independent of rendering state
    if !graph_state.visible || graph_state.collapsed {
        return;
    }

    // Ridisegna solo se i dati sono cambiati o se forzato
    let data_changed = global_history.records.len() != graph_state.last_entry_count;
    if !graph_state.needs_redraw && !data_changed && graph_state.last_entry_count != 0 {
        return;
    }

    // --- 1. PULIZIA ---
    for content_entity in content_query.iter() {
        if let Ok(children) = children_query.get(content_entity) {
            for &child in children.iter() {
                commands.entity(child).despawn_recursive();
            }
        }
    }

    graph_state.needs_redraw = false;
    graph_state.last_entry_count = global_history.records.len();

    // Se non ci sono dati, esci
    if global_history.records.is_empty() {
        return;
    }

    // --- 2. CONFIGURAZIONE LAYOUT ---
    for content_entity in content_query.iter() {
        let margin_left = 40.0;
        let margin_bottom = 30.0;
        let margin_top = 20.0;
        let margin_right = 20.0;

        let graph_width = (graph_state.size.x - margin_left - margin_right).max(1.0);
        let graph_height = (graph_state.size.y - margin_bottom - margin_top).max(1.0);

        commands.entity(content_entity).with_children(|parent| {
            // Sfondo semitrasparente
            parent.spawn(NodeBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    left: Val::Px(margin_left),
                    bottom: Val::Px(margin_bottom),
                    width: Val::Px(graph_width),
                    height: Val::Px(graph_height),
                    ..default()
                },
                background_color: Color::rgba(0.0, 0.0, 0.0, 0.5).into(),
                ..default()
            });

            // --- 3. ALGORITMO DI BUCKETING (AGGREGAZIONE) ---
            // Definiamo quanti "pixel" o "colonne" visive vogliamo al massimo.
            // 2.0 pixel di larghezza per barra è un buon compromesso tra densità e performance.
            let bar_width_px = 2.0;
            let max_bars = (graph_width / bar_width_px).floor() as usize;

            let total_records = global_history.records.len();

            // Calcoliamo la dimensione del chunk (quanti dati reali finiscono in una barra visiva)
            // Se abbiamo meno dati delle barre disponibili, chunk_size sarà 1.
            let chunk_size = (total_records as f32 / max_bars as f32).ceil() as usize;
            let chunk_size = chunk_size.max(1);

            // Dati aggregati da disegnare
            struct AggregatedPoint {
                avg: f32,
                max: u32,
                min: u32,
            }

            let mut visual_points = Vec::new();

            // Calcolo del MAX SCORE globale per la scala Y
            let global_max_score = global_history
                .records
                .iter()
                .map(|r| r.max_score)
                .max()
                .unwrap_or(10)
                .max(10) as f32;

            // Iteriamo sui chunk
            for chunk in global_history.records.chunks(chunk_size) {
                if chunk.is_empty() {
                    continue;
                }

                // QUI sta la magia: invece di prendere il primo elemento,
                // calcoliamo le statistiche del chunk.
                let max_in_chunk = chunk.iter().map(|r| r.max_score).max().unwrap_or(0);
                let min_in_chunk = chunk.iter().map(|r| r.min_score).min().unwrap_or(0);
                let sum_avg: f32 = chunk.iter().map(|r| r.avg_score).sum();
                let avg_in_chunk = sum_avg / chunk.len() as f32;

                visual_points.push(AggregatedPoint {
                    avg: avg_in_chunk,
                    max: max_in_chunk,
                    min: min_in_chunk,
                });
            }

            // --- 4. RENDERING UNIFORME ---
            let num_visual_points = visual_points.len();
            // Ricalcoliamo la larghezza esatta per riempire tutto lo spazio
            let exact_bar_width = graph_width / num_visual_points as f32;

            for (i, point) in visual_points.iter().enumerate() {
                let x_pos = margin_left + (i as f32 * exact_bar_width);

                // Funzione helper per la Y
                let get_height = |val: f32| -> f32 {
                    let ratio = (val / global_max_score).clamp(0.0, 1.0);
                    ratio * graph_height
                };

                let h_max = get_height(point.max as f32);
                let h_avg = get_height(point.avg);
                let h_min = get_height(point.min as f32);

                // Disegniamo 3 layer per ogni "colonna" temporale
                // Usiamo un leggero gap tra le barre se sono larghe, altrimenti no
                let display_width = if exact_bar_width > 2.0 {
                    exact_bar_width - 1.0
                } else {
                    exact_bar_width
                };

                // 1. Barra del MASSIMO (Rosso scuro/trasparente background)
                // Rappresenta il potenziale raggiunto in quel periodo
                if h_max > 0.0 {
                    parent.spawn(NodeBundle {
                        style: Style {
                            position_type: PositionType::Absolute,
                            left: Val::Px(x_pos),
                            bottom: Val::Px(margin_bottom),
                            width: Val::Px(display_width),
                            height: Val::Px(h_max),
                            ..default()
                        },
                        background_color: Color::rgba(1.0, 0.2, 0.2, 0.3).into(), // Rosso semitrasparente
                        ..default()
                    });

                    // "Tappo" del massimo (pixel solido in alto per vedere bene lo spike)
                    parent.spawn(NodeBundle {
                        style: Style {
                            position_type: PositionType::Absolute,
                            left: Val::Px(x_pos),
                            bottom: Val::Px(margin_bottom + h_max - 1.0), // In cima alla barra
                            width: Val::Px(display_width),
                            height: Val::Px(display_width.max(2.0)), // Un quadratino visibile
                            ..default()
                        },
                        background_color: Color::rgba(1.0, 0.2, 0.2, 1.0).into(), // Rosso solido
                        ..default()
                    });
                }

                // 2. Barra della MEDIA (Verde)
                // Sovrapposta, mostra la consistenza
                if h_avg > 0.0 {
                    parent.spawn(NodeBundle {
                        style: Style {
                            position_type: PositionType::Absolute,
                            left: Val::Px(x_pos),
                            bottom: Val::Px(margin_bottom),
                            width: Val::Px(display_width),
                            height: Val::Px(h_avg),
                            ..default()
                        },
                        background_color: Color::rgba(0.2, 1.0, 0.2, 0.5).into(),
                        ..default()
                    });
                }

                // 3. Barra del MINIMO (Blu) - opzionale, per vedere i fail
                if h_min > 0.0 {
                    parent.spawn(NodeBundle {
                        style: Style {
                            position_type: PositionType::Absolute,
                            left: Val::Px(x_pos),
                            bottom: Val::Px(margin_bottom),
                            width: Val::Px(display_width),
                            height: Val::Px(h_min),
                            ..default()
                        },
                        background_color: Color::rgba(0.3, 0.3, 1.0, 0.6).into(),
                        ..default()
                    });
                }
            }

            // --- 5. ETICHETTE E TESTI ---
            // Max Score Label
            parent.spawn(
                TextBundle::from_section(
                    format!("Max: {:.0}", global_max_score),
                    TextStyle {
                        font_size: 12.0,
                        color: Color::GRAY,
                        ..default()
                    },
                )
                .with_style(Style {
                    position_type: PositionType::Absolute,
                    left: Val::Px(margin_left),
                    top: Val::Px(margin_top),
                    ..default()
                }),
            );
        });
    }
}

fn sync_graph_panel_layout(
    graph_state: Res<GraphPanelState>,
    mut panel_query: Query<&mut Style, With<GraphPanel>>,
) {
    // Esegui solo se c'è stato un cambiamento per risparmiare performance
    if graph_state.is_changed() {
        for mut style in panel_query.iter_mut() {
            // Aggiorna Posizione
            style.left = Val::Px(graph_state.position.x);
            style.top = Val::Px(graph_state.position.y);

            // Aggiorna Dimensione
            style.width = Val::Px(graph_state.size.x);

            // Gestisci altezza in base al collasso
            // 30.0 è l'altezza dell'header definita nello spawn
            if graph_state.collapsed {
                style.height = Val::Px(30.0);
            } else {
                style.height = Val::Px(graph_state.size.y);
            }
        }
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Rust Bevy DQN Snake - Multi-Snake AI Edition".into(),
                resolution: (800.0, 600.0).into(),
                resizable: true,
                ..default()
            }),
            ..default()
        }))
        .add_event::<AppExit>()
        .add_systems(Startup, setup)
        .add_systems(Startup, spawn_stats_ui.after(setup))
        .add_systems(
            Update,
            (
                handle_input,
                on_window_resize,
                game_loop,
                render_system,
                toggle_graph_panel,
            )
                .chain(),
        )
        .add_systems(Update, update_stats_ui.after(render_system))
        // --- BLOCCO UI GRAFICO ---
        .add_systems(Update, update_graph_panel_visibility)
        .add_systems(Update, handle_graph_panel_interactions)
        // .add_systems(Update, update_collapse_button_text) // (Se lo hai aggiunto prima)
        // NUOVO SISTEMA: Questo fa muovere la finestra in tempo reale!
        .add_systems(
            Update,
            sync_graph_panel_layout.after(handle_graph_panel_interactions),
        )
        .add_systems(
            Update,
            draw_graph_in_panel
                .after(update_graph_panel_visibility)
                .after(sync_graph_panel_layout),
        )
        .run();
}
