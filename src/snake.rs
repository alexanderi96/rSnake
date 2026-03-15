//! Snake simulation module
//!
//! This module contains the core simulation types and functions for the snake game.
//! It also re-exports from plugins::simulation for convenience.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::indexing_slicing)]
#![allow(clippy::unnecessary_map_or)]

use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Instant;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// Constants
// ============================================================================

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

/// Food sequence length
#[allow(dead_code)]
pub const FOOD_SEQ_LEN: usize = 1000;

// ============================================================================
// Resources
// ============================================================================

/// Numero di step di simulazione per frame Bevy (default 1, range 1-200)
#[derive(Resource, Clone, Debug)]
pub struct SimStepsPerFrame(pub u32);

impl Default for SimStepsPerFrame {
    fn default() -> Self {
        Self(1)
    }
}

/// Dynamic performance tuner for automatic adjustment of simulation speed
#[derive(Resource)]
pub struct PerformanceTuner {
    /// Enable/disable the auto-tuner entirely
    pub enabled: bool,

    /// Target frame time budget in ms (tuner tries to stay under this)
    pub target_frame_ms: f64,

    /// Exponential moving average of recent frame times (ms) — updated every frame
    pub ema_frame_ms: f64,

    /// EMA smoothing factor (higher = reacts faster). Range: 0.01–0.5
    pub ema_alpha: f64,

    /// Frames to wait between adjustments (prevents thrashing)
    pub cooldown_frames: u32,
    pub frames_since_last_adjust: u32,

    /// Bounds for SimStepsPerFrame auto-adjustment
    pub min_steps: u32,
    pub max_steps: u32,

    /// Bounds for population_size auto-adjustment
    pub min_population: usize,
    pub max_population: usize,

    /// If true, prefers adjusting steps before population (lower overhead)
    pub prefer_steps_first: bool,
}

impl Default for PerformanceTuner {
    fn default() -> Self {
        Self {
            enabled: true,
            target_frame_ms: 14.0,
            ema_frame_ms: 14.0,
            ema_alpha: 0.1,
            cooldown_frames: 30,
            frames_since_last_adjust: 0,
            min_steps: 1,
            max_steps: 64,
            min_population: 50,
            max_population: 2000,
            prefer_steps_first: true,
        }
    }
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

/// Modalità apprendimento continuo
/// Quando enabled: gli snake morti vengono sostituiti immediatamente invece di aspettare la fine della generazione
#[derive(Resource, Clone, Debug)]
pub struct ContinuousMode {
    /// Se true, gli snake morti vengono sostituiti immediatamente
    pub enabled: bool,
    /// Numero di replacement effettuati in questa sessione
    pub replacement_count: u64,
    /// Numero di replacement dalla ultima rigenerazione seed
    pub replacements_since_seed: u64,
}

impl Default for ContinuousMode {
    fn default() -> Self {
        Self {
            enabled: false,
            replacement_count: 0,
            replacements_since_seed: 0,
        }
    }
}

/// Global training history with separated read-only history and current session
#[derive(Resource, Default)]
pub struct GlobalTrainingHistory {
    /// Historical records loaded from previous sessions
    pub history_log: Vec<crate::plugins::map_elites::evolution::GenerationRecord>,
    /// Current session records (write-only, saved to new file)
    pub current_session: Vec<crate::plugins::map_elites::evolution::GenerationRecord>,
    /// Total accumulated time from all previous sessions
    pub accumulated_time_secs: u64,
    /// All-time high score across all sessions
    pub all_time_high_score: u32,
}

impl GlobalTrainingHistory {
    /// Get all records combined (for UI display)
    pub fn all_records(
        &self,
    ) -> impl Iterator<Item = &crate::plugins::map_elites::evolution::GenerationRecord> {
        self.history_log.iter().chain(self.current_session.iter())
    }

    /// Add a record to the current session
    pub fn push(&mut self, record: crate::plugins::map_elites::evolution::GenerationRecord) {
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

#[derive(Resource, Serialize, Deserialize, Debug, Default)]
pub struct TrainingSession {
    pub total_time_secs: u64,
    pub records: Vec<crate::plugins::map_elites::evolution::GenerationRecord>,
}

// Re-export from plugins::simulation for convenience
#[allow(unused_imports)]
pub use crate::plugins::simulation::{
    bfs_distance, calculate_grid_dimensions, get_current_17_state, Direction, Food, GameState,
    GenerationSeed, GridDimensions, GridMap, Position, SnakeId, SnakeInstance, RAY_DIRECTIONS,
};

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

// ============================================================================
// File and Directory Management
// ============================================================================

/// Crea o recupera la directory di run.
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
                dirs.sort_by(|a, b| b.cmp(a));
                println!("Resume last run: {}", dirs[0].display());
                return dirs[0].clone();
            }
        }
    }

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
    let mut max_gen = 0u32;

    if let Ok(entries) = std::fs::read_dir(sessions_dir) {
        let mut paths: Vec<_> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .map_or(false, |ext| ext == "json" || ext == "gz")
            })
            .collect();

        paths.sort();

        for path in paths {
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
    let session = TrainingSession {
        total_time_secs: session_duration_secs,
        records: history.current_session.clone(),
    };

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
pub fn load_json_gz<T: for<'de> Deserialize<'de>>(path: &std::path::Path) -> std::io::Result<T> {
    let content = std::fs::read(path)?;

    let is_gzip = content.len() >= 2 && content[0] == 0x1f && content[1] == 0x8b;

    if is_gzip {
        let decoder = GzDecoder::new(&content[..]);
        let reader = std::io::BufReader::new(decoder);
        return Ok(serde_json::from_reader(reader)?);
    }

    Ok(serde_json::from_slice(&content)?)
}
