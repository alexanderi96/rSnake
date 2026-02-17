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
const STATE_SIZE: usize = 11;

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
            steps_per_frame: 50, // Valore default per il "Turbo"
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

#[derive(Resource, Serialize, Deserialize, Debug, Default, Clone)]
struct GameStats {
    total_generations: u32,
    high_score: u32,
    total_games_played: u64,
    total_food_eaten: u64,
    total_training_time_secs: u64,
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
            total_training_time_secs: 0,
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

    fn update(&mut self, game: &GameState, training_time: Duration) {
        self.total_generations = game.total_iterations;
        self.high_score = game.high_score.max(self.high_score);
        self.total_training_time_secs = training_time.as_secs();

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

    // COMANDI
    commands.spawn((
        TextBundle::from_section(
            "[R]Turbo  [G]Graph  [F]Full  [ESC]Exit", // Testo corretto
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
    render_config: Res<RenderConfig>, // Aggiunto
) {
    // ... (codice tempo e FPS invariato) ...
    let now = Instant::now();
    let elapsed = now.duration_since(stats.last_update);
    // ... [Il calcolo FPS resta uguale, ometto per brevità] ...

    // Ricalcolo variabili per brevità del copincolla (assicurati di avere queste nel tuo codice originale o copiale da lì)
    let total_secs = stats.total_training_time.as_secs();
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;
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
        st_text.sections[1].value = format!(
            "Time: {:02}:{:02}:{:02}  Total: {:02}:{:02}:{:02}  FPS: {:5.1}\n",
            hours,
            minutes,
            seconds,
            (game_stats.total_training_time_secs + total_secs) / 3600,
            ((game_stats.total_training_time_secs + total_secs) % 3600) / 60,
            (game_stats.total_training_time_secs + total_secs) % 60,
            stats.fps
        );
        // ASCII puro anche qui
        st_text.sections[2].value = format!(
            "Alive:{} Dead:{} | Food: {}  Games: {}",
            alive_count, dead_count, game_stats.total_food_eaten, game_stats.total_games_played
        );
    }

    // COMANDI
    if let Ok(mut cmd_text) = commands_query.get_single_mut() {
        let render_status = if render_config.enabled { "ON" } else { "TURBO" };
        cmd_text.sections[0].value =
            format!("[R]Render:{}  [G]Graph  [F]Full  [ESC]Exit", render_status);
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

    // Carica TrainingSession - ogni sessione è un file separato
    let session_file = new_session_path();
    let mut training_session = if session_file.exists() {
        match TrainingSession::load(session_file.to_str().unwrap_or("session.json")) {
            Ok(session) => {
                println!(
                    "📊 Sessione caricata: {} records da {}",
                    session.records.len(),
                    session_file.display()
                );
                session
            }
            Err(e) => {
                eprintln!("Errore caricamento sessione: {}, creo nuova", e);
                TrainingSession::new()
            }
        }
    } else {
        println!("📊 Nuova sessione: {}", session_file.display());
        TrainingSession::new()
    };

    // Nota: session_file viene passato tramite GameConfig

    let mut brain = if brain_path().exists() {
        match DqnBrain::load(brain_path().to_str().unwrap_or("brain.json")) {
            Ok(b) => {
                println!("Modello esistente caricato!");
                b
            }
            Err(e) => {
                eprintln!("Errore caricamento modello: {}, creo nuovo brain", e);
                DqnBrain::new()
            }
        }
    } else {
        println!("Nessun modello trovato, inizializzo nuovo brain");
        DqnBrain::new()
    };

    // Crea configurazione parallelism - numero serpenti = core CPU
    let parallel_config = ParallelConfig::new();

    // Crea GameState DOPO aver caricato TrainingSession
    let grid = GridDimensions {
        width: grid_width,
        height: grid_height,
    };
    let mut game_state = GameState::new(&grid, parallel_config.snake_count);

    // FIX SINCRONIZZAZIONE: Se c'è uno storico, sincronizza il contatore
    if let Some(last_gen) = training_session.last_generation() {
        println!("🔄 Sync: Riprendo dalla generazione {}", last_gen);
        game_state.total_iterations = last_gen;
        brain.iterations = last_gen;
    }

    commands.insert_resource(training_session);
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

    commands.insert_resource(TrainingStats {
        total_training_time: Duration::from_secs(0),
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

/// Ottimizzato: restituisce array in stack invece di Vec (zero allocazioni heap)
fn get_state(
    snake: &SnakeInstance,
    all_snakes: &[SnakeInstance],
    snake_idx: usize,
    grid: &GridDimensions,
    check_other_snakes: bool,
) -> [f32; STATE_SIZE] {
    let head = snake.snake[0];

    let dir_onehot = match snake.direction {
        Direction::Up => [1.0, 0.0, 0.0, 0.0],
        Direction::Right => [0.0, 1.0, 0.0, 0.0],
        Direction::Down => [0.0, 0.0, 1.0, 0.0],
        Direction::Left => [0.0, 0.0, 0.0, 1.0],
    };

    let food_left = if snake.food.x < head.x { 1.0 } else { 0.0 };
    let food_right = if snake.food.x > head.x { 1.0 } else { 0.0 };
    let food_up = if snake.food.y > head.y { 1.0 } else { 0.0 };
    let food_down = if snake.food.y < head.y { 1.0 } else { 0.0 };

    let left_dir = snake.direction.turn_left();
    let right_dir = snake.direction.turn_right();

    // Pericolo base: bordi e se stesso
    let danger_straight = is_collision(snake, snake.direction, grid)
        || (check_other_snakes
            && is_collision_with_others(snake, snake.direction, all_snakes, snake_idx));
    let danger_left = is_collision(snake, left_dir, grid)
        || (check_other_snakes && is_collision_with_others(snake, left_dir, all_snakes, snake_idx));
    let danger_right = is_collision(snake, right_dir, grid)
        || (check_other_snakes
            && is_collision_with_others(snake, right_dir, all_snakes, snake_idx));

    [
        danger_left as i32 as f32,
        danger_straight as i32 as f32,
        danger_right as i32 as f32,
        dir_onehot[0],
        dir_onehot[1],
        dir_onehot[2],
        dir_onehot[3],
        food_left,
        food_right,
        food_up,
        food_down,
    ]
}

/// Controlla collisione con altri snake (usato dallo stato quando collisioni sono attive)
fn is_collision_with_others(
    snake: &SnakeInstance,
    direction: Direction,
    all_snakes: &[SnakeInstance],
    self_idx: usize,
) -> bool {
    let head = snake.snake[0];
    let (dx, dy) = direction.as_vec();
    let next_pos = Position {
        x: head.x + dx,
        y: head.y + dy,
    };

    for (idx, other) in all_snakes.iter().enumerate() {
        if idx != self_idx && !other.is_game_over {
            if other.snake.contains(&next_pos) {
                return true;
            }
        }
    }

    false
}

fn is_collision(snake: &SnakeInstance, direction: Direction, grid: &GridDimensions) -> bool {
    let head = snake.snake[0];
    let (dx, dy) = direction.as_vec();
    let next_pos = Position {
        x: head.x + dx,
        y: head.y + dy,
    };

    if next_pos.x < 0 || next_pos.x >= grid.width || next_pos.y < 0 || next_pos.y >= grid.height {
        return true;
    }

    if snake.snake.contains(&next_pos) {
        return true;
    }

    false
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

fn game_loop(
    time: Res<Time>,
    mut config: ResMut<GameConfig>,
    mut game: ResMut<GameState>,
    mut brain: ResMut<DqnBrain>,
    mut game_stats: ResMut<GameStats>,
    mut training_session: ResMut<TrainingSession>,
    mut stats: ResMut<TrainingStats>, // ResMut per aggiornare stats
    grid: Res<GridDimensions>,
    collision_settings: Res<CollisionSettings>,
    parallel_config: Res<ParallelConfig>,
    render_config: Res<RenderConfig>, // Aggiunto
) {
    // --- 1. GESTIONE FPS E TEMPO ---
    stats.frame_count += 1;
    let now = Instant::now();
    if now.duration_since(stats.last_fps_update).as_secs_f32() >= 1.0 {
        stats.fps =
            stats.frame_count as f32 / now.duration_since(stats.last_fps_update).as_secs_f32();
        stats.last_fps_update = now;
        stats.frame_count = 0;
    }

    // --- 2. LOGICA TURBO / VELOCITÀ ---
    let steps_to_run;

    if render_config.enabled {
        // Modalità Normale (Render ON): Rispetta il Timer per non andare troppo veloce visivamente
        config.speed_timer.tick(time.delta());
        if !config.speed_timer.finished() {
            return;
        }
        steps_to_run = 1; // 1 step logico per frame renderizzato

        // Aggiorna tempo training reale
        stats.total_training_time += time.delta();
    } else {
        // Modalità Turbo (Render OFF): Ignora il timer!
        // Esegue N step logici per ogni frame dell'applicazione
        steps_to_run = render_config.steps_per_frame;

        // Simulazione del tempo trascorso (o usa tempo reale se preferisci)
        // Usiamo tempo reale del delta frame per accuratezza
        stats.total_training_time += time.delta();
    }

    // --- 3. CICLO DI SIMULAZIONE (Eseguito 1 o 50 volte) ---
    for _ in 0..steps_to_run {
        // [INIZIO LOGICA ORIGINALE] ----------------------------------------

        let mut all_dead = true;
        let mut active_count = 0;
        let mut step_results: Vec<StepResult> = Vec::with_capacity(parallel_config.snake_count);

        // Processa ogni snake
        for snake_idx in 0..game.snakes.len() {
            if game.snakes[snake_idx].is_game_over {
                continue;
            }

            all_dead = false;
            active_count += 1;

            let state = get_state(
                &game.snakes[snake_idx],
                &game.snakes,
                snake_idx,
                &grid,
                collision_settings.snake_vs_snake,
            );

            let mut rng = rand::thread_rng();
            let action_idx = if rng.gen::<f32>() < brain.epsilon {
                rng.gen_range(0..3)
            } else {
                let q_vals = brain.forward_array(&state);
                let mut max_idx = 0;
                let mut max_val = q_vals[0];
                for i in 1..3 {
                    if q_vals[i] > max_val {
                        max_val = q_vals[i];
                        max_idx = i;
                    }
                }
                max_idx
            };

            match action_idx {
                1 => {
                    game.snakes[snake_idx].direction = game.snakes[snake_idx].direction.turn_right()
                }
                2 => {
                    game.snakes[snake_idx].direction = game.snakes[snake_idx].direction.turn_left()
                }
                _ => {}
            }

            let (dx, dy) = game.snakes[snake_idx].direction.as_vec();
            let new_head = Position {
                x: game.snakes[snake_idx].snake[0].x + dx,
                y: game.snakes[snake_idx].snake[0].y + dy,
            };

            game.snakes[snake_idx].steps_without_food += 1;
            let mut reward: f32 = 0.0;
            let mut done = false;

            let mut collision = new_head.x < 0
                || new_head.x >= grid.width
                || new_head.y < 0
                || new_head.y >= grid.height
                || game.snakes[snake_idx].snake.contains(&new_head);

            if !collision && collision_settings.snake_vs_snake {
                for (other_idx, other_snake) in game.snakes.iter().enumerate() {
                    if other_idx != snake_idx && !other_snake.is_game_over {
                        if other_snake.snake.contains(&new_head) {
                            collision = true;
                            break;
                        }
                    }
                }
            }

            if collision {
                reward = -10.0;
                done = true;
                game.snakes[snake_idx].is_game_over = true;
            } else if new_head == game.snakes[snake_idx].food {
                reward = 10.0 + (game.snakes[snake_idx].score as f32 * 0.5);
                game.snakes[snake_idx].snake.push_front(new_head);
                game.snakes[snake_idx].score += 1;
                if game.snakes[snake_idx].score > game.high_score {
                    game.high_score = game.snakes[snake_idx].score;
                }
                game.snakes[snake_idx].food = spawn_food(&game.snakes[snake_idx], &grid);
                game.snakes[snake_idx].steps_without_food = 0;
            } else {
                game.snakes[snake_idx].snake.push_front(new_head);
                game.snakes[snake_idx].snake.pop_back();

                let max_steps = (grid.width * grid.height) as u32;
                reward -= 0.01 * (game.snakes[snake_idx].steps_without_food as f32 / 100.0);

                if game.snakes[snake_idx].steps_without_food > max_steps {
                    reward = -10.0;
                    done = true;
                    game.snakes[snake_idx].is_game_over = true;
                } else {
                    reward += 0.001;
                    let head = &game.snakes[snake_idx].snake[0];
                    let old_dist_sq = ((head.x - dx - game.snakes[snake_idx].food.x).pow(2)
                        + (head.y - dy - game.snakes[snake_idx].food.y).pow(2))
                        as f32;
                    let new_dist_sq = ((head.x - game.snakes[snake_idx].food.x).pow(2)
                        + (head.y - game.snakes[snake_idx].food.y).pow(2))
                        as f32;
                    if new_dist_sq < old_dist_sq {
                        reward += 0.1;
                    } else {
                        reward -= 0.15;
                    }
                }
            }

            let ate_food = new_head == game.snakes[snake_idx].food;
            step_results.push(StepResult {
                snake_idx,
                state,
                action_idx,
                reward,
                next_state: get_state(
                    &game.snakes[snake_idx],
                    &game.snakes,
                    snake_idx,
                    &grid,
                    collision_settings.snake_vs_snake,
                ),
                done,
                new_head,
                ate_food,
            });
        }

        for result in step_results {
            brain.remember_array(
                result.state,
                result.action_idx,
                result.reward,
                result.next_state,
                result.done,
            );
        }

        if all_dead {
            let food_eaten: u32 = game.snakes.iter().map(|s| s.score).sum();
            game_stats.total_food_eaten += food_eaten as u64;

            for (i, snake) in game.snakes.iter().enumerate() {
                if i < game_stats.best_score_per_snake.len() {
                    game_stats.best_score_per_snake[i] =
                        game_stats.best_score_per_snake[i].max(snake.score);
                }
            }

            let current_scores: Vec<u32> = game.snakes.iter().map(|s| s.score).collect();
            // Rimosso game_history.add_entry che era duplicato/inutile

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
            training_session.add_record(record);
            training_session.total_time_secs = stats.total_training_time.as_secs();

            if let Err(e) =
                training_session.save(config.session_path.to_str().unwrap_or("session.json"))
            {
                eprintln!("Errore salvataggio sessione: {}", e);
            }

            if game.total_iterations % 100 == 0 {
                if let Err(e) = brain.save(brain_path().to_str().unwrap_or("brain.json")) {
                    eprintln!("Errore salvataggio modello: {}", e);
                }
            }

            for snake in game.snakes.iter_mut() {
                snake.reset(&grid);
            }
            game.total_iterations += 1;
            brain.iterations = game.total_iterations;

            brain.train();
            brain.epsilon = (brain.epsilon * 0.995).max(0.01);

            game_stats.total_generations = game.total_iterations;
            game_stats.high_score = game_stats.high_score.max(game.high_score);
            game_stats.total_games_played += parallel_config.snake_count as u64;

            let total_score: u32 = game.snakes.iter().map(|s| s.score).sum();
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

        // [FINE LOGICA ORIGINALE] ----------------------------------------
    }
}

fn render_system(
    mut commands: Commands,
    game: Res<GameState>,
    windows: Query<&Window>,
    grid: Res<GridDimensions>,
    mesh_cache: Res<MeshCache>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    q_segments: Query<Entity, With<SnakeSegment>>,
    q_food: Query<Entity, With<Food>>,
    render_config: Res<RenderConfig>,
) {
    // Skip rendering when disabled (training-only mode)
    if !render_config.enabled {
        return;
    }

    // Ottimizzazione: despawn solo se necessario (non ogni frame)
    // In realtà per Bevy è meglio despawn/ricreate per oggetti dinamici
    // Ma usiamo le mesh cached per evitare allocazioni GPU
    for e in q_segments.iter() {
        commands.entity(e).despawn();
    }
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

    // Usa mesh cache (zero allocazioni GPU per frame!)
    // NOTA: Salta gli snake morti (is_game_over = true)
    for snake in game.snakes.iter() {
        // Salta snake morti - non renderizzare né snake né cibo
        if snake.is_game_over {
            continue;
        }

        // Crea materiale con il colore ATTUALE dello snake (cambia ad ogni reset!)
        let body_material = materials.add(snake.color);

        for (i, pos) in snake.snake.iter().enumerate() {
            // Testa = bianca, corpo = colore dello snake
            let material = if i == 0 {
                mesh_cache.head_material.clone()
            } else {
                body_material.clone()
            };

            commands.spawn((
                MaterialMesh2dBundle {
                    mesh: mesh_cache.segment_mesh.clone().into(),
                    material,
                    transform: Transform::from_xyz(
                        offset_x + (pos.x as f32 * BLOCK_SIZE),
                        offset_y - (pos.y as f32 * BLOCK_SIZE),
                        0.0,
                    ),
                    ..default()
                },
                SnakeSegment,
                SnakeId(snake.id),
            ));
        }

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
    stats: Res<TrainingStats>,
    mut window_settings: ResMut<WindowSettings>,
    mut windows: Query<&mut Window>,
    mut collision_settings: ResMut<CollisionSettings>,
    mut render_config: ResMut<RenderConfig>,  // Aggiunto
    mut graph_state: ResMut<GraphPanelState>, // Aggiunto
) {
    // [ESC] SALVA E ESCI
    if keyboard_input.just_pressed(KeyCode::Escape) {
        // Aggiorna statistiche finali
        game_stats.update(&game, stats.total_training_time);

        // Salva brain.json (unico per run)
        if let Err(e) = brain.save(brain_path().to_str().unwrap_or("brain.json")) {
            eprintln!("Errore salvataggio modello: {}", e);
        }

        // Salva sessione corrente
        training_session.total_time_secs = stats.total_training_time.as_secs();
        training_session.compress_history();
        if let Err(e) =
            training_session.save(config.session_path.to_str().unwrap_or("session.json"))
        {
            eprintln!("Errore salvataggio sessione: {}", e);
        }

        println!("\n=== RIEPILOGO SESSIONE ===");
        println!("Generazioni totali: {}", game_stats.total_generations);
        println!("High Score: {}", game_stats.high_score);
        println!(
            "Tempo di training: {}s",
            game_stats.total_training_time_secs
        );
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

    // [R] RENDER / TURBO MODE (NUOVO)
    if keyboard_input.just_pressed(KeyCode::KeyR) {
        render_config.enabled = !render_config.enabled;
        println!(
            "Rendering: {}",
            if render_config.enabled {
                "ON (Normal Speed)"
            } else {
                "OFF (Turbo Speed)"
            }
        );

        // Se disattivo il render (Turbo), apro automaticamente il grafico per vedere i progressi
        if !render_config.enabled {
            graph_state.visible = true;
            graph_state.needs_redraw = true;
        }
    }

    // [G] GRAPH TOGGLE
    if keyboard_input.just_pressed(KeyCode::KeyG) {
        graph_state.visible = !graph_state.visible;
        graph_state.needs_redraw = true;
    }

    // [F] FULLSCREEN
    if keyboard_input.just_pressed(KeyCode::KeyF) {
        window_settings.is_fullscreen = !window_settings.is_fullscreen;
        let mut window = windows.single_mut();
        window.mode = if window_settings.is_fullscreen {
            WindowMode::Fullscreen
        } else {
            WindowMode::Windowed
        };

        // Se vado in fullscreen, assicuro che il grafico sia visibile
        if window_settings.is_fullscreen {
            graph_state.visible = true;
            graph_state.needs_redraw = true;
            // Opzionale: Disabilita render in fullscreen per max performance
            // render_config.enabled = false;
        }
    }
}

fn on_window_resize(
    mut resize_events: EventReader<bevy::window::WindowResized>,
    mut grid: ResMut<GridDimensions>,
    mut game: ResMut<GameState>,
    mut graph_state: ResMut<GraphPanelState>,
) {
    for event in resize_events.read() {
        let (new_width, new_height) = calculate_grid_dimensions(event.width, event.height);

        // Aggiorna le dimensioni della griglia
        grid.width = new_width;
        grid.height = new_height;

        // Resetta tutti gli snake con le nuove dimensioni
        for snake in game.snakes.iter_mut() {
            snake.reset(&grid);
        }

        // Aggiorna dimensioni grafico se in fullscreen
        if graph_state.fullscreen {
            graph_state.size = Vec2::new(event.width, event.height - 60.0);
            graph_state.needs_redraw = true;
        }

        println!("Griglia ridimensionata: {}x{}", grid.width, grid.height);
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
    mut render_config: ResMut<RenderConfig>,
    windows: Query<&Window>,
) {
    if keyboard_input.just_pressed(KeyCode::KeyG) {
        // Logica toggle: Finestra Normale -> Fullscreen -> Nascosto
        if !graph_state.visible && !graph_state.fullscreen {
            // Prima pressione: mostra grafico a finestra
            graph_state.visible = true;
            graph_state.fullscreen = false;
            render_config.enabled = true;
        } else if graph_state.visible && !graph_state.fullscreen {
            // Seconda pressione: passa a fullscreen (rendering disabilitato)
            graph_state.fullscreen = true;
            render_config.enabled = false;

            // Imposta dimensioni fullscreen dalla finestra
            if let Ok(window) = windows.get_single() {
                graph_state.size = Vec2::new(window.width(), window.height() - 60.0);
                graph_state.position = Vec2::new(0.0, 0.0);
            }
        } else {
            // Terza pressione: nascondi tutto
            graph_state.visible = false;
            graph_state.fullscreen = false;
            render_config.enabled = true;
        }

        // Ridisegna sempre quando il pannello cambia stato
        graph_state.needs_redraw = true;

        println!(
            "Grafico: {} (render: {})",
            if graph_state.visible {
                if graph_state.fullscreen {
                    "FULLSCREEN"
                } else {
                    "FINESTRA"
                }
            } else {
                "NASCOSTO"
            },
            if render_config.enabled { "ON" } else { "OFF" }
        );
    }
}

fn update_graph_panel_visibility(
    mut commands: Commands,
    mut graph_state: ResMut<GraphPanelState>, // Nota: ResMut qui serviva!
    panel_query: Query<Entity, With<GraphPanel>>,
) {
    let panel_exists = !panel_query.is_empty();

    if graph_state.visible && !panel_exists {
        // APERTURA: Resetta il flag di redraw per forzare il disegno delle linee
        graph_state.needs_redraw = true;
        // Importante: resettiamo anche questo per essere sicuri
        graph_state.last_entry_count = 0;

        // Ora possiamo passare una Res immutabile a spawn_graph_panel
        // (Bevy permette di "downgrade" da ResMut a Res automaticamente o tramite into())
        let state_res = graph_state.into();
        spawn_graph_panel(commands, state_res, panel_query);
    } else if !graph_state.visible && panel_exists {
        // CHIUSURA
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
    training_session: Res<TrainingSession>,
    parallel_config: Res<ParallelConfig>,
    content_query: Query<Entity, With<GraphPanelContent>>,
    children_query: Query<&Children>,
) {
    // ... (Controlli di visibilità e redraw invariati) ...
    if !graph_state.visible || graph_state.collapsed {
        return;
    }
    let data_changed = training_session.records.len() != graph_state.last_entry_count;
    if !graph_state.needs_redraw && !data_changed && graph_state.last_entry_count != 0 {
        return;
    }

    // ... (Pulizia figli invariata) ...
    for content_entity in content_query.iter() {
        if let Ok(children) = children_query.get(content_entity) {
            for &child in children.iter() {
                commands.entity(child).despawn_recursive();
            }
        }
    }
    graph_state.needs_redraw = false;
    graph_state.last_entry_count = training_session.records.len();

    for content_entity in content_query.iter() {
        // ... (Definizione margini e dimensioni invariata) ...
        let margin_left = 40.0;
        let margin_bottom = 30.0;
        let margin_top = 20.0;
        let margin_right = 20.0;
        let graph_width = (graph_state.size.x - margin_left - margin_right).max(1.0);
        let graph_height = (graph_state.size.y - 30.0 - margin_bottom - margin_top).max(1.0);

        commands.entity(content_entity).with_children(|parent| {
            // ... (Disegno Sfondo, Assi e check "In attesa di dati" invariati) ...
            parent.spawn(NodeBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    left: Val::Px(margin_left),
                    bottom: Val::Px(margin_bottom),
                    width: Val::Px(graph_width),
                    height: Val::Px(graph_height),
                    ..default()
                },
                background_color: Color::rgba(0.0, 0.0, 0.0, 0.3).into(),
                ..default()
            });
            parent.spawn(NodeBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    left: Val::Px(margin_left),
                    bottom: Val::Px(margin_bottom),
                    width: Val::Px(2.0),
                    height: Val::Px(graph_height),
                    ..default()
                },
                background_color: Color::WHITE.into(),
                ..default()
            });
            parent.spawn(NodeBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    left: Val::Px(margin_left),
                    bottom: Val::Px(margin_bottom),
                    width: Val::Px(graph_width),
                    height: Val::Px(2.0),
                    ..default()
                },
                background_color: Color::WHITE.into(),
                ..default()
            });
            if training_session.records.len() < 2 {
                parent.spawn(
                    TextBundle::from_section(
                        "In attesa di dati...",
                        TextStyle {
                            font_size: 20.0,
                            color: Color::GRAY,
                            ..default()
                        },
                    )
                    .with_style(Style {
                        position_type: PositionType::Absolute,
                        left: Val::Px(margin_left + 10.0),
                        top: Val::Px(50.0),
                        ..default()
                    }),
                );
                return;
            }

            // --- CALCOLO RANGE DATI ---
            let min_gen = training_session.records.first().map(|e| e.gen).unwrap_or(0) as f32;
            let max_gen = training_session.records.last().map(|e| e.gen).unwrap_or(1) as f32;
            let range_gen = (max_gen - min_gen).max(1.0);
            let max_score = training_session
                .records
                .iter()
                .map(|e| e.max_score)
                .max()
                .unwrap_or(10)
                .max(10) as f32;

            // Helper X (invariato)
            let get_x = |gen: u32| -> f32 {
                margin_left + ((gen as f32 - min_gen) / range_gen * graph_width)
            };

            // --- LA FIX E' QUI ---
            // Prima: let get_y = |val: f32| -> f32 { let ratio = (val / max_score)... }
            // Adesso: Accetta `scale_max` come parametro!
            let get_y = |val: f32, scale_max: f32| -> f32 {
                let ratio = (val / scale_max).clamp(0.0, 1.0);
                margin_bottom + (ratio * graph_height)
            };

            // --- DISEGNO LINEE ---
            for i in 0..training_session.records.len().saturating_sub(1) {
                let e1 = &training_session.records[i];
                let e2 = &training_session.records[i + 1];
                let x1 = get_x(e1.gen);
                let x2 = get_x(e2.gen);
                let width = (x2 - x1).max(1.0);

                // Definizione datasets con i loro fattori di scala specifici
                // Nota: Usiamo max_score e avg_score dalla nuova struct GenerationRecord
                let sets = [
                    (
                        e1.max_score as f32,
                        e2.max_score as f32,
                        Color::rgb(1.0, 0.3, 0.3),
                        max_score,
                    ),
                    (
                        e1.avg_score,
                        e2.avg_score,
                        Color::rgb(0.3, 1.0, 0.3),
                        max_score,
                    ),
                    // La linea blu usa snake_count come scala massima
                    (
                        e1.min_score as f32,
                        e2.min_score as f32,
                        Color::rgb(0.3, 0.5, 1.0),
                        parallel_config.snake_count as f32,
                    ),
                ];

                for (val1, val2, color, scale_max) in sets {
                    // --- USARE IL PARAMETRO scale_max QUI ---
                    let y1 = get_y(val1, scale_max);
                    let y2 = get_y(val2, scale_max);

                    // (Resto del disegno segmenti invariato)
                    parent.spawn(NodeBundle {
                        style: Style {
                            position_type: PositionType::Absolute,
                            left: Val::Px(x1),
                            bottom: Val::Px(y1),
                            width: Val::Px(width),
                            height: Val::Px(2.0),
                            ..default()
                        },
                        background_color: color.into(),
                        ..default()
                    });
                    let h_diff = (y2 - y1).abs();
                    if h_diff > 0.5 {
                        parent.spawn(NodeBundle {
                            style: Style {
                                position_type: PositionType::Absolute,
                                left: Val::Px(x2),
                                bottom: Val::Px(y1.min(y2)),
                                width: Val::Px(2.0),
                                height: Val::Px(h_diff + 2.0),
                                ..default()
                            },
                            background_color: color.into(),
                            ..default()
                        });
                    }
                }
            }

            // === LEGENDA DEI COLORI ===
            let legend_y = margin_bottom - 5.0;

            let legend_start_x = margin_left;
            let item_spacing = (graph_width / 3.0).max(80.0);

            // Max Score - Rosso
            let x = legend_start_x;
            parent.spawn(NodeBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    left: Val::Px(x),
                    bottom: Val::Px(legend_y - 8.0),
                    width: Val::Px(12.0),
                    height: Val::Px(12.0),
                    ..default()
                },
                background_color: Color::rgb(1.0, 0.3, 0.3).into(),
                ..default()
            });
            parent.spawn(
                TextBundle::from_section(
                    "Max Score",
                    TextStyle {
                        font_size: 11.0,
                        color: Color::rgb(0.8, 0.8, 0.8),
                        ..default()
                    },
                )
                .with_style(Style {
                    position_type: PositionType::Absolute,
                    left: Val::Px(x + 16.0),
                    bottom: Val::Px(legend_y - 8.0),
                    ..default()
                }),
            );

            // Avg Score - Verde
            let x = legend_start_x + item_spacing;
            parent.spawn(NodeBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    left: Val::Px(x),
                    bottom: Val::Px(legend_y - 8.0),
                    width: Val::Px(12.0),
                    height: Val::Px(12.0),
                    ..default()
                },
                background_color: Color::rgb(0.3, 1.0, 0.3).into(),
                ..default()
            });
            parent.spawn(
                TextBundle::from_section(
                    "Avg Score",
                    TextStyle {
                        font_size: 11.0,
                        color: Color::rgb(0.8, 0.8, 0.8),
                        ..default()
                    },
                )
                .with_style(Style {
                    position_type: PositionType::Absolute,
                    left: Val::Px(x + 16.0),
                    bottom: Val::Px(legend_y - 8.0),
                    ..default()
                }),
            );

            // Min Score - Blu
            let x = legend_start_x + (item_spacing * 2.0);
            parent.spawn(NodeBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    left: Val::Px(x),
                    bottom: Val::Px(legend_y - 8.0),
                    width: Val::Px(12.0),
                    height: Val::Px(12.0),
                    ..default()
                },
                background_color: Color::rgb(0.3, 0.5, 1.0).into(),
                ..default()
            });
            parent.spawn(
                TextBundle::from_section(
                    "Min Score",
                    TextStyle {
                        font_size: 11.0,
                        color: Color::rgb(0.8, 0.8, 0.8),
                        ..default()
                    },
                )
                .with_style(Style {
                    position_type: PositionType::Absolute,
                    left: Val::Px(x + 16.0),
                    bottom: Val::Px(legend_y - 8.0),
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
            draw_graph_in_panel.after(update_graph_panel_visibility),
        )
        .run();
}
