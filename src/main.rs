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
const PARALLEL_SNAKES: usize = 16;
const MEMORY_SIZE: usize = 5000;
const BATCH_SIZE: usize = 256;
const LEARNING_RATE: f32 = 0.0005;
const HIDDEN_NODES: usize = 128;
const TARGET_UPDATE_FREQ: usize = 100;
const STATE_SIZE: usize = 11;

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
    snake_materials: Vec<Handle<ColorMaterial>>,
    food_material: Handle<ColorMaterial>,
    head_material: Handle<ColorMaterial>,
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

        let colors = [
            Color::rgb(0.0, 0.8, 0.0), // Verde
            Color::rgb(0.0, 0.6, 1.0), // Blu
            Color::rgb(1.0, 0.6, 0.0), // Arancione
            Color::rgb(0.8, 0.0, 0.8), // Viola
        ];

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
            color: colors[id % colors.len()],
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
    }
}

#[derive(Resource)]
struct GameState {
    high_score: u32,
    total_iterations: u32,
    snakes: Vec<SnakeInstance>,
}

impl GameState {
    fn new(grid: &GridDimensions) -> Self {
        let snakes = (0..PARALLEL_SNAKES)
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

fn spawn_stats_ui(commands: &mut Commands) {
    commands.spawn((
        TextBundle::from_sections([
            TextSection::new(
                "S: 0  H: 0  G: 0\n",
                TextStyle {
                    font_size: 20.0,
                    color: Color::WHITE,
                    ..default()
                },
            ),
            TextSection::new(
                "Time: 00:00:00  FPS: 0\n",
                TextStyle {
                    font_size: 20.0,
                    color: Color::WHITE,
                    ..default()
                },
            ),
            TextSection::new(
                "Threads: 0  Mode: TRAINING  [F] Fullscreen",
                TextStyle {
                    font_size: 16.0,
                    color: Color::WHITE,
                    ..default()
                },
            ),
        ])
        .with_style(Style {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        }),
        StatsText,
    ));
}

fn update_stats_ui(
    mut query: Query<&mut Text, With<StatsText>>,
    game: Res<GameState>,
    brain: Res<DqnBrain>,
    mut stats: ResMut<TrainingStats>,
    game_stats: Res<GameStats>,
) {
    let mut text = query.single_mut();

    // Aggiorna tempo di training
    let now = Instant::now();
    let elapsed = now.duration_since(stats.last_update);
    stats.total_training_time += elapsed;
    stats.last_update = now;

    // Calcola FPS
    stats.frame_count += 1;
    let fps_elapsed = now.duration_since(stats.last_fps_update);
    if fps_elapsed.as_secs_f32() >= 1.0 {
        stats.fps = stats.frame_count as f32 / fps_elapsed.as_secs_f32();
        stats.frame_count = 0;
        stats.last_fps_update = now;
    }

    let total_secs = stats.total_training_time.as_secs();
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;

    // Score per ogni agente (dinamico, si adatta al numero di snake)
    let scores_text: String = game
        .snakes
        .iter()
        .enumerate()
        .map(|(i, s)| format!("S{}:{:2}", i + 1, s.score))
        .collect::<Vec<_>>()
        .join("  ");

    // Usa il high score persistente se maggiore
    let persistent_high = game_stats.high_score.max(game.high_score);

    text.sections[0].value = format!(
        "{}  |  H: {:3}  G: {:5}  Best: {:3}\n",
        scores_text, game.high_score, game.total_iterations, persistent_high
    );
    text.sections[1].value = format!(
        "Time: {:02}:{:02}:{:02}  Total: {:02}:{:02}:{:02}  FPS: {:5.1}\n",
        hours,
        minutes,
        seconds,
        (game_stats.total_training_time_secs + total_secs) / 3600,
        ((game_stats.total_training_time_secs + total_secs) % 3600) / 60,
        (game_stats.total_training_time_secs + total_secs) % 60,
        stats.fps
    );
    text.sections[2].value = format!(
        "Food: {}  Games: {}  [F] Full  [ESC] Save",
        game_stats.total_food_eaten + game.snakes.iter().map(|s| s.score as u64).sum::<u64>(),
        game_stats.total_games_played
    );
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

    // Colori per ogni snake
    let colors = vec![
        Color::rgb(0.0, 1.0, 0.0), // Verde
        Color::rgb(0.0, 1.0, 1.0), // Ciano
        Color::rgb(1.0, 0.0, 1.0), // Magenta
        Color::rgb(1.0, 1.0, 0.0), // Giallo
        Color::rgb(1.0, 0.5, 0.0), // Arancione
        Color::rgb(0.5, 0.0, 1.0), // Viola
        Color::rgb(0.0, 0.5, 1.0), // Azzurro
        Color::rgb(1.0, 0.0, 0.5), // Rosa
        Color::rgb(0.5, 1.0, 0.0), // Lime
        Color::rgb(0.0, 0.8, 0.4), // Verde acqua
        Color::rgb(0.8, 0.4, 0.0), // Marrone-arancio
        Color::rgb(0.4, 0.0, 0.8), // Indaco
        Color::rgb(0.8, 0.0, 0.4), // Rubino
        Color::rgb(0.4, 0.8, 0.0), // Verde oliva
        Color::rgb(0.0, 0.4, 0.8), // Blu cobalto
        Color::rgb(0.8, 0.8, 0.4), // Giallo chiaro
    ];

    let snake_materials: Vec<_> = colors.iter().map(|&color| materials.add(color)).collect();

    let food_material = materials.add(Color::rgb(1.0, 0.0, 0.0)); // Rosso
    let head_material = materials.add(Color::rgb(1.0, 1.0, 1.0)); // Bianco

    commands.insert_resource(MeshCache {
        segment_mesh,
        food_mesh,
        snake_materials,
        food_material,
        head_material,
    });

    let brain = if Path::new("snake_brain.json").exists() {
        match DqnBrain::load("snake_brain.json") {
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

    commands.insert_resource(brain);
    commands.insert_resource(GameConfig {
        speed_timer: Timer::from_seconds(0.001, TimerMode::Repeating),
    });

    commands.insert_resource(WindowSettings {
        is_fullscreen: false,
    });

    let grid = GridDimensions {
        width: grid_width,
        height: grid_height,
    };

    commands.insert_resource(grid);

    // Crea GameState con le dimensioni corrette
    commands.insert_resource(GameState::new(&GridDimensions {
        width: grid_width,
        height: grid_height,
    }));

    commands.insert_resource(TrainingStats {
        total_training_time: Duration::from_secs(0),
        last_update: Instant::now(),
        parallel_threads: rayon::current_num_threads(),
        fps: 0.0,
        last_fps_update: Instant::now(),
        frame_count: 0,
    });

    // Carica statistiche esistenti o crea nuove
    let mut game_stats = if Path::new("snake_stats.json").exists() {
        match GameStats::load("snake_stats.json") {
            Ok(mut stats) => {
                println!(
                    "Statistiche caricate! High Score: {}, Generazioni: {}",
                    stats.high_score, stats.total_generations
                );
                // Estendi il vettore best_score_per_snake se necessario
                // (ad esempio quando si passa da 4 a 16 snake)
                while stats.best_score_per_snake.len() < PARALLEL_SNAKES {
                    stats.best_score_per_snake.push(0);
                }
                stats
            }
            Err(e) => {
                eprintln!("Errore caricamento statistiche: {}, creo nuove", e);
                GameStats::new(PARALLEL_SNAKES)
            }
        }
    } else {
        println!("Nessuna statistica trovata, inizializzo nuove");
        GameStats::new(PARALLEL_SNAKES)
    };

    commands.insert_resource(game_stats);

    spawn_stats_ui(&mut commands);
}

/// Ottimizzato: restituisce array in stack invece di Vec (zero allocazioni heap)
fn get_state(snake: &SnakeInstance, grid: &GridDimensions) -> [f32; STATE_SIZE] {
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

    let danger_straight = is_collision(snake, snake.direction, grid);
    let danger_left = is_collision(snake, left_dir, grid);
    let danger_right = is_collision(snake, right_dir, grid);

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
    stats: Res<TrainingStats>,
    grid: Res<GridDimensions>,
) {
    config.speed_timer.tick(time.delta());
    if !config.speed_timer.finished() {
        return;
    }

    let mut all_dead = true;
    let mut active_count = 0;

    // Raccolta risultati paralleli
    let mut step_results: Vec<StepResult> = Vec::with_capacity(PARALLEL_SNAKES);

    // Ottimizzazione: processa ogni snake (parallelizzabile in futuro)
    for snake_idx in 0..game.snakes.len() {
        if game.snakes[snake_idx].is_game_over {
            continue;
        }

        all_dead = false;
        active_count += 1;

        // Usa array fisso (zero allocazioni heap)
        let state = get_state(&game.snakes[snake_idx], &grid);

        let mut rng = rand::thread_rng();
        let action_idx = if rng.gen::<f32>() < brain.epsilon {
            rng.gen_range(0..3)
        } else {
            // Usa forward_array ottimizzato (zero allocazioni)
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
            1 => game.snakes[snake_idx].direction = game.snakes[snake_idx].direction.turn_right(),
            2 => game.snakes[snake_idx].direction = game.snakes[snake_idx].direction.turn_left(),
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

        if new_head.x < 0
            || new_head.x >= grid.width
            || new_head.y < 0
            || new_head.y >= grid.height
            || game.snakes[snake_idx].snake.contains(&new_head)
        {
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

        // Salva risultato per applicarlo dopo
        let ate_food = new_head == game.snakes[snake_idx].food;
        step_results.push(StepResult {
            snake_idx,
            state,
            action_idx,
            reward,
            next_state: get_state(&game.snakes[snake_idx], &grid),
            done,
            new_head,
            ate_food,
        });
    }

    // Applica tutti i risultati e salva nella memoria (usa remember_array ottimizzato)
    for result in step_results {
        // Usa remember_array che converte solo alla fine (meno allocazioni)
        brain.remember_array(
            result.state,
            result.action_idx,
            result.reward,
            result.next_state,
            result.done,
        );
    }

    if all_dead {
        for snake in game.snakes.iter_mut() {
            snake.reset(&grid);
        }
        game.total_iterations += 1;
        brain.iterations = game.total_iterations;

        brain.train();
        brain.epsilon = (brain.epsilon * 0.995).max(0.01);

        // Aggiorna statistiche
        game_stats.total_generations = game.total_iterations;
        game_stats.high_score = game_stats.high_score.max(game.high_score);
        game_stats.total_games_played += PARALLEL_SNAKES as u64;
        let food_eaten: u32 = game.snakes.iter().map(|s| s.score).sum();
        game_stats.total_food_eaten += food_eaten as u64;

        // Aggiorna best score per snake
        for (i, snake) in game.snakes.iter().enumerate() {
            if i < game_stats.best_score_per_snake.len() {
                game_stats.best_score_per_snake[i] =
                    game_stats.best_score_per_snake[i].max(snake.score);
            }
        }

        // Salva ogni 100 generazioni
        if game.total_iterations % 100 == 0 {
            if let Err(e) = brain.save("snake_brain.json") {
                eprintln!("Errore salvataggio modello: {}", e);
            }

            // Aggiorna tempo prima di salvare
            game_stats.total_training_time_secs = stats.total_training_time.as_secs();

            if let Err(e) = game_stats.save("snake_stats.json") {
                eprintln!("Errore salvataggio statistiche: {}", e);
            }

            println!("💾 Salvataggio automatico (Gen {})", game.total_iterations);
        }

        let total_score: u32 = game.snakes.iter().map(|s| s.score).sum();
        println!(
            "Gen: {}, Active: {}/{}, Total Score: {}, High: {}, Eps: {:.3}, Loss: {:.5}, Mem: {}/{}",
            game.total_iterations, active_count, PARALLEL_SNAKES, total_score, 
            game.high_score, brain.epsilon, brain.loss, brain.memory.len(), MEMORY_SIZE
        );
    }
}

fn render_system(
    mut commands: Commands,
    game: Res<GameState>,
    windows: Query<&Window>,
    grid: Res<GridDimensions>,
    mesh_cache: Res<MeshCache>,
    q_segments: Query<Entity, With<SnakeSegment>>,
    q_food: Query<Entity, With<Food>>,
) {
    // Ottimizzazione: despawn solo se necessario (non ogni frame)
    // In realtà per Bevy è meglio despawn/ricreate per oggetti dinamici
    // Ma usiamo le mesh cached per evitare allocazioni GPU
    for e in q_segments.iter() {
        commands.entity(e).despawn();
    }
    for e in q_food.iter() {
        commands.entity(e).despawn();
    }

    let window = windows.single();

    let ui_padding = 60.0;
    let offset_x = -window.resolution.width() / 2.0 + BLOCK_SIZE / 2.0;
    let offset_y = window.resolution.height() / 2.0 - ui_padding - BLOCK_SIZE / 2.0;

    // Usa mesh cache (zero allocazioni GPU per frame!)
    for (snake_idx, snake) in game.snakes.iter().enumerate() {
        // Prendi il materiale corretto per questo snake
        let body_material =
            mesh_cache.snake_materials[snake_idx % mesh_cache.snake_materials.len()].clone();

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
    game: Res<GameState>,
    stats: Res<TrainingStats>,
    mut window_settings: ResMut<WindowSettings>,
    mut windows: Query<&mut Window>,
) {
    if keyboard_input.just_pressed(KeyCode::Escape) {
        // Aggiorna statistiche finali
        game_stats.update(&game, stats.total_training_time);

        // Salva modello
        if let Err(e) = brain.save("snake_brain.json") {
            eprintln!("Errore salvataggio modello: {}", e);
        }

        // Salva statistiche
        if let Err(e) = game_stats.save("snake_stats.json") {
            eprintln!("Errore salvataggio statistiche: {}", e);
        }

        // Stampa riepilogo
        println!("\n=== RIEPILOGO SESSIONE ===");
        println!("Generazioni totali: {}", game_stats.total_generations);
        println!("High Score: {}", game_stats.high_score);
        println!(
            "Tempo di training: {}s",
            game_stats.total_training_time_secs
        );
        println!("========================\n");

        app_exit_events.send(AppExit);
    }

    if keyboard_input.just_pressed(KeyCode::KeyF) {
        window_settings.is_fullscreen = !window_settings.is_fullscreen;
        let mut window = windows.single_mut();
        window.mode = if window_settings.is_fullscreen {
            WindowMode::Fullscreen
        } else {
            WindowMode::Windowed
        };
    }
}

fn on_window_resize(
    mut resize_events: EventReader<bevy::window::WindowResized>,
    mut grid: ResMut<GridDimensions>,
    mut game: ResMut<GameState>,
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

        println!("Griglia ridimensionata: {}x{}", grid.width, grid.height);
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
        .add_systems(
            Update,
            (handle_input, on_window_resize, game_loop, render_system).chain(),
        )
        .add_systems(Update, update_stats_ui.after(render_system))
        .run();
}
