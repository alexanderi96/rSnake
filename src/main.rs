#![recursion_limit = "256"]

mod agent;
mod buffer;
mod config;
mod model;
mod snake;
mod types;
mod ui;

#[cfg(feature = "profiling")]
mod profiling;

use bevy::app::AppExit;
use bevy::prelude::*;
use clap::Parser;
use crossbeam_channel::{bounded, Receiver, Sender};
use std::thread; // Questo abilita .choose() sui vettori

use agent::{AgentConfig, DqnAgent, Experience};
use buffer::Transition;
use config::Hyperparameters;
use snake::{
    get_current_17_state, spawn_food, CollisionSettings, GameConfig, GameState, GameStats,
    GenerationRecord, GlobalTrainingHistory, GridDimensions, GridMap, ParallelConfig, Position,
    RenderConfig, SegmentPool, TrainingSession, TrainingStats, BASE_STATE_SIZE, BLOCK_SIZE,
    STATE_SIZE,
};
use types::{GameSnapshot, SnakeSnapshot};
use ui::{ControlSender, ControlSignal, GraphPanelState, UiPlugin, WindowSettings};

/// CLI Arguments per Snake DQN
#[derive(Parser, Debug, Clone)]
#[command(name = "snake-dqn")]
#[command(about = "DQN Snake RL Training with Bevy + Burn")]
pub struct CliArgs {
    /// Path al file di configurazione (TOML o JSON)
    #[arg(short, long)]
    pub config: Option<String>,

    /// Learning rate
    #[arg(long)]
    pub learning_rate: Option<f64>,

    /// Gamma (discount factor)
    #[arg(long)]
    pub gamma: Option<f32>,

    /// Batch size
    #[arg(long)]
    pub batch_size: Option<usize>,

    /// Memory size (replay buffer capacity)
    #[arg(long)]
    pub memory_size: Option<usize>,

    /// Target network update frequency
    #[arg(long)]
    pub target_update_freq: Option<usize>,

    /// Training interval (steps)
    #[arg(long)]
    pub train_interval: Option<usize>,

    /// Reward for eating food
    #[arg(long)]
    pub reward_food: Option<f32>,

    /// Reward for dying (negative)
    #[arg(long)]
    pub reward_death: Option<f32>,

    /// Reward per step (negative for penalty)
    #[arg(long)]
    pub reward_step: Option<f32>,

    /// Base steps without food before timeout
    #[arg(long)]
    pub base_steps_without_food: Option<u32>,

    /// Additional steps per snake segment
    #[arg(long)]
    pub steps_per_segment: Option<u32>,
}

/// Costruisce la configurazione finale combinando file config e CLI args
fn build_hyperparameters(args: &CliArgs) -> Hyperparameters {
    let mut config = if let Some(ref path) = args.config {
        match Hyperparameters::from_file(path) {
            Ok(cfg) => {
                println!("✅ Config loaded from: {}", path);
                cfg
            }
            Err(e) => {
                eprintln!("⚠️  Failed to load config ({}), using defaults", e);
                Hyperparameters::default()
            }
        }
    } else {
        Hyperparameters::default()
    };

    // CLI args override file config
    if let Some(v) = args.learning_rate {
        config.learning_rate = v;
    }
    if let Some(v) = args.gamma {
        config.gamma = v;
    }
    if let Some(v) = args.batch_size {
        config.batch_size = v;
    }
    if let Some(v) = args.memory_size {
        config.memory_size = v;
    }
    if let Some(v) = args.target_update_freq {
        config.target_update_freq = v;
    }
    if let Some(v) = args.train_interval {
        config.train_interval = v;
    }
    if let Some(v) = args.reward_food {
        config.reward_food = v;
    }
    if let Some(v) = args.reward_death {
        config.reward_death = v;
    }
    if let Some(v) = args.reward_step {
        config.reward_step = v;
    }
    if let Some(v) = args.base_steps_without_food {
        config.base_steps_without_food = v;
    }
    if let Some(v) = args.steps_per_segment {
        config.steps_per_segment = v;
    }

    config
}

/// Resource wrapper for the crossbeam receiver
#[derive(Resource)]
pub struct RenderReceiver(pub Receiver<GameSnapshot>);

fn main() {
    // Parse CLI arguments
    let args = CliArgs::parse();
    let hyperparams = build_hyperparameters(&args);

    // Print configuration
    println!("🚀 Snake DQN Training Configuration:");
    println!("  Learning Rate: {:.1e}", hyperparams.learning_rate);
    println!("  Gamma: {:.2}", hyperparams.gamma);
    println!("  Batch Size: {}", hyperparams.batch_size);
    println!("  Memory Size: {}", hyperparams.memory_size);
    println!("  Target Update Freq: {}", hyperparams.target_update_freq);
    println!("  Train Interval: {}", hyperparams.train_interval);
    println!("  Reward Food: {:.2}", hyperparams.reward_food);
    println!("  Reward Death: {:.2}", hyperparams.reward_death);
    println!("  Reward Step: {:.3}", hyperparams.reward_step);
    println!(
        "  Base Steps Without Food: {}",
        hyperparams.base_steps_without_food
    );
    println!("  Steps Per Segment: {}", hyperparams.steps_per_segment);

    // Initialize profiling if feature is enabled
    #[cfg(feature = "profiling")]
    let _profiling_guard = profiling::ProfilingGuard::new();

    // Create crossbeam channel for communication between RL thread and Bevy
    let (tx, rx) = bounded::<GameSnapshot>(2); // Buffer piccolo per non accumulare lag visivo

    // Create control channel for sending commands to RL thread
    let (control_tx, control_rx) = bounded::<ControlSignal>(10);

    // Avvia il thread RL separato
    thread::spawn(move || {
        run_rl_thread(tx, control_rx, hyperparams);
    });

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Rust Bevy + Burn DQN Snake - GPU Accelerated".into(),
                resolution: (800.0, 600.0).into(),
                resizable: true,
                ..default()
            }),
            ..default()
        }))
        .add_event::<AppExit>()
        // Inserisci i receiver come resources prima di tutto
        .insert_resource(RenderReceiver(rx))
        .insert_resource(ControlSender(control_tx))
        // Register plugins first (they add systems but don't run yet)
        .add_plugins(SnakePlugin)
        .add_plugins(UiPlugin)
        // setup must run before any system that uses resources
        .add_systems(Startup, setup)
        // spawn_stats_ui needs resources, so it runs after setup
        .add_systems(Startup, ui::spawn_stats_ui.after(setup))
        .run();
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
        snake::calculate_grid_dimensions(window.resolution.width(), window.resolution.height());

    // Create mesh cache (once)
    let segment_mesh = meshes.add(Rectangle::new(BLOCK_SIZE - 2.0, BLOCK_SIZE - 2.0));
    let food_mesh = meshes.add(Circle::new(BLOCK_SIZE / 2.0));
    let food_material = materials.add(Color::rgb(1.0, 0.0, 0.0));
    let head_material = materials.add(Color::rgb(1.0, 1.0, 1.0));

    commands.insert_resource(snake::MeshCache {
        segment_mesh,
        food_mesh,
        food_material,
        head_material,
    });

    // Initialize configurations
    commands.insert_resource(CollisionSettings::default());
    commands.insert_resource(GraphPanelState::default());
    commands.insert_resource(RenderConfig::default());
    commands.insert_resource(GridMap::new(grid_width, grid_height));

    let parallel_config = ParallelConfig::new();
    commands.insert_resource(SegmentPool::new(parallel_config.snake_count));
    commands.insert_resource(snake::AppStartTime::default());

    // Load training history
    let (global_history, last_gen) = snake::load_global_history();
    let accumulated_time = global_history.accumulated_time_secs;

    // Setup session
    let session_file = snake::new_session_path();
    let training_session = TrainingSession::new();
    println!("📊 New session file: {}", session_file.display());

    // Create grid
    let grid = GridDimensions {
        width: grid_width,
        height: grid_height,
    };

    // Insert all resources
    commands.insert_resource(global_history);
    commands.insert_resource(training_session);
    commands.insert_resource(GameConfig {
        speed_timer: Timer::from_seconds(0.001, TimerMode::Repeating),
        session_path: session_file,
    });
    commands.insert_resource(WindowSettings {
        is_fullscreen: false,
    });
    // GameState is now managed in the RL thread, but we keep a local copy for rendering
    commands.insert_resource(GameState::new(&grid, parallel_config.snake_count));
    commands.insert_resource(grid);
    commands.insert_resource(TrainingStats {
        total_training_time: std::time::Duration::from_secs(accumulated_time),
        last_update: std::time::Instant::now(),
        parallel_threads: rayon::current_num_threads(),
        fps: 0.0,
        last_fps_update: std::time::Instant::now(),
        frame_count: 0,
    });
    commands.insert_resource(parallel_config.clone());
    commands.insert_resource(GameStats::new(parallel_config.snake_count));
}

/// Thread RL principale - Producer nel pattern Producer-Consumer
fn run_rl_thread(
    tx: Sender<GameSnapshot>,
    control_rx: Receiver<ControlSignal>,
    hyperparams: Hyperparameters,
) {
    println!("🧠 RL Thread started - Running simulation at maximum speed");

    // --- AGENT INITIALIZATION ---
    let parallel_config = ParallelConfig::new();
    let agent_config = AgentConfig::new(parallel_config.snake_count);
    let brain_path_buf = snake::brain_path();
    let model_path = brain_path_buf.to_str().unwrap_or("brain.bin");

    let mut agent = if std::path::Path::new(model_path).exists() {
        println!(
            "🔄 Found existing model at {}, attempting to load...",
            model_path
        );
        match DqnAgent::load(model_path) {
            Ok(loaded_agent) => {
                println!("✅ Model loaded successfully!");
                loaded_agent
            }
            Err(e) => {
                eprintln!("⚠️ Error loading model (starting fresh): {}", e);
                DqnAgent::new(agent_config)
            }
        }
    } else {
        println!("🆕 No existing model found. Creating new agent.");
        DqnAgent::new(agent_config)
    };

    // Create grid and game state
    // Usa dimensioni di default per il thread RL
    let mut grid = GridDimensions {
        width: 40,
        height: 30,
    };
    let mut game_state = GameState::new(&grid, parallel_config.snake_count);

    // Load training history for continuity
    let (mut global_history, last_gen) = snake::load_global_history();
    game_state.total_iterations = last_gen;

    let mut training_session = TrainingSession::new();
    let session_file = snake::new_session_path();

    let mut game_stats = GameStats::new(parallel_config.snake_count);
    let collision_settings = CollisionSettings::default();
    let mut grid_map = GridMap::new(grid.width, grid.height);

    // Timer per tracking performance
    let mut generation_start = std::time::Instant::now();
    let mut generation_steps: u64 = 0;

    // --- FASE DI WARM-UP SILENZIOSO ---
    println!("🔥 Inizio fase di Warm-up silenzioso (riempimento buffer offscreen)...");
    let warmup_target = hyperparams.batch_size * 50;

    while agent.replay_buffer.len() < warmup_target {
        run_simulation_step(
            &mut game_state,
            &mut agent,
            &mut game_stats,
            &mut training_session,
            &mut global_history,
            &session_file,
            &grid,
            &collision_settings,
            &parallel_config,
            &mut grid_map,
            &mut generation_start,
            &mut generation_steps,
            &hyperparams,
            true, // is_warmup
        );
    }
    println!("✅ Warm-up completo! Inizio addestramento vero e proprio.");

    // Inizializza gli epsilon per la prima generazione
    game_state.update_epsilons(
        game_state.total_iterations,
        hyperparams.epsilon_decay_rate,
        hyperparams.epsilon_min,
        hyperparams.epsilon_max,
    );

    // Loop principale del training
    loop {
        // Controlla se ci sono segnali di controllo
        while let Ok(signal) = control_rx.try_recv() {
            match signal {
                ControlSignal::SaveBrain => {
                    let brain_path = snake::brain_path();
                    println!("💾 Saving brain to: {}", brain_path.display());
                    if let Err(e) = agent.save(brain_path.to_str().unwrap_or("brain.bin")) {
                        eprintln!("⚠️ Error saving brain: {}", e);
                    } else {
                        println!("✅ Brain saved successfully!");
                    }
                }
                ControlSignal::GridResized(new_width, new_height) => {
                    println!("📐 Grid resized to {}x{}", new_width, new_height);
                    grid.width = new_width;
                    grid.height = new_height;
                    grid_map = GridMap::new(new_width, new_height);
                    // Reset snakes to fit new grid
                    let total_snakes = game_state.snakes.len();
                    for snake in game_state.snakes.iter_mut() {
                        snake.reset(&grid, total_snakes);
                    }
                }
                ControlSignal::Exit => {
                    println!("🛑 Exit signal received, saving brain...");
                    let brain_path = snake::brain_path();
                    if let Err(e) = agent.save(brain_path.to_str().unwrap_or("brain.bin")) {
                        eprintln!("⚠️ Error saving brain on exit: {}", e);
                    } else {
                        println!("✅ Brain saved on exit!");
                    }
                    return;
                }
            }
        }

        // Esegui step di simulazione
        run_simulation_step(
            &mut game_state,
            &mut agent,
            &mut game_stats,
            &mut training_session,
            &mut global_history,
            &session_file,
            &grid,
            &collision_settings,
            &parallel_config,
            &mut grid_map,
            &mut generation_start,
            &mut generation_steps,
            &hyperparams,
            false, // is_warmup
        );

        // Incrementa contatore step
        generation_steps += 1;

        // Costruisci snapshot per il rendering
        let snapshot = build_snapshot(
            &game_state,
            &grid,
            &global_history,
            training_session.records.len(),
        );

        // Invia a Bevy (se il canale è pieno, sovrascrivi o ignora per non bloccare il training)
        let _ = tx.try_send(snapshot);

        // Se vogliamo vedere a velocità normale, metti in sleep
        // Altrimenti gira al massimo della CPU/GPU
        // thread::sleep(Duration::from_millis(16)); // ~60 FPS se abilitato
    }
}

/// Costruisce uno snapshot leggibile dal renderer
fn build_snapshot(
    game_state: &GameState,
    grid: &GridDimensions,
    global_history: &GlobalTrainingHistory,
    session_records_count: usize,
) -> GameSnapshot {
    let snakes: Vec<SnakeSnapshot> = game_state
        .snakes
        .iter()
        .map(|s| SnakeSnapshot {
            id: s.id,
            body: s.snake.iter().map(|p| (p.x, p.y)).collect(),
            food: (s.food.x, s.food.y),
            color: s.color,
            is_game_over: s.is_game_over,
            score: s.score,
        })
        .collect();

    GameSnapshot {
        snakes,
        grid_width: grid.width,
        grid_height: grid.height,
        high_score: game_state.high_score,
        generation: game_state.total_iterations,
        history_records: global_history.records.clone(),
        session_records_count,
    }
}

struct StepResult {
    snake_idx: usize,
    state: [f32; 34],
    action_idx: usize,
    reward: f32,
    next_state: [f32; 34],
    done: bool,
    ate_food: bool,
}

struct Decision {
    snake_idx: usize,
    action_idx: usize,
    state: [f32; 34],
}

/// Calcola la reward basata sull'esito dello step e sul cambiamento di distanza dal cibo.
pub fn calculate_reward(
    is_collision: bool,
    ate_food: bool,
    is_timeout: bool,
    old_pos: Position,
    new_pos: Position,
    food_pos: Position,
    hyperparams: &Hyperparameters,
) -> f32 {
    // 1. Caso Morte (Collisione o Inedia)
    if is_collision || is_timeout {
        return hyperparams.reward_death; // Es: -15.0
    }

    // 2. Caso Cibo
    if ate_food {
        return hyperparams.reward_food; // Es: 10.0
    }

    // 3. Reward Shaping: Incoraggiamo il movimento verso il cibo
    // Usiamo la distanza Manhattan: |x1-x2| + |y1-y2|
    let dist_old = (old_pos.x - food_pos.x).abs() + (old_pos.y - food_pos.y).abs();
    let dist_new = (new_pos.x - food_pos.x).abs() + (new_pos.y - food_pos.y).abs();

    let shaping = if dist_new < dist_old {
        0.15 // Premio per essersi avvicinato
    } else {
        -0.20 // Penalità per essersi allontanato (leggermente più alta per evitare indecisioni)
    };

    // 4. Reward totale: Penalità costante step + shaping
    hyperparams.reward_step + shaping
}

#[allow(clippy::too_many_arguments)]
fn run_simulation_step(
    game: &mut GameState,
    agent: &mut DqnAgent,
    game_stats: &mut GameStats,
    training_session: &mut TrainingSession,
    global_history: &mut GlobalTrainingHistory,
    session_file: &std::path::PathBuf,
    grid: &GridDimensions,
    collision_settings: &CollisionSettings,
    parallel_config: &ParallelConfig,
    grid_map: &mut GridMap,
    generation_start: &mut std::time::Instant,
    generation_steps: &mut u64,
    hyperparams: &Hyperparameters,
    is_warmup: bool,
) {
    use rayon::prelude::*;

    // Identifica serpenti vivi
    let active_snakes: Vec<usize> = game
        .snakes
        .iter()
        .enumerate()
        .filter(|(_, s)| !s.is_game_over)
        .map(|(idx, _)| idx)
        .collect();

    if active_snakes.is_empty() {
        if is_warmup {
            // Warmup: solo reset serpenti, niente logging/training
            let total_snakes = game.snakes.len();
            for snake in game.snakes.iter_mut() {
                snake.reset(grid, total_snakes);
            }
            return;
        } else {
            // Training normale: handle_generation_end completo
            handle_generation_end(
                game,
                agent,
                game_stats,
                training_session,
                global_history,
                session_file,
                parallel_config,
                grid,
                generation_start,
                generation_steps,
                hyperparams,
            );
            return;
        }
    }

    // --- FASE 1: COMPUTE PARALLELO (Sola lettura) ---
    // Calcoliamo lo stato a 34 dimensioni (17 attuali + 17 precedenti salvati nel serpente)
    let computed_data: Vec<(usize, [f32; STATE_SIZE], [f32; BASE_STATE_SIZE], f32)> = active_snakes
        .par_iter()
        .map(|&idx| {
            let snake = &game.snakes[idx];
            let current_17 = get_current_17_state(snake, grid_map, grid);

            let mut state_34 = [0.0f32; STATE_SIZE];
            state_34[..BASE_STATE_SIZE].copy_from_slice(&current_17);
            state_34[BASE_STATE_SIZE..].copy_from_slice(&snake.previous_state);

            (idx, state_34, current_17, snake.epsilon)
        })
        .collect();

    // --- FASE 2: BATCH INFERENCE ---
    let states_for_inference: Vec<([f32; 34], f32)> = computed_data
        .iter()
        .map(|(_, s34, _, eps)| (*s34, *eps))
        .collect();

    let action_indices = agent.select_actions_batch(states_for_inference);

    // --- FASE 3: AGGIORNAMENTO GRID & FISICA ---
    grid_map.clear();
    for (idx, snake) in game.snakes.iter().enumerate() {
        if !snake.is_game_over {
            for pos in snake.snake.iter() {
                grid_map.set(pos.x, pos.y, (idx + 1) as u8);
            }
        }
    }

    let mut step_results = Vec::with_capacity(computed_data.len());

    for (i, (snake_idx, state_34, current_17, _)) in computed_data.into_iter().enumerate() {
        let action_idx = action_indices[i];
        let snake_ref = &mut game.snakes[snake_idx];
        let old_head = snake_ref.snake[0];

        // Aggiorna la memoria temporale del serpente per il prossimo frame
        snake_ref.previous_state = current_17;

        // Applica azione
        match action_idx {
            0 => snake_ref.direction = snake_ref.direction.turn_left(),
            1 => snake_ref.direction = snake_ref.direction.turn_right(),
            _ => {} // Dritto
        }

        let (dx, dy) = snake_ref.direction.as_vec();
        let new_head = Position {
            x: old_head.x + dx,
            y: old_head.y + dy,
        };
        snake_ref.steps_without_food += 1;

        // Check collisioni e timeout
        let is_collision = if collision_settings.snake_vs_snake {
            grid_map.is_collision(new_head.x, new_head.y, snake_idx)
        } else {
            grid_map.is_collision_no_snakes(new_head.x, new_head.y)
        } || snake_ref.snake.contains(&new_head);

        let ate_food = new_head == snake_ref.food;
        let is_timeout =
            snake_ref.steps_without_food > hyperparams.calculate_timeout(snake_ref.snake.len());

        // --- CALCOLO REWARD ---
        let reward = calculate_reward(
            is_collision,
            ate_food,
            is_timeout,
            old_head,
            new_head,
            snake_ref.food,
            hyperparams,
        );

        // --- ESECUZIONE MOVIMENTO ---
        let mut done = false;
        if is_collision || is_timeout {
            done = true;
            snake_ref.is_game_over = true;
        } else {
            snake_ref.snake.push_front(new_head);
            if ate_food {
                snake_ref.score += 1;
                if snake_ref.score > game.high_score {
                    game.high_score = snake_ref.score;
                }
                snake_ref.food = spawn_food(snake_ref, grid);
                snake_ref.steps_without_food = 0;
            } else {
                snake_ref.snake.pop_back();
            }
        }

        // Calcola il next_state (34-dim) per il Replay Buffer
        // Nota: snake_ref.previous_state è già stato aggiornato a current_17 sopra
        let next_17 = get_current_17_state(snake_ref, grid_map, grid);
        let mut next_state_34 = [0.0f32; STATE_SIZE];
        next_state_34[..BASE_STATE_SIZE].copy_from_slice(&next_17);
        next_state_34[BASE_STATE_SIZE..].copy_from_slice(&snake_ref.previous_state);

        step_results.push(StepResult {
            snake_idx,
            state: state_34,
            action_idx,
            reward,
            next_state: next_state_34,
            done,
            ate_food,
        });
    }

    // Memorizza esperienze e allena
    for res in step_results {
        agent.remember(Transition::new(
            res.state,
            res.action_idx,
            res.reward,
            res.next_state,
            res.done,
        ));
    }

    // Training solo se non in warmup
    if !is_warmup {
        agent.iterations += 1;
        if agent.iterations % hyperparams.train_interval as u32 == 0 {
            agent.train();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_generation_end(
    game: &mut GameState,
    agent: &mut DqnAgent,
    game_stats: &mut GameStats,
    training_session: &mut TrainingSession,
    global_history: &mut GlobalTrainingHistory,
    session_file: &std::path::PathBuf,
    parallel_config: &ParallelConfig,
    grid: &GridDimensions,
    generation_start: &mut std::time::Instant,
    generation_steps: &mut u64,
    hyperparams: &Hyperparameters,
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

    // Calcola la media degli epsilon dei serpenti (Ape-X distributed exploration)
    let avg_epsilon = game.snakes.iter().map(|s| s.epsilon).sum::<f32>() / game.snakes.len() as f32;

    // === NUOVE METRICHE DIAGNOSTICHE ===

    // 1. Average Q-Value dalla generazione
    let avg_q_value = agent.get_generation_q_stats();

    // 2. Min/Max/Avg Loss dalla generazione
    let (min_loss, max_loss, avg_loss) = agent.get_generation_loss_stats();

    // 3. Average Episode Length (steps sopravvissuti)
    let avg_episode_length = *generation_steps as f32 / parallel_config.snake_count as f32;

    // 4. Buffer Reward Distribution
    let (pos_ratio, neg_ratio, neut_ratio) = agent.replay_buffer.analyze_reward_distribution();

    let record = GenerationRecord {
        gen: game.total_iterations,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        avg_score,
        max_score,
        min_score,
        avg_loss,
        epsilon: avg_epsilon,
        // Nuove metriche
        avg_q_value,
        min_loss,
        max_loss,
        avg_episode_length,
        buffer_positive_ratio: pos_ratio,
        buffer_negative_ratio: neg_ratio,
        buffer_neutral_ratio: neut_ratio,
    };

    global_history.records.push(record.clone());
    training_session.add_record(record);

    if let Err(e) = training_session.save(session_file.to_str().unwrap_or("session.json")) {
        eprintln!("Error saving session: {}", e);
    }

    let total_snakes = game.snakes.len();
    for snake in game.snakes.iter_mut() {
        snake.reset(grid, total_snakes);
    }
    game.total_iterations += 1;
    // Aggiorna gli epsilon dinamicamente basandosi sulla generazione corrente
    game.update_epsilons(
        game.total_iterations,
        hyperparams.epsilon_decay_rate,
        hyperparams.epsilon_min,
        hyperparams.epsilon_max,
    );
    agent.iterations = game.total_iterations;

    agent.train();
    // Nota: decay_epsilon rimosso - usiamo epsilon statici individuali (Ape-X style)

    game_stats.total_generations = game.total_iterations;
    game_stats.high_score = game_stats.high_score.max(game.high_score);
    game_stats.total_games_played += parallel_config.snake_count as u64;

    // Auto-save brain every N generations
    if game.total_iterations % snake::AUTO_SAVE_INTERVAL == 0 {
        let brain_path = snake::brain_path();
        println!(
            "💾 Auto-saving brain (gen {}) to: {}",
            game.total_iterations,
            brain_path.display()
        );
        if let Err(e) = agent.save(brain_path.to_str().unwrap_or("brain.bin")) {
            eprintln!("⚠️ Error auto-saving brain: {}", e);
        } else {
            println!("✅ Brain auto-saved successfully!");
        }
    }

    let active_count = game.snakes.iter().filter(|s| !s.is_game_over).count();

    // Calcola durata e steps/sec della generazione
    let generation_duration = generation_start.elapsed();
    let steps_per_second = if generation_duration.as_secs_f32() > 0.0 {
        *generation_steps as f32 / generation_duration.as_secs_f32()
    } else {
        0.0
    };

    println!(
        "Gen: {}, Active: {}/{}, Score: {:.1}, High: {}, AvgEps: {:.3}, Loss: {:.5}, Time: {:.2}s, Steps/sec: {:.0}",
        game.total_iterations,
        active_count,
        parallel_config.snake_count,
        avg_score,
        max_score,
        avg_epsilon,
        avg_loss,
        generation_duration.as_secs_f32(),
        steps_per_second
    );

    // Log metriche avanzate
    println!(
        "     Q-Value: {:.3}, Loss Range: {:.5}-{:.5}, Episode Len: {:.1}, Buffer: +{:.1}% -{:.1}% ~{:.1}%",
        avg_q_value,
        min_loss,
        max_loss,
        avg_episode_length,
        pos_ratio * 100.0,
        neg_ratio * 100.0,
        neut_ratio * 100.0
    );

    // Reset timer e contatore per la prossima generazione
    *generation_start = std::time::Instant::now();
    *generation_steps = 0;

    // Reset metriche del agent per la prossima generazione
    agent.reset_generation_metrics();
}

pub struct SnakePlugin;

impl Plugin for SnakePlugin {
    fn build(&self, _app: &mut App) {
        // Snake plugin systems are configured in main.rs
    }
}
