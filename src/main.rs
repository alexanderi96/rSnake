#![recursion_limit = "256"]

mod agent;
mod model;
mod snake;
mod types;
mod ui;

#[cfg(feature = "profiling")]
mod profiling;

use bevy::app::AppExit;
use bevy::prelude::*;
use crossbeam_channel::{bounded, Receiver, Sender};
use std::thread;

use agent::{AgentConfig, DqnAgent};
use snake::{
    get_current_17_state, get_state_egocentric, get_train_interval, spawn_food, CollisionSettings,
    GameConfig, GameState, GameStats, GenerationRecord, GlobalTrainingHistory, GridDimensions,
    GridMap, ParallelConfig, Position, RenderConfig, SegmentPool, TrainingSession, TrainingStats,
    BASE_STATE_SIZE, BATCH_SIZE, BLOCK_SIZE, STATE_SIZE,
};
use types::{GameSnapshot, SnakeSnapshot};
use ui::{ControlSender, ControlSignal, GraphPanelState, UiPlugin, WindowSettings};

/// Resource wrapper for the crossbeam receiver
#[derive(Resource)]
pub struct RenderReceiver(pub Receiver<GameSnapshot>);

fn main() {
    // Initialize profiling if feature is enabled
    #[cfg(feature = "profiling")]
    let _profiling_guard = profiling::ProfilingGuard::new();

    // Create crossbeam channel for communication between RL thread and Bevy
    let (tx, rx) = bounded::<GameSnapshot>(2); // Buffer piccolo per non accumulare lag visivo

    // Create control channel for sending commands to RL thread
    let (control_tx, control_rx) = bounded::<ControlSignal>(10);

    // Avvia il thread RL separato
    thread::spawn(move || {
        run_rl_thread(tx, control_rx);
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
fn run_rl_thread(tx: Sender<GameSnapshot>, control_rx: Receiver<ControlSignal>) {
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

    // --- FASE DI WARM-UP ---
    println!("🔥 Inizio fase di Warm-up del Replay Buffer...");
    let warmup_target = BATCH_SIZE * 50;

    // Salviamo gli epsilon calcolati e forziamo esplorazione pura (1.0)
    let backup_epsilons: Vec<f32> = game_state.snakes.iter().map(|s| s.epsilon).collect();
    for snake in game_state.snakes.iter_mut() {
        snake.epsilon = 1.0;
    }

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
        );
    }

    // Ripristina la distribuzione di esplorazione originale
    for (i, snake) in game_state.snakes.iter_mut().enumerate() {
        snake.epsilon = backup_epsilons[i];
    }
    println!("✅ Warm-up completo! Inizio addestramento.");

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
                    for snake in game_state.snakes.iter_mut() {
                        snake.reset(&grid);
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
) {
    use rand::Rng;
    use rayon::prelude::*;
    use std::time::{Duration, Instant};

    let mut all_dead = true;

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
            agent,
            game_stats,
            training_session,
            global_history,
            session_file,
            parallel_config,
            grid,
            generation_start,
            generation_steps,
        );
        return;
    }

    // 1. COMPUTE PHASE - Calcolo parallelo puro (sola lettura)
    // Restituisce: (snake_idx, stato_completo_34, nuovi_17_sensori, epsilon)
    let computed_data: Vec<(usize, [f32; STATE_SIZE], [f32; BASE_STATE_SIZE], f32)> = active_snakes
        .par_iter()
        .map(|&snake_idx| {
            let snake = &game.snakes[snake_idx];

            // Calcola solo i 17 sensori attuali in sola lettura (pure function)
            let current_17 = get_current_17_state(snake, grid_map, grid);

            // Costruisci l'array finale da 34 concatenando attuale e precedente
            let mut state_34 = [0.0f32; STATE_SIZE];
            state_34[..BASE_STATE_SIZE].copy_from_slice(&current_17);
            state_34[BASE_STATE_SIZE..STATE_SIZE].copy_from_slice(&snake.previous_state);

            (snake_idx, state_34, current_17, snake.epsilon)
        })
        .collect();

    // 2. APPLY PHASE - Aggiornamento sequenziale veloce
    let mut states: Vec<(usize, [f32; STATE_SIZE], f32)> = Vec::with_capacity(computed_data.len());
    for (snake_idx, state_34, current_17, epsilon) in computed_data {
        // Aggiorna la memoria del serpente per il frame successivo
        game.snakes[snake_idx].previous_state = current_17;

        // Prepara i dati per il batch inference
        states.push((snake_idx, state_34, epsilon));
    }

    // 2. Decisione dell'Agente con BATCH INFERENCE (usa epsilon individuali)
    let states_with_eps: Vec<([f32; 34], f32)> = states.iter().map(|(_, s, e)| (*s, *e)).collect();
    let action_indices = agent.select_actions_batch(states_with_eps);

    // Ricostruisci le decisioni
    let decisions: Vec<Decision> = states
        .iter()
        .zip(action_indices.into_iter())
        .map(|((snake_idx, state, _epsilon), action_idx)| Decision {
            snake_idx: *snake_idx,
            action_idx,
            state: *state,
        })
        .collect();

    // Update grid map
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

        // Apply action: 0=Left, 1=Right, 2=Straight
        match action_idx {
            0 => snake_ref.direction = snake_ref.direction.turn_left(),
            1 => snake_ref.direction = snake_ref.direction.turn_right(),
            _ => {} // Straight
        }

        let (dx, dy) = snake_ref.direction.as_vec();
        let new_head = Position {
            x: snake_ref.snake[0].x + dx,
            y: snake_ref.snake[0].y + dy,
        };

        snake_ref.steps_without_food += 1;

        // 1. Penalità costante minima per ogni step (scoraggia i loop e l'inattività)
        let mut reward: f32 = -0.001;
        let mut done = false;

        let collision = if collision_settings.snake_vs_snake {
            grid_map.is_collision(new_head.x, new_head.y, snake_idx)
        } else {
            grid_map.is_collision_no_snakes(new_head.x, new_head.y)
        } || snake_ref.snake.contains(&new_head);

        if collision {
            reward = -10.0; // Death penalty
            done = true;
            snake_ref.is_game_over = true;
        } else if new_head == snake_ref.food {
            // Constant food reward (removed length-based scaling for stable convergence)
            reward = 10.0;

            snake_ref.snake.push_front(new_head);
            snake_ref.score += 1;
            if snake_ref.score > game.high_score {
                game.high_score = snake_ref.score;
            }
            snake_ref.food = spawn_food(snake_ref, grid);
            snake_ref.steps_without_food = 0;
            grid_map.set(new_head.x, new_head.y, (snake_idx + 1) as u8);
        } else {
            let old_tail = snake_ref.snake.back().copied();
            snake_ref.snake.push_front(new_head);
            snake_ref.snake.pop_back();

            // Calcola un limite massimo ragionevole, ad esempio 100 step base + 10 per ogni segmento
            let max_steps = 100 + (snake_ref.snake.len() as u32 * 10);

            if snake_ref.steps_without_food > max_steps {
                reward = -10.0; // Punito severamente per non aver cercato attivamente
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
            next_state: get_state_egocentric(snake_ref, grid_map, grid),
            done,
            ate_food: new_head == game.snakes[snake_idx].food,
        });
    }

    for result in step_results {
        agent.remember((
            result.state,
            result.action_idx,
            result.reward,
            result.next_state,
            result.done,
        ));
    }

    agent.iterations += 1;
    if agent.iterations % get_train_interval() as u32 == 0 {
        agent.train();
    }

    if all_dead {
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
        );
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

    let record = GenerationRecord {
        gen: game.total_iterations,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        avg_score,
        max_score,
        min_score,
        avg_loss: agent.loss,
        epsilon: avg_epsilon,
    };

    global_history.records.push(record.clone());
    training_session.add_record(record);

    if let Err(e) = training_session.save(session_file.to_str().unwrap_or("session.json")) {
        eprintln!("Error saving session: {}", e);
    }

    for snake in game.snakes.iter_mut() {
        snake.reset(grid);
    }
    game.total_iterations += 1;
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
        agent.loss,
        generation_duration.as_secs_f32(),
        steps_per_second
    );

    // Reset timer e contatore per la prossima generazione
    *generation_start = std::time::Instant::now();
    *generation_steps = 0;
}

pub struct SnakePlugin;

impl Plugin for SnakePlugin {
    fn build(&self, _app: &mut App) {
        // Snake plugin systems are configured in main.rs
    }
}
