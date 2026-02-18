#![recursion_limit = "256"]

mod agent;
mod model;
mod snake;
mod ui;

#[cfg(feature = "profiling")]
mod profiling;

use bevy::app::AppExit;
use bevy::prelude::*;
// use bevy::window::WindowMode; // Non strettamente necessario se non lo usi qui, ma lo lascio

use agent::{AgentConfig, DqnAgent};
use snake::{
    AppStartTime, CollisionSettings, GameConfig, GameState, GlobalTrainingHistory, GridDimensions,
    GridMap, ParallelConfig, RenderConfig, SegmentPool, SnakePlugin, TrainingSession,
    TrainingStats,
};
use ui::{GraphPanelState, UiPlugin, WindowSettings};

fn main() {
    // Initialize profiling if feature is enabled
    #[cfg(feature = "profiling")]
    let _profiling_guard = profiling::ProfilingGuard::new();

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
        // Register plugins first (they add systems but don't run yet)
        .add_plugins(SnakePlugin)
        .add_plugins(UiPlugin)
        // setup must run before any system that uses GameState
        .add_systems(Startup, setup)
        // spawn_stats_ui needs GameState, so it runs after setup
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
    let segment_mesh = meshes.add(Rectangle::new(
        snake::BLOCK_SIZE - 2.0,
        snake::BLOCK_SIZE - 2.0,
    ));
    let food_mesh = meshes.add(Circle::new(snake::BLOCK_SIZE / 2.0));
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
    commands.insert_resource(AppStartTime::default());

    // Load training history
    let (global_history, last_gen) = snake::load_global_history();
    let accumulated_time = global_history.accumulated_time_secs;

    // Setup session
    let session_file = snake::new_session_path();
    let training_session = TrainingSession::new();
    println!("📊 New session file: {}", session_file.display());

    // --- AGENT INITIALIZATION (CORRECTED) ---
    let agent_config = AgentConfig::new(parallel_config.snake_count);
    let model_path = "brain.bin"; // Burn usa formato binario

    let agent = if std::path::Path::new(model_path).exists() {
        println!(
            "🔄 Found existing model at '{}', attempting to load...",
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
    // ----------------------------------------

    // Create grid and game state
    let grid = GridDimensions {
        width: grid_width,
        height: grid_height,
    };
    let mut game_state = GameState::new(&grid, parallel_config.snake_count);
    game_state.total_iterations = last_gen;

    // Insert all resources
    commands.insert_resource(global_history);
    commands.insert_resource(training_session);
    commands.insert_resource(agent);
    commands.insert_resource(GameConfig {
        speed_timer: Timer::from_seconds(0.001, TimerMode::Repeating),
        session_path: session_file,
    });
    commands.insert_resource(WindowSettings {
        is_fullscreen: false,
    });
    commands.insert_resource(grid);
    commands.insert_resource(game_state);
    commands.insert_resource(TrainingStats {
        total_training_time: std::time::Duration::from_secs(accumulated_time),
        last_update: std::time::Instant::now(),
        parallel_threads: rayon::current_num_threads(),
        fps: 0.0,
        last_fps_update: std::time::Instant::now(),
        frame_count: 0,
    });
    commands.insert_resource(parallel_config.clone());
    commands.insert_resource(snake::GameStats::new(parallel_config.snake_count));
}
