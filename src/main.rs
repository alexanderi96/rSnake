//! MAP-Elites Snake - Bevy ECS with Evolutionary Algorithm
//!
//! A simplified implementation to get things compiling first.

#![recursion_limit = "256"]

mod brain;
mod config;
mod evolution;
mod map_elites;
mod snake;
mod ui;

use bevy::app::AppExit;
use bevy::prelude::*;
use clap::Parser;

use brain::Action;
use config::Hyperparameters;
use evolution::EvolutionManager;
use snake::{
    calculate_grid_dimensions, get_current_17_state, AppStartTime, CollisionSettings, GameConfig,
    GameState, GameStats, GenerationSeed, GlobalTrainingHistory, GridDimensions, GridMap,
    MeshCache, ParallelConfig, Position, RenderConfig, SegmentPool, TrainingStats, BLOCK_SIZE,
    STATE_SIZE,
};
use ui::{GraphPanelState, PauseState, UiPlugin, WindowSettings};

/// CLI Arguments
#[derive(Parser, Debug, Clone)]
#[command(name = "snake-map-elites")]
#[command(about = "MAP-Elites Snake RL Training")]
pub struct CliArgs {
    #[arg(short, long)]
    pub config: Option<String>,
    #[arg(long)]
    pub population_size: Option<usize>,
    #[arg(long)]
    pub mutation_rate: Option<f64>,
    #[arg(long)]
    pub mutation_strength: Option<f64>,
    #[arg(long)]
    pub crossover_rate: Option<f64>,
    #[arg(long)]
    pub max_frames: Option<u32>,
    #[arg(long)]
    pub base_steps_without_food: Option<u32>,
    #[arg(long)]
    pub steps_per_segment: Option<u32>,
}

fn build_hyperparameters(args: &CliArgs) -> Hyperparameters {
    let mut config = if let Some(ref path) = args.config {
        match Hyperparameters::from_file(path) {
            Ok(cfg) => {
                println!("Config loaded from: {}", path);
                cfg
            }
            Err(e) => {
                eprintln!("Failed to load config ({}), using defaults", e);
                Hyperparameters::default()
            }
        }
    } else {
        Hyperparameters::default()
    };

    if let Some(v) = args.population_size {
        config.population_size = v;
    }
    if let Some(v) = args.mutation_rate {
        config.mutation_rate = v;
    }
    if let Some(v) = args.mutation_strength {
        config.mutation_strength = v;
    }
    if let Some(v) = args.crossover_rate {
        config.crossover_rate = v;
    }
    if let Some(v) = args.max_frames {
        config.max_frames = v;
    }
    if let Some(v) = args.base_steps_without_food {
        config.base_steps_without_food = v;
    }
    if let Some(v) = args.steps_per_segment {
        config.steps_per_segment = v;
    }

    config
}

#[derive(Resource)]
struct Population(pub Vec<brain::Brain>);

fn main() {
    let args = CliArgs::parse();
    let hyperparams = build_hyperparameters(&args);

    println!("MAP-Elites Snake Configuration:");
    println!("  Population: {}", hyperparams.population_size);
    println!(
        "  Mutation: {:.2}/{:.2}",
        hyperparams.mutation_rate, hyperparams.mutation_strength
    );

    #[cfg(feature = "profiling")]
    let _profiling_guard = profiling::ProfilingGuard::new();

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "MAP-Elites Snake".into(),
                resolution: (800.0, 600.0).into(),
                resizable: true,
                ..default()
            }),
            ..default()
        }))
        .add_event::<AppExit>()
        .insert_resource(hyperparams) // Insert CLI/config hyperparameters as resource
        .add_plugins(SnakePlugin)
        .add_plugins(UiPlugin)
        .add_systems(Startup, setup)
        .add_systems(Startup, ui::spawn_stats_ui.after(setup))
        .add_systems(Update, simulation_step)
        .run();
}

fn setup(
    mut commands: Commands,
    windows: Query<&Window>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    hyperparams: Res<Hyperparameters>, // Read from resource instead of default
) {
    commands.spawn(Camera2dBundle::default());

    let window = windows.single();
    let (grid_width, grid_height) =
        calculate_grid_dimensions(window.resolution.width(), window.resolution.height());

    let segment_mesh = meshes.add(Rectangle::new(BLOCK_SIZE - 2.0, BLOCK_SIZE - 2.0));
    let food_mesh = meshes.add(Circle::new(BLOCK_SIZE / 2.0));
    let food_material = materials.add(Color::rgb(1.0, 0.0, 0.0));
    let head_material = materials.add(Color::rgb(1.0, 1.0, 1.0));

    commands.insert_resource(MeshCache {
        segment_mesh,
        food_mesh,
        food_material,
        head_material,
    });

    commands.insert_resource(CollisionSettings::default());
    commands.insert_resource(GraphPanelState::default());
    commands.insert_resource(RenderConfig::default());
    commands.insert_resource(GridMap::new(grid_width, grid_height));

    // Use population_size from config, not CPU cores
    let snake_count = hyperparams.population_size;

    let parallel_config = ParallelConfig::new(snake_count);
    commands.insert_resource(SegmentPool::new(snake_count));
    commands.insert_resource(AppStartTime::default());

    let (global_history, _) = snake::load_global_history();
    let _accumulated_time = global_history.accumulated_time_secs;

    let session_file = snake::new_session_path();
    println!("Session file: {}", session_file.display());

    let grid = GridDimensions {
        width: grid_width,
        height: grid_height,
    };

    // Create shared generation seed for fair comparison
    let gen_seed = GenerationSeed::new_for_grid(&grid);
    commands.insert_resource(gen_seed.clone());

    // Use hyperparams directly (population_size from config)
    let mut evo_manager = EvolutionManager::new(hyperparams.clone());
    evo_manager.load_archive();
    evo_manager.start_generation();

    // Create population brains and colors
    let individuals = evo_manager.get_population();
    let brains: Vec<_> = individuals.iter().map(|i| i.brain.clone()).collect();
    commands.insert_resource(Population(brains));

    // Extract behavioral values for color calculation
    // Format: (congestion, agility, fitness, best_fitness)
    let best_fitness = evo_manager.generation_state.best_fitness.max(1.0); // Avoid division by zero
    let behaviors: Vec<(f64, f64, f64, f64)> = individuals
        .iter()
        .map(|i| (i.congestion, i.agility, i.fitness, best_fitness))
        .collect();

    commands.insert_resource(global_history);
    commands.insert_resource(GameConfig::default());
    commands.insert_resource(WindowSettings {
        is_fullscreen: false,
    });
    commands.insert_resource(GameState::new_with_behavioral_colors(
        &grid,
        snake_count,
        Some(behaviors),
    ));
    commands.insert_resource(grid);
    commands.insert_resource(TrainingStats {
        fps: 0.0,
        last_fps_update: std::time::Instant::now(),
        frame_count: 0,
    });
    commands.insert_resource(parallel_config);
    commands.insert_resource(GameStats::new(snake_count));
    commands.insert_resource(evo_manager);
}

fn simulation_step(
    _commands: Commands,
    mut game: ResMut<GameState>,
    mut grid_map: ResMut<GridMap>,
    grid: Res<GridDimensions>,
    mut population: ResMut<Population>,
    mut evo_manager: ResMut<EvolutionManager>,
    mut global_history: ResMut<GlobalTrainingHistory>,
    collision_settings: Res<CollisionSettings>,
    mut gen_seed: ResMut<GenerationSeed>,
    pause_state: Res<PauseState>,
) {
    if pause_state.paused {
        return;
    }

    let config = &evo_manager.config;

    // Build grid map with overflow protection for >254 snakes
    grid_map.clear();
    for (idx, snake) in game.snakes.iter().enumerate() {
        if !snake.is_game_over {
            let cell_val = ((idx + 1) as u16).min(255) as u8; // clamp sicuro
            for pos in snake.snake.iter() {
                grid_map.set(pos.x, pos.y, cell_val);
            }
        }
    }

    // Process each snake
    let mut new_high_score = game.high_score;
    for snake in game.snakes.iter_mut() {
        if snake.is_game_over {
            continue;
        }

        let brain = match population.0.get(snake.id) {
            Some(b) => b,
            None => continue,
        };

        // Calculate state
        let current_17 = get_current_17_state(snake, &grid_map, &grid);

        let mut state_34 = [0.0f32; STATE_SIZE];
        state_34[..17].copy_from_slice(&current_17);
        state_34[17..].copy_from_slice(&snake.previous_state);

        let action = brain.predict(&state_34);

        // Update tracking
        snake.previous_state = current_17;

        match action {
            Action::Left => {
                snake.direction = snake.direction.turn_left();
                snake.turn_count += 1; // Increment for Agility tracking
            }
            Action::Right => {
                snake.direction = snake.direction.turn_right();
                snake.turn_count += 1; // Increment for Agility tracking
            }
            Action::Straight => {}
        }

        let (dx, dy) = snake.direction.as_vec();
        let old_head = snake.snake[0];
        let new_head = Position {
            x: old_head.x + dx,
            y: old_head.y + dy,
        };

        snake.steps_without_food += 1;
        snake.frames_survived += 1;

        // Track visited cells for exploration_ratio descriptor
        snake.visited_cells.insert((new_head.x, new_head.y));

        // Check collisions
        // Self-collision is always checked first
        let is_collision = snake.snake.contains(&new_head)
            || if collision_settings.snake_vs_snake {
                grid_map.is_collision(new_head.x, new_head.y, snake.id)
            } else {
                grid_map.is_wall_collision(new_head.x, new_head.y)
            };

        let ate_food = new_head == snake.food;
        let is_timeout = snake.steps_without_food > config.calculate_timeout(snake.snake.len());

        if is_collision || is_timeout {
            snake.is_game_over = true;
        } else {
            snake.snake.push_front(new_head);
            if ate_food {
                snake.score += 1;
                // Accumulate time spent to reach this apple
                snake.food_time_sum += snake.steps_without_food as u64;
                if snake.score > new_high_score {
                    new_high_score = snake.score;
                }
                // Next food from deterministic sequence
                snake.food = gen_seed.food_at(snake.score as usize);
                snake.steps_without_food = 0;
            } else {
                snake.snake.pop_back();
            }
        }
    }
    game.high_score = new_high_score;

    // Check generation end
    if game.alive_count() == 0 {
        end_generation(&mut game, &mut evo_manager, &mut global_history, &grid);

        // Generate new seed for next generation
        let new_seed = GenerationSeed::new_for_grid(&grid);
        *gen_seed = new_seed.clone();

        // 1. Reset all snakes with the new seed and archive colors
        let total_snakes = game.snakes.len();
        let individuals = evo_manager.get_population();
        for (i, snake) in game.snakes.iter_mut().enumerate() {
            // Reset snake with shared seed
            snake.reset_with_seed(&grid, total_snakes, &gen_seed, 0.0, 0.0, 0.0, 1.0);
            // Apply archive_color after reset
            if let Some(ind) = individuals.get(i) {
                snake.color = ind.archive_color.to_bevy_color();
            }
        }

        // 2. Sostituisci i vecchi cervelli con i nuovi appena evoluti
        population.0 = evo_manager
            .get_population()
            .iter()
            .map(|i| i.brain.clone())
            .collect();

        // 3. Aggiorna il contatore per la UI
        game.total_iterations += 1;
    }
}

fn end_generation(
    game: &mut GameState,
    evo_manager: &mut EvolutionManager,
    global_history: &mut GlobalTrainingHistory,
    grid: &GridDimensions,
) {
    // Update individuals in evolution manager
    for (i, snake) in game.snakes.iter().enumerate() {
        if let Some(ind) = evo_manager.get_individual_mut(i) {
            ind.fitness = snake.fitness(grid);
            ind.congestion = snake.exploration_ratio(grid);
            ind.agility = snake.agility();
            ind.frames_survived = snake.frames_survived;
            ind.apples_eaten = snake.score;
            ind.is_alive = false;
        }
    }

    let record = evo_manager.end_generation();

    // Sync to GlobalTrainingHistory for UI graph and JSON persistence
    global_history.records.push(record.clone());

    println!(
        "Gen {:4} | Fitness: {:.0} (best: {:.0}) | Coverage: {:.1}% | {:.2}s",
        record.generation,
        record.avg_fitness,
        record.best_fitness,
        record.archive_coverage * 100.0,
        record.elapsed_secs
    );

    // Start next generation
    evo_manager.start_generation();
}

pub struct SnakePlugin;
impl Plugin for SnakePlugin {
    fn build(&self, _app: &mut App) {}
}
