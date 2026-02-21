//! MAP-Elites Snake - Bevy ECS with Evolutionary Algorithm
//!
//! A simplified implementation to get things compiling first.

#![recursion_limit = "256"]

mod brain;
mod config;
mod evolution;
mod map_elites;
mod snake;
mod types;
mod ui;

use bevy::app::AppExit;
use bevy::prelude::*;
use clap::Parser;

use brain::Action;
use config::Hyperparameters;
use evolution::{EvolutionConfig, EvolutionManager};
use snake::{
    calculate_grid_dimensions, get_current_17_state, spawn_food, AppStartTime, CollisionSettings,
    GameConfig, GameState, GameStats, GlobalTrainingHistory, GridDimensions, GridMap, MeshCache,
    ParallelConfig, Position, RenderConfig, SegmentPool, TrainingStats, BLOCK_SIZE, STATE_SIZE,
};
use ui::{GraphPanelState, UiPlugin, WindowSettings};

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
) {
    // Get hyperparameters from a default since we can't pass CLI args to Bevy systems easily
    let hyperparams = Hyperparameters::default();

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

    let parallel_config = ParallelConfig::new();
    commands.insert_resource(SegmentPool::new(parallel_config.snake_count));
    commands.insert_resource(AppStartTime::default());

    let (global_history, _) = snake::load_global_history();
    let accumulated_time = global_history.accumulated_time_secs;

    let session_file = snake::new_session_path();
    println!("Session file: {}", session_file.display());

    let grid = GridDimensions {
        width: grid_width,
        height: grid_height,
    };

    // Use CPU core count as population size for parallel evaluation
    let snake_count = parallel_config.snake_count;

    let evo_config = EvolutionConfig {
        population_size: snake_count,
        mutation_rate: hyperparams.mutation_rate,
        mutation_strength: hyperparams.mutation_strength,
        crossover_rate: hyperparams.crossover_rate,
        max_frames: hyperparams.max_frames,
        base_steps_without_food: hyperparams.base_steps_without_food,
        steps_per_segment: hyperparams.steps_per_segment,
        auto_save_interval: hyperparams.auto_save_interval,
    };

    let mut evo_manager = EvolutionManager::new(evo_config);
    evo_manager.load_archive();
    evo_manager.start_generation();

    // Create population brains
    let brains: Vec<_> = evo_manager
        .get_population()
        .iter()
        .map(|i| i.brain.clone())
        .collect();
    commands.insert_resource(Population(brains));

    commands.insert_resource(global_history);
    commands.insert_resource(GameConfig {
        speed_timer: Timer::from_seconds(0.001, TimerMode::Repeating),
        session_path: session_file,
    });
    commands.insert_resource(WindowSettings {
        is_fullscreen: false,
    });
    commands.insert_resource(GameState::new(&grid, snake_count));
    commands.insert_resource(grid);
    commands.insert_resource(TrainingStats {
        total_training_time: std::time::Duration::from_secs(accumulated_time),
        last_update: std::time::Instant::now(),
        parallel_threads: rayon::current_num_threads(),
        fps: 0.0,
        last_fps_update: std::time::Instant::now(),
        frame_count: 0,
    });
    commands.insert_resource(parallel_config);
    commands.insert_resource(GameStats::new(snake_count));
    commands.insert_resource(evo_manager);
}

fn simulation_step(
    mut game: ResMut<GameState>,
    mut grid_map: ResMut<GridMap>,
    grid: Res<GridDimensions>,
    population: Res<Population>,
    mut evo_manager: ResMut<EvolutionManager>,
    collision_settings: Res<CollisionSettings>,
) {
    let config = &evo_manager.config;

    // Build grid map
    grid_map.clear();
    for (idx, snake) in game.snakes.iter().enumerate() {
        if !snake.is_game_over {
            for pos in snake.snake.iter() {
                grid_map.set(pos.x, pos.y, (idx + 1) as u8);
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
            Action::Left => snake.direction = snake.direction.turn_left(),
            Action::Right => snake.direction = snake.direction.turn_right(),
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

        // Wall distance for courage
        let dist_x = (new_head.x as f64 / grid.width as f64)
            .min(1.0 - (new_head.x as f64 / grid.width as f64));
        let dist_y = (new_head.y as f64 / grid.height as f64)
            .min(1.0 - (new_head.y as f64 / grid.height as f64));
        snake.wall_distance_sum += (dist_x + dist_y) / 2.0;

        // Check collisions
        let is_collision = if collision_settings.snake_vs_snake {
            grid_map.is_collision(new_head.x, new_head.y, snake.id)
        } else {
            grid_map.is_collision_no_snakes(new_head.x, new_head.y)
        } || snake.snake.contains(&new_head);

        let ate_food = new_head == snake.food;
        let is_timeout = snake.steps_without_food > config.calculate_timeout(snake.snake.len());

        if is_collision || is_timeout {
            snake.is_game_over = true;
        } else {
            snake.snake.push_front(new_head);
            if ate_food {
                snake.score += 1;
                if snake.score > new_high_score {
                    new_high_score = snake.score;
                }
                snake.food = spawn_food(snake, &grid);
                snake.steps_without_food = 0;
            } else {
                snake.snake.pop_back();
            }
        }
    }
    game.high_score = new_high_score;

    // Check generation end
    if game.alive_count() == 0 {
        end_generation(&mut game, &mut evo_manager);
    }
}

fn end_generation(game: &mut GameState, evo_manager: &mut EvolutionManager) {
    // Update individuals in evolution manager
    for (i, snake) in game.snakes.iter().enumerate() {
        if let Some(ind) = evo_manager.get_individual_mut(i) {
            ind.fitness = snake.fitness();
            ind.courage = snake.courage();
            ind.agility = snake.agility();
            ind.frames_survived = snake.frames_survived;
            ind.apples_eaten = snake.score;
            ind.is_alive = false;
        }
    }

    let record = evo_manager.end_generation();

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
