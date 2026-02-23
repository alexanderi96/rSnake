//! MAP-Elites Snake - Bevy ECS with Evolutionary Algorithm
//!
//! A simplified implementation to get things compiling first.

#![recursion_limit = "256"]

use std::collections::HashMap;

mod brain;
mod config;
mod evolution;
mod map_elites;
#[cfg(feature = "profiling")]
mod profiling;

#[cfg(not(feature = "profiling"))]
mod profiling {
    pub struct ProfilingGuard;
    impl ProfilingGuard {
        pub fn new() -> Self {
            Self
        }
    }
    pub fn is_profiling() -> bool {
        false
    }
}

mod snake;
mod ui;

use bevy::app::AppExit;
use bevy::prelude::*;
use bevy::sprite::MaterialMesh2dBundle;
use clap::Parser;

use brain::Action;
use config::Hyperparameters;
use evolution::EvolutionManager;
use snake::{
    calculate_grid_dimensions, get_current_17_state, AppStartTime, CollisionSettings, Food,
    GameConfig, GameState, GameStats, GenerationSeed, GlobalTrainingHistory, GridDimensions,
    GridMap, MeshCache, ParallelConfig, Position, RenderConfig, SnakeId, TrainingStats,
    BASE_STATE_SIZE, BLOCK_SIZE, STATE_SIZE,
};
use ui::{CellRenderMap, GraphPanelState, PauseState, UiPlugin, WindowSettings};

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

/// Intermediate results from parallel brain forward pass
/// Stores (action, current_17_state) for each snake, indexed by snake id
#[derive(Resource, Default)]
struct ComputedMoves(Vec<Option<(Action, [f32; BASE_STATE_SIZE])>>);

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
        // Two-phase parallel simulation: compute moves (parallel) then apply (serial)
        .insert_resource(ComputedMoves::default())
        .add_systems(
            Update,
            (
                compute_moves_parallel,
                apply_moves_serial.after(compute_moves_parallel),
            ),
        )
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

    // Create mesh cache and materials
    let mesh_cache = MeshCache {
        segment_mesh: meshes.add(Rectangle::new(BLOCK_SIZE - 2.0, BLOCK_SIZE - 2.0)),
        food_mesh: meshes.add(Circle::new(BLOCK_SIZE / 2.0)),
        food_material: materials.add(Color::rgb(1.0, 0.0, 0.0)),
    };

    // Pre-spawn one entity per grid cell for cell-based rendering
    let cell_count = (grid_width * grid_height) as usize;
    let mut cell_entities = Vec::with_capacity(cell_count);
    // Use a neutral dark color as default (will be overwritten when visible)
    let default_material = materials.add(Color::rgb(0.05, 0.05, 0.05));
    for _ in 0..cell_count {
        let entity = commands
            .spawn(MaterialMesh2dBundle {
                mesh: mesh_cache.segment_mesh.clone().into(),
                material: default_material.clone(),
                transform: Transform::from_xyz(0.0, 0.0, 0.0),
                visibility: Visibility::Hidden,
                ..default()
            })
            .id();
        cell_entities.push(entity);
    }
    commands.insert_resource(CellRenderMap {
        cells: HashMap::new(),
        entities: cell_entities,
        grid_width,
        grid_height,
        rebuilding: false, // Initial spawn - safe, no deferred commands to worry about
    });

    commands.insert_resource(CollisionSettings::default());
    commands.insert_resource(GraphPanelState::default());
    commands.insert_resource(RenderConfig::default());
    commands.insert_resource(GridMap::new(grid_width, grid_height));

    // Use population_size from config, not CPU cores
    let snake_count = hyperparams.population_size;

    let parallel_config = ParallelConfig::new(snake_count);
    commands.insert_resource(AppStartTime::default());

    // Pre-spawn food entities for the pool (one per snake)
    let mut food_entities = Vec::with_capacity(snake_count);
    for i in 0..snake_count {
        let entity = commands
            .spawn((
                MaterialMesh2dBundle {
                    mesh: mesh_cache.food_mesh.clone().into(),
                    material: mesh_cache.food_material.clone(),
                    transform: Transform::from_xyz(0.0, 0.0, 0.0),
                    visibility: Visibility::Hidden,
                    ..default()
                },
                Food,
                SnakeId(i),
            ))
            .id();
        food_entities.push(entity);
    }
    commands.insert_resource(ui::FoodPool {
        entities: food_entities,
    });

    commands.insert_resource(mesh_cache);

    let (global_history, max_gen) = snake::load_global_history();
    let _accumulated_time = global_history.accumulated_time_secs;
    let persisted_high_score = global_history.all_time_high_score;

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
    // Format: (path_directness, body_avoidance, fitness, best_fitness)
    let best_fitness = evo_manager.generation_state.best_fitness.max(1.0); // Avoid division by zero
    let behaviors: Vec<(f64, f64, f64, f64)> = individuals
        .iter()
        .map(|i| (i.path_directness, i.body_avoidance, i.fitness, best_fitness))
        .collect();

    commands.insert_resource(global_history);
    commands.insert_resource(GameConfig::default());
    commands.insert_resource(WindowSettings {
        is_fullscreen: false,
    });

    // Create GameState and restore persistent counters from loaded history
    let mut game_state =
        GameState::new_with_behavioral_colors(&grid, snake_count, Some(behaviors.clone()));
    game_state.total_iterations = max_gen;
    game_state.high_score = persisted_high_score;

    // --- FIX PULITO (DRY) ---
    // Invece di duplicare la logica, usiamo reset_with_seed esattamente
    // come avviene a fine generazione in apply_moves_serial.
    let total_snakes = game_state.snakes.len();
    for (i, snake) in game_state.snakes.iter_mut().enumerate() {
        // Recuperiamo i comportamenti estratti poco sopra
        let (courage, agility, fitness, best) =
            behaviors.get(i).copied().unwrap_or((0.5, 0.5, 0.0, 1.0));

        // Applichiamo il seed condiviso
        snake.reset_with_seed(
            &grid,
            total_snakes,
            &gen_seed,
            courage,
            agility,
            fitness,
            best,
        );

        // Riapplichiamo il colore dell'archivio (opzionale, ma mantiene la coerenza visiva)
        if let Some(ind) = individuals.get(i) {
            snake.color = ind.archive_color.to_bevy_color();
        }
    }
    // ------------------------

    commands.insert_resource(game_state);
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

/// PHASE 1: Parallel brain forward pass
/// Computes actions for all snakes in parallel using rayon.
/// Read-only access to GameState, GridMap, Population.
fn compute_moves_parallel(
    game: Res<GameState>,
    grid_map: Res<GridMap>,
    grid: Res<GridDimensions>,
    population: Res<Population>,
    mut computed: ResMut<ComputedMoves>,
    pause_state: Res<PauseState>,
) {
    if pause_state.paused {
        return;
    }

    let snake_count = game.snakes.len();
    let mut results = vec![None; snake_count];

    // Rayon parallel iterator — read-only access to GameState and GridMap
    use rayon::prelude::*;
    results
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, result)| {
            let snake = &game.snakes[idx];
            if snake.is_game_over {
                return;
            }

            let brain = match population.0.get(snake.id) {
                Some(b) => b,
                None => return,
            };

            let current_17 = get_current_17_state(snake, &grid_map, &grid);
            let mut state_34 = [0.0f32; STATE_SIZE];
            state_34[..17].copy_from_slice(&current_17);
            state_34[17..].copy_from_slice(&snake.previous_state);

            let action = brain.predict(&state_34);
            *result = Some((action, current_17));
        });

    computed.0 = results;
}

/// PHASE 2: Serial state application
/// Applies computed moves and updates all mutable state.
/// Single-threaded due to GridMap writes.
#[allow(clippy::too_many_arguments)]
fn apply_moves_serial(
    mut game: ResMut<GameState>,
    mut grid_map: ResMut<GridMap>,
    grid: Res<GridDimensions>,
    computed: Res<ComputedMoves>,
    config: Res<Hyperparameters>,
    mut evo_manager: ResMut<EvolutionManager>,
    mut global_history: ResMut<GlobalTrainingHistory>,
    mut gen_seed: ResMut<GenerationSeed>,
    mut population: ResMut<Population>,
    collision_settings: Res<CollisionSettings>,
    pause_state: Res<PauseState>,
    mut game_stats: ResMut<GameStats>,
) {
    if pause_state.paused {
        return;
    }

    // Rebuild grid_map (serial — cannot be parallelized)
    grid_map.clear();
    for (idx, snake) in game.snakes.iter().enumerate() {
        if !snake.is_game_over {
            let cell_val = ((idx + 1) as u16).min(255) as u8;
            for pos in snake.snake.iter() {
                grid_map.set(pos.x, pos.y, cell_val);
            }
        }
    }

    let mut new_high_score = game.high_score;

    for (idx, result) in computed.0.iter().enumerate() {
        let Some((action, current_17)) = result else {
            continue;
        };
        let snake = &mut game.snakes[idx];
        if snake.is_game_over {
            continue;
        }

        snake.previous_state = *current_17;

        match action {
            Action::Left => {
                snake.direction = snake.direction.turn_left();
                snake.turn_count += 1;
            }
            Action::Right => {
                snake.direction = snake.direction.turn_right();
                snake.turn_count += 1;
            }
            Action::Straight => {}
        }

        let (dx, dy) = snake.direction.as_vec();
        let old_head = snake.snake[0];
        let new_head = Position {
            x: old_head.x + dx,
            y: old_head.y + dy,
        };

        // Calcola ate_food PRIMA del collision check e degli aggiornamenti metriche:
        // serve per la tail exception e per non inquinare le metriche col frame di morte
        let ate_food = new_head == snake.food;

        // Tail exception: se non mangiamo, la coda si sposterà questo frame —
        // muovere la testa dove c'è la coda è legale
        let tail_pos = snake.snake.back().copied();
        let is_self_collision =
            snake.body_set.contains(&new_head) && (ate_food || Some(new_head) != tail_pos);

        let is_collision = is_self_collision
            || if collision_settings.snake_vs_snake {
                grid_map.is_collision(new_head.x, new_head.y, snake.id)
            } else {
                grid_map.is_wall_collision(new_head.x, new_head.y)
            };

        let is_timeout = snake.steps_without_food > config.calculate_timeout(snake.snake.len());

        // Aggiorna metriche SOLO se il serpente sopravvive questo frame
        if !is_collision && !is_timeout {
            snake.steps_without_food += 1;
            snake.frames_survived += 1;
            snake.visited_cells.insert((new_head.x, new_head.y));
            let body_len = snake.snake.len() as f64;
            let visited = snake.visited_cells.len().max(1) as f64;
            snake.body_pressure_sum += (body_len / visited).clamp(0.0, 1.0);
        }

        if is_collision || is_timeout {
            snake.is_game_over = true;
        } else {
            // Update snake body and body_set
            snake.snake.push_front(new_head);
            snake.body_set.insert(new_head);
            if ate_food {
                // Path directness accumulation (grid-invariant)
                if snake.food_spawn_distance > 0 {
                    let ratio = (snake.food_spawn_distance as f64
                        / snake.steps_without_food as f64)
                        .clamp(0.0, 1.0);
                    snake.path_directness_sum += ratio;
                }
                snake.score += 1;
                game_stats.total_food_eaten += 1;
                snake.food_time_sum += snake.steps_without_food as u64;
                // Budget timeout disponibile per questa mela (scala con la lunghezza)
                // Dopo push_front il len() include già la nuova testa
                snake.timeout_budget_sum += config.calculate_timeout(snake.snake.len()) as u64;
                if snake.score > new_high_score {
                    new_high_score = snake.score;
                }

                // Spawn next food and calculate new Manhattan distance
                let new_food = gen_seed.food_at(snake.score as usize);
                let new_manhattan =
                    (new_food.x - new_head.x).abs() + (new_food.y - new_head.y).abs();
                snake.food_spawn_distance = new_manhattan as u32;
                snake.food = new_food;
                snake.steps_without_food = 0;
            } else {
                // Remove tail from body_set BEFORE pop_back
                let tail = *snake.snake.back().unwrap();
                snake.body_set.remove(&tail);
                snake.snake.pop_back();
            }
        }
    }
    game.high_score = new_high_score;

    // Check generation end
    if game.alive_count() == 0 {
        game_stats.total_games_played += game.snakes.len() as u64;
        end_generation(&mut game, &mut evo_manager, &mut global_history, &grid);

        // Generate new seed for next generation
        let new_seed = GenerationSeed::new_for_grid(&grid);
        *gen_seed = new_seed.clone();

        // Reset all snakes with the new seed and archive colors
        let total_snakes = game.snakes.len();
        let individuals = evo_manager.get_population();
        let best_fitness = evo_manager.archive.best_fitness.max(1.0);
        for (i, snake) in game.snakes.iter_mut().enumerate() {
            let (courage, agility, fitness, best) = individuals
                .get(i)
                .map(|ind| {
                    (
                        ind.path_directness,
                        ind.body_avoidance,
                        ind.fitness,
                        best_fitness,
                    )
                })
                .unwrap_or((0.0, 0.0, 0.0, 1.0));
            snake.reset_with_seed(
                &grid,
                total_snakes,
                &gen_seed,
                courage,
                agility,
                fitness,
                best,
            );
            if let Some(ind) = individuals.get(i) {
                snake.color = ind.archive_color.to_bevy_color();
            }
        }

        // Replace old brains with newly evolved ones
        population.0 = evo_manager
            .get_population()
            .iter()
            .map(|i| i.brain.clone())
            .collect();

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
            // Grid-invariant behavioral descriptors
            ind.path_directness = snake.path_directness();
            ind.body_avoidance = snake.body_avoidance();
            ind.frames_survived = snake.frames_survived;
            ind.apples_eaten = snake.score;
            ind.is_alive = false;
        }
    }

    // Compute generation high score before calling end_generation
    let gen_high_score = game.snakes.iter().map(|s| s.score).max().unwrap_or(0);

    let mut record = evo_manager.end_generation();

    // Set generation_high_score on the record after creation
    record.generation_high_score = gen_high_score;

    // Update all_time_high_score
    if gen_high_score > global_history.all_time_high_score {
        global_history.all_time_high_score = gen_high_score;
    }

    // Sync to GlobalTrainingHistory for UI graph and JSON persistence
    global_history.records.push(record.clone());

    // Cap history to prevent unbounded memory growth
    if global_history.records.len() > 10_000 {
        global_history.records.drain(0..5_000);
    }

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
