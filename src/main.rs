//! MAP-Elites Snake - Bevy ECS with Evolutionary Algorithm

#![recursion_limit = "256"]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::implicit_saturating_sub)]
#![allow(clippy::unnecessary_map_or)]
#![allow(clippy::borrow_deref_ref)]
#![allow(clippy::indexing_slicing)]
#![allow(clippy::iter_with_drain)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::wrong_self_convention)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::needless_borrows_for_generic_args)]

use std::sync::Arc;

mod config;
mod plugins;
mod profiling;

mod snake;

use bevy::app::AppExit;
use bevy::diagnostic::{
    DiagnosticsStore, EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin,
    SystemInformationDiagnosticsPlugin,
};
use bevy::prelude::*;
use bevy::sprite::MaterialMesh2dBundle;
use clap::Parser;

use config::Hyperparameters;
use plugins::brain_inspector::SimulationCamera;
use plugins::food_spawn::FoodSpawnPlugin;
use plugins::map_elites::evolution::EvolutionManager;
use plugins::map_elites::individual::{Action, Individual};
use plugins::terrain::TerrainPlugin;
pub use plugins::terrain::{generate, TerrainMap};
use plugins::ui::{
    CellRenderMap, GraphPanelState, HeatmapPanelState, MaterialPalette, PauseState, WindowSettings,
};
use snake::{
    bfs_distance, calculate_grid_dimensions, get_current_17_state, AppStartTime, CollisionSettings,
    ContinuousMode, Food, GameConfig, GameState, GameStats, GenerationSeed, GlobalTrainingHistory,
    GridDimensions, GridMap, MeshCache, ParallelConfig, Position, RenderConfig, RunDirectory,
    SnakeId, TrainingStats, BASE_STATE_SIZE, BLOCK_SIZE, STATE_SIZE,
};

/// Colore sfondo — deve corrispondere al ClearColor inserito in main()
/// Usato anche in ui/mod.rs per il lerp alpha-simulato se necessario.
pub const BG_COLOR: Color = Color::rgb(0.1, 0.1, 0.1);

/// CLI Arguments
#[derive(Parser, Debug, Clone, Resource)]
#[command(name = "snake-map-elites")]
#[command(about = "MAP-Elites Snake RL Training")]
pub struct CliArgs {
    #[arg(short, long)]
    pub config: Option<String>,
    #[arg(long)]
    pub population_size: Option<usize>,
    #[arg(long)]
    pub mutation_rate: Option<f32>,
    #[arg(long)]
    pub mutation_strength: Option<f32>,
    #[arg(long)]
    pub crossover_rate: Option<f32>,
    #[arg(long)]
    pub base_steps_without_food: Option<u32>,
    #[arg(long)]
    pub steps_per_segment: Option<u32>,
    #[arg(long)]
    pub terrain_blob_scale: Option<f32>,
    #[arg(long)]
    pub terrain_fill_rate: Option<f32>,

    /// Forza l'inizio di una nuova run (nuova cartella) ignorando quelle esistenti
    #[arg(long, default_value_t = false)]
    pub new_run: bool,

    /// Azzera la fitness di tutti gli individui caricati dall'archivio (per migrazione a nuova formula)
    #[arg(long, default_value_t = false)]
    pub migrate_fitness: bool,

    /// Modalità apprendimento continuo: gli snake morti vengono sostituiti immediatamente
    #[arg(long, default_value_t = false)]
    pub continuous: bool,
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
    if let Some(v) = args.terrain_fill_rate {
        config.terrain_fill_rate = v;
    }
    if let Some(v) = args.terrain_blob_scale {
        config.terrain_blob_scale = v;
    }

    config
}

#[derive(Resource)]
struct Population(pub Vec<Arc<plugins::map_elites::individual::Brain>>);

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

    // ProfilingGuard MUST be the first binding in main to be dropped last
    let _profiling = profiling::ProfilingGuard::new();

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "MAP-Elites Snake".into(),
                resolution: (800.0, 600.0).into(),
                resizable: true,
                present_mode: bevy::window::PresentMode::AutoNoVsync,
                ..default()
            }),
            ..default()
        }))
        .add_plugins(TerrainPlugin)
        .add_plugins(FrameTimeDiagnosticsPlugin)
        .add_plugins(EntityCountDiagnosticsPlugin)
        .add_plugins(SystemInformationDiagnosticsPlugin)
        .add_event::<AppExit>()
        // Sfondo quasi-nero con lieve tinta blu-notte
        .insert_resource(ClearColor(BG_COLOR))
        .insert_resource(args)
        .insert_resource(hyperparams)
        .add_plugins(FoodSpawnPlugin)
        .add_plugins(plugins::simulation::SimulationPlugin)
        .add_plugins(plugins::ui::UiPlugin)
        .add_plugins(plugins::brain_inspector::BrainInspectorPlugin)
        .add_systems(Startup, setup)
        .add_systems(Startup, plugins::ui::spawn_stats_ui.after(setup))
        // Two-phase parallel simulation: compute moves (parallel) then apply (serial)
        .insert_resource(ComputedMoves::default())
        .insert_resource(snake::SimStepsPerFrame::default())
        .add_systems(
            Update,
            (
                compute_moves_parallel,
                apply_moves_serial.after(compute_moves_parallel),
                log_diagnostics_periodic,
            ),
        )
        .run();
}

/// Logga diagnostics ogni 5 secondi su stderr (non interferisce con UI)
fn log_diagnostics_periodic(
    diagnostics: Res<DiagnosticsStore>,
    time: Res<Time>,
    mut last_log: Local<f64>,
) {
    let now = time.elapsed_seconds_f64();
    if now - *last_log < 5.0 {
        return;
    }
    *last_log = now;

    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);

    let frame_ms = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FRAME_TIME)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0)
        * 1000.0;

    let entities = diagnostics
        .get(&EntityCountDiagnosticsPlugin::ENTITY_COUNT)
        .and_then(|d| d.value())
        .unwrap_or(0.0);

    let cpu = diagnostics
        .get(&SystemInformationDiagnosticsPlugin::CPU_USAGE)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);

    let mem_mb = diagnostics
        .get(&SystemInformationDiagnosticsPlugin::MEM_USAGE)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);

    eprintln!(
        "[DIAG] FPS:{:.1} frame:{:.2}ms entities:{:.0} CPU:{:.1}% MEM:{:.1}MB",
        fps, frame_ms, entities, cpu, mem_mb
    );
}

fn setup(
    mut commands: Commands,
    windows: Query<&Window>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    hyperparams: Res<Hyperparameters>,
    args: Res<CliArgs>,
) {
    commands.spawn((Camera2dBundle::default(), SimulationCamera));

    let window = windows.single();
    let (grid_width, grid_height) =
        calculate_grid_dimensions(window.resolution.width(), window.resolution.height());

    // 1. Determine Run Directory (New or Latest)
    let run_dir_path = snake::get_or_create_run_dir(args.new_run);
    println!("📂 Run Directory: {}", run_dir_path.display());

    commands.insert_resource(RunDirectory(run_dir_path.clone()));

    // Create mesh cache and materials
    let mesh_cache = MeshCache {
        segment_mesh: meshes.add(Rectangle::new(BLOCK_SIZE - 2.0, BLOCK_SIZE - 2.0)),
        food_mesh: meshes.add(Circle::new(BLOCK_SIZE * 0.20)),            // ~4px radius, pellet
        food_material: materials.add(Color::rgba(1.0, 0.78, 0.10, 0.28)), // oro, molto trasparente
        food_material_best: materials.add(Color::rgba(1.0, 0.88, 0.25, 0.92)), // oro brillante
    };

    // Pre-spawn one entity per grid cell for cell-based rendering
    let cell_count = (grid_width * grid_height) as usize;
    let mut cell_entities = Vec::with_capacity(cell_count);
    let default_material = materials.add(Color::rgba(0.05, 0.05, 0.05, 1.0));
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
    let cell_size = (grid_width * grid_height) as usize;
    commands.insert_resource(CellRenderMap {
        cells: vec![None; cell_size],
        prev_colors: vec![None; cell_size],
        entities: cell_entities,
        grid_width,
        grid_height,
        rebuilding: false,
        terrain_dirty: true,
    });

    // ── Material palette con alpha ────────────────────────────────────────────
    // 4 livelli alpha × 8×8×8 RGB = 2048 entries
    // Alpha: 255=opaco (testa), 212, 170, 128=50% (coda)
    const PALETTE_STEPS: usize = 8;
    const ALPHA_LEVELS: [u8; 4] = [255, 212, 170, 96];

    let mut palette_handles: Vec<Handle<ColorMaterial>> = Vec::with_capacity(2048);
    let mut palette_colors: Vec<[u8; 4]> = Vec::with_capacity(2048);

    for &a in ALPHA_LEVELS.iter() {
        for r in (0..=255u8).step_by(255 / (PALETTE_STEPS - 1)) {
            for g in (0..=255u8).step_by(255 / (PALETTE_STEPS - 1)) {
                for b in (0..=255u8).step_by(255 / (PALETTE_STEPS - 1)) {
                    palette_colors.push([r, g, b, a]);
                    palette_handles.push(materials.add(Color::rgba(
                        r as f32 / 255.0,
                        g as f32 / 255.0,
                        b as f32 / 255.0,
                        a as f32 / 255.0,
                    )));
                }
            }
        }
    }

    // lookup[ai * 512 + ri*64 + gi*8 + bi] → indice in palette_handles
    let mut lookup = vec![0usize; 4 * 512];
    for (i, &[r, g, b, a]) in palette_colors.iter().enumerate() {
        let ai: usize = match a {
            212..=255 => 0,
            170..=211 => 1,
            128..=169 => 2,
            _          => 3,
        };
        let ri = (r as usize * 7 + 127) / 255;
        let gi = (g as usize * 7 + 127) / 255;
        let bi = (b as usize * 7 + 127) / 255;
        lookup[ai * 512 + ri * 64 + gi * 8 + bi] = i;
    }

    commands.insert_resource(MaterialPalette {
        handles: palette_handles,
        colors: palette_colors,
        lookup,
    });

    commands.insert_resource(CollisionSettings::default());
    commands.insert_resource(GraphPanelState::default());
    commands.insert_resource(HeatmapPanelState::default());
    commands.insert_resource(RenderConfig::default());
    commands.insert_resource(ContinuousMode {
        enabled: args.continuous,
        replacement_count: 0,
        replacements_since_seed: 0,
    });

    let snake_count = hyperparams.population_size;
    let parallel_config = ParallelConfig::new(snake_count);
    commands.insert_resource(AppStartTime::default());

    // Pre-spawn food entities
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
    commands.insert_resource(plugins::ui::FoodPool {
        entities: food_entities,
    });

    commands.insert_resource(mesh_cache);

    // 2. Load Global History from the specific Run Directory
    let (global_history, max_gen) = snake::load_global_history(&run_dir_path);
    let _accumulated_time = global_history.accumulated_time_secs;
    let persisted_high_score = global_history.all_time_high_score;

    let grid = GridDimensions {
        width: grid_width,
        height: grid_height,
    };

    // Create shared generation seed and apply terrain to grid map
    let mut grid_map = GridMap::new(grid_width, grid_height);
    let gen_seed = GenerationSeed::new_for_grid_with_config(&grid, &hyperparams);
    grid_map.apply_terrain(&gen_seed.terrain);
    commands.insert_resource(grid_map);
    commands.insert_resource(gen_seed.clone());

    let wall_count = gen_seed.terrain.iter().filter(|&&w| w).count();
    let total = gen_seed.terrain.len();
    println!(
        "🗺  Terrain: fill={:.0}% blob_scale={:.1} walls={}/{} ({:.1}%)",
        hyperparams.terrain_fill_rate * 100.0,
        hyperparams.terrain_blob_scale,
        wall_count,
        total,
        wall_count as f32 / total as f32 * 100.0,
    );

    let mut evo_manager = EvolutionManager::new(hyperparams.clone());
    evo_manager.load_archive(&run_dir_path);

    if args.migrate_fitness {
        if !evo_manager.archive.grid.is_empty() {
            println!(
                "⚠️  MIGRATION FLAG ACTIVE: Resetting fitness to 0.0 for all loaded individuals."
            );
            println!("    Brains are preserved, but they must re-validate their score.");
            for individual in evo_manager.archive.grid.values_mut() {
                individual.fitness = 0.0;
            }
        }
    }

    evo_manager.start_generation();

    // Create population brains (using Arc to avoid copying 10MB per generation)
    let individuals = evo_manager.get_population();
    let brains: Vec<_> = individuals
        .iter()
        .map(|i| Arc::new(i.brain.clone()))
        .collect();
    commands.insert_resource(Population(brains));

    // Extract behavioral values
    let best_fitness = evo_manager.generation_state.best_fitness.max(1.0);
    let behaviors: Vec<(f32, f32, f32, f32)> = individuals
        .iter()
        .map(|i| {
            (
                i.desc_path_efficiency,
                i.desc_danger_affinity,
                i.fitness,
                best_fitness,
            )
        })
        .collect();

    commands.insert_resource(global_history);
    commands.insert_resource(GameConfig::default());
    commands.insert_resource(WindowSettings {
        is_fullscreen: false,
    });

    let mut game_state =
        GameState::new_with_behavioral_colors(&grid, snake_count, Some(behaviors.clone()));
    game_state.total_iterations = max_gen;
    game_state.high_score = persisted_high_score;

    // Apply shared seed to snakes
    let total_snakes = game_state.snakes.len();
    for (i, snake) in game_state.snakes.iter_mut().enumerate() {
        let (courage, agility, fitness, best) =
            behaviors.get(i).copied().unwrap_or((0.5, 0.5, 0.0, 1.0));

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
fn compute_moves_fn(
    snakes: &[crate::snake::SnakeInstance],
    grid_map: &GridMap,
    grid: &GridDimensions,
    population: &Population,
    snake_vs_snake: bool,
) -> Vec<Option<(Action, [f32; BASE_STATE_SIZE])>> {
    use rayon::prelude::*;
    let mut results = vec![None; snakes.len()];
    results
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, result)| {
            let snake = &snakes[idx];
            if snake.is_game_over {
                return;
            }
            let brain = match population.0.get(snake.id) {
                Some(b) => b.as_ref(),
                None => return,
            };
            let current_17 = get_current_17_state(snake, grid_map, grid, snake_vs_snake);
            let mut state_34 = [0.0f32; STATE_SIZE];
            state_34[..17].copy_from_slice(&current_17);
            state_34[17..].copy_from_slice(&snake.previous_state);
            *result = Some((brain.predict(&state_34), current_17));
        });
    results
}

fn compute_moves_parallel(
    game: Res<GameState>,
    grid_map: Res<GridMap>,
    grid: Res<GridDimensions>,
    population: Res<Population>,
    mut computed: ResMut<ComputedMoves>,
    pause_state: Res<PauseState>,
    collision_settings: Res<CollisionSettings>,
) {
    #[cfg(feature = "tracy")]
    let _span = tracing::info_span!("compute_moves_parallel").entered();

    if pause_state.paused {
        return;
    }

    computed.0 = compute_moves_fn(
        &game.snakes,
        &grid_map,
        &grid,
        &population,
        collision_settings.snake_vs_snake,
    );
}

/// PHASE 2: Serial state application
#[allow(clippy::too_many_arguments)]
fn apply_moves_serial(
    mut game: ResMut<GameState>,
    mut grid_map: ResMut<GridMap>,
    grid: Res<GridDimensions>,
    mut computed: ResMut<ComputedMoves>,
    config: Res<Hyperparameters>,
    mut evo_manager: ResMut<EvolutionManager>,
    mut global_history: ResMut<GlobalTrainingHistory>,
    mut gen_seed: ResMut<GenerationSeed>,
    mut population: ResMut<Population>,
    collision_settings: Res<CollisionSettings>,
    pause_state: Res<PauseState>,
    mut game_stats: ResMut<GameStats>,
    mut cell_map: ResMut<CellRenderMap>,
    sim_steps: Res<snake::SimStepsPerFrame>,
    food_spawn_zone: Res<plugins::food_spawn::FoodSpawnZone>,
    mut continuous_mode: ResMut<ContinuousMode>,
) {
    #[cfg(feature = "tracy")]
    let _span = tracing::info_span!("apply_moves_serial").entered();

    if pause_state.paused {
        return;
    }

    let steps = sim_steps.0;
    let mut current_moves = std::mem::take(&mut computed.0);

    for step_idx in 0..steps {
        if step_idx > 0 {
            current_moves = compute_moves_fn(
                &game.snakes,
                &grid_map,
                &grid,
                &population,
                collision_settings.snake_vs_snake,
            );
        }

        // Rebuild grid_map (serial)
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

        for (idx, result) in current_moves.iter().enumerate() {
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
                }
                Action::Right => {
                    snake.direction = snake.direction.turn_right();
                }
                Action::Straight => {}
            }

            let (dx, dy) = snake.direction.as_vec();
            let old_head = snake.snake[0];
            let new_head = Position {
                x: old_head.x + dx,
                y: old_head.y + dy,
            };

            let ate_food = new_head == snake.food;

            // Tail exception
            let tail_pos = snake.snake.back().copied();
            let is_self_collision =
                snake.body_set.contains(&new_head) && (ate_food || Some(new_head) != tail_pos);

            let is_collision = is_self_collision
                || if collision_settings.snake_vs_snake {
                    grid_map.is_collision(new_head.x, new_head.y, snake.id)
                } else {
                    grid_map.is_wall_collision(new_head.x, new_head.y)
                };

            let is_timeout = snake.steps_without_food
                > config.calculate_timeout_bfs(snake.snake.len(), snake.food_real_distance);

            if !is_collision && !is_timeout {
                let progress_ratio = if snake.food_real_distance > 0 {
                    let par = snake.food_real_distance as f32 * 2.0;
                    ((par - snake.steps_without_food as f32) / par).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                snake.path_progress_sum += progress_ratio;
                snake.steps_without_food += 1;
                snake.frames_survived += 1;
                snake.visited_cells.insert((new_head.x, new_head.y));

                let obstacle_proximity = current_17[0..8].iter().sum::<f32>() / 8.0;
                snake.obstacle_adjacency_sum += obstacle_proximity;
            }

            if is_collision || is_timeout {
                snake.is_game_over = true;
            } else {
                snake.snake.push_front(new_head);
                snake.body_set.insert(new_head);
                if ate_food {
                    if snake.food_real_distance > 0 && snake.steps_without_food > 0 {
                        let efficiency = (snake.food_real_distance as f32
                            / snake.steps_without_food as f32)
                            .clamp(0.0, 1.0);
                        snake.path_directness_sum += efficiency;
                    }
                    snake.score += 1;
                    game_stats.total_food_eaten += 1;
                    if snake.score > new_high_score {
                        new_high_score = snake.score;
                    }

                    let tail = snake.snake.back().copied();
                    let zone_center = Some((food_spawn_zone.center.x, food_spawn_zone.center.y));
                    let new_food = gen_seed.food_at_free(
                        snake.score as usize,
                        &snake.body_set,
                        &grid_map.terrain,
                        grid.width,
                        zone_center,
                        food_spawn_zone.radius,
                    );

                    let real_dist = bfs_distance(
                        new_head,
                        new_food,
                        &snake.body_set,
                        tail,
                        &grid_map.terrain,
                        grid.width,
                        grid.height,
                    );

                    match real_dist {
                        None => {
                            snake.is_game_over = true;
                        }
                        Some(dist) => {
                            snake.food = new_food;
                            snake.food_real_distance = dist;
                            snake.steps_without_food = 0;
                        }
                    }
                } else {
                    let tail = *snake.snake.back().unwrap();
                    snake.body_set.remove(&tail);
                    snake.snake.pop_back();
                }
            }
        }
        game.high_score = new_high_score;

        let dead_snakes: Vec<usize> = game
            .snakes
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_game_over)
            .map(|(i, _)| i)
            .collect();

        if continuous_mode.enabled && !dead_snakes.is_empty() {
            for idx in dead_snakes {
                let snake = &game.snakes[idx];

                let mut individual_clone: Option<Individual> = None;
                if let Some(ind) = evo_manager.get_individual_mut(idx) {
                    ind.fitness = snake.fitness(&grid);
                    ind.desc_path_efficiency = snake.path_efficiency();
                    ind.desc_danger_affinity = snake.danger_affinity();
                    ind.desc_spatial_spread = snake.spatial_spread();
                    ind.frames_survived = snake.frames_survived;
                    ind.apples_eaten = snake.score;
                    ind.is_alive = false;
                    individual_clone = Some(ind.clone());
                }

                if let Some(ref ind) = individual_clone {
                    if ind.fitness > 0.0 {
                        evo_manager.archive.insert(ind.clone());
                    }
                }

                continuous_mode.replacement_count += 1;
                continuous_mode.replacements_since_seed += 1;
                game_stats.total_games_played += 1;

                let new_individual = evo_manager.generate_single_individual(idx);
                let new_brain = Arc::new(new_individual.brain.clone());
                let archive_color = new_individual.archive_color;
                let new_fitness = new_individual.fitness;

                if let Some(ind) = evo_manager.get_individual_mut(idx) {
                    *ind = new_individual;
                }

                population.0[idx] = new_brain;

                let best_fitness = evo_manager.archive.best_fitness.max(1.0);
                let total_snakes = game.snakes.len();
                let snake = &mut game.snakes[idx];
                snake.reset_with_seed(
                    &grid,
                    total_snakes,
                    &gen_seed,
                    new_fitness.max(0.0),
                    new_fitness.max(0.0),
                    new_fitness,
                    best_fitness,
                );
                snake.color = archive_color.to_bevy_color();
            }

            if continuous_mode.replacement_count % 50 == 0 {
                println!(
                    "🔄 Continuous: {:>4} replacements | Coverage: {:.1}% | Gen: {}",
                    continuous_mode.replacement_count,
                    evo_manager.archive.coverage() * 100.0,
                    evo_manager.generation_state.generation
                );
            }
        }

        if !continuous_mode.enabled && game.alive_count() == 0 {
            game_stats.total_games_played += game.snakes.len() as u64;
            end_generation(&mut game, &mut evo_manager, &mut global_history, &grid);

            let new_seed = GenerationSeed::new_for_grid_with_config(&grid, &config);
            grid_map.apply_terrain(&new_seed.terrain);
            cell_map.terrain_dirty = true;
            *gen_seed = new_seed;

            let total_snakes = game.snakes.len();
            let individuals = evo_manager.get_population();
            let best_fitness = evo_manager.archive.best_fitness.max(1.0);

            for (i, snake) in game.snakes.iter_mut().enumerate() {
                let (courage, agility, fitness, best) = individuals
                    .get(i)
                    .map(|ind| {
                        (
                            ind.desc_path_efficiency,
                            ind.desc_danger_affinity,
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

            let new_pop = evo_manager.get_population();
            population.0.clear();
            for ind in new_pop.iter() {
                population.0.push(Arc::new(ind.brain.clone()));
            }

            game.total_iterations += 1;
        }
    }

    computed.0 = Vec::new();
}

fn end_generation(
    game: &mut GameState,
    evo_manager: &mut EvolutionManager,
    global_history: &mut GlobalTrainingHistory,
    grid: &GridDimensions,
) {
    for (i, snake) in game.snakes.iter().enumerate() {
        if let Some(ind) = evo_manager.get_individual_mut(i) {
            ind.fitness = snake.fitness(grid);
            ind.desc_path_efficiency = snake.path_efficiency();
            ind.desc_danger_affinity = snake.danger_affinity();
            ind.desc_spatial_spread = snake.spatial_spread();
            ind.frames_survived = snake.frames_survived;
            ind.apples_eaten = snake.score;
            ind.is_alive = false;
        }
    }

    let gen_high_score = game.snakes.iter().map(|s| s.score).max().unwrap_or(0);
    let mut record = evo_manager.end_generation();
    record.generation_high_score = gen_high_score;

    if gen_high_score > global_history.all_time_high_score {
        global_history.all_time_high_score = gen_high_score;
    }

    global_history.push(record.clone());

    if global_history.current_session.len() > 10_000 {
        global_history.current_session.drain(0..5_000);
    }

    println!(
        "Gen {:4} | Fitness: {:.0} (best: {:.0}) | Coverage: {:.1}% | {:.2}s",
        record.generation,
        record.avg_fitness,
        record.best_fitness,
        record.archive_coverage * 100.0,
        record.elapsed_secs
    );

    evo_manager.start_generation();
}
