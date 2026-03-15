//! UI systems for MAP-Elites Snake

use bevy::app::AppExit;
use bevy::prelude::*;
use bevy::sprite::MaterialMesh2dBundle;
use std::collections::HashMap;

use crate::plugins::map_elites::evolution::EvolutionManager;
use crate::snake::{
    AppStartTime, CollisionSettings, ContinuousMode, GameState, GameStats, GenerationSeed,
    GlobalTrainingHistory, GridDimensions, GridMap, MeshCache, RenderConfig, TrainingStats,
    BLOCK_SIZE,
};

/// UI Component markers
#[derive(Component)]
pub struct StatsText;

#[derive(Resource)]
#[allow(dead_code)]
pub struct GraphPanelState {
    pub visible: bool,
    pub collapsed: bool,
    pub fullscreen: bool,
    pub position: Vec2,
    pub size: Vec2,
    pub is_dragging: bool,
    pub drag_offset: Vec2,
    pub is_resizing: bool,
    pub resize_start_pos: Vec2,
    pub resize_start_size: Vec2,
    pub needs_redraw: bool,
    pub last_entry_count: usize,
}

#[derive(Resource)]
#[allow(dead_code)]
pub struct HeatmapPanelState {
    pub visible: bool,
    pub position: Vec2,
    pub size: Vec2,
    pub needs_redraw: bool,
    pub last_archive_gen: u32, // Track archive generation changes
}

impl Default for HeatmapPanelState {
    fn default() -> Self {
        Self {
            visible: false,
            position: Vec2::new(100.0, 100.0),
            size: Vec2::new(420.0, 450.0), // Slightly larger for labels
            needs_redraw: true,
            last_archive_gen: 0,
        }
    }
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

#[derive(Resource)]
pub struct WindowSettings {
    pub is_fullscreen: bool,
}

/// Pause state for simulation
#[derive(Resource, Default)]
pub struct PauseState {
    pub paused: bool,
}

/// Duration in seconds to ignore resize events after startup.
/// Hyprland/Wayland sends 2-3 automatic resize events during window placement.
const STARTUP_GRACE_PERIOD_SECS: f64 = 2.5;

/// Debounce for window resize events
#[derive(Resource)]
pub struct ResizeDebounce {
    pub pending: Option<(f32, f32)>, // (width, height) waiting
    pub last_event_time: std::time::Instant,
    pub startup_time: std::time::Instant, // when app started
    pub post_startup_sync_done: bool,     // force one resize after grace period
}

impl Default for ResizeDebounce {
    fn default() -> Self {
        Self {
            pending: None,
            last_event_time: std::time::Instant::now(),
            startup_time: std::time::Instant::now(),
            post_startup_sync_done: false,
        }
    }
}

/// Graph panel components
#[allow(dead_code)]
#[derive(Component)]
pub struct GraphPanel;

#[allow(dead_code)]
#[derive(Component)]
pub struct GraphPanelHeader;

#[allow(dead_code)]
#[derive(Component)]
pub struct GraphPanelContent;

#[allow(dead_code)]
#[derive(Component)]
pub struct GraphCloseButton;

#[allow(dead_code)]
#[derive(Component)]
pub struct GraphCollapseButton;

#[allow(dead_code)]
#[derive(Component)]
pub struct GraphResizeHandle;

/// Heatmap panel components
#[allow(dead_code)]
#[derive(Component)]
pub struct HeatmapPanel;

#[allow(dead_code)]
#[derive(Component)]
pub struct HeatmapGrid;

/// Material cache to avoid creating duplicate ColorMaterial assets
#[allow(dead_code)]
#[derive(Resource, Default)]
pub struct MaterialCache {
    pub cache: HashMap<[u8; 3], Handle<ColorMaterial>>,
}

/// Fixed material palette to prevent unbounded material allocations
/// Pre-created at startup - colors are quantized to nearest palette entry
#[derive(Resource)]
pub struct MaterialPalette {
    pub handles: Vec<Handle<ColorMaterial>>,
    #[allow(dead_code)]
    pub colors: Vec<[u8; 3]>,
    /// O(1) lookup table: 8×8×8 = 512 entries, index = ri*64 + gi*8 + bi
    pub lookup: Vec<usize>,
}

/// Cell-based render map: one entity per grid cell, pre-spawned.
/// Each frame only the highest-fitness snake occupying each cell is displayed.
/// Entity index = y * grid_width + x
#[derive(Resource)]
pub struct CellRenderMap {
    /// For each cell: Option<(color, fitness_of_best_snake_here)>
    /// Indexed by y * grid_width + x. Rebuilt from scratch every frame before rendering.
    pub cells: Vec<Option<(Color, f32)>>,
    /// Quantized color from previous frame for delta tracking (skip unchanged cells)
    pub prev_colors: Vec<Option<[u8; 3]>>,
    /// Pre-spawned Bevy entities — one per grid cell, indexed y*w+x
    pub entities: Vec<Entity>,
    pub grid_width: i32,
    pub grid_height: i32,
    /// True for exactly 1 frame after entity respawn (Bevy deferred commands not yet applied)
    pub rebuilding: bool,
    /// True only when terrain changes (new generation or resize) — avoids re-inserting walls every frame
    pub terrain_dirty: bool,
}

/// Visibility state for all floating panels
#[allow(dead_code)]
#[derive(Resource)]
pub struct PanelVisibility {
    pub inspector: bool,   // [I] Brain inspector panel
    pub graph: bool,       // [G] Fitness history graph
    pub heatmap: bool,     // [B] Map elites heatmap
    pub leaderboard: bool, // [L] Snake rankings
    pub keybindings: bool, // [K] Keybindings help
}

impl Default for PanelVisibility {
    fn default() -> Self {
        Self {
            inspector: true, // Inspector visible by default
            graph: true,     // Graph visible by default
            heatmap: false,
            leaderboard: true, // Leaderboard visible by default
            keybindings: false,
        }
    }
}

impl CellRenderMap {
    pub fn new(grid_width: i32, grid_height: i32) -> Self {
        let size = (grid_width * grid_height) as usize;
        Self {
            cells: vec![None; size],
            prev_colors: vec![None; size],
            entities: Vec::new(),
            grid_width,
            grid_height,
            rebuilding: false,
            terrain_dirty: true,
        }
    }

    /// Get flat index for grid position (x, y), with bounds check
    pub fn cell_index(&self, x: i32, y: i32) -> Option<usize> {
        if x < 0 || x >= self.grid_width || y < 0 || y >= self.grid_height {
            return None;
        }
        Some((y * self.grid_width + x) as usize)
    }
}

/// Food entity pool - one pre-spawned food entity per snake
#[derive(Resource, Default)]
pub struct FoodPool {
    pub entities: Vec<Entity>,
}

/// Timer to limit UI updates to ~10Hz
#[derive(Resource)]
pub struct UiUpdateTimer(pub Timer);

impl Default for UiUpdateTimer {
    fn default() -> Self {
        Self(Timer::from_seconds(0.1, TimerMode::Repeating))
    }
}

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(WindowSettings {
            is_fullscreen: false,
        })
        .insert_resource(PauseState::default())
        .insert_resource(ResizeDebounce::default())
        .insert_resource(GraphPanelState::default())
        .insert_resource(HeatmapPanelState::default())
        .insert_resource(MaterialCache::default())
        .insert_resource(FoodPool::default())
        .insert_resource(UiUpdateTimer::default())
        .insert_resource(CellRenderMap::new(0, 0)) // placeholder, re-created in setup
        .insert_resource(PanelVisibility::default()) // Floating panel visibility
        // MaterialPalette is created in main.rs setup (needs access to materials asset)
        .add_systems(Update, handle_input)
        .add_systems(Update, handle_continuous_mode_input)
        .add_systems(Update, on_window_resize_collect)
        .add_systems(
            Update,
            on_window_resize_apply.after(on_window_resize_collect),
        )
        .add_systems(Update, render_system.after(on_window_resize_apply))
        .add_systems(Update, update_stats_ui);
    }
}

pub fn spawn_stats_ui(mut commands: Commands, _game: Res<GameState>) {
    // Only FPS + steps in bottom right — everything else is in inspector panel
    commands.spawn((
        TextBundle::from_section(
            "FPS: 0 | Steps: 1 | BATCH",
            TextStyle {
                font_size: 13.0,
                color: Color::rgba(0.6, 0.6, 0.6, 0.8),
                ..default()
            },
        )
        .with_style(Style {
            position_type: PositionType::Absolute,
            bottom: Val::Px(8.0),
            right: Val::Px(10.0),
            ..default()
        }),
        StatsText,
    ));
}

#[allow(clippy::too_many_arguments)]
pub fn update_stats_ui(
    mut stats_query: Query<&mut Text, With<StatsText>>,
    _stats: Res<TrainingStats>,
    mut ui_timer: ResMut<UiUpdateTimer>,
    time: Res<Time>,
    sim_steps: Res<crate::snake::SimStepsPerFrame>,
    continuous_mode: Res<crate::snake::ContinuousMode>,
    tuner: Res<crate::snake::PerformanceTuner>,
    hyperparams: Res<crate::config::Hyperparameters>,
) {
    // Limit UI updates to ~10Hz
    ui_timer.0.tick(time.delta());
    if !ui_timer.0.just_finished() {
        return;
    }

    if let Ok(mut text) = stats_query.get_single_mut() {
        let mode_str = if continuous_mode.enabled {
            format!("CONTINUOUS | Repl: {}", continuous_mode.replacement_count)
        } else {
            "BATCH".to_string()
        };

        // Tuner status line
        let tuner_str = if tuner.enabled {
            format!(
                "[AUTO] EMA:{:.1}ms | Tgt:{:.0}ms | Steps:{} | Pop:{}",
                tuner.ema_frame_ms, tuner.target_frame_ms, sim_steps.0, hyperparams.population_size
            )
        } else {
            format!(
                "[MANUAL] Steps:{} | Pop:{}",
                sim_steps.0, hyperparams.population_size
            )
        };

        text.sections[0].value = format!("{} | {}", tuner_str, mode_str);
    }
}

// In src/ui.rs

pub fn handle_input(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut app_exit_events: EventWriter<AppExit>,
    game: Res<GameState>,
    app_start_time: Res<AppStartTime>,
    global_history: Res<GlobalTrainingHistory>,
    game_stats: Res<GameStats>,
    evo_manager: ResMut<EvolutionManager>,
    mut window_settings: ResMut<WindowSettings>,
    mut windows: Query<&mut Window>,
    mut collision_settings: ResMut<CollisionSettings>,
    mut render_config: ResMut<RenderConfig>,
    mut graph_state: ResMut<GraphPanelState>,
    _heatmap_state: ResMut<HeatmapPanelState>,
    mut pause_state: ResMut<PauseState>,
    // AGGIUNTA: Recuperiamo la directory della run corrente
    run_dir: Res<crate::snake::RunDirectory>,
    mut sim_steps: ResMut<crate::snake::SimStepsPerFrame>,
) {
    use crate::snake::{new_session_path, save_training_session}; // Rimosso get_or_create_run_dir non più necessario qui
    use std::time::Instant;

    if keyboard_input.just_pressed(KeyCode::Escape) {
        let current_session_duration = Instant::now().duration_since(app_start_time.0);
        let total_training_time =
            std::time::Duration::from_secs(global_history.accumulated_time_secs)
                + current_session_duration;

        println!("\n=== SESSION SUMMARY ===");
        println!("Total generations: {}", game.total_iterations);
        println!("High Score: {}", game.high_score);
        println!(
            "Current session time: {}s",
            current_session_duration.as_secs()
        );
        println!("Total time (runtime): {}s", total_training_time.as_secs());

        // Force save the archive before exit
        evo_manager.save_archive();
        println!("💾 Archive saved on exit");

        // Save session data
        // CORREZIONE 1: Passiamo il path contenuto nella risorsa RunDirectory
        let session_path = new_session_path(&run_dir.0);

        if let Err(e) = save_training_session(
            &session_path,
            &global_history,
            &game_stats,
            current_session_duration.as_secs(),
        ) {
            eprintln!("⚠️ Error saving session: {}", e);
        }

        // CORREZIONE 2: Usiamo direttamente il path della risorsa invece di ricalcolarlo
        println!("Saved to: {}", run_dir.0.display());
        println!("====================\n");

        app_exit_events.send(AppExit);
    }

    // ... (il resto della funzione rimane identico: tasti C, R, G, B, P, F) ...
    if keyboard_input.just_pressed(KeyCode::KeyC) {
        collision_settings.snake_vs_snake = !collision_settings.snake_vs_snake;
        println!(
            "Snake-vs-snake collisions: {}",
            if collision_settings.snake_vs_snake {
                "ON"
            } else {
                "OFF"
            }
        );
    }

    if keyboard_input.just_pressed(KeyCode::KeyR) {
        render_config.enabled = !render_config.enabled;
        println!(
            "Rendering: {}",
            if render_config.enabled {
                "ON (Normal)"
            } else {
                "OFF (Turbo)"
            }
        );
    }

    if keyboard_input.just_pressed(KeyCode::KeyP) {
        pause_state.paused = !pause_state.paused;
        println!(
            "{}",
            if pause_state.paused {
                "PAUSED"
            } else {
                "RESUMED"
            }
        );
    }

    if keyboard_input.just_pressed(KeyCode::KeyF) {
        window_settings.is_fullscreen = !window_settings.is_fullscreen;
        let mut window = windows.single_mut();
        window.mode = if window_settings.is_fullscreen {
            bevy::window::WindowMode::Fullscreen
        } else {
            bevy::window::WindowMode::Windowed
        };
        if window_settings.is_fullscreen {
            graph_state.visible = true;
            graph_state.needs_redraw = true;
        }
    }

    if keyboard_input.just_pressed(KeyCode::BracketRight) {
        sim_steps.0 = (sim_steps.0 + 1).min(200);
        println!("Steps/frame: {}", sim_steps.0);
    }
    if keyboard_input.just_pressed(KeyCode::BracketLeft) {
        sim_steps.0 = sim_steps.0.saturating_sub(1).max(1);
        println!("Steps/frame: {}", sim_steps.0);
    }
}

/// Task 04: Sistema separato per continuous mode e rigenerazione seed
/// (handle_input ha troppi parametri per Bevy ECS, quindi usiamo un sistema separato)
pub fn handle_continuous_mode_input(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut continuous_mode: ResMut<ContinuousMode>,
    grid: Res<GridDimensions>,
    config: Res<crate::config::Hyperparameters>,
    mut game: ResMut<GameState>,
    evo_manager: ResMut<EvolutionManager>,
    mut grid_map: ResMut<GridMap>,
    mut gen_seed: ResMut<GenerationSeed>,
    mut cell_map: ResMut<CellRenderMap>,
) {
    // Toggle continuous mode con tasto 'O' (On/Off)
    if keyboard_input.just_pressed(KeyCode::KeyO) {
        continuous_mode.enabled = !continuous_mode.enabled;
        println!(
            "Continuous Mode: {} (replacements: {})",
            if continuous_mode.enabled { "ON" } else { "OFF" },
            continuous_mode.replacement_count
        );
    }

    // Rigenera seed manualmente con tasto 'N' (N come "New seed")
    if keyboard_input.just_pressed(KeyCode::KeyN) {
        // Crea nuovo seed
        let new_seed = GenerationSeed::new_for_grid_with_config(&grid, &config);

        // Applica i muri alla mappa
        grid_map.apply_terrain(&new_seed.terrain);
        cell_map.terrain_dirty = true;

        // Resetta tutti gli snake con il nuovo seed
        let individuals = evo_manager.get_population();
        let best_fitness = evo_manager.archive.best_fitness.max(1.0);
        let total_snakes = game.snakes.len();

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
                &new_seed,
                courage,
                agility,
                fitness,
                best,
            );
            if let Some(ind) = individuals.get(i) {
                snake.color = ind.archive_color.to_bevy_color();
            }
        }

        // Aggiorna il seed
        *gen_seed = new_seed;

        // Resetta contatore replacements since seed
        continuous_mode.replacements_since_seed = 0;

        println!(
            "🔄 NEW SEED: {} replacements since last seed",
            continuous_mode.replacement_count
        );
    }
}

/// Collect resize events without applying immediately (debounce)
pub fn on_window_resize_collect(
    mut resize_events: EventReader<bevy::window::WindowResized>,
    mut debounce: ResMut<ResizeDebounce>,
    windows: Query<&Window>,
) {
    // Discard resize events during startup grace period
    if debounce.startup_time.elapsed().as_secs_f64() < STARTUP_GRACE_PERIOD_SECS {
        // After grace period ends, do one forced sync if not yet done
        if !debounce.post_startup_sync_done {
            debounce.post_startup_sync_done = true;
            if let Ok(window) = windows.get_single() {
                debounce.pending = Some((window.width(), window.height()));
                debounce.last_event_time =
                    std::time::Instant::now() - std::time::Duration::from_millis(600);
                // already past debounce threshold
            }
        }
        return;
    }

    for event in resize_events.read() {
        debounce.pending = Some((event.width, event.height));
        debounce.last_event_time = std::time::Instant::now();
    }
}

/// Apply resize after 500ms debounce
pub fn on_window_resize_apply(
    mut debounce: ResMut<ResizeDebounce>,
    mut grid: ResMut<GridDimensions>,
    mut game: ResMut<GameState>,
    mut grid_map: ResMut<GridMap>,
    mut graph_state: ResMut<GraphPanelState>,
    mut cell_map: ResMut<CellRenderMap>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mesh_cache: Res<MeshCache>,
    config: Option<Res<crate::config::Hyperparameters>>,
    evo_manager: Res<EvolutionManager>,
    mut commands: Commands,
    mut food_pool: ResMut<FoodPool>,
) {
    // Ignore all resizes during Wayland/Hyprland startup window placement
    if debounce.startup_time.elapsed().as_secs_f64() < STARTUP_GRACE_PERIOD_SECS {
        debounce.pending = None; // discard any pending resize
        return;
    }

    let Some((w, h)) = debounce.pending else {
        return;
    };

    let elapsed = debounce.last_event_time.elapsed();
    if elapsed.as_millis() < 500 {
        return;
    }

    // Apply resize
    debounce.pending = None;

    let (new_width, new_height) = crate::snake::calculate_grid_dimensions(w, h);
    grid.width = new_width;
    grid.height = new_height;
    *grid_map = GridMap::new(new_width, new_height);

    // Despawn old cell entities and re-create for new grid size
    // Only despawn if entity is valid - skip silently if already despawned
    for &entity in cell_map.entities.iter() {
        if entity != Entity::from_raw(u32::MAX) {
            commands.entity(entity).despawn();
        }
    }
    let cell_count = (new_width * new_height) as usize;
    let default_material = materials.add(Color::rgb(0.05, 0.05, 0.05));
    let mut new_entities = Vec::with_capacity(cell_count);
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
        new_entities.push(entity);
    }
    cell_map.entities = new_entities;
    cell_map.cells = vec![None; (new_width * new_height) as usize];
    cell_map.prev_colors = vec![None; (new_width * new_height) as usize];
    cell_map.grid_width = new_width;
    cell_map.grid_height = new_height;
    cell_map.rebuilding = true; // Skip next render frame, entities not yet in World
    cell_map.terrain_dirty = true;

    // Sync FoodPool with snake count - despawn excess or create new entities
    let snake_count = game.snakes.len();
    while food_pool.entities.len() > snake_count {
        if let Some(entity) = food_pool.entities.pop() {
            commands.entity(entity).despawn();
        }
    }
    while food_pool.entities.len() < snake_count {
        let entity = commands
            .spawn((
                MaterialMesh2dBundle {
                    mesh: mesh_cache.food_mesh.clone().into(),
                    material: mesh_cache.food_material.clone(),
                    transform: Transform::from_xyz(0.0, 0.0, 0.0),
                    visibility: Visibility::Hidden,
                    ..default()
                },
                crate::snake::Food,
                crate::snake::SnakeId(food_pool.entities.len()),
            ))
            .id();
        food_pool.entities.push(entity);
    }

    // Regenerate seed for new grid, preserving user config if available
    let new_seed = if let Some(ref cfg) = config {
        crate::snake::GenerationSeed::new_for_grid_with_config(&grid, cfg)
    } else {
        crate::snake::GenerationSeed::new_for_grid(&grid)
    };
    grid_map.apply_terrain(&new_seed.terrain); // apply terrain to resized map
    let total_snakes = game.snakes.len();

    // Get population for color assignment
    let population = &evo_manager.generation_state.population;

    for (idx, snake) in game.snakes.iter_mut().enumerate() {
        snake.reset_with_seed(&grid, total_snakes, &new_seed, 0.0, 0.0, 0.0, 1.0);
        // Set color from population archive_color (same logic as initial spawn)
        if let Some(ind) = population.get(idx) {
            snake.color = ind.archive_color.to_bevy_color();
        }
    }
    commands.insert_resource(new_seed);
    graph_state.needs_redraw = true;

    println!(
        "Resized: GridMap re-initialized to {}x{}",
        new_width, new_height
    );
}

/// Rendering system - cell-based rendering (replaces per-segment entity approach)
pub fn render_system(
    mut commands: Commands,
    game: Res<GameState>,
    grid: Res<GridDimensions>,
    grid_map: Res<GridMap>, // terrain walls
    windows: Query<&Window>,
    _mesh_cache: Res<MeshCache>,
    _materials: ResMut<Assets<ColorMaterial>>, // Unused - palette pre-allocates all materials
    mat_palette: Res<MaterialPalette>,         // Fixed palette - no allocations
    food_pool: Res<FoodPool>,
    render_config: Res<RenderConfig>,
    mut stats: ResMut<TrainingStats>,
    mut cell_map: ResMut<CellRenderMap>,
    evo_manager: Res<EvolutionManager>,
    panel_visibility: Res<PanelVisibility>,
    inspected_agent: Res<crate::plugins::brain_inspector::InspectedAgent>,
) {
    #[cfg(feature = "tracy")]
    let _span = tracing::info_span!("render_system").entered();

    // Skip 1 frame after entity rebuild — Bevy deferred commands not yet applied
    if cell_map.rebuilding {
        cell_map.rebuilding = false;
        return;
    }

    if !render_config.enabled {
        // Hide everything when turbo mode is on
        for snake in game.snakes.iter() {
            if let Some(&food_entity) = food_pool.entities.get(snake.id) {
                commands.entity(food_entity).insert(Visibility::Hidden);
            }
        }
        return;
    }

    // FPS counter
    stats.frame_count += 1;
    let now = std::time::Instant::now();
    if now.duration_since(stats.last_fps_update).as_secs_f32() >= 1.0 {
        stats.fps =
            stats.frame_count as f32 / now.duration_since(stats.last_fps_update).as_secs_f32();
        stats.last_fps_update = now;
        stats.frame_count = 0;
    }

    let Ok(window) = windows.get_single() else {
        return;
    };

    // Usiamo tutta l'altezza della finestra
    let available_height = window.resolution.height();

    // Calcola la dimensione totale in pixel della griglia
    let grid_px_w = cell_map.grid_width as f32 * BLOCK_SIZE;
    let grid_px_h = cell_map.grid_height as f32 * BLOCK_SIZE;

    // Calcola lo spazio rimanente per centrare
    let leftover_x = window.resolution.width() - grid_px_w;
    let leftover_y = available_height - grid_px_h;

    // Centra la griglia dividendo lo spazio rimanente equamente
    let offset_x = -window.resolution.width() / 2.0 + (leftover_x / 2.0) + BLOCK_SIZE / 2.0;
    let offset_y = available_height / 2.0 - (leftover_y / 2.0) - BLOCK_SIZE / 2.0;

    // Inspector mode when panel is visible and an agent is selected
    let is_inspector_view = panel_visibility.inspector && inspected_agent.snake_idx.is_some();
    let selected_snake_id = inspected_agent.snake_idx;

    // In inspector mode: only show food for selected snake, hide all others
    if is_inspector_view {
        for snake in game.snakes.iter() {
            if let Some(&food_entity) = food_pool.entities.get(snake.id) {
                let show_food = selected_snake_id == Some(snake.id);
                commands.entity(food_entity).insert(if show_food {
                    Visibility::Inherited
                } else {
                    Visibility::Hidden
                });
            }
        }
    }

    // === PHASE 0: Render terrain walls (only when terrain changes) ===
    const WALL_COLOR: Color = Color::rgb(0.50, 0.22, 0.15);
    // Dimmed wall color for inspector view (semi-transparent effect via darker color)
    const WALL_COLOR_DIM: Color = Color::rgb(0.12, 0.06, 0.04);

    // Clear dynamic cells (snakes) but preserve terrain via dirty flag
    cell_map.cells.fill(None);

    let terrain_wall_color = if is_inspector_view {
        WALL_COLOR_DIM
    } else {
        WALL_COLOR
    };

    if cell_map.terrain_dirty {
        for y in 0..grid_map.height {
            for x in 0..grid_map.width {
                let idx = (y * grid_map.width + x) as usize;
                if grid_map.terrain[idx] {
                    if let Some(cidx) = cell_map.cell_index(x, y) {
                        cell_map.cells[cidx] = Some((terrain_wall_color, f32::INFINITY));
                    }
                }
            }
        }
        cell_map.terrain_dirty = false;
    } else {
        // Re-insert terrain walls from grid_map (they were cleared by fill(None))
        for y in 0..grid_map.height {
            for x in 0..grid_map.width {
                let idx = (y * grid_map.width + x) as usize;
                if grid_map.terrain[idx] {
                    if let Some(cidx) = cell_map.cell_index(x, y) {
                        cell_map.cells[cidx] = Some((terrain_wall_color, f32::INFINITY));
                    }
                }
            }
        }
    }

    // === PHASE 1: Build cell color map ===
    // For each occupied cell, keep only the color of the highest-fitness snake.
    // In inspector view: ONLY render selected snake, hide all others completely.

    // Use HISTORICAL best fitness from archive for normalization (not current generation max)
    // This ensures consistent color mapping across all time
    let max_fitness = evo_manager.archive.best_fitness.max(1.0);

    for snake in game.snakes.iter() {
        if snake.is_game_over {
            continue;
        }

        // In inspector mode, skip snakes that are not selected
        if is_inspector_view && selected_snake_id != Some(snake.id) {
            continue;
        }

        // Get REAL-TIME fitness directly from snake
        let snake_fitness = snake.fitness(&grid);

        // Calculate color directly from fitness (normalized)
        // Low fitness = blue (0,0,1), High fitness = green (0,1,0)
        let normalized = (snake_fitness / max_fitness).clamp(0.0, 1.0);
        let fitness_color = Color::rgb(0.0, normalized, 1.0 - normalized);

        for (seg_idx, pos) in snake.snake.iter().enumerate() {
            let Some(cidx) = cell_map.cell_index(pos.x, pos.y) else {
                continue;
            };
            // Head is always white; body uses fitness-based color
            let color = if seg_idx == 0 {
                Color::rgb(1.0, 1.0, 1.0)
            } else {
                fitness_color
            };
            // Only update if this snake has higher fitness than the current winner
            // Use < instead of <= to ensure deterministic behavior (first snake wins on tie)
            match cell_map.cells[cidx] {
                Some((_, existing_fitness)) if snake_fitness < existing_fitness => {}
                _ => {
                    cell_map.cells[cidx] = Some((color, snake_fitness));
                }
            }
        }
    }

    // === PHASE 2: Update cell entities with delta color tracking ===
    #[inline]
    fn quantize_to_palette_index(color: Color, palette: &MaterialPalette) -> usize {
        let r = (color.r() * 255.0) as usize;
        let g = (color.g() * 255.0) as usize;
        let b = (color.b() * 255.0) as usize;
        let ri = (r * 7 + 127) / 255;
        let gi = (g * 7 + 127) / 255;
        let bi = (b * 7 + 127) / 255;
        palette.lookup[ri * 64 + gi * 8 + bi]
    }

    let cell_count = cell_map.cells.len();
    let gw = cell_map.grid_width;
    for idx in 0..cell_count {
        let cell = cell_map.cells[idx];
        let Some((color, _)) = cell else {
            // Cell now empty: hide only if it was visible last frame
            if cell_map.prev_colors[idx].is_some() {
                if let Some(&entity) = cell_map.entities.get(idx) {
                    commands.entity(entity).insert(Visibility::Hidden);
                }
                cell_map.prev_colors[idx] = None;
            }
            continue;
        };

        let r = (color.r() * 255.0) as u8;
        let g = (color.g() * 255.0) as u8;
        let b = (color.b() * 255.0) as u8;
        let color_key = [r, g, b];

        // Emit command ONLY if the color changed from previous frame
        if cell_map.prev_colors[idx] == Some(color_key) {
            continue;
        }

        let x = (idx as i32) % gw;
        let y = (idx as i32) / gw;
        let Some(&entity) = cell_map.entities.get(idx) else {
            continue;
        };

        let pal_idx = quantize_to_palette_index(color, &mat_palette);
        let material = mat_palette.handles[pal_idx].clone();
        let transform = Transform::from_xyz(
            offset_x + x as f32 * BLOCK_SIZE,
            offset_y - y as f32 * BLOCK_SIZE,
            0.0,
        );
        commands
            .entity(entity)
            .insert((material, transform, Visibility::Visible));
        cell_map.prev_colors[idx] = Some(color_key);
    }

    // === PHASE 3: Update food entities ===
    // In inspector view: only show food for the selected snake.
    for snake in game.snakes.iter() {
        let Some(&food_entity) = food_pool.entities.get(snake.id) else {
            continue;
        };

        let hide_food =
            snake.is_game_over || (is_inspector_view && selected_snake_id != Some(snake.id));

        if hide_food {
            commands.entity(food_entity).insert(Visibility::Hidden);
            continue;
        }

        let food_transform = Transform::from_xyz(
            offset_x + snake.food.x as f32 * BLOCK_SIZE,
            offset_y - snake.food.y as f32 * BLOCK_SIZE,
            1.0, // z=1.0 renders food above snake cells (z=0.0)
        );
        commands
            .entity(food_entity)
            .insert((food_transform, Visibility::Visible));
    }
}

#[allow(dead_code)]
pub fn update_graph_panel_visibility(
    mut commands: Commands,
    mut graph_state: ResMut<GraphPanelState>,
    panel_query: Query<Entity, With<GraphPanel>>,
) {
    let panel_exists = !panel_query.is_empty();
    let should_be_visible = graph_state.visible;

    if should_be_visible && !panel_exists {
        graph_state.needs_redraw = true;
        graph_state.last_entry_count = 0;
        spawn_graph_panel_internal(commands, &graph_state);
    } else if !should_be_visible && panel_exists {
        for entity in panel_query.iter() {
            commands.entity(entity).despawn_recursive();
        }
    }
}

#[allow(dead_code)]
fn spawn_graph_panel_internal(mut commands: Commands, graph_state: &GraphPanelState) {
    let header_height = 30.0;

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
                        "MAP-Elites Archive",
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

            if !graph_state.collapsed {
                parent.spawn((
                    NodeBundle {
                        style: Style {
                            width: Val::Percent(100.0),
                            height: Val::Px(graph_state.size.y - header_height),
                            overflow: Overflow::clip(),
                            ..default()
                        },
                        background_color: Color::rgba(0.05, 0.05, 0.05, 0.9).into(),
                        ..default()
                    },
                    GraphPanelContent,
                ));

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

#[allow(dead_code)]
fn spawn_heatmap_panel_internal(mut commands: Commands, heatmap_state: &HeatmapPanelState) {
    let header_height = 30.0;

    commands
        .spawn((
            NodeBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    left: Val::Px(heatmap_state.position.x),
                    top: Val::Px(heatmap_state.position.y),
                    width: Val::Px(heatmap_state.size.x),
                    height: Val::Px(heatmap_state.size.y),
                    flex_direction: FlexDirection::Column,
                    ..default()
                },
                background_color: Color::rgba(0.1, 0.1, 0.1, 0.95).into(),
                ..default()
            },
            HeatmapPanel,
        ))
        .with_children(|parent| {
            parent
                .spawn((NodeBundle {
                    style: Style {
                        width: Val::Percent(100.0),
                        height: Val::Px(header_height),
                        flex_direction: FlexDirection::Row,
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        padding: UiRect::horizontal(Val::Px(10.0)),
                        ..default()
                    },
                    background_color: Color::rgb(0.2, 0.2, 0.3).into(),
                    ..default()
                },))
                .with_children(|header| {
                    header.spawn(TextBundle::from_section(
                        "MAP-Elites Heatmap (Path Efficiency vs Danger Affinity)",
                        TextStyle {
                            font_size: 16.0,
                            color: Color::WHITE,
                            ..default()
                        },
                    ));
                });

            parent.spawn((
                NodeBundle {
                    style: Style {
                        width: Val::Percent(100.0),
                        height: Val::Px(heatmap_state.size.y - header_height),
                        overflow: Overflow::clip(),
                        ..default()
                    },
                    background_color: Color::rgba(0.05, 0.05, 0.05, 0.9).into(),
                    ..default()
                },
                HeatmapGrid,
            ));
        });
}

#[allow(dead_code)]
pub fn handle_graph_panel_interactions(
    mut graph_state: ResMut<GraphPanelState>,
    mouse_button: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    header_query: Query<&Interaction, (Changed<Interaction>, With<GraphPanelHeader>)>,
    collapse_query: Query<&Interaction, (Changed<Interaction>, With<GraphCollapseButton>)>,
    close_query: Query<&Interaction, (Changed<Interaction>, With<GraphCloseButton>)>,
    resize_query: Query<&Interaction, (Changed<Interaction>, With<GraphResizeHandle>)>,
) {
    let Ok(window) = windows.get_single() else {
        return;
    };
    let cursor_pos = window.cursor_position().unwrap_or(Vec2::ZERO);

    if graph_state.is_dragging {
        if mouse_button.just_released(MouseButton::Left) {
            graph_state.is_dragging = false;
        } else {
            let new_pos = cursor_pos - graph_state.drag_offset;
            graph_state.position.x = new_pos.x.clamp(0.0, window.width() - 50.0);
            graph_state.position.y = new_pos.y.clamp(0.0, window.height() - 50.0);
        }
        return;
    }

    if graph_state.is_resizing {
        if mouse_button.just_released(MouseButton::Left) {
            graph_state.is_resizing = false;
            graph_state.needs_redraw = true;
        } else {
            let mouse_delta = cursor_pos - graph_state.resize_start_pos;
            let new_size = graph_state.resize_start_size + mouse_delta;
            graph_state.size.x = new_size.x.max(300.0);
            graph_state.size.y = new_size.y.max(200.0);
            graph_state.needs_redraw = true;
        }
        return;
    }

    for interaction in header_query.iter() {
        if *interaction == Interaction::Pressed {
            graph_state.is_dragging = true;
            graph_state.drag_offset = cursor_pos - graph_state.position;
        }
    }

    for interaction in resize_query.iter() {
        if *interaction == Interaction::Pressed {
            graph_state.is_resizing = true;
            graph_state.resize_start_pos = cursor_pos;
            graph_state.resize_start_size = graph_state.size;
        }
    }

    for interaction in collapse_query.iter() {
        if *interaction == Interaction::Pressed {
            graph_state.collapsed = !graph_state.collapsed;
            graph_state.needs_redraw = true;
        }
    }

    for interaction in close_query.iter() {
        if *interaction == Interaction::Pressed {
            graph_state.visible = false;
        }
    }
}

#[allow(dead_code)]
pub fn draw_heatmap_in_panel(
    mut commands: Commands,
    mut heatmap_state: ResMut<HeatmapPanelState>,
    evo_manager: Res<EvolutionManager>,
    grid_query: Query<Entity, With<HeatmapGrid>>,
    children_query: Query<&Children>,
) {
    if !heatmap_state.visible {
        return;
    }

    // UPDATE TRIGGER:
    // Redraw if forced OR if archive generation has increased
    let archive_gen = evo_manager.archive.generation;
    if !heatmap_state.needs_redraw && archive_gen == heatmap_state.last_archive_gen {
        return;
    }

    // Clear old nodes
    for grid_entity in grid_query.iter() {
        if let Ok(children) = children_query.get(grid_entity) {
            for &child in children.iter() {
                commands.entity(child).despawn_recursive();
            }
        }
    }

    heatmap_state.needs_redraw = false;
    heatmap_state.last_archive_gen = archive_gen; // Sync generation

    for grid_entity in grid_query.iter() {
        let margin = 25.0;
        let grid_width = heatmap_state.size.x - margin * 2.0;
        let grid_height = heatmap_state.size.y - 40.0 - margin * 2.0;
        let res = crate::plugins::map_elites::GRID_RESOLUTION as f32;
        let cell_w = grid_width / res;
        let cell_h = grid_height / res;

        commands.entity(grid_entity).with_children(|parent| {
            let max_fitness = evo_manager.archive.best_fitness.max(1.0);
            // Use central slice (10 for GRID_RESOLUTION=20)
            let slice_z = 10;

            for x in 0..crate::plugins::map_elites::GRID_RESOLUTION {
                for y in 0..crate::plugins::map_elites::GRID_RESOLUTION {
                    let cell_opt = evo_manager.archive.grid.get(&(x, y, slice_z));

                    let cell_color = if let Some(ind) = cell_opt {
                        // Color based on fitness: blue (low) → green (high)
                        let intensity = (ind.fitness / max_fitness).clamp(0.0, 1.0) as f32;
                        Color::rgb(0.1, intensity, 1.0 - intensity)
                    } else {
                        Color::rgb(0.05, 0.05, 0.07) // Empty
                    };

                    // Y-AXIS CORRECTION:
                    // Bevy top:0 is at top. To have y=0 at bottom, we do (Res - 1 - y)
                    let display_y = (crate::plugins::map_elites::GRID_RESOLUTION - 1 - y) as f32;

                    parent.spawn(NodeBundle {
                        style: Style {
                            position_type: PositionType::Absolute,
                            left: Val::Px(margin + x as f32 * cell_w),
                            top: Val::Px(margin + display_y * cell_h),
                            width: Val::Px(cell_w - 1.0),
                            height: Val::Px(cell_h - 1.0),
                            ..default()
                        },
                        background_color: cell_color.into(),
                        ..default()
                    });
                }
            }

            // Axis Labels
            spawn_axis_label(
                parent,
                "Path Efficiency →",
                Val::Px(margin),
                Val::Px(grid_height + margin + 5.0),
            );
            spawn_axis_label(parent, "Danger Affinity ↑", Val::Px(5.0), Val::Px(margin));

            // Stats text
            let filled = evo_manager.archive.filled_cells();
            let total = evo_manager.archive.capacity();
            let coverage = (filled as f64 / total as f64) * 100.0; // Keep as f64 for decimal precision

            spawn_axis_label(
                parent,
                &format!(
                    "Gen:{} | Coverage: {:.2}% ({}/{})", // {:.2} to show 2 decimal places
                    archive_gen, coverage, filled, total
                ),
                Val::Px(margin),
                Val::Px(5.0),
            );
        });
    }
}

// Helper for axis labels
#[allow(dead_code)]
fn spawn_axis_label(parent: &mut ChildBuilder, text: &str, left: Val, top: Val) {
    parent.spawn(
        TextBundle::from_section(
            text,
            TextStyle {
                font_size: 12.0,
                color: Color::GRAY,
                ..default()
            },
        )
        .with_style(Style {
            position_type: PositionType::Absolute,
            left,
            top,
            ..default()
        }),
    );
}

#[allow(dead_code)]
pub fn update_heatmap_panel_visibility(
    mut commands: Commands,
    mut heatmap_state: ResMut<HeatmapPanelState>,
    panel_query: Query<Entity, With<HeatmapPanel>>,
) {
    let panel_exists = !panel_query.is_empty();
    let should_be_visible = heatmap_state.visible;

    if should_be_visible && !panel_exists {
        heatmap_state.needs_redraw = true;
        spawn_heatmap_panel_internal(commands, &heatmap_state);
    } else if !should_be_visible && panel_exists {
        for entity in panel_query.iter() {
            commands.entity(entity).despawn_recursive();
        }
    }
}

#[allow(dead_code)]
pub fn sync_graph_panel_layout(
    graph_state: Res<GraphPanelState>,
    mut panel_query: Query<&mut Style, With<GraphPanel>>,
) {
    if graph_state.is_changed() {
        for mut style in panel_query.iter_mut() {
            style.left = Val::Px(graph_state.position.x);
            style.top = Val::Px(graph_state.position.y);
            style.width = Val::Px(graph_state.size.x);

            if graph_state.collapsed {
                style.height = Val::Px(30.0);
            } else {
                style.height = Val::Px(graph_state.size.y);
            }
        }
    }
}

#[allow(dead_code)]
pub fn draw_graph_in_panel(
    mut commands: Commands,
    mut graph_state: ResMut<GraphPanelState>,
    global_history: Res<GlobalTrainingHistory>,
    content_query: Query<Entity, With<GraphPanelContent>>,
    children_query: Query<&Children>,
) {
    if !graph_state.visible || graph_state.collapsed {
        return;
    }

    let data_changed = global_history.all_records().count() != graph_state.last_entry_count;
    if !graph_state.needs_redraw && !data_changed && graph_state.last_entry_count != 0 {
        return;
    }

    for content_entity in content_query.iter() {
        if let Ok(children) = children_query.get(content_entity) {
            for &child in children.iter() {
                commands.entity(child).despawn_recursive();
            }
        }
    }

    graph_state.needs_redraw = false;
    graph_state.last_entry_count = global_history.all_records().count();

    if global_history.all_records().next().is_none() {
        return;
    }

    for content_entity in content_query.iter() {
        let margin_left = 40.0;
        let margin_bottom = 30.0;
        let margin_top = 20.0;
        let margin_right = 20.0;

        let graph_width = (graph_state.size.x - margin_left - margin_right).max(1.0);
        let graph_height = (graph_state.size.y - margin_bottom - margin_top).max(1.0);

        commands.entity(content_entity).with_children(|parent| {
            parent.spawn(NodeBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    left: Val::Px(margin_left),
                    bottom: Val::Px(margin_bottom),
                    width: Val::Px(graph_width),
                    height: Val::Px(graph_height),
                    ..default()
                },
                background_color: Color::rgba(0.0, 0.0, 0.0, 0.5).into(),
                ..default()
            });

            let bar_width_px = 2.0;
            let max_bars = (graph_width / bar_width_px).floor() as usize;
            let total_records = global_history.all_records().count();
            let chunk_size = (total_records as f32 / max_bars as f32).ceil() as usize;
            let chunk_size = chunk_size.max(1);

            struct AggregatedPoint {
                avg: f32,
                max: f32,
                min: f32,
            }

            let mut visual_points = Vec::new();

            let records: Vec<_> = global_history.all_records().collect();

            let global_max = records
                .iter()
                .map(|r| r.best_fitness)
                .fold(0.0_f32, |a, b| a.max(b))
                .max(10.0);

            for chunk in records.chunks(chunk_size) {
                if chunk.is_empty() {
                    continue;
                }

                let max_in_chunk = chunk
                    .iter()
                    .map(|r| r.best_fitness)
                    .fold(0.0_f32, |a, b| a.max(b));
                let min_in_chunk = chunk
                    .iter()
                    .map(|r| r.avg_fitness)
                    .fold(f32::INFINITY, |a, b| a.min(b));
                let sum_avg: f32 = chunk.iter().map(|r| r.avg_fitness).sum();
                let avg_in_chunk = sum_avg / chunk.len() as f32;

                visual_points.push(AggregatedPoint {
                    avg: avg_in_chunk,
                    max: max_in_chunk,
                    min: min_in_chunk,
                });
            }

            let num_visual_points = visual_points.len();
            let exact_bar_width = graph_width / num_visual_points.max(1) as f32;

            for (i, point) in visual_points.iter().enumerate() {
                let x_pos = margin_left + (i as f32 * exact_bar_width);

                let get_height = |val: f32| -> f32 {
                    let ratio = (val / global_max).clamp(0.0, 1.0);
                    ratio * graph_height
                };

                let h_max = get_height(point.max);
                let h_avg = get_height(point.avg);
                let h_min = get_height(point.min);

                let display_width = if exact_bar_width > 2.0 {
                    exact_bar_width - 1.0
                } else {
                    exact_bar_width
                };

                // Min bar (orange, at the back)
                if h_min > 0.0 {
                    parent.spawn(NodeBundle {
                        style: Style {
                            position_type: PositionType::Absolute,
                            left: Val::Px(x_pos),
                            bottom: Val::Px(margin_bottom),
                            width: Val::Px(display_width),
                            height: Val::Px(h_min),
                            ..default()
                        },
                        background_color: Color::rgba(1.0, 0.5, 0.0, 0.3).into(),
                        ..default()
                    });
                }

                if h_max > 0.0 {
                    parent.spawn(NodeBundle {
                        style: Style {
                            position_type: PositionType::Absolute,
                            left: Val::Px(x_pos),
                            bottom: Val::Px(margin_bottom),
                            width: Val::Px(display_width),
                            height: Val::Px(h_max),
                            ..default()
                        },
                        background_color: Color::rgba(1.0, 0.2, 0.2, 0.3).into(),
                        ..default()
                    });
                }

                if h_avg > 0.0 {
                    parent.spawn(NodeBundle {
                        style: Style {
                            position_type: PositionType::Absolute,
                            left: Val::Px(x_pos),
                            bottom: Val::Px(margin_bottom),
                            width: Val::Px(display_width),
                            height: Val::Px(h_avg),
                            ..default()
                        },
                        background_color: Color::rgba(0.2, 1.0, 0.2, 0.5).into(),
                        ..default()
                    });
                }
            }

            parent.spawn(
                TextBundle::from_section(
                    format!("Best: {:.0}", global_max),
                    TextStyle {
                        font_size: 12.0,
                        color: Color::GRAY,
                        ..default()
                    },
                )
                .with_style(Style {
                    position_type: PositionType::Absolute,
                    left: Val::Px(margin_left),
                    top: Val::Px(margin_top),
                    ..default()
                }),
            );
        });
    }
}
