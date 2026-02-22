//! UI systems for MAP-Elites Snake

use bevy::app::AppExit;
use bevy::prelude::*;
use bevy::sprite::MaterialMesh2dBundle;
use bevy::window::WindowMode;
use std::collections::HashMap;

use crate::evolution::EvolutionManager;
use crate::snake::{
    AppStartTime, CollisionSettings, GameState, GameStats, GlobalTrainingHistory, GridDimensions,
    GridMap, MeshCache, RenderConfig, SnakeInstance, TrainingStats, BLOCK_SIZE,
};

/// UI Component markers
#[derive(Component)]
pub struct StatsText;

#[derive(Component)]
pub struct LeaderboardText;

#[derive(Component)]
pub struct CommandsText;

#[derive(Resource)]
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
#[derive(Component)]
pub struct GraphPanel;

#[derive(Component)]
pub struct GraphPanelHeader;

#[derive(Component)]
pub struct GraphPanelContent;

#[derive(Component)]
pub struct GraphCloseButton;

#[derive(Component)]
pub struct GraphCollapseButton;

#[derive(Component)]
pub struct GraphResizeHandle;

/// Heatmap panel components
#[derive(Component)]
pub struct HeatmapPanel;

#[derive(Component)]
pub struct HeatmapGrid;

/// Material cache to avoid creating duplicate ColorMaterial assets
#[derive(Resource, Default)]
pub struct MaterialCache {
    pub cache: HashMap<[u8; 3], Handle<ColorMaterial>>,
}

/// Cell-based render map: one entity per grid cell, pre-spawned.
/// Each frame only the highest-fitness snake occupying each cell is displayed.
/// Entity index = y * grid_width + x
#[derive(Resource)]
pub struct CellRenderMap {
    /// For each occupied cell: (color, fitness_of_best_snake_here)
    /// Rebuilt from scratch every frame before rendering
    pub cells: HashMap<(i32, i32), (Color, f64)>,
    /// Pre-spawned Bevy entities — one per grid cell, indexed y*w+x
    pub entities: Vec<Entity>,
    pub grid_width: i32,
    pub grid_height: i32,
    /// True for exactly 1 frame after entity respawn (Bevy deferred commands not yet applied)
    pub rebuilding: bool,
}

impl CellRenderMap {
    pub fn new(grid_width: i32, grid_height: i32) -> Self {
        Self {
            cells: HashMap::new(),
            entities: Vec::new(),
            grid_width,
            grid_height,
            rebuilding: false,
        }
    }

    /// Get entity index for grid position (x, y)
    pub fn entity_index(&self, x: i32, y: i32) -> Option<usize> {
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
        .add_systems(
            Update,
            (
                handle_input,
                on_window_resize_collect,
                on_window_resize_apply.after(on_window_resize_collect),
                render_system.after(on_window_resize_apply),
            ),
        )
        .add_systems(Update, update_stats_ui)
        .add_systems(Update, update_graph_panel_visibility)
        .add_systems(Update, handle_graph_panel_interactions)
        .add_systems(
            Update,
            sync_graph_panel_layout.after(handle_graph_panel_interactions),
        )
        .add_systems(
            Update,
            draw_graph_in_panel
                .after(update_graph_panel_visibility)
                .after(sync_graph_panel_layout),
        )
        .add_systems(Update, update_heatmap_panel_visibility)
        .add_systems(
            Update,
            draw_heatmap_in_panel.after(update_heatmap_panel_visibility),
        );
    }
}

pub fn spawn_stats_ui(mut commands: Commands, _game: Res<GameState>) {
    let mut leaderboard_sections = vec![TextSection::new(
        "[LEADERBOARD]\n",
        TextStyle {
            font_size: 18.0,
            color: Color::GOLD,
            ..default()
        },
    )];

    // Create only 20 leaderboard slots (fixed size for performance)
    for _ in 0..20 {
        leaderboard_sections.push(TextSection::new(
            "",
            TextStyle {
                font_size: 15.0,
                color: Color::WHITE,
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
                "Alive:0 Dead:0 | Food: 0  Games: 0",
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

    commands.spawn((
        TextBundle::from_section(
            "[R]Render:ON  [G]Graph  [B]Board  [F]Fullscreen  [C]Collision:OFF  [ESC]Exit",
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

#[allow(clippy::too_many_arguments)]
pub fn update_stats_ui(
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
    _stats: Res<TrainingStats>,
    game_stats: Res<GameStats>,
    collision_settings: Res<CollisionSettings>,
    render_config: Res<RenderConfig>,
    app_start_time: Res<AppStartTime>,
    global_history: Res<GlobalTrainingHistory>,
    mut ui_timer: ResMut<UiUpdateTimer>,
    time: Res<Time>,
) {
    // Limit UI updates to ~10Hz
    ui_timer.0.tick(time.delta());
    if !ui_timer.0.just_finished() {
        return;
    }

    use std::time::Instant;

    let now = Instant::now();
    let current_session_duration = now.duration_since(app_start_time.0);
    let total_training_time = std::time::Duration::from_secs(global_history.accumulated_time_secs)
        + current_session_duration;

    let session_secs = current_session_duration.as_secs();
    let session_hours = session_secs / 3600;
    let session_minutes = (session_secs % 3600) / 60;
    let session_seconds = session_secs % 60;

    let total_secs = total_training_time.as_secs();
    let total_hours = total_secs / 3600;
    let total_minutes = (total_secs % 3600) / 60;
    let total_seconds = total_secs % 60;

    let persistent_high = game_stats.high_score.max(game.high_score);
    let alive_count = game.alive_count();

    let mut snake_data: Vec<(usize, &SnakeInstance)> = game.snakes.iter().enumerate().collect();
    snake_data.sort_by(|a, b| b.1.score.cmp(&a.1.score));

    // Limit leaderboard to top 20 for performance with large populations
    let display_count = snake_data.len().min(20);

    if let Ok(mut lb_text) = leaderboard_query.get_single_mut() {
        for (rank, (original_idx, snake)) in snake_data.iter().take(display_count).enumerate() {
            let section_idx = rank + 1;
            if section_idx < lb_text.sections.len() {
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
        // Clear remaining sections if population is smaller than sections
        for section_idx in (display_count + 1)..lb_text.sections.len() {
            lb_text.sections[section_idx].value = String::new();
        }
    }

    if let Ok(mut st_text) = stats_query.get_single_mut() {
        st_text.sections[0].value = format!(
            "Gen:{:4} | High:{:3} | Best:{:3}\n",
            game.total_iterations, game.high_score, persistent_high
        );
        st_text.sections[1].value = format!(
            "Time: {:02}:{:02}:{:02} | Tot: {:02}:{:02}:{:02}\n",
            session_hours,
            session_minutes,
            session_seconds,
            total_hours,
            total_minutes,
            total_seconds
        );
        st_text.sections[2].value = format!(
            "Alive:{:2} Dead:{:2} | Food:{:4} | Games:{:5}",
            alive_count,
            game.snakes.len() - alive_count,
            game_stats.total_food_eaten,
            game_stats.total_games_played
        );
    }

    if let Ok(mut cmd_text) = commands_query.get_single_mut() {
        let render_status = if render_config.enabled { "ON" } else { "TURBO" };
        let collision_status = if collision_settings.snake_vs_snake {
            "ON"
        } else {
            "OFF"
        };
        let graph_status = if GraphPanelState::default().visible {
            "ON"
        } else {
            "OFF"
        };
        cmd_text.sections[0].value = format!(
            "[R]Render:{}  [G]Graph:{}  [B]Board  [P]Pause  [F]Full  [C]Collision:{}  [ESC]Exit",
            render_status, graph_status, collision_status
        );
    }
}

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
    mut heatmap_state: ResMut<HeatmapPanelState>,
    mut pause_state: ResMut<PauseState>,
) {
    use crate::snake::{get_or_create_run_dir, new_session_path, save_training_session};
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
        let session_path = new_session_path();
        if let Err(e) = save_training_session(
            &session_path,
            &global_history,
            &game_stats,
            current_session_duration.as_secs(),
        ) {
            eprintln!("⚠️ Error saving session: {}", e);
        }

        println!("Saved to: {}", get_or_create_run_dir().display());
        println!("====================\n");

        app_exit_events.send(AppExit);
    }

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

    if keyboard_input.just_pressed(KeyCode::KeyG) {
        if !graph_state.visible && !graph_state.fullscreen {
            graph_state.visible = true;
            graph_state.fullscreen = false;
        } else if graph_state.visible && !graph_state.fullscreen {
            graph_state.fullscreen = true;
            if let Ok(window) = windows.get_single() {
                graph_state.size = Vec2::new(window.width(), window.height() - 60.0);
                graph_state.position = Vec2::new(0.0, 0.0);
            }
        } else {
            graph_state.visible = false;
            graph_state.fullscreen = false;
        }

        graph_state.needs_redraw = true;

        println!(
            "Graph: {}",
            if graph_state.visible {
                if graph_state.fullscreen {
                    "FULLSCREEN"
                } else {
                    "WINDOW"
                }
            } else {
                "HIDDEN"
            }
        );
    }

    if keyboard_input.just_pressed(KeyCode::KeyB) {
        heatmap_state.visible = !heatmap_state.visible;
        heatmap_state.needs_redraw = true;

        println!(
            "Heatmap Board: {}",
            if heatmap_state.visible {
                "VISIBLE"
            } else {
                "HIDDEN"
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
            WindowMode::Fullscreen
        } else {
            WindowMode::Windowed
        };
        if window_settings.is_fullscreen {
            graph_state.visible = true;
            graph_state.needs_redraw = true;
        }
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
    _gen_seed: Option<Res<crate::snake::GenerationSeed>>,
    mut commands: Commands,
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
    for &entity in cell_map.entities.iter() {
        commands.entity(entity).despawn();
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
    cell_map.cells.clear();
    cell_map.grid_width = new_width;
    cell_map.grid_height = new_height;
    cell_map.rebuilding = true; // Skip next render frame, entities not yet in World

    // Regenerate seed for new grid
    let new_seed = crate::snake::GenerationSeed::new_for_grid(&grid);
    let total_snakes = game.snakes.len();
    for snake in game.snakes.iter_mut() {
        snake.reset_with_seed(&grid, total_snakes, &new_seed, 0.0, 0.0, 0.0, 1.0);
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
    windows: Query<&Window>,
    _mesh_cache: Res<MeshCache>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut mat_cache: ResMut<MaterialCache>,
    food_pool: Res<FoodPool>,
    render_config: Res<RenderConfig>,
    mut stats: ResMut<TrainingStats>,
    mut cell_map: ResMut<CellRenderMap>,
    evo_manager: Res<EvolutionManager>,
) {
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

    // Build fitness lookup: snake_id → fitness from current generation population
    // This is used to determine which snake "wins" a contested cell
    let fitness_map: Vec<f64> = evo_manager
        .generation_state
        .population
        .iter()
        .map(|ind| ind.fitness)
        .collect();

    // === PHASE 1: Build cell color map ===
    // For each occupied cell, keep only the color of the highest-fitness snake
    cell_map.cells.clear();
    for snake in game.snakes.iter() {
        if snake.is_game_over {
            continue;
        }
        let snake_fitness = fitness_map.get(snake.id).copied().unwrap_or(0.0);

        for (seg_idx, pos) in snake.snake.iter().enumerate() {
            let key = (pos.x, pos.y);
            // Head is always white; body uses the snake's behavioral color
            let color = if seg_idx == 0 {
                Color::rgb(1.0, 1.0, 1.0)
            } else {
                snake.color
            };
            // Only update if this snake has higher fitness than the current winner
            let entry = cell_map
                .cells
                .entry(key)
                .or_insert((color, f64::NEG_INFINITY));
            if snake_fitness > entry.1 {
                *entry = (color, snake_fitness);
            }
        }
    }

    // === PHASE 2: Update cell entities ===
    // Helper closure for material caching
    let get_or_create_material = |color: Color,
                                  cache: &mut MaterialCache,
                                  materials: &mut Assets<ColorMaterial>|
     -> Handle<ColorMaterial> {
        let key = [
            (color.r() * 255.0) as u8,
            (color.g() * 255.0) as u8,
            (color.b() * 255.0) as u8,
        ];
        cache
            .cache
            .entry(key)
            .or_insert_with(|| materials.add(color))
            .clone()
    };

    // Show/update occupied cells
    for (&(x, y), &(color, _)) in cell_map.cells.iter() {
        let Some(idx) = cell_map.entity_index(x, y) else {
            continue;
        };
        let Some(&entity) = cell_map.entities.get(idx) else {
            continue;
        };

        let material = get_or_create_material(color, &mut mat_cache, &mut materials);
        let transform = Transform::from_xyz(
            offset_x + x as f32 * BLOCK_SIZE,
            offset_y - y as f32 * BLOCK_SIZE,
            0.0, // z=0 puts cells behind food (food is at z=1.0)
        );
        commands
            .entity(entity)
            .insert((material, transform, Visibility::Visible));
    }

    // Hide unoccupied cells
    // Iterating all grid cells is cheap: max 57*33=1881 iterations
    for y in 0..cell_map.grid_height {
        for x in 0..cell_map.grid_width {
            if !cell_map.cells.contains_key(&(x, y)) {
                let Some(idx) = cell_map.entity_index(x, y) else {
                    continue;
                };
                let Some(&entity) = cell_map.entities.get(idx) else {
                    continue;
                };
                commands.entity(entity).insert(Visibility::Hidden);
            }
        }
    }

    // === PHASE 3: Update food entities ===
    for snake in game.snakes.iter() {
        let Some(&food_entity) = food_pool.entities.get(snake.id) else {
            continue;
        };

        if snake.is_game_over {
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
                        "MAP-Elites Heatmap (Body Pressure vs Path Directness)",
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
        let res = crate::map_elites::GRID_RESOLUTION as f32;
        let cell_w = grid_width / res;
        let cell_h = grid_height / res;

        commands.entity(grid_entity).with_children(|parent| {
            let max_fitness = evo_manager.archive.best_fitness.max(1.0);

            for x in 0..crate::map_elites::GRID_RESOLUTION {
                for y in 0..crate::map_elites::GRID_RESOLUTION {
                    let cell_opt = evo_manager.archive.grid.get(&(x, y));

                    let cell_color = if let Some(ind) = cell_opt {
                        // Color based on fitness: blue (low) → green (high)
                        let intensity = (ind.fitness / max_fitness).clamp(0.0, 1.0) as f32;
                        Color::rgb(0.1, intensity, 1.0 - intensity)
                    } else {
                        Color::rgb(0.05, 0.05, 0.07) // Empty
                    };

                    // Y-AXIS CORRECTION:
                    // Bevy top:0 is at top. To have y=0 at bottom, we do (Res - 1 - y)
                    let display_y = (crate::map_elites::GRID_RESOLUTION - 1 - y) as f32;

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
                "Path Directness →",
                Val::Px(margin),
                Val::Px(grid_height + margin + 5.0),
            );
            spawn_axis_label(parent, "Body Pressure ↑", Val::Px(5.0), Val::Px(margin));

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

    let data_changed = global_history.records.len() != graph_state.last_entry_count;
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
    graph_state.last_entry_count = global_history.records.len();

    if global_history.records.is_empty() {
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
            let total_records = global_history.records.len();
            let chunk_size = (total_records as f32 / max_bars as f32).ceil() as usize;
            let chunk_size = chunk_size.max(1);

            struct AggregatedPoint {
                avg: f64,
                max: f64,
                min: f64,
            }

            let mut visual_points = Vec::new();

            let global_max = global_history
                .records
                .iter()
                .map(|r| r.best_fitness)
                .fold(0.0_f64, |a, b| a.max(b))
                .max(10.0);

            for chunk in global_history.records.chunks(chunk_size) {
                if chunk.is_empty() {
                    continue;
                }

                let max_in_chunk = chunk
                    .iter()
                    .map(|r| r.best_fitness)
                    .fold(0.0_f64, |a, b| a.max(b));
                let min_in_chunk = chunk
                    .iter()
                    .map(|r| r.avg_fitness)
                    .fold(f64::INFINITY, |a, b| a.min(b));
                let sum_avg: f64 = chunk.iter().map(|r| r.avg_fitness).sum();
                let avg_in_chunk = sum_avg / chunk.len() as f64;

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

                let get_height = |val: f64| -> f32 {
                    let ratio = (val / global_max).clamp(0.0, 1.0) as f32;
                    ratio * graph_height
                };

                let h_max = get_height(point.max);
                let h_avg = get_height(point.avg);
                let _h_min = get_height(point.min);

                let display_width = if exact_bar_width > 2.0 {
                    exact_bar_width - 1.0
                } else {
                    exact_bar_width
                };

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
