//! Brain Inspector Module - Visual Neural Network Analysis
//!
//! Provides a dedicated view for inspecting agent brains, sensor inputs,
//! and neural network activations in real-time.

pub mod gizmos;
pub mod loader;
pub mod ui;

pub use gizmos::*;
pub use loader::*;

use bevy::prelude::*;

use crate::snake::{GameState, GridDimensions, GridMap, SnakeInstance};

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

/// Application states for view switching
#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AppState {
    #[default]
    SimulationView,
    BrainInspectorView,
}

// ============================================================================
// RESOURCES
// ============================================================================

/// Resource holding the currently selected agent for inspection
#[derive(Resource, Default)]
pub struct InspectedAgent {
    /// Index of the snake being inspected (None if no selection)
    pub snake_idx: Option<usize>,
    /// Last known sensor state (cached to avoid recalculation)
    pub last_sensor_state: Option<[f32; 17]>,
    /// Last known neural network output
    pub last_output: Option<[f32; 3]>,
}

/// Resource for brain inspector UI state
#[derive(Resource)]
pub struct BrainInspectorState {
    /// Whether the inspector panel is visible
    pub panel_visible: bool,
    /// Selected tab: "sensors", "weights", "activations"
    pub active_tab: InspectorTab,
    /// Target snake ID to follow (for auto-selection)
    pub follow_snake_id: Option<usize>,
}

impl Default for BrainInspectorState {
    fn default() -> Self {
        Self {
            panel_visible: true,
            active_tab: InspectorTab::Sensors,
            follow_snake_id: Some(0), // Default to first snake
        }
    }
}

/// Inspector UI tabs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InspectorTab {
    Sensors,
    Weights,
    Activations,
}

// ============================================================================
// COMPONENTS
// ============================================================================

/// Marker component for brain inspector UI elements
#[derive(Component)]
pub struct BrainInspectorUi;

/// Marker for the inspector camera (separate from simulation camera)
#[derive(Component)]
pub struct InspectorCamera;

/// State for inspector camera (position, zoom, pan)
#[derive(Component)]
pub struct InspectorCameraState {
    /// Camera offset from the followed snake
    pub offset: Vec2,
    /// Zoom level (1.0 = default, >1.0 = zoomed in, <1.0 = zoomed out)
    pub zoom: f32,
    /// Whether the camera is currently being dragged
    pub is_dragging: bool,
    /// Last mouse position for drag calculation
    pub last_mouse_pos: Option<Vec2>,
}

impl Default for InspectorCameraState {
    fn default() -> Self {
        Self {
            offset: Vec2::ZERO,
            zoom: 1.0,
            is_dragging: false,
            last_mouse_pos: None,
        }
    }
}

/// Marker for simulation camera
#[derive(Component)]
pub struct SimulationCamera;

/// Marker for sensor visualization gizmos
#[derive(Component)]
pub struct SensorGizmo;

// ============================================================================
// PLUGIN
// ============================================================================

pub struct BrainInspectorPlugin;

impl Plugin for BrainInspectorPlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<AppState>()
            .insert_resource(InspectedAgent::default())
            .insert_resource(BrainInspectorState::default())
            // View switching always available
            .add_systems(Update, view_switch_system)
            // Inspector controls only in BrainInspectorView
            .add_systems(
                Update,
                inspector_controls_system.run_if(in_state(AppState::BrainInspectorView)),
            )
            // State enter/exit systems
            .add_systems(OnEnter(AppState::BrainInspectorView), enter_inspector_view)
            .add_systems(OnExit(AppState::BrainInspectorView), exit_inspector_view)
            .add_systems(OnEnter(AppState::SimulationView), enter_simulation_view)
            // Inspector update systems (only in BrainInspectorView)
            .add_systems(
                Update,
                (
                    handle_agent_death_and_switch,
                    update_sensor_cache,
                    inspector_camera_controls,
                    follow_inspected_agent_with_camera,
                )
                    .run_if(in_state(AppState::BrainInspectorView)),
            );
    }
}

// ============================================================================
// INPUT SYSTEMS
// ============================================================================

/// View switching system - always available
fn view_switch_system(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    current_state: Res<State<AppState>>,
    mut next_state: ResMut<NextState<AppState>>,
) {
    // View switching with number keys (always active)
    if keyboard_input.just_pressed(KeyCode::Digit1) {
        if current_state.get() != &AppState::SimulationView {
            println!("[VIEW] Switching to Simulation View");
            next_state.set(AppState::SimulationView);
        }
    }

    if keyboard_input.just_pressed(KeyCode::Digit2) {
        if current_state.get() != &AppState::BrainInspectorView {
            println!("[VIEW] Switching to Brain Inspector View");
            next_state.set(AppState::BrainInspectorView);
        }
    }
}

/// Inspector controls - only active in BrainInspectorView
fn inspector_controls_system(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut inspector_state: ResMut<BrainInspectorState>,
    game_state: Res<GameState>,
    mut inspected_agent: ResMut<InspectedAgent>,
) {
    // Tab switching
    if keyboard_input.just_pressed(KeyCode::KeyS) {
        inspector_state.active_tab = InspectorTab::Sensors;
        println!("[INSPECTOR] Tab: Sensors");
    }
    if keyboard_input.just_pressed(KeyCode::KeyW) {
        inspector_state.active_tab = InspectorTab::Weights;
        println!("[INSPECTOR] Tab: Weights");
    }
    if keyboard_input.just_pressed(KeyCode::KeyA) {
        inspector_state.active_tab = InspectorTab::Activations;
        println!("[INSPECTOR] Tab: Activations");
    }

    // Agent selection with number keys (0-9)
    // Skip 1 and 2 as they're used for view switching
    if keyboard_input.just_pressed(KeyCode::Digit0) {
        select_agent_by_index(&mut inspected_agent, &mut inspector_state, &game_state, 0);
    }
    if keyboard_input.just_pressed(KeyCode::Digit3) {
        select_agent_by_index(&mut inspected_agent, &mut inspector_state, &game_state, 3);
    }
    if keyboard_input.just_pressed(KeyCode::Digit4) {
        select_agent_by_index(&mut inspected_agent, &mut inspector_state, &game_state, 4);
    }
    if keyboard_input.just_pressed(KeyCode::Digit5) {
        select_agent_by_index(&mut inspected_agent, &mut inspector_state, &game_state, 5);
    }
    if keyboard_input.just_pressed(KeyCode::Digit6) {
        select_agent_by_index(&mut inspected_agent, &mut inspector_state, &game_state, 6);
    }
    if keyboard_input.just_pressed(KeyCode::Digit7) {
        select_agent_by_index(&mut inspected_agent, &mut inspector_state, &game_state, 7);
    }
    if keyboard_input.just_pressed(KeyCode::Digit8) {
        select_agent_by_index(&mut inspected_agent, &mut inspector_state, &game_state, 8);
    }
    if keyboard_input.just_pressed(KeyCode::Digit9) {
        select_agent_by_index(&mut inspected_agent, &mut inspector_state, &game_state, 9);
    }

    // Next/Previous agent navigation
    if keyboard_input.just_pressed(KeyCode::ArrowRight)
        || keyboard_input.just_pressed(KeyCode::KeyN)
    {
        navigate_to_next_agent(&mut inspected_agent, &mut inspector_state, &game_state);
    }

    if keyboard_input.just_pressed(KeyCode::ArrowLeft) || keyboard_input.just_pressed(KeyCode::KeyP)
    {
        navigate_to_prev_agent(&mut inspected_agent, &mut inspector_state, &game_state);
    }

    // Toggle panel visibility
    if keyboard_input.just_pressed(KeyCode::KeyH) {
        inspector_state.panel_visible = !inspector_state.panel_visible;
        println!(
            "[INSPECTOR] Panel: {}",
            if inspector_state.panel_visible {
                "visible"
            } else {
                "hidden"
            }
        );
    }
}

/// Helper to select an agent by index
fn select_agent_by_index(
    inspected_agent: &mut InspectedAgent,
    inspector_state: &mut BrainInspectorState,
    game_state: &GameState,
    idx: usize,
) {
    if idx < game_state.snakes.len() {
        inspected_agent.snake_idx = Some(idx);
        inspector_state.follow_snake_id = Some(idx);
        let snake = &game_state.snakes[idx];
        let status = if snake.is_game_over { "DEAD" } else { "alive" };
        println!(
            "[INSPECTOR] Selected snake {} (score: {}, {})",
            idx, snake.score, status
        );
    }
}

/// Navigate to the next agent (skipping dead ones if possible)
fn navigate_to_next_agent(
    inspected_agent: &mut InspectedAgent,
    inspector_state: &mut BrainInspectorState,
    game_state: &GameState,
) {
    if game_state.snakes.is_empty() {
        return;
    }

    let current = inspected_agent.snake_idx.unwrap_or(0);
    let snake_count = game_state.snakes.len();

    // Try to find next alive snake
    for i in 1..=snake_count {
        let next = (current + i) % snake_count;
        if !game_state.snakes[next].is_game_over {
            inspected_agent.snake_idx = Some(next);
            inspector_state.follow_snake_id = Some(next);
            println!(
                "[INSPECTOR] Next snake {} (score: {})",
                next, game_state.snakes[next].score
            );
            return;
        }
    }

    // If all dead, just go to next
    let next = (current + 1) % snake_count;
    inspected_agent.snake_idx = Some(next);
    inspector_state.follow_snake_id = Some(next);
    println!("[INSPECTOR] Next snake {} (DEAD)", next);
}

/// Navigate to the previous agent (skipping dead ones if possible)
fn navigate_to_prev_agent(
    inspected_agent: &mut InspectedAgent,
    inspector_state: &mut BrainInspectorState,
    game_state: &GameState,
) {
    if game_state.snakes.is_empty() {
        return;
    }

    let current = inspected_agent.snake_idx.unwrap_or(0);
    let snake_count = game_state.snakes.len();

    // Try to find previous alive snake
    for i in 1..=snake_count {
        let prev = if current >= i {
            current - i
        } else {
            snake_count - (i - current)
        };
        if !game_state.snakes[prev].is_game_over {
            inspected_agent.snake_idx = Some(prev);
            inspector_state.follow_snake_id = Some(prev);
            println!(
                "[INSPECTOR] Previous snake {} (score: {})",
                prev, game_state.snakes[prev].score
            );
            return;
        }
    }

    // If all dead, just go to previous
    let prev = if current == 0 {
        snake_count - 1
    } else {
        current - 1
    };
    inspected_agent.snake_idx = Some(prev);
    inspector_state.follow_snake_id = Some(prev);
    println!("[INSPECTOR] Previous snake {} (DEAD)", prev);
}

// ============================================================================
// STATE TRANSITION SYSTEMS
// ============================================================================

/// Called when entering Brain Inspector view
fn enter_inspector_view(
    mut commands: Commands,
    mut inspector_state: ResMut<BrainInspectorState>,
    mut inspected_agent: ResMut<InspectedAgent>,
    game_state: Res<GameState>,
    sim_camera_query: Query<Entity, With<SimulationCamera>>,
    window_query: Query<&Window>,
) {
    println!("Entering Brain Inspector View");

    // Hide simulation camera
    for entity in sim_camera_query.iter() {
        commands.entity(entity).insert(Visibility::Hidden);
    }

    // Spawn inspector camera with initial position
    let _window = window_query.single();
    commands.spawn((
        Camera2dBundle {
            camera: Camera {
                order: 1, // Render on top
                ..default()
            },
            ..default()
        },
        InspectorCamera,
        InspectorCameraState::default(),
    ));

    // Auto-select the alive snake with highest score
    let best_alive = game_state
        .snakes
        .iter()
        .enumerate()
        .filter(|(_, s)| !s.is_game_over)
        .max_by_key(|(_, s)| s.score);

    if let Some((idx, snake)) = best_alive {
        inspected_agent.snake_idx = Some(idx);
        inspector_state.follow_snake_id = Some(idx);
        println!("Auto-selected snake {} with score {}", idx, snake.score);
    } else if !game_state.snakes.is_empty() {
        // If all dead, select first one
        inspected_agent.snake_idx = Some(0);
        inspector_state.follow_snake_id = Some(0);
    }

    // Mark panel as needing spawn
    inspector_state.panel_visible = true;

    // Spawn the UI
    ui::spawn_inspector_ui(commands, inspector_state.into());
}

/// Called when exiting Brain Inspector view
fn exit_inspector_view(
    mut commands: Commands,
    inspector_ui_query: Query<Entity, With<BrainInspectorUi>>,
    inspector_camera_query: Query<Entity, With<InspectorCamera>>,
    mut sim_camera_query: Query<&mut Visibility, With<SimulationCamera>>,
) {
    println!("Exiting Brain Inspector View");

    // Despawn all inspector UI
    for entity in inspector_ui_query.iter() {
        commands.entity(entity).despawn_recursive();
    }

    // Despawn inspector camera
    for entity in inspector_camera_query.iter() {
        commands.entity(entity).despawn();
    }

    // Show simulation camera
    for mut visibility in sim_camera_query.iter_mut() {
        *visibility = Visibility::Visible;
    }
}

/// Called when entering Simulation view
fn enter_simulation_view(
    mut commands: Commands,
    window_query: Query<&Window>,
    sim_camera_query: Query<Entity, With<SimulationCamera>>,
) {
    println!("Entering Simulation View");

    // Ensure simulation camera exists (in case it was despawned)
    if sim_camera_query.is_empty() {
        let _window = window_query.single();
        commands.spawn((Camera2dBundle::default(), SimulationCamera));
    }
}

// ============================================================================
// UPDATE SYSTEMS
// ============================================================================

/// Handle agent death and auto-switch to next alive agent
fn handle_agent_death_and_switch(
    game_state: Res<GameState>,
    mut inspected_agent: ResMut<InspectedAgent>,
    mut inspector_state: ResMut<BrainInspectorState>,
) {
    let Some(current_idx) = inspected_agent.snake_idx else {
        // No agent selected, try to find best alive one
        if let Some((idx, snake)) = find_best_alive_snake(&game_state) {
            inspected_agent.snake_idx = Some(idx);
            inspector_state.follow_snake_id = Some(idx);
            println!(
                "[INSPECTOR] Auto-selected snake {} with score {} (no previous selection)",
                idx, snake.score
            );
        }
        return;
    };

    // Check if current snake is valid
    if current_idx >= game_state.snakes.len() {
        // Snake no longer exists, find a replacement
        if let Some((idx, snake)) = find_best_alive_snake(&game_state) {
            inspected_agent.snake_idx = Some(idx);
            inspector_state.follow_snake_id = Some(idx);
            inspected_agent.last_sensor_state = None;
            inspected_agent.last_output = None;
            println!(
                "[INSPECTOR] Switched to snake {} with score {} (previous invalid)",
                idx, snake.score
            );
        } else {
            inspected_agent.snake_idx = None;
            inspected_agent.last_sensor_state = None;
            inspected_agent.last_output = None;
        }
        return;
    }

    // Check if current snake died
    let snake = &game_state.snakes[current_idx];
    if snake.is_game_over {
        // Snake died, switch to next alive one
        if let Some((idx, new_snake)) = find_next_alive_snake(&game_state, current_idx) {
            inspected_agent.snake_idx = Some(idx);
            inspector_state.follow_snake_id = Some(idx);
            inspected_agent.last_sensor_state = None;
            inspected_agent.last_output = None;
            println!(
                "[INSPECTOR] Snake {} died, switched to snake {} with score {}",
                current_idx, idx, new_snake.score
            );
        }
        // If no alive snakes found, stay on dead one (user can manually navigate)
    }
}

/// Find the alive snake with the highest score
fn find_best_alive_snake(game_state: &GameState) -> Option<(usize, &crate::snake::SnakeInstance)> {
    game_state
        .snakes
        .iter()
        .enumerate()
        .filter(|(_, s)| !s.is_game_over)
        .max_by_key(|(_, s)| s.score)
}

/// Find the next alive snake after the given index
fn find_next_alive_snake(
    game_state: &GameState,
    start_idx: usize,
) -> Option<(usize, &crate::snake::SnakeInstance)> {
    let snake_count = game_state.snakes.len();

    // Search forward from start_idx
    for i in 1..=snake_count {
        let idx = (start_idx + i) % snake_count;
        let snake = &game_state.snakes[idx];
        if !snake.is_game_over {
            return Some((idx, snake));
        }
    }

    // No alive snakes found
    None
}

/// Cache sensor state for the inspected agent to avoid recalculation
fn update_sensor_cache(
    game_state: Res<GameState>,
    grid_map: Res<GridMap>,
    grid: Res<GridDimensions>,
    mut inspected_agent: ResMut<InspectedAgent>,
    population: Res<crate::Population>,
) {
    use crate::snake::get_current_17_state;

    let Some(idx) = inspected_agent.snake_idx else {
        return;
    };

    let Some(snake) = game_state.snakes.get(idx) else {
        return;
    };

    // Calculate current sensor state using the same function as training
    let sensor_state = get_current_17_state(snake, &grid_map, &grid);
    inspected_agent.last_sensor_state = Some(sensor_state);

    // Calculate neural network output if we have a brain for this snake
    if let Some(brain_arc) = population.0.get(idx) {
        let mut full_state = [0.0f32; 34];
        full_state[..17].copy_from_slice(&sensor_state);
        full_state[17..].copy_from_slice(&snake.previous_state);
        let output = brain_arc.forward(&full_state);
        inspected_agent.last_output = Some(output);
    }
}

// ============================================================================
// CAMERA CONTROL SYSTEMS
// ============================================================================

/// Handle camera controls (zoom, pan, reset)
fn inspector_camera_controls(
    mut camera_query: Query<(&mut InspectorCameraState, &mut Transform), With<InspectorCamera>>,
    mut mouse_wheel: EventReader<bevy::input::mouse::MouseWheel>,
    mouse_button: Res<ButtonInput<MouseButton>>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window>,
) {
    let Ok((mut cam_state, mut transform)) = camera_query.get_single_mut() else {
        return;
    };

    let Ok(window) = windows.get_single() else {
        return;
    };

    // Reset camera on 'R' key
    if keyboard_input.just_pressed(KeyCode::KeyR) {
        cam_state.offset = Vec2::ZERO;
        cam_state.zoom = 1.0;
        transform.scale = Vec3::ONE;
        println!("Camera reset");
    }

    // Zoom with mouse wheel
    for event in mouse_wheel.read() {
        let zoom_delta = event.y * 0.1;
        cam_state.zoom = (cam_state.zoom + zoom_delta).clamp(0.5, 5.0);
        transform.scale = Vec3::splat(cam_state.zoom);
    }

    // Pan with middle mouse button or right mouse button
    let is_pan_button =
        mouse_button.pressed(MouseButton::Middle) || mouse_button.pressed(MouseButton::Right);

    if is_pan_button {
        if let Some(cursor_pos) = window.cursor_position() {
            if let Some(last_pos) = cam_state.last_mouse_pos {
                // Calculate delta in world space (accounting for zoom)
                let delta = (cursor_pos - last_pos) * cam_state.zoom;
                cam_state.offset.x -= delta.x;
                cam_state.offset.y += delta.y; // Y is inverted in screen space
            }
            cam_state.last_mouse_pos = Some(cursor_pos);
            cam_state.is_dragging = true;
        }
    } else {
        cam_state.is_dragging = false;
        cam_state.last_mouse_pos = None;
    }
}

/// Update camera position to follow the inspected agent
fn follow_inspected_agent_with_camera(
    inspected: Res<InspectedAgent>,
    game_state: Res<GameState>,
    windows: Query<&Window>,
    mut camera_query: Query<(&InspectorCameraState, &mut Transform), With<InspectorCamera>>,
) {
    let Ok((cam_state, mut transform)) = camera_query.get_single_mut() else {
        return;
    };

    let Ok(window) = windows.get_single() else {
        return;
    };

    // Calculate grid offset (same as render_system)
    let grid_px_w = game_state
        .snakes
        .first()
        .map(|_| crate::snake::BLOCK_SIZE * 20.0)
        .unwrap_or(400.0);
    let grid_px_h = grid_px_w;
    let leftover_x = window.resolution.width() - grid_px_w;
    let leftover_y = window.resolution.height() - grid_px_h;
    let base_offset_x =
        -window.resolution.width() / 2.0 + (leftover_x / 2.0) + crate::snake::BLOCK_SIZE / 2.0;
    let base_offset_y =
        window.resolution.height() / 2.0 - (leftover_y / 2.0) - crate::snake::BLOCK_SIZE / 2.0;

    // Get snake head position
    let snake_world_pos = if let Some(idx) = inspected.snake_idx {
        game_state
            .snakes
            .get(idx)
            .map(|snake| {
                let head = snake.snake[0];
                Vec3::new(
                    base_offset_x + head.x as f32 * crate::snake::BLOCK_SIZE,
                    base_offset_y - head.y as f32 * crate::snake::BLOCK_SIZE,
                    0.0,
                )
            })
            .unwrap_or_else(|| Vec3::ZERO)
    } else {
        Vec3::ZERO
    };

    // Apply position with offset
    transform.translation = snake_world_pos
        + Vec3::new(
            cam_state.offset.x / cam_state.zoom,
            cam_state.offset.y / cam_state.zoom,
            0.0,
        );
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Get a reference to the inspected snake if valid
pub fn get_inspected_snake<'a>(
    inspected: &'a InspectedAgent,
    game_state: &'a GameState,
) -> Option<(usize, &'a SnakeInstance)> {
    let idx = inspected.snake_idx?;
    game_state.snakes.get(idx).map(|s| (idx, s))
}

/// Check if the inspected agent is still alive
pub fn is_inspected_agent_alive(inspected: &InspectedAgent, game_state: &GameState) -> bool {
    match inspected.snake_idx {
        Some(idx) => game_state
            .snakes
            .get(idx)
            .map(|s| !s.is_game_over)
            .unwrap_or(false),
        None => false,
    }
}
