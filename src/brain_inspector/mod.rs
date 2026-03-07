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
            // Input system runs in all states
            .add_systems(Update, keyboard_input_system)
            // State enter/exit systems
            .add_systems(OnEnter(AppState::BrainInspectorView), enter_inspector_view)
            .add_systems(OnExit(AppState::BrainInspectorView), exit_inspector_view)
            .add_systems(OnEnter(AppState::SimulationView), enter_simulation_view)
            // Inspector update systems (only in BrainInspectorView)
            .add_systems(
                Update,
                (update_inspected_agent, update_sensor_cache)
                    .run_if(in_state(AppState::BrainInspectorView)),
            );
    }
}

// ============================================================================
// INPUT SYSTEM
// ============================================================================

/// Keyboard input handler for view switching and inspector controls
fn keyboard_input_system(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    current_state: Res<State<AppState>>,
    mut next_state: ResMut<NextState<AppState>>,
    mut inspector_state: ResMut<BrainInspectorState>,
    game_state: Res<GameState>,
    mut inspected_agent: ResMut<InspectedAgent>,
) {
    // View switching with number keys
    if keyboard_input.just_pressed(KeyCode::Digit1) {
        if current_state.get() != &AppState::SimulationView {
            println!("Switching to Simulation View");
            next_state.set(AppState::SimulationView);
        }
    }

    if keyboard_input.just_pressed(KeyCode::Digit2) {
        if current_state.get() != &AppState::BrainInspectorView {
            println!("Switching to Brain Inspector View");
            next_state.set(AppState::BrainInspectorView);
        }
    }

    // Inspector-specific controls (only when in inspector view)
    if current_state.get() == &AppState::BrainInspectorView {
        // Tab switching
        if keyboard_input.just_pressed(KeyCode::KeyS) {
            inspector_state.active_tab = InspectorTab::Sensors;
            println!("Inspector tab: Sensors");
        }
        if keyboard_input.just_pressed(KeyCode::KeyW) {
            inspector_state.active_tab = InspectorTab::Weights;
            println!("Inspector tab: Weights");
        }
        if keyboard_input.just_pressed(KeyCode::KeyA) {
            inspector_state.active_tab = InspectorTab::Activations;
            println!("Inspector tab: Activations");
        }

        // Agent selection (number keys 0-9 for quick selection)
        let key_mappings = [
            (KeyCode::Digit0, 0usize),
            (KeyCode::Digit3, 3usize),
            (KeyCode::Digit4, 4usize),
            (KeyCode::Digit5, 5usize),
            (KeyCode::Digit6, 6usize),
            (KeyCode::Digit7, 7usize),
            (KeyCode::Digit8, 8usize),
            (KeyCode::Digit9, 9usize),
        ];

        for (key, idx) in key_mappings.iter() {
            if keyboard_input.just_pressed(*key) {
                if *idx < game_state.snakes.len() {
                    inspected_agent.snake_idx = Some(*idx);
                    inspector_state.follow_snake_id = Some(*idx);
                    println!("Inspecting snake {}", idx);
                }
            }
        }

        // Next/Previous agent navigation
        if keyboard_input.just_pressed(KeyCode::ArrowRight)
            || keyboard_input.just_pressed(KeyCode::KeyN)
        {
            let current = inspected_agent.snake_idx.unwrap_or(0);
            let next = (current + 1) % game_state.snakes.len();
            inspected_agent.snake_idx = Some(next);
            inspector_state.follow_snake_id = Some(next);
            println!("Inspecting snake {}", next);
        }

        if keyboard_input.just_pressed(KeyCode::ArrowLeft)
            || keyboard_input.just_pressed(KeyCode::KeyP)
        {
            let current = inspected_agent.snake_idx.unwrap_or(0);
            let prev = if current == 0 {
                game_state.snakes.len() - 1
            } else {
                current - 1
            };
            inspected_agent.snake_idx = Some(prev);
            inspector_state.follow_snake_id = Some(prev);
            println!("Inspecting snake {}", prev);
        }

        // Toggle panel visibility
        if keyboard_input.just_pressed(KeyCode::KeyH) {
            inspector_state.panel_visible = !inspector_state.panel_visible;
            println!(
                "Inspector panel: {}",
                if inspector_state.panel_visible {
                    "visible"
                } else {
                    "hidden"
                }
            );
        }
    }
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

    // Spawn inspector camera
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
    ));

    // Auto-select first alive snake if none selected
    if inspected_agent.snake_idx.is_none() {
        // Find first alive snake
        for (idx, snake) in game_state.snakes.iter().enumerate() {
            if !snake.is_game_over {
                inspected_agent.snake_idx = Some(idx);
                inspector_state.follow_snake_id = Some(idx);
                break;
            }
        }
        // If all dead, select first one anyway
        if inspected_agent.snake_idx.is_none() && !game_state.snakes.is_empty() {
            inspected_agent.snake_idx = Some(0);
            inspector_state.follow_snake_id = Some(0);
        }
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

/// Update the inspected agent (handle agent death, follow selection)
fn update_inspected_agent(
    game_state: Res<GameState>,
    mut inspected_agent: ResMut<InspectedAgent>,
    inspector_state: Res<BrainInspectorState>,
) {
    // If following a specific snake, update the index
    if let Some(follow_id) = inspector_state.follow_snake_id {
        if follow_id < game_state.snakes.len() {
            inspected_agent.snake_idx = Some(follow_id);
        }
    }

    // Validate current selection
    if let Some(idx) = inspected_agent.snake_idx {
        if idx >= game_state.snakes.len() {
            // Selected snake no longer exists, clear selection
            inspected_agent.snake_idx = None;
            inspected_agent.last_sensor_state = None;
            inspected_agent.last_output = None;
        }
    }
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
