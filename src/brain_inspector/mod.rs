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
        // Resources always available (no state-based filtering)
        app.insert_resource(InspectedAgent::default())
            .insert_resource(BrainInspectorState::default())
            // Input handling - unified, no state restrictions
            .add_systems(Update, inspector_input_system)
            // Always run sensor cache and death handling
            .add_systems(Update, (handle_agent_death_and_switch, update_sensor_cache));
    }
}

// ============================================================================
// INPUT SYSTEMS
// ============================================================================

/// Unified input system - handles all keybindings in single view
/// Panel toggles: [I] Inspector, [L] Leaderboard, [K] Keybindings
/// Note: [G] Graph and [B] Heatmap are handled in handle_input (ui.rs) with complex state
fn inspector_input_system(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut inspector_state: ResMut<BrainInspectorState>,
    game_state: Res<GameState>,
    mut inspected_agent: ResMut<InspectedAgent>,
    mut panel_visibility: ResMut<crate::ui::PanelVisibility>,
) {
    // === PANEL TOGGLES ===
    // [I] Toggle Inspector Panel
    if keyboard_input.just_pressed(KeyCode::KeyI) {
        panel_visibility.inspector = !panel_visibility.inspector;
        println!(
            "[PANEL] Inspector: {}",
            if panel_visibility.inspector {
                "ON"
            } else {
                "OFF"
            }
        );
    }
    // [L] Toggle Leaderboard Panel
    if keyboard_input.just_pressed(KeyCode::KeyL) {
        panel_visibility.leaderboard = !panel_visibility.leaderboard;
        println!(
            "[PANEL] Leaderboard: {}",
            if panel_visibility.leaderboard {
                "ON"
            } else {
                "OFF"
            }
        );
    }
    // [K] Toggle Keybindings Panel
    if keyboard_input.just_pressed(KeyCode::KeyK) {
        panel_visibility.keybindings = !panel_visibility.keybindings;
        println!(
            "[PANEL] Keybindings: {}",
            if panel_visibility.keybindings {
                "ON"
            } else {
                "OFF"
            }
        );
    }

    // === INSPECTOR CONTROLS ===
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

    // Next/Previous agent navigation (arrow keys only)
    if keyboard_input.just_pressed(KeyCode::ArrowRight) {
        navigate_to_next_agent(&mut inspected_agent, &mut inspector_state, &game_state);
    }

    if keyboard_input.just_pressed(KeyCode::ArrowLeft) {
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
    collision_settings: Res<crate::snake::CollisionSettings>,
) {
    use crate::snake::get_current_17_state;

    let Some(idx) = inspected_agent.snake_idx else {
        return;
    };

    let Some(snake) = game_state.snakes.get(idx) else {
        return;
    };

    // Calculate current sensor state using the same function as training
    let sensor_state =
        get_current_17_state(snake, &grid_map, &grid, collision_settings.snake_vs_snake);
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
