//! Brain Inspector UI Components
//!
//! UI layout and rendering for the brain inspector view.
//! Follows the same patterns as the main UI in ui.rs

use bevy::prelude::*;

use crate::brain::{HIDDEN2_SIZE, HIDDEN_SIZE, INPUT_SIZE, OUTPUT_SIZE};
use crate::brain_inspector::{BrainInspectorState, BrainInspectorUi, InspectedAgent, InspectorTab};
use crate::snake::GameState;

// ============================================================================
// UI SPAWN SYSTEMS
// ============================================================================

/// Spawn the brain inspector UI panel
pub fn spawn_inspector_ui(mut commands: Commands, inspector_state: Res<BrainInspectorState>) {
    let panel_width = 400.0;
    let panel_height = 600.0;

    commands
        .spawn((
            NodeBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    right: Val::Px(10.0),
                    top: Val::Px(10.0),
                    width: Val::Px(panel_width),
                    height: Val::Px(panel_height),
                    flex_direction: FlexDirection::Column,
                    padding: UiRect::all(Val::Px(10.0)),
                    ..default()
                },
                background_color: Color::rgba(0.1, 0.1, 0.15, 0.95).into(),
                ..default()
            },
            BrainInspectorUi,
        ))
        .with_children(|parent| {
            // Header
            spawn_header(parent);

            // Tab bar
            spawn_tab_bar(parent, &inspector_state);

            // Content area
            parent
                .spawn((
                    NodeBundle {
                        style: Style {
                            width: Val::Percent(100.0),
                            height: Val::Percent(100.0),
                            flex_direction: FlexDirection::Column,
                            overflow: Overflow::clip(),
                            ..default()
                        },
                        background_color: Color::rgba(0.05, 0.05, 0.08, 0.9).into(),
                        ..default()
                    },
                    InspectorContent,
                ))
                .with_children(|content| {
                    // Content will be populated based on active tab
                    spawn_placeholder_content(content);
                });
        });
}

/// Spawn the inspector panel header
fn spawn_header(parent: &mut ChildBuilder) {
    parent
        .spawn(NodeBundle {
            style: Style {
                width: Val::Percent(100.0),
                height: Val::Px(40.0),
                flex_direction: FlexDirection::Row,
                justify_content: JustifyContent::SpaceBetween,
                align_items: AlignItems::Center,
                margin: UiRect::bottom(Val::Px(10.0)),
                ..default()
            },
            background_color: Color::rgb(0.2, 0.2, 0.3).into(),
            ..default()
        })
        .with_children(|header| {
            header.spawn(TextBundle::from_section(
                "Brain Inspector",
                TextStyle {
                    font_size: 20.0,
                    color: Color::WHITE,
                    ..default()
                },
            ));

            header.spawn(TextBundle::from_section(
                "[1]Sim [2]Inspect | [S]ensors [W]eights [A]ctivations | [→] Next [←] Prev [H]ide",
                TextStyle {
                    font_size: 11.0,
                    color: Color::GRAY,
                    ..default()
                },
            ));
        });
}

/// Spawn the tab selection bar
fn spawn_tab_bar(parent: &mut ChildBuilder, inspector_state: &BrainInspectorState) {
    parent
        .spawn(NodeBundle {
            style: Style {
                width: Val::Percent(100.0),
                height: Val::Px(35.0),
                flex_direction: FlexDirection::Row,
                justify_content: JustifyContent::SpaceEvenly,
                margin: UiRect::bottom(Val::Px(10.0)),
                ..default()
            },
            background_color: Color::rgba(0.15, 0.15, 0.2, 1.0).into(),
            ..default()
        })
        .with_children(|tabs| {
            let tab_names = [
                (
                    InspectorTab::Sensors,
                    "Sensors (S)",
                    "Input sensors & raycasting",
                ),
                (
                    InspectorTab::Weights,
                    "Weights (W)",
                    "Neural network weights",
                ),
                (
                    InspectorTab::Activations,
                    "Activations (A)",
                    "Layer activations",
                ),
            ];

            for (tab, name, _desc) in tab_names {
                let is_active = inspector_state.active_tab == tab;
                let bg_color = if is_active {
                    Color::rgb(0.3, 0.5, 0.7)
                } else {
                    Color::rgb(0.2, 0.2, 0.25)
                };

                tabs.spawn(NodeBundle {
                    style: Style {
                        width: Val::Percent(32.0),
                        height: Val::Percent(100.0),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    background_color: bg_color.into(),
                    ..default()
                })
                .with_children(|tab_node| {
                    tab_node.spawn(TextBundle::from_section(
                        name,
                        TextStyle {
                            font_size: 13.0,
                            color: if is_active { Color::WHITE } else { Color::GRAY },
                            ..default()
                        },
                    ));
                });
            }
        });
}

/// Spawn placeholder content (will be replaced by actual content)
fn spawn_placeholder_content(parent: &mut ChildBuilder) {
    parent.spawn(TextBundle::from_section(
        "Select an agent to inspect\nUse arrow keys or N/P to navigate\nPress H to hide this panel",
        TextStyle {
            font_size: 14.0,
            color: Color::GRAY,
            ..default()
        },
    ));
}

// ============================================================================
// CONTENT UPDATE SYSTEMS
// ============================================================================

/// Marker component for inspector content area
#[derive(Component)]
pub struct InspectorContent;

/// Marker for sensor visualization elements
#[derive(Component)]
pub struct SensorVisualization;

/// Marker for weight visualization elements
#[derive(Component)]
pub struct WeightVisualization;

/// Marker for activation visualization elements
#[derive(Component)]
pub struct ActivationVisualization;

/// Update the inspector content based on active tab and selected agent
pub fn update_inspector_content(
    mut commands: Commands,
    inspector_state: Res<BrainInspectorState>,
    inspected_agent: Res<InspectedAgent>,
    game_state: Res<GameState>,
    content_query: Query<Entity, With<InspectorContent>>,
    children_query: Query<&Children>,
) {
    // Only update when state changes
    if !inspector_state.is_changed() && !inspected_agent.is_changed() {
        return;
    }

    // Clear old content
    for content_entity in content_query.iter() {
        if let Ok(children) = children_query.get(content_entity) {
            for &child in children.iter() {
                commands.entity(child).despawn_recursive();
            }
        }

        // Spawn new content based on tab
        commands
            .entity(content_entity)
            .with_children(|parent| match inspector_state.active_tab {
                InspectorTab::Sensors => spawn_sensors_tab(parent, &inspected_agent, &game_state),
                InspectorTab::Weights => spawn_weights_tab(parent, &inspected_agent),
                InspectorTab::Activations => spawn_activations_tab(parent, &inspected_agent),
            });
    }
}

// ============================================================================
// SENSORS TAB
// ============================================================================

fn spawn_sensors_tab(
    parent: &mut ChildBuilder,
    inspected: &InspectedAgent,
    game_state: &GameState,
) {
    // Get inspected snake info
    let (snake_info, status) = match inspected.snake_idx {
        Some(idx) => match game_state.snakes.get(idx) {
            Some(snake) => {
                let status = if snake.is_game_over { "DEAD" } else { "ALIVE" };
                (
                    format!("Snake {} - Score: {} - {}", idx, snake.score, status),
                    status,
                )
            }
            None => ("Invalid snake index".to_string(), "ERROR"),
        },
        None => ("No snake selected".to_string(), "NONE"),
    };

    // Header
    parent.spawn(TextBundle::from_section(
        &snake_info,
        TextStyle {
            font_size: 16.0,
            color: if status == "ALIVE" {
                Color::GREEN
            } else {
                Color::RED
            },
            ..default()
        },
    ));

    // Sensor values
    if let Some(sensors) = inspected.last_sensor_state {
        parent.spawn(NodeBundle {
            style: Style {
                width: Val::Percent(100.0),
                height: Val::Px(10.0),
                ..default()
            },
            ..default()
        });

        // Obstacle sensors (8 rays)
        parent.spawn(TextBundle::from_section(
            "Obstacle Sensors (8 rays):",
            TextStyle {
                font_size: 14.0,
                color: Color::YELLOW,
                ..default()
            },
        ));

        spawn_sensor_grid(parent, &sensors[0..8], "Ray");

        // Target direction sensors (8 values)
        parent.spawn(NodeBundle {
            style: Style {
                width: Val::Percent(100.0),
                height: Val::Px(10.0),
                ..default()
            },
            ..default()
        });

        parent.spawn(TextBundle::from_section(
            "Target Direction (dot products):",
            TextStyle {
                font_size: 14.0,
                color: Color::CYAN,
                ..default()
            },
        ));

        spawn_sensor_grid(parent, &sensors[8..16], "Dir");

        // Distance sensor
        parent.spawn(NodeBundle {
            style: Style {
                width: Val::Percent(100.0),
                height: Val::Px(10.0),
                ..default()
            },
            ..default()
        });

        parent.spawn(TextBundle::from_section(
            &format!("Distance to food: {:.3}", sensors[16]),
            TextStyle {
                font_size: 14.0,
                color: Color::rgb(1.0, 0.0, 1.0), // Magenta
                ..default()
            },
        ));
    } else {
        parent.spawn(TextBundle::from_section(
            "No sensor data available",
            TextStyle {
                font_size: 14.0,
                color: Color::GRAY,
                ..default()
            },
        ));
    }

    // Neural output
    if let Some(output) = inspected.last_output {
        parent.spawn(NodeBundle {
            style: Style {
                width: Val::Percent(100.0),
                height: Val::Px(20.0),
                ..default()
            },
            ..default()
        });

        parent.spawn(TextBundle::from_section(
            "Neural Network Output:",
            TextStyle {
                font_size: 14.0,
                color: Color::WHITE,
                ..default()
            },
        ));

        let actions = ["Left", "Straight", "Right"];
        let max_idx = output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(1);

        for (i, (action, value)) in actions.iter().zip(output.iter()).enumerate() {
            let color = if i == max_idx {
                Color::GREEN
            } else {
                Color::GRAY
            };
            let bar = "█".repeat((value.abs() * 20.0) as usize);
            parent.spawn(TextBundle::from_section(
                &format!("{}: {:>8.3} {}", action, value, bar),
                TextStyle {
                    font_size: 13.0,
                    color,
                    ..default()
                },
            ));
        }
    }
}

fn spawn_sensor_grid(parent: &mut ChildBuilder, values: &[f32], _label: &str) {
    parent
        .spawn(NodeBundle {
            style: Style {
                width: Val::Percent(100.0),
                height: Val::Px(60.0),
                flex_direction: FlexDirection::Row,
                flex_wrap: FlexWrap::Wrap,
                margin: UiRect::vertical(Val::Px(5.0)),
                ..default()
            },
            ..default()
        })
        .with_children(|grid| {
            for (i, &value) in values.iter().enumerate() {
                let intensity = (value * 255.0) as u8;
                let color = Color::rgb(
                    intensity as f32 / 255.0,
                    (255 - intensity) as f32 / 255.0,
                    0.2,
                );

                grid.spawn(NodeBundle {
                    style: Style {
                        width: Val::Percent(25.0),
                        height: Val::Px(25.0),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        margin: UiRect::all(Val::Px(2.0)),
                        ..default()
                    },
                    background_color: color.into(),
                    ..default()
                })
                .with_children(|cell| {
                    cell.spawn(TextBundle::from_section(
                        &format!("{}:{:.2}", i, value),
                        TextStyle {
                            font_size: 10.0,
                            color: Color::WHITE,
                            ..default()
                        },
                    ));
                });
            }
        });
}

// ============================================================================
// WEIGHTS TAB
// ============================================================================

fn spawn_weights_tab(parent: &mut ChildBuilder, inspected: &InspectedAgent) {
    parent.spawn(TextBundle::from_section(
        "Neural Network Weights",
        TextStyle {
            font_size: 16.0,
            color: Color::WHITE,
            ..default()
        },
    ));

    parent.spawn(NodeBundle {
        style: Style {
            width: Val::Percent(100.0),
            height: Val::Px(10.0),
            ..default()
        },
        ..default()
    });

    // Network architecture info
    parent.spawn(TextBundle::from_section(
        &format!(
            "Architecture: {} → {} → {} → {}",
            INPUT_SIZE, HIDDEN_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE
        ),
        TextStyle {
            font_size: 13.0,
            color: Color::GRAY,
            ..default()
        },
    ));

    parent.spawn(TextBundle::from_section(
        "Total parameters: 12,931",
        TextStyle {
            font_size: 13.0,
            color: Color::GRAY,
            ..default()
        },
    ));

    parent.spawn(NodeBundle {
        style: Style {
            width: Val::Percent(100.0),
            height: Val::Px(20.0),
            ..default()
        },
        ..default()
    });

    parent.spawn(TextBundle::from_section(
        "Weight visualization coming soon...",
        TextStyle {
            font_size: 14.0,
            color: Color::GRAY,
            ..default()
        },
    ));

    parent.spawn(TextBundle::from_section(
        "Use [N]/[P] to inspect different agents",
        TextStyle {
            font_size: 12.0,
            color: Color::DARK_GRAY,
            ..default()
        },
    ));
}

// ============================================================================
// ACTIVATIONS TAB
// ============================================================================

fn spawn_activations_tab(parent: &mut ChildBuilder, inspected: &InspectedAgent) {
    parent.spawn(TextBundle::from_section(
        "Layer Activations",
        TextStyle {
            font_size: 16.0,
            color: Color::WHITE,
            ..default()
        },
    ));

    parent.spawn(NodeBundle {
        style: Style {
            width: Val::Percent(100.0),
            height: Val::Px(10.0),
            ..default()
        },
        ..default()
    });

    if let Some(output) = inspected.last_output {
        parent.spawn(TextBundle::from_section(
            "Output Layer Activations:",
            TextStyle {
                font_size: 14.0,
                color: Color::YELLOW,
                ..default()
            },
        ));

        let actions = ["Left", "Straight", "Right"];
        for (action, value) in actions.iter().zip(output.iter()) {
            let bar_length = (value.abs().min(1.0) * 30.0) as usize;
            let bar = if *value >= 0.0 {
                "█".repeat(bar_length)
            } else {
                "░".repeat(bar_length)
            };

            parent.spawn(TextBundle::from_section(
                &format!("{:>10}: {:>7.3} {}", action, value, bar),
                TextStyle {
                    font_size: 13.0,
                    color: Color::WHITE,
                    ..default()
                },
            ));
        }
    } else {
        parent.spawn(TextBundle::from_section(
            "No activation data available\nSelect an agent to view activations",
            TextStyle {
                font_size: 14.0,
                color: Color::GRAY,
                ..default()
            },
        ));
    }
}

// ============================================================================
// VISIBILITY SYSTEM
// ============================================================================

/// Update inspector UI visibility based on state.
/// Handles [H] toggle: despawn when hidden, respawn when shown again.
pub fn update_inspector_visibility(
    mut commands: Commands,
    inspector_state: Res<BrainInspectorState>,
    current_state: Res<State<crate::brain_inspector::AppState>>,
    ui_query: Query<Entity, With<BrainInspectorUi>>,
) {
    // Only relevant in BrainInspectorView
    if current_state.get() != &crate::brain_inspector::AppState::BrainInspectorView {
        return;
    }

    // Only act when inspector_state actually changed (avoids work every frame)
    if !inspector_state.is_changed() {
        return;
    }

    let ui_exists = !ui_query.is_empty();

    if inspector_state.panel_visible && !ui_exists {
        // Respawn panel (e.g. after [H] toggled it back on)
        spawn_inspector_ui(commands, inspector_state);
    } else if !inspector_state.panel_visible && ui_exists {
        for entity in ui_query.iter() {
            commands.entity(entity).despawn_recursive();
        }
    }
}
