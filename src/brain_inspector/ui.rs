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

/// Direction labels matching index.html: FWD, F-R, R, B-R, BCK, B-L, L, F-L
const DIR_LABELS: [&str; 8] = ["FWD", "F-R", "R", "B-R", "BCK", "B-L", "L", "F-L"];

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

    // Header with status color
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

    // Spacer
    parent.spawn(NodeBundle {
        style: Style {
            width: Val::Percent(100.0),
            height: Val::Px(8.0),
            ..default()
        },
        ..default()
    });

    // === OBSTACLE RAYS [0-7] - Orange bars from left ===
    parent.spawn(TextBundle::from_section(
        "OBSTACLE RAYS [0-7]",
        TextStyle {
            font_size: 12.0,
            color: Color::rgb(1.0, 0.42, 0.21), // Orange
            ..default()
        },
    ));

    if let Some(sensors) = inspected.last_sensor_state {
        // Render 8 horizontal bars for obstacle sensors
        for i in 0..8 {
            let value = sensors[i];
            spawn_horizontal_bar(
                parent,
                i,
                DIR_LABELS[i],
                value,
                Color::rgb(1.0, 0.42, 0.21), // Orange
                true,                        // left-aligned
            );
        }

        // === FOOD DIRECTION [8-15] - Blue bars centered ===
        parent.spawn(NodeBundle {
            style: Style {
                width: Val::Percent(100.0),
                height: Val::Px(12.0),
                ..default()
            },
            ..default()
        });

        parent.spawn(TextBundle::from_section(
            "FOOD DIRECTION [8-15]",
            TextStyle {
                font_size: 12.0,
                color: Color::rgb(0.22, 0.68, 1.0), // Blue
                ..default()
            },
        ));

        for i in 0..8 {
            let value = sensors[8 + i];
            // Food direction can be negative (away from food) - render centered
            let bar_color = if value >= 0.0 {
                Color::rgb(0.22, 0.68, 1.0) // Blue for positive
            } else {
                Color::rgb(0.5, 0.3, 0.8) // Violet for negative
            };
            spawn_horizontal_bar(
                parent,
                8 + i,
                DIR_LABELS[i],
                value,
                bar_color,
                false, // centered
            );
        }

        // === FOOD PROXIMITY [16] ===
        parent.spawn(NodeBundle {
            style: Style {
                width: Val::Percent(100.0),
                height: Val::Px(12.0),
                ..default()
            },
            ..default()
        });

        parent.spawn(TextBundle::from_section(
            "FOOD PROXIMITY [16]",
            TextStyle {
                font_size: 12.0,
                color: Color::rgb(1.0, 0.0, 1.0), // Magenta
                ..default()
            },
        ));

        spawn_horizontal_bar(
            parent,
            16,
            "PROX",
            sensors[16],
            Color::rgb(1.0, 0.0, 1.0), // Magenta
            true,
        );
    } else {
        parent.spawn(TextBundle::from_section(
            "No sensor data - select snake with [1-9] or [←][→]",
            TextStyle {
                font_size: 12.0,
                color: Color::GRAY,
                ..default()
            },
        ));
    }

    // === NEURAL OUTPUT ===
    parent.spawn(NodeBundle {
        style: Style {
            width: Val::Percent(100.0),
            height: Val::Px(16.0),
            ..default()
        },
        ..default()
    });

    parent.spawn(TextBundle::from_section(
        "NEURAL OUTPUT",
        TextStyle {
            font_size: 12.0,
            color: Color::GREEN,
            ..default()
        },
    ));

    // Neural output - 3 action cards
    if let Some(output) = inspected.last_output {
        let actions = ["LEFT", "STRAIGHT", "RIGHT"];
        let max_idx = output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(1);

        // Horizontal layout for 3 actions
        parent
            .spawn(NodeBundle {
                style: Style {
                    width: Val::Percent(100.0),
                    flex_direction: FlexDirection::Row,
                    justify_content: JustifyContent::SpaceEvenly,
                    ..default()
                },
                ..default()
            })
            .with_children(|row| {
                for (i, (action, &value)) in actions.iter().zip(output.iter()).enumerate() {
                    let is_max = i == max_idx;
                    let action_color = if is_max { Color::GREEN } else { Color::GRAY };
                    let bar_width = (value.clamp(-1.0, 1.0).abs() * 50.0) as f32;

                    row.spawn((NodeBundle {
                        style: Style {
                            width: Val::Percent(30.0),
                            flex_direction: FlexDirection::Column,
                            align_items: AlignItems::Center,
                            padding: UiRect::all(Val::Px(4.0)),
                            ..default()
                        },
                        background_color: BackgroundColor(Color::rgba(0.1, 0.1, 0.15, 0.8)),
                        ..default()
                    },))
                        .with_children(|card| {
                            // Action name
                            card.spawn(TextBundle::from_section(
                                action.to_string(),
                                TextStyle {
                                    font_size: 11.0,
                                    color: action_color,
                                    ..default()
                                },
                            ));
                            // Value
                            card.spawn(TextBundle::from_section(
                                &format!("{:.3}", value),
                                TextStyle {
                                    font_size: 14.0,
                                    color: if is_max { Color::GREEN } else { Color::WHITE },
                                    ..default()
                                },
                            ));
                            // Progress bar
                            card.spawn(NodeBundle {
                                style: Style {
                                    width: Val::Px(bar_width),
                                    height: Val::Px(6.0),
                                    ..default()
                                },
                                background_color: BackgroundColor(action_color),
                                ..default()
                            });
                        });
                }
            });
    } else {
        parent.spawn(TextBundle::from_section(
            "No NN output - snake may be dead",
            TextStyle {
                font_size: 11.0,
                color: Color::GRAY,
                ..default()
            },
        ));
    }
}

/// Spawn a horizontal bar with label and value (index.html style)
fn spawn_horizontal_bar(
    parent: &mut ChildBuilder,
    index: usize,
    label: &str,
    value: f32,
    color: Color,
    left_aligned: bool,
) {
    // Bar container
    parent
        .spawn(NodeBundle {
            style: Style {
                width: Val::Percent(100.0),
                height: Val::Px(16.0),
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                margin: UiRect::vertical(Val::Px(1.0)),
                ..default()
            },
            ..default()
        })
        .with_children(|row| {
            // Label
            row.spawn(TextBundle::from_section(
                format!("[{:02}] {}", index, label),
                TextStyle {
                    font_size: 10.0,
                    color: Color::GRAY,
                    ..default()
                },
            ));

            // Bar track
            row.spawn((NodeBundle {
                style: Style {
                    width: Val::Percent(100.0),
                    height: Val::Px(8.0),
                    margin: UiRect::horizontal(Val::Px(4.0)),
                    ..default()
                },
                background_color: BackgroundColor(Color::rgba(0.15, 0.15, 0.2, 1.0)),
                ..default()
            },))
                .with_children(|track| {
                    let bar_width = if left_aligned {
                        Val::Percent(value.clamp(0.0, 1.0) * 100.0)
                    } else {
                        // Centered bar: value 0 = center, -1 = left edge, 1 = right edge
                        let pct = value.clamp(-1.0, 1.0).abs() * 50.0;
                        Val::Percent(pct)
                    };

                    // For centered bars: start from center (50%), offset based on sign
                    // value >= 0: bar extends from center to right
                    // value < 0: bar extends from center to left
                    let bar_left = if left_aligned {
                        Val::Percent(0.0)
                    } else if value >= 0.0 {
                        Val::Percent(50.0) // Start at center for positive values
                    } else {
                        Val::Percent(50.0 - value.abs() * 50.0) // Offset left for negative
                    };

                    track.spawn((NodeBundle {
                        style: Style {
                            position_type: PositionType::Absolute,
                            left: bar_left,
                            width: bar_width,
                            height: Val::Percent(100.0),
                            ..default()
                        },
                        background_color: BackgroundColor(color),
                        ..default()
                    },));
                });

            // Value
            row.spawn(TextBundle::from_section(
                &format!("{:.3}", value),
                TextStyle {
                    font_size: 10.0,
                    color: Color::WHITE,
                    ..default()
                },
            ));
        });
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

fn spawn_weights_tab(parent: &mut ChildBuilder, _inspected: &InspectedAgent) {
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
/// Handles panel visibility based on PanelVisibility resource
pub fn update_inspector_visibility(
    mut commands: Commands,
    inspector_state: Res<BrainInspectorState>,
    panel_visibility: Res<crate::ui::PanelVisibility>,
    ui_query: Query<Entity, With<BrainInspectorUi>>,
) {
    // Only act when panel_visibility actually changed (avoids work every frame)
    if !panel_visibility.is_changed() {
        return;
    }

    let ui_exists = !ui_query.is_empty();

    // Show panel if inspector flag is true and no UI exists
    if panel_visibility.inspector && !ui_exists {
        spawn_inspector_ui(commands, inspector_state);
    } else if !panel_visibility.inspector && ui_exists {
        // Hide panel
        for entity in ui_query.iter() {
            commands.entity(entity).despawn_recursive();
        }
    }
}
