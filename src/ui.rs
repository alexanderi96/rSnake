//! UI systems for MAP-Elites Snake

use bevy::app::AppExit;
use bevy::prelude::*;
use bevy::sprite::MaterialMesh2dBundle;
use bevy::window::WindowMode;

use crate::snake::{
    AppStartTime, CollisionSettings, Food, GameConfig, GameState, GameStats, GlobalTrainingHistory,
    GridDimensions, GridMap, MeshCache, Position, RenderConfig, SegmentPool, SnakeId,
    SnakeInstance, TrainingStats, BLOCK_SIZE,
};
use crate::types::GameSnapshot;

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

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(WindowSettings {
            is_fullscreen: false,
        })
        .insert_resource(GraphPanelState::default())
        .add_systems(Update, (handle_input, on_window_resize, render_system))
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
        );
    }
}

pub fn spawn_stats_ui(mut commands: Commands, game: Res<GameState>) {
    let mut leaderboard_sections = vec![TextSection::new(
        "[LEADERBOARD]\n",
        TextStyle {
            font_size: 18.0,
            color: Color::GOLD,
            ..default()
        },
    )];

    for _snake in game.snakes.iter() {
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
            "[R]Render:ON  [G]Graph  [F]Fullscreen  [C]Collision:OFF  [ESC]Exit",
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
    stats: Res<TrainingStats>,
    game_stats: Res<GameStats>,
    collision_settings: Res<CollisionSettings>,
    render_config: Res<RenderConfig>,
    app_start_time: Res<AppStartTime>,
    global_history: Res<GlobalTrainingHistory>,
) {
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

    if let Ok(mut lb_text) = leaderboard_query.get_single_mut() {
        for (rank, (original_idx, snake)) in snake_data.iter().enumerate() {
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
            "[R]Render:{}  [G]Graph:{}  [F]Fullscreen  [C]Collision:{}  [ESC]Exit",
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
    mut window_settings: ResMut<WindowSettings>,
    mut windows: Query<&mut Window>,
    mut collision_settings: ResMut<CollisionSettings>,
    mut render_config: ResMut<RenderConfig>,
    mut graph_state: ResMut<GraphPanelState>,
) {
    use crate::snake::get_or_create_run_dir;
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

pub fn on_window_resize(
    mut resize_events: EventReader<bevy::window::WindowResized>,
    mut grid: ResMut<GridDimensions>,
    mut game: ResMut<GameState>,
    mut grid_map: ResMut<GridMap>,
    mut graph_state: ResMut<GraphPanelState>,
) {
    for event in resize_events.read() {
        let (new_width, new_height) =
            crate::snake::calculate_grid_dimensions(event.width, event.height);

        grid.width = new_width;
        grid.height = new_height;
        *grid_map = GridMap::new(new_width, new_height);

        let total_snakes = game.snakes.len();
        for snake in game.snakes.iter_mut() {
            snake.reset(&grid, total_snakes);
        }

        graph_state.needs_redraw = true;
        println!(
            "Resized: GridMap re-initialized to {}x{}",
            new_width, new_height
        );
    }
}

/// Rendering system - now uses game state directly from ECS
pub fn render_system(
    mut commands: Commands,
    mut game: ResMut<GameState>,
    mut global_history: ResMut<GlobalTrainingHistory>,
    windows: Query<&Window>,
    mesh_cache: Res<MeshCache>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut segment_pool: ResMut<SegmentPool>,
    render_config: Res<RenderConfig>,
    q_food: Query<Entity, With<Food>>,
    mut stats: ResMut<TrainingStats>,
) {
    if !render_config.enabled {
        return;
    }

    // Update FPS
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

    // Remove old food
    for e in q_food.iter() {
        commands.entity(e).despawn();
    }

    let ui_padding = 60.0;
    let offset_x = -window.resolution.width() / 2.0 + BLOCK_SIZE / 2.0;
    let offset_y = window.resolution.height() / 2.0 - ui_padding - BLOCK_SIZE / 2.0;

    // Render the snakes
    for snake in game.snakes.iter() {
        if snake.is_game_over {
            segment_pool.hide_excess(&mut commands, snake.id, 0);
            continue;
        }

        let body_material = materials.add(snake.color);
        let snake_len = snake.snake.len();

        for (i, pos) in snake.snake.iter().enumerate() {
            let material = if i == 0 {
                mesh_cache.head_material.clone()
            } else {
                body_material.clone()
            };

            let transform = Transform::from_xyz(
                offset_x + (pos.x as f32 * BLOCK_SIZE),
                offset_y - (pos.y as f32 * BLOCK_SIZE),
                0.0,
            );

            segment_pool.get_or_spawn(
                &mut commands,
                snake.id,
                i,
                mesh_cache.segment_mesh.clone(),
                material,
                transform,
            );
        }

        segment_pool.hide_excess(&mut commands, snake.id, snake_len);
        segment_pool.set_active_count(snake.id, snake_len);

        // Render food
        commands.spawn((
            MaterialMesh2dBundle {
                mesh: mesh_cache.food_mesh.clone().into(),
                material: mesh_cache.food_material.clone(),
                transform: Transform::from_xyz(
                    offset_x + (snake.food.x as f32 * BLOCK_SIZE),
                    offset_y - (snake.food.y as f32 * BLOCK_SIZE),
                    0.0,
                ),
                ..default()
            },
            Food,
            SnakeId(snake.id),
        ));
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
                let h_min = get_height(point.min);

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
