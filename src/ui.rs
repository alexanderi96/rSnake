use bevy::app::AppExit;
use bevy::prelude::*;
use bevy::sprite::MaterialMesh2dBundle;
use bevy::window::WindowMode;

use crate::agent::{AgentConfig, DqnAgent};
use crate::snake::{
    get_state_egocentric, spawn_food, AppStartTime, CollisionSettings, Direction, Food, GameConfig,
    GameState, GameStats, GenerationRecord, GlobalTrainingHistory, GridDimensions, GridMap,
    MeshCache, ParallelConfig, Position, RenderConfig, SegmentPool, SnakeId, SnakeInstance,
    SnakeSegment, TrainingSession, TrainingStats, BLOCK_SIZE, TRAIN_INTERVAL,
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
        // spawn_stats_ui is NOT added here - it's added in main.rs after setup
        .add_systems(
            Update,
            (
                handle_input,
                on_window_resize,
                game_loop_system,
                render_system,
                toggle_graph_panel,
            )
                .chain(),
        )
        .add_systems(Update, update_stats_ui.after(render_system))
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
    agent: Res<DqnAgent>,
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
    let alive_count = game.snakes.iter().filter(|s| !s.is_game_over).count();
    let dead_count = game.snakes.len() - alive_count;

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
            "H: {:3}  G: {:5}  Best: {:3}\n",
            game.high_score, game.total_iterations, persistent_high
        );
        st_text.sections[1].value = format!(
            "Session: {:02}:{:02}:{:02}  Total: {:02}:{:02}:{:02}  FPS: {:5.1}\n",
            session_hours,
            session_minutes,
            session_seconds,
            total_hours,
            total_minutes,
            total_seconds,
            stats.fps
        );
        st_text.sections[2].value = format!(
            "Alive:{} Dead:{} | Food: {}  Games: {} Eps: {:.3}",
            alive_count,
            dead_count,
            game_stats.total_food_eaten,
            game_stats.total_games_played,
            agent.epsilon
        );
    }

    if let Ok(mut cmd_text) = commands_query.get_single_mut() {
        let render_status = if render_config.enabled { "ON" } else { "TURBO" };
        let collision_status = if collision_settings.snake_vs_snake {
            "ON"
        } else {
            "OFF"
        };
        cmd_text.sections[0].value = format!(
            "[R]Render:{}  [G]Graph  [F]Fullscreen  [C]Collision:{}  [ESC]Exit",
            render_status, collision_status
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn handle_input(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut app_exit_events: EventWriter<AppExit>,
    mut game_stats: ResMut<GameStats>,
    mut training_session: ResMut<TrainingSession>,
    mut agent: ResMut<DqnAgent>,
    config: Res<GameConfig>,
    game: Res<GameState>,
    mut window_settings: ResMut<WindowSettings>,
    mut windows: Query<&mut Window>,
    mut collision_settings: ResMut<CollisionSettings>,
    mut render_config: ResMut<RenderConfig>,
    mut graph_state: ResMut<GraphPanelState>,
    app_start_time: Res<AppStartTime>,
    global_history: Res<GlobalTrainingHistory>,
) {
    use crate::snake::{get_or_create_run_dir, session_path};
    use std::time::Instant;

    if keyboard_input.just_pressed(KeyCode::Escape) {
        game_stats.update(&game);

        // FIX: Save brain model before exiting
        let brain_path = crate::snake::brain_path();
        println!("💾 Saving brain to: {}", brain_path.display());
        if let Err(e) = agent.save(brain_path.to_str().unwrap_or("brain.bin")) {
            eprintln!("⚠️ Error saving brain: {}", e);
        } else {
            println!("✅ Brain saved successfully!");
        }

        let current_session_duration = Instant::now().duration_since(app_start_time.0);
        training_session.total_time_secs = current_session_duration.as_secs();
        training_session.compress_history();
        if let Err(e) =
            training_session.save(config.session_path.to_str().unwrap_or("session.json"))
        {
            eprintln!("Error saving session: {}", e);
        }

        let current_session_duration = Instant::now().duration_since(app_start_time.0);
        let total_training_time =
            std::time::Duration::from_secs(global_history.accumulated_time_secs)
                + current_session_duration;

        println!("\n=== SESSION SUMMARY ===");
        println!("Total generations: {}", game_stats.total_generations);
        println!("High Score: {}", game_stats.high_score);
        println!(
            "Current session time: {}s",
            current_session_duration.as_secs()
        );
        println!("Total time (runtime): {}s", total_training_time.as_secs());
        println!("Records in session: {}", training_session.records.len());
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
        graph_state.visible = !graph_state.visible;
        graph_state.needs_redraw = true;
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

        for snake in game.snakes.iter_mut() {
            snake.reset(&grid);
        }

        graph_state.needs_redraw = true;
        println!(
            "Resized: GridMap re-initialized to {}x{}",
            new_width, new_height
        );
    }
}

pub fn game_loop_system(
    time: Res<Time>,
    mut config: ResMut<GameConfig>,
    mut game: ResMut<GameState>,
    mut agent: ResMut<DqnAgent>,
    mut game_stats: ResMut<GameStats>,
    mut training_session: ResMut<TrainingSession>,
    mut global_history: ResMut<GlobalTrainingHistory>,
    mut stats: ResMut<TrainingStats>,
    grid: Res<GridDimensions>,
    collision_settings: Res<CollisionSettings>,
    parallel_config: Res<ParallelConfig>,
    render_config: Res<RenderConfig>,
    mut grid_map: ResMut<GridMap>,
) {
    use std::time::{Duration, Instant};

    stats.frame_count += 1;
    let now = Instant::now();
    if now.duration_since(stats.last_fps_update).as_secs_f32() >= 1.0 {
        stats.fps =
            stats.frame_count as f32 / now.duration_since(stats.last_fps_update).as_secs_f32();
        stats.last_fps_update = now;
        stats.frame_count = 0;
    }

    if render_config.enabled {
        config.speed_timer.tick(time.delta());
        if !config.speed_timer.finished() {
            return;
        }
        run_simulation_step(
            &mut game,
            &mut agent,
            &mut game_stats,
            &mut training_session,
            &mut global_history,
            &config,
            &grid,
            &collision_settings,
            &parallel_config,
            &mut stats,
            &mut grid_map,
        );
    } else {
        let ui_target = Duration::from_millis(33);
        let buffer = Duration::from_millis(3);
        let start = Instant::now();

        const BATCH_SIZE: usize = 10;

        loop {
            for _ in 0..BATCH_SIZE {
                run_simulation_step(
                    &mut game,
                    &mut agent,
                    &mut game_stats,
                    &mut training_session,
                    &mut global_history,
                    &config,
                    &grid,
                    &collision_settings,
                    &parallel_config,
                    &mut stats,
                    &mut grid_map,
                );
            }

            if start.elapsed() >= ui_target.saturating_sub(buffer) {
                break;
            }
        }

        stats.total_training_time += start.elapsed();
    }
}

struct StepResult {
    snake_idx: usize,
    state: [f32; 8],
    action_idx: usize,
    reward: f32,
    next_state: [f32; 8],
    done: bool,
    ate_food: bool,
}

struct Decision {
    snake_idx: usize,
    action_idx: usize,
    state: [f32; 8],
}

#[allow(clippy::too_many_arguments)]
fn run_simulation_step(
    game: &mut GameState,
    agent: &mut DqnAgent,
    game_stats: &mut GameStats,
    training_session: &mut TrainingSession,
    global_history: &mut GlobalTrainingHistory,
    config: &GameConfig,
    grid: &GridDimensions,
    collision_settings: &CollisionSettings,
    parallel_config: &ParallelConfig,
    stats: &mut TrainingStats,
    grid_map: &mut GridMap,
) {
    use rand::Rng;
    use rayon::prelude::*;

    let mut all_dead = true;

    let active_snakes: Vec<usize> = game
        .snakes
        .iter()
        .enumerate()
        .filter(|(_, s)| !s.is_game_over)
        .map(|(idx, _)| idx)
        .collect();

    if active_snakes.is_empty() {
        handle_generation_end(
            game,
            agent,
            game_stats,
            training_session,
            global_history,
            config,
            parallel_config,
            stats,
            grid,
        );
        return;
    }

    // 1. Calcolo degli stati in parallelo (Veloce)
    let states: Vec<(usize, [f32; 8])> = active_snakes
        .par_iter()
        .map(|&snake_idx| {
            let snake = &game.snakes[snake_idx];
            let state = get_state_egocentric(snake, grid_map, grid);
            (snake_idx, state)
        })
        .collect();

    // 2. Decisione dell'Agente con BATCH INFERENCE
    // Estrai solo gli stati puri per l'agente
    let state_vectors: Vec<[f32; 8]> = states.iter().map(|(_, s)| *s).collect();

    // Ottieni tutte le decisioni in UN COLPO SOLO (batch GPU)
    let action_indices = agent.select_actions_batch(state_vectors);

    // Ricostruisci le decisioni associate all'ID del serpente
    let decisions: Vec<Decision> = states
        .iter()
        .zip(action_indices.into_iter())
        .map(|((snake_idx, state), action_idx)| Decision {
            snake_idx: *snake_idx,
            action_idx,
            state: *state,
        })
        .collect();

    // Update grid map
    grid_map.clear();
    for (idx, snake) in game.snakes.iter().enumerate() {
        if !snake.is_game_over {
            for pos in snake.snake.iter() {
                grid_map.set(pos.x, pos.y, (idx + 1) as u8);
            }
        }
    }

    let mut step_results: Vec<StepResult> = Vec::with_capacity(decisions.len());

    for decision in decisions {
        let snake_idx = decision.snake_idx;
        let action_idx = decision.action_idx;
        let state = decision.state;

        all_dead = false;
        let snake_ref = &mut game.snakes[snake_idx];

        let old_dist_sq = ((snake_ref.snake[0].x - snake_ref.food.x).pow(2)
            + (snake_ref.snake[0].y - snake_ref.food.y).pow(2)) as f32;

        // Apply action: 0=Left, 1=Right, 2=Straight
        match action_idx {
            0 => snake_ref.direction = snake_ref.direction.turn_left(),
            1 => snake_ref.direction = snake_ref.direction.turn_right(),
            _ => {} // Straight
        }

        let (dx, dy) = snake_ref.direction.as_vec();
        let new_head = Position {
            x: snake_ref.snake[0].x + dx,
            y: snake_ref.snake[0].y + dy,
        };

        snake_ref.steps_without_food += 1;
        let mut reward: f32 = 0.0;
        let mut done = false;

        let collision = if collision_settings.snake_vs_snake {
            grid_map.is_collision(new_head.x, new_head.y, snake_idx)
        } else {
            grid_map.is_collision_no_snakes(new_head.x, new_head.y)
        } || snake_ref.snake.contains(&new_head);

        if collision {
            reward = -1.5;
            done = true;
            snake_ref.is_game_over = true;
        } else if new_head == snake_ref.food {
            reward = 1.0;
            snake_ref.snake.push_front(new_head);
            snake_ref.score += 1;
            if snake_ref.score > game.high_score {
                game.high_score = snake_ref.score;
            }
            snake_ref.food = spawn_food(snake_ref, grid);
            snake_ref.steps_without_food = 0;
            grid_map.set(new_head.x, new_head.y, (snake_idx + 1) as u8);
        } else {
            reward -= 0.01;

            let new_dist_sq = ((new_head.x - snake_ref.food.x).pow(2)
                + (new_head.y - snake_ref.food.y).pow(2)) as f32;

            if new_dist_sq < old_dist_sq {
                reward += 0.05;
            } else {
                reward -= 0.05;
            }

            let old_tail = snake_ref.snake.back().copied();
            snake_ref.snake.push_front(new_head);
            snake_ref.snake.pop_back();

            if snake_ref.steps_without_food > (grid.width * grid.height) as u32 {
                reward = -1.0;
                done = true;
                snake_ref.is_game_over = true;
            }

            grid_map.set(new_head.x, new_head.y, (snake_idx + 1) as u8);
            if let Some(tail) = old_tail {
                grid_map.set(tail.x, tail.y, 0);
            }
        }

        step_results.push(StepResult {
            snake_idx,
            state,
            action_idx,
            reward,
            next_state: get_state_egocentric(&game.snakes[snake_idx], grid_map, grid),
            done,
            ate_food: new_head == game.snakes[snake_idx].food,
        });
    }

    for result in step_results {
        agent.remember((
            result.state,
            result.action_idx,
            result.reward,
            result.next_state,
            result.done,
        ));
    }

    agent.iterations += 1;
    if agent.iterations % TRAIN_INTERVAL as u32 == 0 {
        // Chiama la funzione reale implementata in agent.rs
        agent.train();
    }

    if all_dead {
        handle_generation_end(
            game,
            agent,
            game_stats,
            training_session,
            global_history,
            config,
            parallel_config,
            stats,
            grid,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_generation_end(
    game: &mut GameState,
    agent: &mut DqnAgent,
    game_stats: &mut GameStats,
    training_session: &mut TrainingSession,
    global_history: &mut GlobalTrainingHistory,
    config: &GameConfig,
    parallel_config: &ParallelConfig,
    stats: &mut TrainingStats,
    grid: &GridDimensions,
) {
    let food_eaten: u32 = game.snakes.iter().map(|s| s.score).sum();
    game_stats.total_food_eaten += food_eaten as u64;

    for (i, snake) in game.snakes.iter().enumerate() {
        if i < game_stats.best_score_per_snake.len() {
            game_stats.best_score_per_snake[i] =
                game_stats.best_score_per_snake[i].max(snake.score);
        }
    }

    let current_scores: Vec<u32> = game.snakes.iter().map(|s| s.score).collect();
    let max_score = current_scores.iter().copied().max().unwrap_or(0);
    let min_score = current_scores.iter().copied().min().unwrap_or(0);
    let avg_score = current_scores.iter().sum::<u32>() as f32 / current_scores.len() as f32;

    let record = GenerationRecord {
        gen: game.total_iterations,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        avg_score,
        max_score,
        min_score,
        avg_loss: agent.loss,
        epsilon: agent.epsilon,
    };

    global_history.records.push(record.clone());
    training_session.add_record(record);

    training_session.total_time_secs = stats
        .total_training_time
        .as_secs()
        .saturating_sub(global_history.accumulated_time_secs);

    if let Err(e) = training_session.save(config.session_path.to_str().unwrap_or("session.json")) {
        eprintln!("Error saving session: {}", e);
    }

    for snake in game.snakes.iter_mut() {
        snake.reset(grid);
    }
    game.total_iterations += 1;
    agent.iterations = game.total_iterations;

    agent.train(); // Training finale sulla memoria accumulata
    agent.decay_epsilon();

    game_stats.total_generations = game.total_iterations;
    game_stats.high_score = game_stats.high_score.max(game.high_score);
    game_stats.total_games_played += parallel_config.snake_count as u64;

    // FIX: Auto-save brain every N generations
    if game.total_iterations % crate::snake::AUTO_SAVE_INTERVAL == 0 {
        let brain_path = crate::snake::brain_path();
        println!(
            "💾 Auto-saving brain (gen {}) to: {}",
            game.total_iterations,
            brain_path.display()
        );
        if let Err(e) = agent.save(brain_path.to_str().unwrap_or("brain.bin")) {
            eprintln!("⚠️ Error auto-saving brain: {}", e);
        } else {
            println!("✅ Brain auto-saved successfully!");
        }
    }

    let total_score: u32 = game.snakes.iter().map(|s| s.score).sum();
    let active_count = game.snakes.iter().filter(|s| !s.is_game_over).count();
    println!(
        "Gen: {}, Active: {}/{}, Total Score: {}, High: {}, Eps: {:.3}, Loss: {:.5}",
        game.total_iterations,
        active_count,
        parallel_config.snake_count,
        total_score,
        game.high_score,
        agent.epsilon,
        agent.loss
    );
}

pub fn render_system(
    mut commands: Commands,
    game: Res<GameState>,
    windows: Query<&Window>,
    mesh_cache: Res<MeshCache>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut segment_pool: ResMut<SegmentPool>,
    render_config: Res<RenderConfig>,
    q_food: Query<Entity, With<Food>>,
) {
    if !render_config.enabled {
        return;
    }

    for e in q_food.iter() {
        commands.entity(e).despawn();
    }

    let Ok(window) = windows.get_single() else {
        return;
    };

    let ui_padding = 60.0;
    let offset_x = -window.resolution.width() / 2.0 + BLOCK_SIZE / 2.0;
    let offset_y = window.resolution.height() / 2.0 - ui_padding - BLOCK_SIZE / 2.0;

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

pub fn toggle_graph_panel(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut graph_state: ResMut<GraphPanelState>,
    windows: Query<&Window>,
) {
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
                        "Training History",
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

#[allow(clippy::too_many_arguments)]
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
                avg: f32,
                max: u32,
                min: u32,
            }

            let mut visual_points = Vec::new();

            let global_max_score = global_history
                .records
                .iter()
                .map(|r| r.max_score)
                .max()
                .unwrap_or(10)
                .max(10) as f32;

            for chunk in global_history.records.chunks(chunk_size) {
                if chunk.is_empty() {
                    continue;
                }

                let max_in_chunk = chunk.iter().map(|r| r.max_score).max().unwrap_or(0);
                let min_in_chunk = chunk.iter().map(|r| r.min_score).min().unwrap_or(0);
                let sum_avg: f32 = chunk.iter().map(|r| r.avg_score).sum();
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
                    let ratio = (val / global_max_score).clamp(0.0, 1.0);
                    ratio * graph_height
                };

                let h_max = get_height(point.max as f32);
                let h_avg = get_height(point.avg);
                let h_min = get_height(point.min as f32);

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

                    parent.spawn(NodeBundle {
                        style: Style {
                            position_type: PositionType::Absolute,
                            left: Val::Px(x_pos),
                            bottom: Val::Px(margin_bottom + h_max - 1.0),
                            width: Val::Px(display_width),
                            height: Val::Px(display_width.max(2.0)),
                            ..default()
                        },
                        background_color: Color::rgba(1.0, 0.2, 0.2, 1.0).into(),
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
                        background_color: Color::rgba(0.3, 0.3, 1.0, 0.6).into(),
                        ..default()
                    });
                }
            }

            parent.spawn(
                TextBundle::from_section(
                    format!("Max: {:.0}", global_max_score),
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
