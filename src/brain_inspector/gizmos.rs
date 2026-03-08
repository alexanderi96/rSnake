//! Brain Inspector Gizmo Visualization
//!
//! Renders sensor rays and neural network visualizations using Bevy Gizmos.
//! This provides visual feedback for what the agent is "seeing".

use bevy::prelude::*;

use crate::brain_inspector::InspectedAgent;
use crate::snake::{
    Direction, GameState, GridDimensions, GridMap, SnakeInstance, BLOCK_SIZE, RAY_DIRECTIONS,
};

// ============================================================================
// GIZMO CONFIGURATION
// ============================================================================

/// Configuration for gizmo rendering
#[derive(Resource)]
pub struct InspectorGizmoConfig {
    /// Maximum ray length in grid cells
    pub max_ray_distance: f32,
    /// Color for rays that hit obstacles
    pub hit_color: Color,
    /// Color for rays that don't hit (reach max distance)
    pub miss_color: Color,
    /// Color for food direction indicator
    pub food_color: Color,
    /// Line thickness for rays
    pub ray_thickness: f32,
    /// Whether to show ray endpoints
    pub show_endpoints: bool,
}

impl Default for InspectorGizmoConfig {
    fn default() -> Self {
        Self {
            max_ray_distance: 20.0,
            hit_color: Color::rgb(1.0, 0.3, 0.3), // Red for hits
            miss_color: Color::rgb(0.3, 0.3, 0.3), // Gray for misses
            food_color: Color::rgb(0.3, 1.0, 0.3), // Green for food
            ray_thickness: 2.0,
            show_endpoints: true,
        }
    }
}

// ============================================================================
// GIZMO RENDERING SYSTEMS
// ============================================================================

/// Main gizmo rendering system - draws sensor rays for the inspected agent
pub fn draw_inspector_gizmos(
    mut gizmos: Gizmos,
    inspected: Res<InspectedAgent>,
    game_state: Res<GameState>,
    grid: Res<GridDimensions>,
    grid_map: Res<GridMap>,
    config: Res<InspectorGizmoConfig>,
    windows: Query<&Window>,
    panel_visibility: Res<crate::ui::PanelVisibility>,
) {
    if !panel_visibility.inspector {
        return;
    }
    let Some(idx) = inspected.snake_idx else {
        return;
    };

    let Some(snake) = game_state.snakes.get(idx) else {
        return;
    };

    let Ok(window) = windows.get_single() else {
        return;
    };

    // Calculate grid offset (same as render_system)
    let grid_px_w = grid.width as f32 * BLOCK_SIZE;
    let grid_px_h = grid.height as f32 * BLOCK_SIZE;
    let leftover_x = window.resolution.width() - grid_px_w;
    let leftover_y = window.resolution.height() - grid_px_h;
    let offset_x = -window.resolution.width() / 2.0 + (leftover_x / 2.0) + BLOCK_SIZE / 2.0;
    let offset_y = window.resolution.height() / 2.0 - (leftover_y / 2.0) - BLOCK_SIZE / 2.0;

    // Get head position
    let head = snake.snake[0];
    let head_world_pos = Vec3::new(
        offset_x + head.x as f32 * BLOCK_SIZE,
        offset_y - head.y as f32 * BLOCK_SIZE,
        5.0, // Above the snake
    );

    // Draw food indicator
    let food_world_pos = Vec3::new(
        offset_x + snake.food.x as f32 * BLOCK_SIZE,
        offset_y - snake.food.y as f32 * BLOCK_SIZE,
        5.0,
    );

    // Draw line from head to food
    gizmos.line(head_world_pos, food_world_pos, config.food_color);

    // Draw food marker
    gizmos.circle(
        food_world_pos,
        Direction3d::Z,
        BLOCK_SIZE / 2.0,
        config.food_color,
    );

    // Draw agent info label above head
    let _label_pos = head_world_pos + Vec3::new(0.0, BLOCK_SIZE * 1.5, 0.0);
    // Note: Text rendering with gizmos is limited, we'll use the UI for detailed info

    // Draw selection highlight around the snake
    draw_selection_highlight(&mut gizmos, snake, offset_x, offset_y);

    // Draw sensor rays for the inspected agent
    draw_sensor_rays(
        &mut gizmos,
        snake,
        &grid_map,
        &grid,
        head_world_pos,
        offset_x,
        offset_y,
        &config,
    );
}

/// Draw the 8 sensor rays emanating from the snake's head
fn draw_sensor_rays(
    gizmos: &mut Gizmos,
    snake: &SnakeInstance,
    grid_map: &GridMap,
    grid: &GridDimensions,
    head_pos: Vec3,
    offset_x: f32,
    offset_y: f32,
    config: &InspectorGizmoConfig,
) {
    // Ray directions are now imported from crate::snake

    // Direction offset based on current facing direction
    let dir_offset: usize = match snake.direction {
        Direction::Up => 0,
        Direction::Right => 2,
        Direction::Down => 4,
        Direction::Left => 6,
    };

    let head = snake.snake[0];
    let decay_rate = 0.1_f32;

    for i in 0..8 {
        let ray_idx = (i + dir_offset) % 8;
        let (dx, dy) = RAY_DIRECTIONS[ray_idx];

        // Cast ray to find hit point
        let mut curr_x = head.x;
        let mut curr_y = head.y;
        let mut hit_point: Option<(i32, i32)> = None;
        let mut hit_distance: f32 = config.max_ray_distance;

        loop {
            curr_x += dx;
            curr_y += dy;

            let hit_wall =
                curr_x < 0 || curr_x >= grid.width || curr_y < 0 || curr_y >= grid.height;
            let hit_obstacle = !hit_wall && grid_map.is_collision_with_self(curr_x, curr_y);

            if hit_wall || hit_obstacle {
                hit_point = Some((curr_x, curr_y));
                let diff_x = (curr_x - head.x) as f32;
                let diff_y = (curr_y - head.y) as f32;
                hit_distance = (diff_x * diff_x + diff_y * diff_y).sqrt();
                break;
            }

            // Max distance check
            let diff_x = (curr_x - head.x) as f32;
            let diff_y = (curr_y - head.y) as f32;
            let dist = (diff_x * diff_x + diff_y * diff_y).sqrt();
            if dist >= config.max_ray_distance {
                hit_distance = dist;
                break;
            }
        }

        // Calculate ray color based on distance (same logic as get_current_17_state)
        let sensor_value = if hit_distance <= 1.415 {
            1.0
        } else {
            (-decay_rate * hit_distance).exp()
        };

        let ray_color = if hit_point.is_some() {
            // Hit something - interpolate between hit_color and miss_color based on distance
            Color::rgb(
                config.hit_color.r() * sensor_value + config.miss_color.r() * (1.0 - sensor_value),
                config.hit_color.g() * sensor_value + config.miss_color.g() * (1.0 - sensor_value),
                config.hit_color.b() * sensor_value + config.miss_color.b() * (1.0 - sensor_value),
            )
        } else {
            config.miss_color
        };

        // Calculate end point in world space
        let end_x = head.x as f32 + dx as f32 * hit_distance.min(config.max_ray_distance);
        let end_y = head.y as f32 + dy as f32 * hit_distance.min(config.max_ray_distance);
        let end_pos = Vec3::new(
            offset_x + end_x * BLOCK_SIZE,
            offset_y - end_y * BLOCK_SIZE,
            5.0,
        );

        // Draw the ray
        gizmos.line(head_pos, end_pos, ray_color);

        // Draw endpoint marker
        if config.show_endpoints {
            let marker_size = 3.0 + sensor_value * 5.0;
            gizmos.circle(end_pos, Direction3d::Z, marker_size, ray_color);
        }
    }
}

/// Draw a selection highlight around the inspected snake
fn draw_selection_highlight(
    gizmos: &mut Gizmos,
    snake: &SnakeInstance,
    offset_x: f32,
    offset_y: f32,
) {
    if snake.snake.is_empty() {
        return;
    }

    let highlight_color = Color::rgba(1.0, 1.0, 0.0, 0.5); // Yellow, semi-transparent

    // Draw bounding box around the snake
    let min_x = snake.snake.iter().map(|p| p.x).min().unwrap_or(0);
    let max_x = snake.snake.iter().map(|p| p.x).max().unwrap_or(0);
    let min_y = snake.snake.iter().map(|p| p.y).min().unwrap_or(0);
    let max_y = snake.snake.iter().map(|p| p.y).max().unwrap_or(0);

    let min_world = Vec3::new(
        offset_x + min_x as f32 * BLOCK_SIZE - BLOCK_SIZE / 2.0,
        offset_y - max_y as f32 * BLOCK_SIZE - BLOCK_SIZE / 2.0,
        4.0,
    );
    let max_world = Vec3::new(
        offset_x + max_x as f32 * BLOCK_SIZE + BLOCK_SIZE / 2.0,
        offset_y - min_y as f32 * BLOCK_SIZE + BLOCK_SIZE / 2.0,
        4.0,
    );

    // Draw rectangle corners
    let corners = [
        Vec3::new(min_world.x, min_world.y, 4.0),
        Vec3::new(max_world.x, min_world.y, 4.0),
        Vec3::new(max_world.x, max_world.y, 4.0),
        Vec3::new(min_world.x, max_world.y, 4.0),
    ];

    for i in 0..4 {
        let start = corners[i];
        let end = corners[(i + 1) % 4];
        gizmos.line(start, end, highlight_color);
    }

    // Draw pulsing circle around head
    let head = snake.snake[0];
    let head_pos = Vec3::new(
        offset_x + head.x as f32 * BLOCK_SIZE,
        offset_y - head.y as f32 * BLOCK_SIZE,
        6.0,
    );

    // Pulsing effect (using time would require Time resource, keeping it simple)
    gizmos.circle(head_pos, Direction3d::Z, BLOCK_SIZE, highlight_color);
}

// ============================================================================
// PLUGIN SETUP
// ============================================================================

/// Plugin for gizmo visualization
pub struct InspectorGizmoPlugin;

impl Plugin for InspectorGizmoPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(InspectorGizmoConfig::default())
            .add_systems(Update, draw_inspector_gizmos);
    }
}
