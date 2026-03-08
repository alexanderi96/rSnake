//! Brain Inspector Gizmo Visualization
//!
//! Renders sensor roses and neural network visualizations using Bevy Gizmos.
//! This provides visual feedback for what the agent is "seeing".

use bevy::prelude::*;

use crate::brain_inspector::InspectedAgent;
use crate::snake::{GameState, GridDimensions, BLOCK_SIZE};

// ============================================================================
// GIZMO CONFIGURATION
// ============================================================================

/// Configuration for gizmo rendering
#[derive(Resource)]
pub struct InspectorGizmoConfig {
    /// Color for food direction indicator
    pub food_color: Color,
}

impl Default for InspectorGizmoConfig {
    fn default() -> Self {
        Self {
            food_color: Color::rgb(0.3, 1.0, 0.3), // Green for food
        }
    }
}

// ============================================================================
// GIZMO RENDERING SYSTEMS
// ============================================================================

/// Main gizmo rendering system - draws bounding box and food indicator for the inspected agent
pub fn draw_inspector_gizmos(
    mut gizmos: Gizmos,
    inspected: Res<InspectedAgent>,
    game_state: Res<GameState>,
    grid: Res<GridDimensions>,
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

    // Bounding box giallo attorno al serpente selezionato
    if !snake.snake.is_empty() {
        let min_x = snake.snake.iter().map(|p| p.x).min().unwrap_or(0);
        let max_x = snake.snake.iter().map(|p| p.x).max().unwrap_or(0);
        let min_y = snake.snake.iter().map(|p| p.y).min().unwrap_or(0);
        let max_y = snake.snake.iter().map(|p| p.y).max().unwrap_or(0);

        let corners = [
            Vec3::new(
                offset_x + min_x as f32 * BLOCK_SIZE - BLOCK_SIZE / 2.0,
                offset_y - max_y as f32 * BLOCK_SIZE - BLOCK_SIZE / 2.0,
                4.0,
            ),
            Vec3::new(
                offset_x + max_x as f32 * BLOCK_SIZE + BLOCK_SIZE / 2.0,
                offset_y - max_y as f32 * BLOCK_SIZE - BLOCK_SIZE / 2.0,
                4.0,
            ),
            Vec3::new(
                offset_x + max_x as f32 * BLOCK_SIZE + BLOCK_SIZE / 2.0,
                offset_y - min_y as f32 * BLOCK_SIZE + BLOCK_SIZE / 2.0,
                4.0,
            ),
            Vec3::new(
                offset_x + min_x as f32 * BLOCK_SIZE - BLOCK_SIZE / 2.0,
                offset_y - min_y as f32 * BLOCK_SIZE + BLOCK_SIZE / 2.0,
                4.0,
            ),
        ];
        for i in 0..4 {
            gizmos.line(
                corners[i],
                corners[(i + 1) % 4],
                Color::rgba(1.0, 1.0, 0.0, 0.5),
            );
        }
    }

    // Linea testa → cibo e cerchio sul cibo
    let head = snake.snake[0];
    let head_pos = Vec3::new(
        offset_x + head.x as f32 * BLOCK_SIZE,
        offset_y - head.y as f32 * BLOCK_SIZE,
        5.0,
    );
    let food_pos = Vec3::new(
        offset_x + snake.food.x as f32 * BLOCK_SIZE,
        offset_y - snake.food.y as f32 * BLOCK_SIZE,
        5.0,
    );
    gizmos.line(head_pos, food_pos, config.food_color);
    gizmos.circle(
        food_pos,
        Direction3d::Z,
        BLOCK_SIZE / 2.0,
        config.food_color,
    );
    gizmos.circle(
        head_pos,
        Direction3d::Z,
        BLOCK_SIZE,
        Color::rgba(1.0, 1.0, 0.0, 0.5),
    );

    // NOTA: draw_sensor_rays RIMOSSO — sostituito da draw_sensor_roses
}

/// Disegna le due rose sensoriali in screen-space nel pannello inspector
pub fn draw_sensor_roses(
    mut gizmos: Gizmos,
    inspected: Res<InspectedAgent>,
    game_state: Res<GameState>,
    panel_visibility: Res<crate::ui::PanelVisibility>,
    windows: Query<&Window>,
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
    let Some(sensors) = inspected.last_sensor_state else {
        return;
    };

    let Ok(window) = windows.get_single() else {
        return;
    };

    // Posizione rose in screen-space (angolo top-right dove sta il pannello)
    // Pannello largo 480px, ancorato a destra a 10px
    let panel_center_x = window.width() - 10.0 - 480.0 / 2.0;

    // Rosa ostacoli: sotto l'header del pannello
    let obstacle_center = Vec2::new(panel_center_x - 100.0, window.height() / 2.0 - 200.0);

    // Rosa cibo: sotto la rosa ostacoli
    let food_center = Vec2::new(panel_center_x - 100.0, window.height() / 2.0 - 370.0);

    draw_rose(
        &mut gizmos,
        obstacle_center,
        &sensors[0..8],
        snake.direction,
        Color::rgba(1.0, 0.45, 0.1, 0.9),
    );

    draw_rose(
        &mut gizmos,
        food_center,
        &sensors[8..16],
        snake.direction,
        Color::rgba(0.2, 0.7, 1.0, 0.9),
    );
}

const ROSE_RADIUS: f32 = 52.0;

fn draw_rose(
    gizmos: &mut Gizmos,
    center: Vec2,
    values: &[f32],
    facing: crate::snake::Direction,
    line_color: Color,
) {
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    // Sfondo scuro
    gizmos.circle_2d(
        center,
        ROSE_RADIUS + 6.0,
        Color::rgba(0.05, 0.05, 0.08, 0.85),
    );

    // Cerchi guida
    gizmos.circle_2d(center, ROSE_RADIUS, Color::rgba(0.35, 0.35, 0.35, 0.25));
    gizmos.circle_2d(
        center,
        ROSE_RADIUS * 0.5,
        Color::rgba(0.25, 0.25, 0.25, 0.20),
    );

    // Assi cardinali tenui
    for angle in [0.0_f32, FRAC_PI_2, PI, 3.0 * FRAC_PI_4] {
        let d = Vec2::new(angle.cos(), angle.sin()) * (ROSE_RADIUS + 6.0);
        gizmos.line_2d(center - d, center + d, Color::rgba(0.3, 0.3, 0.3, 0.15));
    }

    // Angoli base dei sensori in ordine [FWD, F-R, R, B-R, BCK, B-L, L, F-L]
    // riferiti alla direzione UP del serpente (0° = destra, senso antiorario)
    const BASE_ANGLES: [f32; 8] = [
        FRAC_PI_2,        // FWD
        FRAC_PI_4,        // F-R
        0.0,              // R
        -FRAC_PI_4,       // B-R
        -FRAC_PI_2,       // BCK
        -3.0 * FRAC_PI_4, // B-L
        PI,               // L
        3.0 * FRAC_PI_4,  // F-L
    ];

    // Offset angolare basato sulla direzione attuale del serpente
    let facing_offset = match facing {
        crate::snake::Direction::Up => 0.0,
        crate::snake::Direction::Right => -FRAC_PI_2,
        crate::snake::Direction::Down => PI,
        crate::snake::Direction::Left => FRAC_PI_2,
    };

    for (i, &raw_val) in values.iter().enumerate() {
        let angle = BASE_ANGLES[i] + facing_offset;
        let dir = Vec2::new(angle.cos(), angle.sin());

        // I valori food_dir sono in [-1,1], ostacoli in [0,1]
        let normalized = raw_val.clamp(-1.0, 1.0);
        let length = normalized.abs() * ROSE_RADIUS;

        if length < 1.0 {
            continue;
        }

        let endpoint = if normalized >= 0.0 {
            center + dir * length
        } else {
            center - dir * length // direzione inversa per valori negativi
        };

        let alpha = if normalized >= 0.0 { 0.9 } else { 0.35 };
        gizmos.line_2d(center, endpoint, line_color.with_a(alpha));

        // Pallino all'estremità
        gizmos.circle_2d(endpoint, 2.5, line_color.with_a(alpha));
    }

    // Indicatore freccia direzione serpente (bianco semi-trasparente)
    let facing_angle = match facing {
        crate::snake::Direction::Up => FRAC_PI_2,
        crate::snake::Direction::Right => 0.0,
        crate::snake::Direction::Down => -FRAC_PI_2,
        crate::snake::Direction::Left => PI,
    };
    let fdir = Vec2::new(facing_angle.cos(), facing_angle.sin());
    gizmos.line_2d(
        center,
        center + fdir * (ROSE_RADIUS + 10.0),
        Color::rgba(1.0, 1.0, 1.0, 0.45),
    );
    gizmos.circle_2d(
        center + fdir * (ROSE_RADIUS + 10.0),
        3.0,
        Color::rgba(1.0, 1.0, 1.0, 0.45),
    );
}

// ============================================================================
// PLUGIN SETUP
// ============================================================================

/// Plugin for gizmo visualization
pub struct InspectorGizmoPlugin;

impl Plugin for InspectorGizmoPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(InspectorGizmoConfig::default())
            .add_systems(Update, draw_inspector_gizmos)
            .add_systems(
                Update,
                draw_sensor_roses.after(crate::brain_inspector::update_sensor_cache),
            );
    }
}
