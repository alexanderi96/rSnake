//! Simulation plugin for snake game logic.
//!
//! This plugin contains the core simulation types and functions for the snake game:
//! - Grid primitives (GridDimensions, GridMap, Position)
//! - Snake logic (SnakeInstance, Direction, Food, GameState)
//! - Simulation step (end_generation in main.rs)

pub mod grid;
pub mod snake;
pub mod step;

// Re-export grid primitives
#[allow(unused_imports)]
pub use grid::{
    Direction, Food, GridDimensions, GridMap, Position, SnakeId, BASE_STATE_SIZE, BLOCK_SIZE,
    RAY_DIRECTIONS, STATE_SIZE,
};

// Re-export snake logic
pub use snake::{
    bfs_distance, calculate_grid_dimensions, get_current_17_state, GameState, GenerationSeed,
    SnakeInstance,
};

use bevy::prelude::*;

/// Simulation plugin for snake game logic.
#[allow(dead_code)]
pub struct SimulationPlugin;

impl Plugin for SimulationPlugin {
    fn build(&self, _app: &mut App) {
        // Systems are registered in main.rs
    }
}
