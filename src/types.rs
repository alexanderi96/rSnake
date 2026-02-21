//! Type definitions for MAP-Elites Snake

use bevy::prelude::Color;
use serde::{Deserialize, Serialize};

use crate::evolution::GenerationRecord;

/// Lightweight snapshot for rendering
#[derive(Clone, Debug)]
pub struct GameSnapshot {
    pub snakes: Vec<SnakeSnapshot>,
    pub grid_width: i32,
    pub grid_height: i32,
    pub high_score: u32,
    pub generation: u32,
    pub alive_count: usize,
    pub history_records: Vec<GenerationRecord>,
}

/// Snapshot of a single snake's state for rendering
#[derive(Clone, Debug)]
pub struct SnakeSnapshot {
    pub id: usize,
    pub body: Vec<(i32, i32)>,
    pub food: (i32, i32),
    pub color: Color,
    pub is_game_over: bool,
    pub score: u32,
    pub fitness: f64,
}

/// Archive cell for visualization
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArchiveCell {
    pub courage_bin: usize,
    pub agility_bin: usize,
    pub fitness: f64,
    pub apples: u32,
    pub frames: u32,
}
