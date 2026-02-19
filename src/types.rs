use crate::snake::GenerationRecord;
use bevy::prelude::Color;

/// Lightweight snapshot for thread-safe communication between RL thread and Bevy renderer
#[derive(Clone, Debug)]
pub struct GameSnapshot {
    pub snakes: Vec<SnakeSnapshot>,
    pub grid_width: i32,
    pub grid_height: i32,
    pub high_score: u32,
    pub generation: u32,
    /// Training history records for graph visualization
    pub history_records: Vec<GenerationRecord>,
    /// Total number of records in the current session
    pub session_records_count: usize,
}

/// Snapshot of a single snake's state for rendering
#[derive(Clone, Debug)]
pub struct SnakeSnapshot {
    pub id: usize,
    pub body: Vec<(i32, i32)>, // Solo le coordinate (x, y)
    pub food: (i32, i32),
    pub color: Color,
    pub is_game_over: bool,
    pub score: u32,
}
