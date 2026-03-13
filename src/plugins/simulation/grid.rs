use bevy::prelude::*;

/// Grid constants
pub const BLOCK_SIZE: f32 = 20.0;

/// Base state size for a single frame (8 obstacle + 8 target direction + 1 distance)
pub const BASE_STATE_SIZE: usize = 17;
/// Total state size with frame stacking (current frame + previous frame)
#[allow(dead_code)]
pub const STATE_SIZE: usize = BASE_STATE_SIZE * 2; // 34

/// Ray directions for sensors (8 directions)
pub const RAY_DIRECTIONS: [(i32, i32); 8] = [
    (0, 1),   // N
    (1, 1),   // NE
    (1, 0),   // E
    (1, -1),  // SE
    (0, -1),  // S
    (-1, -1), // SW
    (-1, 0),  // W
    (-1, 1),  // NW
];

// ============================================================================
// Grid Primitives
// ============================================================================

/// Grid position
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Position {
    pub x: i32,
    pub y: i32,
}

/// Grid dimensions
#[derive(Resource)]
pub struct GridDimensions {
    pub width: i32,
    pub height: i32,
}

/// GridMap for O(1) collision detection
#[derive(Resource)]
pub struct GridMap {
    pub width: i32,
    pub height: i32,
    pub data: Vec<u8>,
    /// Static terrain walls generated from seed (true = wall)
    pub terrain: Vec<bool>,
}

impl GridMap {
    pub fn new(width: i32, height: i32) -> Self {
        let size = (width * height) as usize;
        Self {
            width,
            height,
            data: vec![0; size],
            terrain: vec![false; size],
        }
    }

    pub fn clear(&mut self) {
        self.data.fill(0);
        // NOTE: terrain is NOT cleared here — it persists for the whole generation
    }

    /// Copy terrain from a generated terrain slice into the map
    pub fn apply_terrain(&mut self, terrain: &[bool]) {
        debug_assert_eq!(terrain.len(), self.terrain.len());
        self.terrain.copy_from_slice(terrain);
    }

    pub fn set(&mut self, x: i32, y: i32, value: u8) {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return;
        }
        let idx = (y * self.width + x) as usize;
        self.data[idx] = value;
    }

    pub fn is_collision(&self, x: i32, y: i32, self_snake_id: usize) -> bool {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return true;
        }
        let idx = (y * self.width + x) as usize;
        if self.terrain[idx] {
            return true;
        }
        let cell = self.data[idx];
        cell != 0 && cell != (self_snake_id + 1) as u8
    }

    /// Check collision against terrain walls only (no snake bodies)
    pub fn is_wall_collision(&self, x: i32, y: i32) -> bool {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return true;
        }
        self.terrain[(y * self.width + x) as usize]
    }

    /// Check collision including own body (for sensors only - game logic uses is_collision)
    pub fn is_collision_with_self(&self, x: i32, y: i32) -> bool {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return true;
        }
        let idx = (y * self.width + x) as usize;
        if self.terrain[idx] {
            return true;
        }
        self.data[idx] != 0
    }
}

/// ECS Components
#[derive(Component)]
pub struct Food;

#[derive(Component)]
#[allow(dead_code)]
pub struct SnakeId(pub usize);

// ============================================================================
// Direction
// ============================================================================

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum Direction {
    #[default]
    Right,
    Up,
    Down,
    Left,
}

impl Direction {
    pub fn as_vec(&self) -> (i32, i32) {
        match self {
            Direction::Up => (0, 1),
            Direction::Down => (0, -1),
            Direction::Left => (-1, 0),
            Direction::Right => (1, 0),
        }
    }

    pub fn turn_right(&self) -> Self {
        match self {
            Direction::Up => Direction::Right,
            Direction::Right => Direction::Down,
            Direction::Down => Direction::Left,
            Direction::Left => Direction::Up,
        }
    }

    pub fn turn_left(&self) -> Self {
        match self {
            Direction::Up => Direction::Left,
            Direction::Left => Direction::Down,
            Direction::Down => Direction::Right,
            Direction::Right => Direction::Up,
        }
    }
}
