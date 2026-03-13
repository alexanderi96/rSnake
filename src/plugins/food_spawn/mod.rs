use bevy::prelude::*;

/// Shared resource: the zone within which food may spawn.
/// Updated by whatever plugin owns the spawn strategy.
/// The simulation reads this every time food needs to be placed.
#[derive(Resource, Debug, Clone)]
#[allow(dead_code)]
pub struct FoodSpawnZone {
    /// Center in grid coordinates (col, row as f32).
    pub center: Vec2,
    /// Radius in grid cells. f32::INFINITY = full map (default).
    pub radius: f32,
    /// For UI / debug display only — does not affect logic.
    pub source: SpawnSource,
}

impl Default for FoodSpawnZone {
    fn default() -> Self {
        Self {
            center: Vec2::ZERO, // overwritten by grid center at startup
            radius: f32::INFINITY,
            source: SpawnSource::Uniform,
        }
    }
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub enum SpawnSource {
    #[default]
    Uniform,
    AudioReactive,
    Fixed(Vec2),
}

#[allow(dead_code)]
pub struct FoodSpawnPlugin;

impl Plugin for FoodSpawnPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FoodSpawnZone>()
            .add_systems(Startup, init_spawn_zone);
    }
}

/// Sets zone center to the map center at startup.
#[allow(dead_code, unused_variables)]
fn init_spawn_zone(
    zone: ResMut<FoodSpawnZone>,
    // TODO: read GridDimensions once available from SimulationPlugin
) {
    // Left as a stub — SimulationPlugin startup sets the real center.
}
