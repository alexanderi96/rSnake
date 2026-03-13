//! MAP-Elites plugin for quality-diversity evolutionary algorithm.

pub mod archive;
pub mod evolution;
pub mod individual;

pub use archive::GRID_RESOLUTION;

use bevy::prelude::*;

/// MAP-Elites evolutionary algorithm plugin.
#[allow(dead_code)]
pub struct MapElitesPlugin;

impl Plugin for MapElitesPlugin {
    fn build(&self, _app: &mut App) {
        // MAP-Elites logic is invoked from SimulationPlugin - no Bevy systems needed
    }
}
