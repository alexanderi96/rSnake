//! Configuration for MAP-Elites Evolutionary Algorithm

use bevy::prelude::Resource;
use serde::{Deserialize, Serialize};
use std::path::Path;

fn default_terrain_fill_rate() -> f32 {
    0.35
}
fn default_terrain_blob_scale() -> f32 {
    4.0
}
fn default_terrain_smooth_passes() -> u32 {
    1
}
fn default_terrain_spawn_clearance() -> i32 {
    5
}

/// MAP-Elites Hyperparameters — caricabili da config.toml o config.json
#[derive(Debug, Clone, Serialize, Deserialize, Resource)]
pub struct Hyperparameters {
    // ── Evoluzione ─────────────────────────────────────────────────────────
    pub population_size: usize,
    pub mutation_rate: f32,
    pub mutation_strength: f32,
    pub crossover_rate: f32,

    // ── Timeout ────────────────────────────────────────────────────────────
    pub base_steps_without_food: u32,
    pub steps_per_segment: u32,

    // ── Archivio MAP-Elites ─────────────────────────────────────────────────
    pub grid_resolution: usize,
    pub auto_save_interval: u32,

    // ── Terrain (cluster noise) ─────────────────────────────────────────────
    /// Densità dei muri [0.20 sparso … 0.55 denso]
    #[serde(default = "default_terrain_fill_rate")]
    pub terrain_fill_rate: f32,

    /// Dimensione dei cluster noise.
    /// 2.0 = cluster molto grandi | 4.0 = medi (default) | 7.0 = piccoli
    #[serde(default = "default_terrain_blob_scale")]
    pub terrain_blob_scale: f32,

    /// Passate CA per arrotondare i bordi (0–3, default 1)
    #[serde(default = "default_terrain_smooth_passes")]
    pub terrain_smooth_passes: u32,

    /// Raggio della zona libera attorno allo spawn (celle)
    #[serde(default = "default_terrain_spawn_clearance")]
    pub terrain_spawn_clearance: i32,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            population_size: 200,
            mutation_rate: 0.05,
            mutation_strength: 0.3,
            crossover_rate: 0.3,
            base_steps_without_food: 60,
            steps_per_segment: 8,
            grid_resolution: 20,
            auto_save_interval: 50,
            terrain_fill_rate: default_terrain_fill_rate(),
            terrain_blob_scale: default_terrain_blob_scale(),
            terrain_smooth_passes: default_terrain_smooth_passes(),
            terrain_spawn_clearance: default_terrain_spawn_clearance(),
        }
    }
}

impl Hyperparameters {
    pub fn calculate_timeout(&self, snake_length: usize, grid_width: i32, grid_height: i32) -> u32 {
        // Add the grid perimeter to ensure the snake always has enough time 
        // to cross the map, regardless of the window resolution.
        let map_allowance = (grid_width + grid_height) as u32; 
        
        self.base_steps_without_food + map_allowance + (snake_length as u32 * self.steps_per_segment)
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(&path)?;
        let path_str = path.as_ref().to_string_lossy();
        if path_str.ends_with(".toml") {
            Ok(toml::from_str(&content)?)
        } else if path_str.ends_with(".json") {
            Ok(serde_json::from_str(&content)?)
        } else {
            Err(ConfigError::InvalidFormat(
                "Il file deve avere estensione .toml o .json".to_string(),
            ))
        }
    }
}

// ── Errori ─────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum ConfigError {
    Io(std::io::Error),
    Toml(toml::de::Error),
    Json(serde_json::Error),
    InvalidFormat(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Io(e) => write!(f, "IO error: {}", e),
            ConfigError::Toml(e) => write!(f, "TOML parse error: {}", e),
            ConfigError::Json(e) => write!(f, "JSON parse error: {}", e),
            ConfigError::InvalidFormat(s) => write!(f, "Invalid format: {}", s),
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ConfigError::Io(e) => Some(e),
            ConfigError::Toml(e) => Some(e),
            ConfigError::Json(e) => Some(e),
            ConfigError::InvalidFormat(_) => None,
        }
    }
}

impl From<std::io::Error> for ConfigError {
    fn from(e: std::io::Error) -> Self {
        ConfigError::Io(e)
    }
}
impl From<toml::de::Error> for ConfigError {
    fn from(e: toml::de::Error) -> Self {
        ConfigError::Toml(e)
    }
}
impl From<serde_json::Error> for ConfigError {
    fn from(e: serde_json::Error) -> Self {
        ConfigError::Json(e)
    }
}
impl From<toml::ser::Error> for ConfigError {
    fn from(e: toml::ser::Error) -> Self {
        ConfigError::InvalidFormat(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let h = Hyperparameters::default();
        assert_eq!(h.population_size, 200);
        assert_eq!(h.terrain_fill_rate, 0.35);
        assert_eq!(h.terrain_blob_scale, 4.0);
    }

    #[test]
    fn test_timeout() {
        let h = Hyperparameters::default();
        // Griglia fittizia 20x20 per il test.
        // Base (60) + Margine Mappa (20 + 20) + Segmenti (5 * 8) = 140
        assert_eq!(h.calculate_timeout(5, 20, 20), 140); 
    }
}
