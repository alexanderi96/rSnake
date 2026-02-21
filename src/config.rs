//! Configuration for MAP-Elites Evolutionary Algorithm

use bevy::prelude::Resource;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// MAP-Elites Hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize, Resource)]
pub struct Hyperparameters {
    // Population
    pub population_size: usize,

    // Variation operators
    pub mutation_rate: f64,
    pub mutation_strength: f64,
    pub crossover_rate: f64,

    // Simulation limits
    pub max_frames: u32,
    pub base_steps_without_food: u32,
    pub steps_per_segment: u32,

    // Archive
    pub grid_resolution: usize,

    // Auto-save
    pub auto_save_interval: u32,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            population_size: 200,
            mutation_rate: 0.1,
            mutation_strength: 0.5,
            crossover_rate: 0.3,
            max_frames: 2000,
            base_steps_without_food: 100,
            steps_per_segment: 10,
            grid_resolution: 20,
            auto_save_interval: 50,
        }
    }
}

impl Hyperparameters {
    /// Load configuration from TOML or JSON file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(&path)?;
        let path_str = path.as_ref().to_string_lossy();

        if path_str.ends_with(".toml") {
            Ok(toml::from_str(&content)?)
        } else if path_str.ends_with(".json") {
            Ok(serde_json::from_str(&content)?)
        } else {
            Err(ConfigError::InvalidFormat(
                "File must have .toml or .json extension".to_string(),
            ))
        }
    }
}

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
    fn test_default_hyperparameters() {
        let h = Hyperparameters::default();
        assert_eq!(h.population_size, 200);
        assert_eq!(h.mutation_rate, 0.1);
        assert_eq!(h.mutation_strength, 0.5);
        assert_eq!(h.crossover_rate, 0.3);
        assert_eq!(h.grid_resolution, 20);
    }

    #[test]
    fn test_calculate_timeout() {
        let h = Hyperparameters::default();
        assert_eq!(h.calculate_timeout(1), 110); // 100 + 1*10
        assert_eq!(h.calculate_timeout(5), 150); // 100 + 5*10
        assert_eq!(h.calculate_timeout(10), 200); // 100 + 10*10
    }
}
