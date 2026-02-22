//! Configuration for MAP-Elites Evolutionary Algorithm

use bevy::prelude::Resource;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// MAP-Elites Hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize, Resource)]
pub struct Hyperparameters {
    /// Numero di individui (serpenti) valutati in parallelo per generazione
    pub population_size: usize,

    /// Probabilità (0.0 - 1.0) che un gene subisca una variazione casuale
    pub mutation_rate: f64,
    /// Entità massima della variazione applicata a un gene mutato
    pub mutation_strength: f64,
    /// Probabilità (0.0 - 1.0) di combinare due genitori (crossover) invece di clonarne uno
    pub crossover_rate: f64,

    /// Frame di sopravvivenza garantiti dopo aver mangiato una mela (o allo spawn)
    pub base_steps_without_food: u32,
    /// Frame extra concessi per ogni unità di lunghezza del corpo del serpente
    pub steps_per_segment: u32,

    /// Numero di celle per asse dell'archivio (es: 20 = griglia 20x20 = 400 nicchie comportamentali)
    pub grid_resolution: usize,

    /// Frequenza di salvataggio automatico dell'archivio su disco (in generazioni)
    pub auto_save_interval: u32,
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
        }
    }
}

impl Hyperparameters {
    /// Calculate timeout based on snake length
    pub fn calculate_timeout(&self, snake_length: usize) -> u32 {
        self.base_steps_without_food + (snake_length as u32 * self.steps_per_segment)
    }

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
        assert_eq!(h.mutation_rate, 0.05);
        assert_eq!(h.mutation_strength, 0.3);
        assert_eq!(h.crossover_rate, 0.3);
        assert_eq!(h.grid_resolution, 20);
    }

    #[test]
    fn test_calculate_timeout() {
        let h = Hyperparameters::default();
        assert_eq!(h.calculate_timeout(1), 68); // 60 + 1*8
        assert_eq!(h.calculate_timeout(5), 100); // 60 + 5*8
        assert_eq!(h.calculate_timeout(10), 140); // 60 + 10*8
    }
}
