use serde::{Deserialize, Serialize};
use std::path::Path;

/// Hyperparameters centralizzati per il training DQN
/// Caricabili da file TOML/JSON e sovrascrivibili da CLI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hyperparameters {
    // DQN / Rete
    pub learning_rate: f64,
    pub gamma: f32,
    pub batch_size: usize,
    pub memory_size: usize,

    // Update Frequencies
    pub target_update_freq: usize,
    pub train_interval: usize,

    // Reward System
    pub reward_food: f32,
    pub reward_death: f32,
    pub reward_step: f32,

    // Timeout (morte per inedia)
    pub base_steps_without_food: u32,
    pub steps_per_segment: u32,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            // DQN defaults
            learning_rate: 1e-4,
            gamma: 0.99,
            batch_size: 256,
            memory_size: 100_000,

            // Update frequencies
            target_update_freq: 5_000,
            train_interval: 64, // Sviluppato dal numero di thread

            // Reward system (bilanciato)
            reward_food: 10.0,
            reward_death: -10.0,
            reward_step: -0.01, // Minima penalità per scoraggiare inattività

            // Timeout dinamico
            base_steps_without_food: 100,
            steps_per_segment: 10,
        }
    }
}

impl Hyperparameters {
    /// Carica configurazione da file TOML o JSON
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

    /// Salva configurazione in formato TOML
    pub fn save_toml<P: AsRef<Path>>(&self, path: P) -> Result<(), ConfigError> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Salva configurazione in formato JSON
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<(), ConfigError> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Calcola il timeout dinamico basato sulla lunghezza del serpente
    pub fn calculate_timeout(&self, snake_length: usize) -> u32 {
        self.base_steps_without_food + (snake_length as u32 * self.steps_per_segment)
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
        assert_eq!(h.learning_rate, 1e-4);
        assert_eq!(h.gamma, 0.99);
        assert_eq!(h.batch_size, 256);
        assert_eq!(h.memory_size, 100_000);
        assert_eq!(h.target_update_freq, 5_000);
        assert_eq!(h.train_interval, 64);
        assert_eq!(h.reward_food, 10.0);
        assert_eq!(h.reward_death, -10.0);
        assert_eq!(h.reward_step, -0.01);
        assert_eq!(h.base_steps_without_food, 100);
        assert_eq!(h.steps_per_segment, 10);
    }

    #[test]
    fn test_calculate_timeout() {
        let h = Hyperparameters::default();
        assert_eq!(h.calculate_timeout(1), 110); // 100 + 1*10
        assert_eq!(h.calculate_timeout(5), 150); // 100 + 5*10
        assert_eq!(h.calculate_timeout(10), 200); // 100 + 10*10
    }
}
