//! Evolution Management for MAP-Elites
//!
//! This module handles the generational evaluation loop, coordinating
//! between the MAP-Elites archive and the Bevy simulation.

use bevy::prelude::Resource;
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::brain::Individual;
use crate::config::Hyperparameters;
use crate::map_elites::MapElitesArchive;

/// Generation state for tracking evaluation progress
#[derive(Debug, Clone, Default)]
pub struct GenerationState {
    /// Current generation number
    pub generation: u32,
    /// Individuals currently being evaluated
    pub population: Vec<Individual>,
    /// Index of next individual to spawn
    pub next_individual_id: usize,
    /// Number of alive individuals
    pub alive_count: usize,
    /// Generation start time
    pub start_time: Option<Instant>,
    /// Total frames in this generation
    pub total_frames: u64,
    /// Best fitness this generation
    pub best_fitness: f64,
    /// Average fitness this generation
    pub avg_fitness: f64,
}

impl GenerationState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialize a new generation with the given population
    pub fn start_generation(&mut self, population: Vec<Individual>) {
        self.generation += 1;
        self.population = population;
        self.next_individual_id = 0;
        self.alive_count = self.population.len();
        self.start_time = Some(Instant::now());
        self.total_frames = 0;
        self.best_fitness = 0.0;
        self.avg_fitness = 0.0;

        // Reset all individuals
        for individual in &mut self.population {
            individual.reset();
        }
    }

    /// Get elapsed time for this generation
    pub fn elapsed_secs(&self) -> f64 {
        self.start_time
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0)
    }

    /// Calculate generation statistics
    pub fn calculate_stats(&mut self) {
        if self.population.is_empty() {
            return;
        }

        let total_fitness: f64 = self.population.iter().map(|i| i.fitness).sum();
        self.avg_fitness = total_fitness / self.population.len() as f64;
        self.best_fitness = self
            .population
            .iter()
            .map(|i| i.fitness)
            .fold(0.0, f64::max);
    }
}

/// Evolution manager coordinating MAP-Elites and generations
#[derive(Debug, Resource)]
pub struct EvolutionManager {
    /// MAP-Elites archive
    pub archive: MapElitesArchive,
    /// Evolution configuration (uses Hyperparameters)
    pub config: Hyperparameters,
    /// Current generation state
    pub generation_state: GenerationState,
    /// History of generation records
    pub history: Vec<GenerationRecord>,
}

impl Default for EvolutionManager {
    fn default() -> Self {
        let config = Hyperparameters::default();
        Self::new(config)
    }
}

impl EvolutionManager {
    pub fn new(config: Hyperparameters) -> Self {
        Self {
            archive: MapElitesArchive::new(config.grid_resolution),
            config,
            generation_state: GenerationState::new(),
            history: Vec::new(),
        }
    }

    /// Start a new generation
    pub fn start_generation(&mut self) {
        let population = self.archive.generate_population_with_crossover(
            self.config.population_size,
            self.config.mutation_rate,
            self.config.mutation_strength,
            self.config.crossover_rate,
        );

        self.generation_state.start_generation(population);
    }

    /// End the current generation and update the archive
    pub fn end_generation(&mut self) -> GenerationRecord {
        self.generation_state.calculate_stats();

        // Update archive with evaluated individuals
        let insertions = self.archive.update(&self.generation_state.population);

        // Create record
        let record = GenerationRecord {
            generation: self.generation_state.generation,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            avg_fitness: self.generation_state.avg_fitness,
            best_fitness: self.generation_state.best_fitness,
            alive_count: self.generation_state.alive_count,
            population_size: self.config.population_size,
            elapsed_secs: self.generation_state.elapsed_secs(),
            total_frames: self.generation_state.total_frames,
            archive_coverage: self.archive.coverage(),
            archive_filled: self.archive.filled_cells(),
            insertions,
            generation_high_score: 0, // Will be set by caller
        };

        self.history.push(record.clone());

        // Cap history to prevent unbounded growth
        if self.history.len() > 500 {
            self.history.drain(0..250);
        }

        // Auto-save if needed
        if self.generation_state.generation % self.config.auto_save_interval == 0 {
            self.save_archive();
        }

        record
    }

    /// Get the current population
    pub fn get_population(&self) -> &[Individual] {
        &self.generation_state.population
    }

    /// Get mutable access to an individual
    pub fn get_individual_mut(&mut self, id: usize) -> Option<&mut Individual> {
        self.generation_state.population.get_mut(id)
    }

    /// Save the archive to disk
    pub fn save_archive(&self) {
        let path = crate::snake::get_or_create_run_dir().join("archive.json");
        if let Err(e) = self.archive.save(path.to_str().unwrap_or("archive.json")) {
            eprintln!("⚠️ Error saving archive: {}", e);
        } else {
            println!("💾 Archive saved to: {}", path.display());
        }
    }

    /// Load the archive from disk
    pub fn load_archive(&mut self) {
        let path = crate::snake::get_or_create_run_dir().join("archive.json");
        if path.exists() {
            match MapElitesArchive::load(path.to_str().unwrap_or("archive.json")) {
                Ok(archive) => {
                    self.archive = archive;
                    println!("✅ Archive loaded: {} elites", self.archive.filled_cells());
                }
                Err(e) => {
                    eprintln!("⚠️ Error loading archive: {}", e);
                }
            }
        }
    }
}

/// Record of a generation's performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRecord {
    pub generation: u32,
    pub timestamp: u64,
    pub avg_fitness: f64,
    pub best_fitness: f64,
    pub alive_count: usize,
    pub population_size: usize,
    pub elapsed_secs: f64,
    pub total_frames: u64,
    pub archive_coverage: f64,
    pub archive_filled: usize,
    pub insertions: usize,
    #[serde(default)]
    pub generation_high_score: u32, // max apples eaten this generation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evolution_manager_default() {
        let manager = EvolutionManager::default();
        assert_eq!(manager.config.population_size, 200);
        assert_eq!(manager.config.mutation_rate, 0.05);
        assert_eq!(manager.config.grid_resolution, 20);
    }
}
