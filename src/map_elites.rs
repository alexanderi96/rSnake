//! MAP-Elites Quality-Diversity Algorithm Implementation
//!
//! MAP-Elites maintains a grid of high-performing solutions where each cell
//! represents a unique behavioral niche. The algorithm illuminates the
//! behavioral space by discovering diverse, high-quality solutions.

use rand::seq::SliceRandom;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;

use crate::brain::Individual;

/// Number of bins for each behavioral descriptor dimension
pub const GRID_RESOLUTION: usize = 20;

/// Custom serializer for HashMap with (usize, usize) keys
/// Converts tuple keys to "x,y" string format
fn serialize_grid<S>(
    grid: &HashMap<(usize, usize), Individual>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let string_keyed: std::collections::HashMap<String, &Individual> = grid
        .iter()
        .map(|((x, y), v)| (format!("{},{}", x, y), v))
        .collect();
    string_keyed.serialize(serializer)
}

/// Custom deserializer for HashMap with (usize, usize) keys
/// Converts "x,y" string keys back to tuple format
fn deserialize_grid<'de, D>(
    deserializer: D,
) -> Result<HashMap<(usize, usize), Individual>, D::Error>
where
    D: Deserializer<'de>,
{
    let string_keyed: std::collections::HashMap<String, Individual> =
        Deserialize::deserialize(deserializer)?;

    let mut grid = HashMap::new();
    for (key, value) in string_keyed {
        let parts: Vec<&str> = key.split(',').collect();
        if parts.len() == 2 {
            if let (Ok(x), Ok(y)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                grid.insert((x, y), value);
            }
        }
    }
    Ok(grid)
}

/// MAP-Elites Archive: a 2D grid storing elite individuals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapElitesArchive {
    /// Grid storing elite individuals: key = (courage_bin, agility_bin)
    #[serde(
        serialize_with = "serialize_grid",
        deserialize_with = "deserialize_grid"
    )]
    pub grid: HashMap<(usize, usize), Individual>,
    /// Resolution of each dimension
    pub resolution: usize,
    /// Statistics
    pub total_insertions: u64,
    pub successful_insertions: u64,
    /// Best fitness ever seen
    pub best_fitness: f64,
    /// Generation counter
    pub generation: u32,
}

impl Default for MapElitesArchive {
    fn default() -> Self {
        Self::new(GRID_RESOLUTION)
    }
}

impl MapElitesArchive {
    /// Create a new MAP-Elites archive with given resolution
    pub fn new(resolution: usize) -> Self {
        Self {
            grid: HashMap::new(),
            resolution,
            total_insertions: 0,
            successful_insertions: 0,
            best_fitness: 0.0,
            generation: 0,
        }
    }

    /// Discretize a behavioral descriptor value into a grid bin
    /// Value should be in [0.0, 1.0]
    fn discretize(&self, value: f64) -> usize {
        let clamped = value.clamp(0.0, 1.0);
        let bin = (clamped * (self.resolution - 1) as f64).round() as usize;
        bin.min(self.resolution - 1)
    }

    /// Get the grid cell coordinates for an individual
    pub fn get_cell(&self, individual: &Individual) -> (usize, usize) {
        let congestion_bin = self.discretize(individual.congestion);
        let agility_bin = self.discretize(individual.agility);
        (congestion_bin, agility_bin)
    }

    /// Try to insert an individual into the archive
    /// Returns true if the individual was inserted (either new cell or better fitness)
    pub fn insert(&mut self, individual: Individual) -> bool {
        self.total_insertions += 1;

        let cell = self.get_cell(&individual);

        // Update best fitness
        if individual.fitness > self.best_fitness {
            self.best_fitness = individual.fitness;
        }

        // Check if we should insert
        let should_insert = match self.grid.get(&cell) {
            None => true,                                            // Empty cell, always insert
            Some(existing) => individual.fitness > existing.fitness, // Better fitness
        };

        if should_insert {
            self.grid.insert(cell, individual);
            self.successful_insertions += 1;
            true
        } else {
            false
        }
    }

    /// Get the number of filled cells
    pub fn filled_cells(&self) -> usize {
        self.grid.len()
    }

    /// Get the total capacity of the grid
    pub fn capacity(&self) -> usize {
        self.resolution * self.resolution
    }

    /// Get the coverage ratio (filled / total)
    pub fn coverage(&self) -> f64 {
        self.grid.len() as f64 / self.capacity() as f64
    }

    /// Generate a new population by selecting and varying elites
    pub fn generate_population(
        &self,
        population_size: usize,
        mutation_rate: f64,
        mutation_strength: f64,
    ) -> Vec<Individual> {
        let mut rng = rand::thread_rng();

        let mut population = Vec::with_capacity(population_size);

        // If archive is empty, generate random individuals
        if self.grid.is_empty() {
            for id in 0..population_size {
                population.push(Individual::new_random(id));
            }
            return population;
        }

        // Collect elites with their cell coordinates for archive_color calculation
        let elites_with_cells: Vec<(&(usize, usize), &Individual)> = self.grid.iter().collect();

        // Color mutation strength (smaller than brain mutation)
        const COLOR_MUTATION_STRENGTH: f64 = 0.05;

        for id in 0..population_size {
            // Select a random elite with cell
            let (_cell, parent) = elites_with_cells.choose(&mut rng).unwrap();

            // Create a mutated offspring
            let mutated_brain = parent.brain.mutate(mutation_rate, mutation_strength);

            // Mutate color with small jitter
            let mutated_color = parent.color.mutate(COLOR_MUTATION_STRENGTH);

            // Calculate archive_color from parent cell fitness
            let normalized = (parent.fitness / self.best_fitness.max(1.0)).clamp(0.0, 1.0) as f32;
            let archive_color = crate::brain::GenomeColor {
                r: 0.1,
                g: normalized as f64,
                b: (1.0 - normalized) as f64,
            };

            let mut individual = Individual::from_genome_with_archive_color(
                id,
                mutated_brain.get_genome(),
                mutated_color,
                archive_color,
            );
            individual.is_alive = true;

            population.push(individual);
        }

        population
    }

    /// Generate population with crossover
    pub fn generate_population_with_crossover(
        &self,
        population_size: usize,
        mutation_rate: f64,
        mutation_strength: f64,
        crossover_rate: f64,
    ) -> Vec<Individual> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut population = Vec::with_capacity(population_size);

        // If archive is empty or has only one elite, generate random/mutated individuals
        if self.grid.len() <= 1 {
            return self.generate_population(population_size, mutation_rate, mutation_strength);
        }

        // Collect elites with their cell coordinates for archive_color calculation
        let elites_with_cells: Vec<(&(usize, usize), &Individual)> = self.grid.iter().collect();

        // Color mutation strength (smaller than brain mutation)
        const COLOR_MUTATION_STRENGTH: f64 = 0.05;

        for id in 0..population_size {
            let individual = if rng.gen::<f64>() < crossover_rate && elites_with_cells.len() >= 2 {
                // Crossover between two random elites
                let (_cell1, parent1) = elites_with_cells.choose(&mut rng).unwrap();
                let (_cell2, parent2) = elites_with_cells.choose(&mut rng).unwrap();

                // Brain crossover
                let child_brain = parent1.brain.crossover(&parent2.brain);
                let mutated_brain = child_brain.mutate(mutation_rate, mutation_strength);

                // Color inheritance: blend from parents + mutation
                let blend_factor = rng.gen::<f64>();
                let child_color = parent1.color.lerp(&parent2.color, blend_factor);
                let mutated_color = child_color.mutate(COLOR_MUTATION_STRENGTH);

                // Archive color from parent1's cell fitness
                let normalized =
                    (parent1.fitness / self.best_fitness.max(1.0)).clamp(0.0, 1.0) as f32;
                let archive_color = crate::brain::GenomeColor {
                    r: 0.1,
                    g: normalized as f64,
                    b: (1.0 - normalized) as f64,
                };

                let mut ind = Individual::from_genome_with_archive_color(
                    id,
                    mutated_brain.get_genome(),
                    mutated_color,
                    archive_color,
                );
                ind.is_alive = true;
                ind
            } else {
                // Just mutation
                let (_cell, parent) = elites_with_cells.choose(&mut rng).unwrap();
                let mutated_brain = parent.brain.mutate(mutation_rate, mutation_strength);

                // Mutate color with small jitter
                let mutated_color = parent.color.mutate(COLOR_MUTATION_STRENGTH);

                // Archive color from parent's cell fitness
                let normalized =
                    (parent.fitness / self.best_fitness.max(1.0)).clamp(0.0, 1.0) as f32;
                let archive_color = crate::brain::GenomeColor {
                    r: 0.1,
                    g: normalized as f64,
                    b: (1.0 - normalized) as f64,
                };

                let mut ind = Individual::from_genome_with_archive_color(
                    id,
                    mutated_brain.get_genome(),
                    mutated_color,
                    archive_color,
                );
                ind.is_alive = true;
                ind
            };

            population.push(individual);
        }

        population
    }

    /// Update the archive with evaluated individuals
    pub fn update(&mut self, individuals: &[Individual]) -> usize {
        let mut insertions = 0;

        for individual in individuals {
            if individual.fitness > 0.0 {
                if self.insert(individual.clone()) {
                    insertions += 1;
                }
            }
        }

        self.generation += 1;
        insertions
    }

    /// Save archive to file
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load archive from file
    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let archive: MapElitesArchive = serde_json::from_str(&json)?;

        // Verify genome size compatibility with current brain architecture
        if let Some(ind) = archive.grid.values().next() {
            if ind.brain.genome.len() != crate::brain::GENOME_SIZE {
                eprintln!(
                    "⚠️  Archive incompatible: genome size {} != {}. Starting fresh.",
                    ind.brain.genome.len(),
                    crate::brain::GENOME_SIZE
                );
                return Ok(MapElitesArchive::new(archive.resolution));
            }
        }

        Ok(archive)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_archive_creation() {
        let archive = MapElitesArchive::new(10);
        assert_eq!(archive.capacity(), 100);
        assert_eq!(archive.filled_cells(), 0);
    }

    #[test]
    fn test_discretization() {
        let archive = MapElitesArchive::new(10);

        assert_eq!(archive.discretize(0.0), 0);
        assert_eq!(archive.discretize(1.0), 9);
        assert_eq!(archive.discretize(0.5), 5);
    }

    #[test]
    fn test_insertion() {
        let mut archive = MapElitesArchive::new(10);

        let mut individual = Individual::new_random(0);
        individual.congestion = 0.5;
        individual.agility = 0.5;
        individual.fitness = 100.0;

        assert!(archive.insert(individual));
        assert_eq!(archive.filled_cells(), 1);
    }

    #[test]
    fn test_better_fitness_replacement() {
        let mut archive = MapElitesArchive::new(10);

        let mut individual1 = Individual::new_random(0);
        individual1.congestion = 0.5;
        individual1.agility = 0.5;
        individual1.fitness = 100.0;

        assert!(archive.insert(individual1));

        let mut individual2 = Individual::new_random(1);
        individual2.congestion = 0.5;
        individual2.agility = 0.5;
        individual2.fitness = 200.0;

        assert!(archive.insert(individual2));
        assert_eq!(archive.filled_cells(), 1);
        assert_eq!(archive.best_fitness, 200.0);
    }

    #[test]
    fn test_worse_fitness_rejection() {
        let mut archive = MapElitesArchive::new(10);

        let mut individual1 = Individual::new_random(0);
        individual1.congestion = 0.5;
        individual1.agility = 0.5;
        individual1.fitness = 200.0;

        assert!(archive.insert(individual1));

        let mut individual2 = Individual::new_random(1);
        individual2.congestion = 0.5;
        individual2.agility = 0.5;
        individual2.fitness = 100.0;

        assert!(!archive.insert(individual2));
        assert_eq!(archive.filled_cells(), 1);
    }

    #[test]
    fn test_population_generation() {
        let mut archive = MapElitesArchive::new(10);

        // Add some elites
        for i in 0..5 {
            let mut individual = Individual::new_random(i);
            individual.congestion = i as f64 / 10.0;
            individual.agility = i as f64 / 10.0;
            individual.fitness = (i * 100) as f64;
            archive.insert(individual);
        }

        let population = archive.generate_population(50, 0.1, 0.5);
        assert_eq!(population.len(), 50);
    }
}
