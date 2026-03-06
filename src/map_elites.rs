//! MAP-Elites Quality-Diversity Algorithm Implementation
//!
//! MAP-Elites maintains a grid of high-performing solutions where each cell
//! represents a unique behavioral niche. The algorithm illuminates the
//! behavioral space by discovering diverse, high-quality solutions.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use std::path::Path;

use crate::brain::Individual;
use crate::snake::{load_json_gz, save_json_gz};

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
    /// Grid storing elite individuals: key = (descriptor1_bin, descriptor2_bin)
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
    pub best_fitness: f32,
    /// Generation counter
    pub generation: u32,
    /// Name of first behavioral descriptor (X-axis)
    #[serde(default = "default_descriptor_1")]
    pub descriptor_1: String,
    /// Name of second behavioral descriptor (Y-axis)
    #[serde(default = "default_descriptor_2")]
    pub descriptor_2: String,
}

fn default_descriptor_1() -> String {
    "path_directness".to_string()
}

fn default_descriptor_2() -> String {
    "body_avoidance".to_string()
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
            descriptor_1: default_descriptor_1(),
            descriptor_2: default_descriptor_2(),
        }
    }

    /// Discretize a behavioral descriptor value into a grid bin
    /// Value should be in [0.0, 1.0]
    fn discretize(&self, value: f32) -> usize {
        let clamped = value.clamp(0.0, 1.0);
        let bin = (clamped * (self.resolution - 1) as f32).round() as usize;
        bin.min(self.resolution - 1)
    }

    /// Get the grid cell coordinates for an individual
    pub fn get_cell(&self, individual: &Individual) -> (usize, usize) {
        let path_directness_bin = self.discretize(individual.path_directness);
        let body_avoidance_bin = self.discretize(individual.body_avoidance);
        (path_directness_bin, body_avoidance_bin)
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
    pub fn coverage(&self) -> f32 {
        self.grid.len() as f32 / self.capacity() as f32
    }

    /// Generate a new population by selecting and varying elites
    pub fn generate_population(
        &self,
        population_size: usize,
        mutation_rate: f32,
        mutation_strength: f32,
    ) -> Vec<Individual> {
        use rand::Rng;
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
        let elites: Vec<(&(usize, usize), &Individual)> = self.grid.iter().collect();

        // Build fitness-weighted index using sqrt to preserve diversity
        let weights: Vec<f32> = elites
            .iter()
            .map(|(_, ind)| ind.fitness.max(1.0).powf(0.5))
            .collect();
        let total_weight: f32 = weights.iter().sum();
        let normalized_weights: Vec<f32> = weights.iter().map(|w| w / total_weight).collect();

        // Weighted selection closure
        let weighted_select = |rng: &mut rand::rngs::ThreadRng| -> usize {
            let r: f32 = rng.gen();
            let mut cumulative = 0.0;
            for (i, &w) in normalized_weights.iter().enumerate() {
                cumulative += w;
                if r <= cumulative {
                    return i;
                }
            }
            elites.len() - 1
        };

        // Color mutation strength (smaller than brain mutation)
        const COLOR_MUTATION_STRENGTH: f32 = 0.05;

        for id in 0..population_size {
            // Select elite using fitness-weighted selection
            let idx = weighted_select(&mut rng);
            let (_cell, parent) = elites[idx];

            // Create a mutated offspring
            let mutated_brain = parent.brain.mutate(mutation_rate, mutation_strength);

            // Mutate color with small jitter
            let mutated_color = parent.color.mutate(COLOR_MUTATION_STRENGTH);

            // Calculate archive_color from parent cell fitness
            let normalized = (parent.fitness / self.best_fitness.max(1.0)).clamp(0.0, 1.0);
            let archive_color = crate::brain::GenomeColor {
                r: 0.1,
                g: normalized,
                b: 1.0 - normalized,
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
        mutation_rate: f32,
        mutation_strength: f32,
        crossover_rate: f32,
    ) -> Vec<Individual> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut population = Vec::with_capacity(population_size);

        // If archive is empty or has only one elite, generate random/mutated individuals
        if self.grid.len() <= 1 {
            return self.generate_population(population_size, mutation_rate, mutation_strength);
        }

        // Collect elites with their cell coordinates for archive_color calculation
        let elites: Vec<(&(usize, usize), &Individual)> = self.grid.iter().collect();

        // Build fitness-weighted index using sqrt to preserve diversity
        let weights: Vec<f32> = elites
            .iter()
            .map(|(_, ind)| ind.fitness.max(1.0).powf(0.5))
            .collect();
        let total_weight: f32 = weights.iter().sum();
        let normalized_weights: Vec<f32> = weights.iter().map(|w| w / total_weight).collect();

        // Weighted selection closure
        let weighted_select = |rng: &mut rand::rngs::ThreadRng| -> usize {
            let r: f32 = rng.gen();
            let mut cumulative = 0.0;
            for (i, &w) in normalized_weights.iter().enumerate() {
                cumulative += w;
                if r <= cumulative {
                    return i;
                }
            }
            elites.len() - 1
        };

        // Color mutation strength (smaller than brain mutation)
        const COLOR_MUTATION_STRENGTH: f32 = 0.05;

        for id in 0..population_size {
            let individual = if rng.gen::<f32>() < crossover_rate && elites.len() >= 2 {
                // Crossover between two fitness-weighted selected elites
                let idx1 = weighted_select(&mut rng);
                let idx2 = weighted_select(&mut rng);
                let (_cell1, parent1) = elites[idx1];
                let (_cell2, parent2) = elites[idx2];

                // Brain crossover
                let child_brain = parent1.brain.crossover(&parent2.brain);
                let mutated_brain = child_brain.mutate(mutation_rate, mutation_strength);

                // Color inheritance: blend from parents + mutation
                let blend_factor = rng.gen::<f32>();
                let child_color = parent1.color.lerp(&parent2.color, blend_factor);
                let mutated_color = child_color.mutate(COLOR_MUTATION_STRENGTH);

                // Archive color from parent1's cell fitness
                let normalized = (parent1.fitness / self.best_fitness.max(1.0)).clamp(0.0, 1.0);
                let archive_color = crate::brain::GenomeColor {
                    r: 0.1,
                    g: normalized,
                    b: 1.0 - normalized,
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
                // Just mutation with fitness-weighted selection
                let idx = weighted_select(&mut rng);
                let (_cell, parent) = elites[idx];
                let mutated_brain = parent.brain.mutate(mutation_rate, mutation_strength);

                // Mutate color with small jitter
                let mutated_color = parent.color.mutate(COLOR_MUTATION_STRENGTH);

                // Archive color from parent's cell fitness
                let normalized = (parent.fitness / self.best_fitness.max(1.0)).clamp(0.0, 1.0);
                let archive_color = crate::brain::GenomeColor {
                    r: 0.1,
                    g: normalized,
                    b: 1.0 - normalized,
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

    /// Save archive to file (gzip compressed)
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let path_obj = Path::new(path);

        // Use gzip compression
        save_json_gz(path_obj, self)?;
        Ok(())
    }

    /// Load archive from file (supports both .json and .json.gz)
    pub fn load(path: &str) -> std::io::Result<Self> {
        let path_obj = Path::new(path);

        // Try loading with gzip support
        let archive: MapElitesArchive = if path_obj.extension().map_or(false, |ext| ext == "gz") {
            load_json_gz(path_obj)?
        } else {
            // Try plain JSON first
            let content = std::fs::read_to_string(path)?;
            serde_json::from_str(&content)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
        };

        // Verify genome size compatibility with current brain architecture
        if let Some(ind) = archive.grid.values().next() {
            if ind.brain.genome.len() != crate::brain::GENOME_SIZE {
                eprintln!("⚠️  GENOME MISMATCH: archive has {} params, current brain has {}. Old archive discarded.", ind.brain.genome.len(), crate::brain::GENOME_SIZE);
                return Ok(MapElitesArchive::new(archive.resolution));
            }
        }

        // Check descriptor compatibility
        let current_desc1 = default_descriptor_1();
        let current_desc2 = default_descriptor_2();
        if archive.descriptor_1 != current_desc1 || archive.descriptor_2 != current_desc2 {
            eprintln!(
                "⚠️  Archive descriptors mismatch: loaded ({}, {}) != current ({}, {}).",
                archive.descriptor_1, archive.descriptor_2, current_desc1, current_desc2
            );
            eprintln!("    Archive cells are misaligned.");
            // Try to get the run directory from the path
            if let Some(parent) = std::path::Path::new(path).parent() {
                if let Some(run_dir) = parent.parent() {
                    eprintln!(
                        "    Recommend starting a new run with: rm -rf {}",
                        run_dir.display()
                    );
                }
            }
            eprintln!("    Loading anyway — archive will be repopulated with correct descriptors over time.");
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
        individual.path_directness = 0.5;
        individual.body_avoidance = 0.5;
        individual.fitness = 100.0;

        assert!(archive.insert(individual));
        assert_eq!(archive.filled_cells(), 1);
    }

    #[test]
    fn test_better_fitness_replacement() {
        let mut archive = MapElitesArchive::new(10);

        let mut individual1 = Individual::new_random(0);
        individual1.path_directness = 0.5;
        individual1.body_avoidance = 0.5;
        individual1.fitness = 100.0;

        assert!(archive.insert(individual1));

        let mut individual2 = Individual::new_random(1);
        individual2.path_directness = 0.5;
        individual2.body_avoidance = 0.5;
        individual2.fitness = 200.0;

        assert!(archive.insert(individual2));
        assert_eq!(archive.filled_cells(), 1);
        assert_eq!(archive.best_fitness, 200.0);
    }

    #[test]
    fn test_worse_fitness_rejection() {
        let mut archive = MapElitesArchive::new(10);

        let mut individual1 = Individual::new_random(0);
        individual1.path_directness = 0.5;
        individual1.body_avoidance = 0.5;
        individual1.fitness = 200.0;

        assert!(archive.insert(individual1));

        let mut individual2 = Individual::new_random(1);
        individual2.path_directness = 0.5;
        individual2.body_avoidance = 0.5;
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
            individual.path_directness = i as f32 / 10.0;
            individual.body_avoidance = i as f32 / 10.0;
            individual.fitness = (i * 100) as f32;
            archive.insert(individual);
        }

        let population = archive.generate_population(50, 0.1, 0.5);
        assert_eq!(population.len(), 50);
    }
}
