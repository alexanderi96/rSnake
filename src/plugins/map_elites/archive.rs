//! MAP-Elites Quality-Diversity Algorithm Implementation
//!
//! MAP-Elites maintains a grid of high-performing solutions where each cell
//! represents a unique behavioral niche. The algorithm illuminates the
//! behavioral space by discovering diverse, high-quality solutions.

use rand::rngs::SmallRng;
use rand::SeedableRng;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use std::path::Path;

use crate::plugins::map_elites::individual::Individual;
use crate::snake::{load_json_gz, save_json_gz};

thread_local! {
    static MAP_RNG: std::cell::RefCell<SmallRng> =
        std::cell::RefCell::new(SmallRng::from_entropy());
}

/// Number of bins for each behavioral descriptor dimension
pub const GRID_RESOLUTION: usize = 33;

/// Colore mostrato allo spawn: gradiente fitness (blu→verde) del GENITORE,
/// congelato alla nascita. Con snake_color_from_parent=false il render usa
/// invece il gradiente live della fitness corrente (vedi ui::render_system).
fn display_color(
    parent_fitness: f32,
    best_fitness: f32,
) -> crate::plugins::map_elites::individual::GenomeColor {
    let normalized = (parent_fitness / best_fitness.max(1.0)).clamp(0.0, 1.0);
    crate::plugins::map_elites::individual::GenomeColor {
        r: 0.1,
        g: normalized,
        b: 1.0 - normalized,
    }
}

/// Custom serializer for HashMap with (usize, usize, usize) keys
/// Converts tuple keys to "x,y,z" string format
fn serialize_grid<S>(
    grid: &HashMap<(usize, usize, usize), Individual>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let string_keyed: std::collections::HashMap<String, &Individual> = grid
        .iter()
        .map(|((x, y, z), v)| (format!("{},{},{}", x, y, z), v))
        .collect();
    string_keyed.serialize(serializer)
}

/// Custom deserializer for HashMap with (usize, usize, usize) keys
/// Converts "x,y,z" string keys back to tuple format
fn deserialize_grid<'de, D>(
    deserializer: D,
) -> Result<HashMap<(usize, usize, usize), Individual>, D::Error>
where
    D: Deserializer<'de>,
{
    let string_keyed: std::collections::HashMap<String, Individual> =
        Deserialize::deserialize(deserializer)?;

    let mut grid = HashMap::new();
    for (key, value) in string_keyed {
        let parts: Vec<&str> = key.split(',').collect();
        if parts.len() == 3 {
            if let (Ok(x), Ok(y), Ok(z)) = (
                parts[0].parse::<usize>(),
                parts[1].parse::<usize>(),
                parts[2].parse::<usize>(),
            ) {
                grid.insert((x, y, z), value);
            }
        }
    }
    Ok(grid)
}

/// MAP-Elites Archive: a 3D grid storing elite individuals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapElitesArchive {
    /// Grid storing elite individuals: key = (descriptor1_bin, descriptor2_bin, descriptor3_bin)
    #[serde(
        serialize_with = "serialize_grid",
        deserialize_with = "deserialize_grid"
    )]
    pub grid: HashMap<(usize, usize, usize), Individual>,
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
    /// Name of third behavioral descriptor (Z-axis)
    #[serde(default = "default_descriptor_3")]
    pub descriptor_3: String,
}

fn default_descriptor_1() -> String {
    "turn_rate".to_string()
}

fn default_descriptor_2() -> String {
    "center_affinity".to_string()
}

fn default_descriptor_3() -> String {
    "coverage".to_string()
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
            descriptor_3: default_descriptor_3(),
        }
    }

    /// Discretize a behavioral descriptor value into a grid bin
    /// Value should be in [0.0, 1.0]
    fn discretize(&self, value: f32) -> usize {
        let clamped = value.clamp(0.0, 1.0);
        // floor: ogni bin copre 1/resolution del range (round dimezzava i bin di bordo)
        let bin = (clamped * self.resolution as f32) as usize;
        bin.min(self.resolution - 1)
    }

    /// Get the grid cell coordinates for an individual
    pub fn get_cell(&self, individual: &Individual) -> (usize, usize, usize) {
        (
            self.discretize(individual.desc_turn_rate),
            self.discretize(individual.desc_center_affinity),
            self.discretize(individual.desc_coverage),
        )
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
        self.resolution.pow(3)
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

        let mut population = Vec::with_capacity(population_size);

        // If archive is empty, generate random individuals
        if self.grid.is_empty() {
            for id in 0..population_size {
                population.push(Individual::new_random(id));
            }
            return population;
        }

        // Collect elites with their cell coordinates for archive_color calculation
        let elites: Vec<(&(usize, usize, usize), &Individual)> = self.grid.iter().collect();

        // MAP-Elites canonico: selezione uniforme sulle nicchie occupate.
        // Protegge gli elite mediocri-ma-diversi, che sono il motore dell'esplorazione.
        let weighted_select = |rng: &mut SmallRng| -> usize { rng.gen_range(0..elites.len()) };

        // Color mutation strength (smaller than brain mutation)
        const COLOR_MUTATION_STRENGTH: f32 = 0.05;

        MAP_RNG.with(|rng_cell| {
            let mut rng = rng_cell.borrow_mut();
            for id in 0..population_size {
                // Select elite using fitness-weighted selection
                let idx = weighted_select(&mut rng);
                let (_cell, parent) = elites[idx];

                // Create a mutated offspring
                let mutated_brain = parent.brain.mutate(mutation_rate, mutation_strength);

                // Mutate color with small jitter
                let mutated_color = parent.color.mutate(COLOR_MUTATION_STRENGTH);

                // Displayed color: inherited from parent, or fitness gradient
                let archive_color = display_color(parent.fitness, self.best_fitness);

                let mut individual = Individual::from_genome_with_archive_color(
                    id,
                    mutated_brain.get_genome(),
                    mutated_color,
                    archive_color,
                );
                individual.is_alive = true;

                population.push(individual);
            }
        });

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

        let mut population = Vec::with_capacity(population_size);

        // If archive is empty or has only one elite, generate random/mutated individuals
        if self.grid.len() <= 1 {
            return self.generate_population(population_size, mutation_rate, mutation_strength);
        }

        // Collect elites with their cell coordinates for archive_color calculation
        let elites: Vec<(&(usize, usize, usize), &Individual)> = self.grid.iter().collect();

        // MAP-Elites canonico: selezione uniforme sulle nicchie occupate.
        // Protegge gli elite mediocri-ma-diversi, che sono il motore dell'esplorazione.
        let weighted_select = |rng: &mut SmallRng| -> usize { rng.gen_range(0..elites.len()) };

        // Color mutation strength (smaller than brain mutation)
        const COLOR_MUTATION_STRENGTH: f32 = 0.05;

        MAP_RNG.with(|rng_cell| {
            let mut rng = rng_cell.borrow_mut();
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

                    // Displayed color: inherited from parents, or fitness gradient
                    let archive_color = display_color(parent1.fitness.max(parent2.fitness), self.best_fitness);

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

                    // Displayed color: inherited from parent, or fitness gradient
                    let archive_color = display_color(parent.fitness, self.best_fitness);

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
        });

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
            if ind.brain.genome.len() != crate::plugins::map_elites::individual::GENOME_SIZE {
                eprintln!("⚠️  GENOME MISMATCH: archive has {} params, current brain has {}. Old archive discarded.", ind.brain.genome.len(), crate::plugins::map_elites::individual::GENOME_SIZE);
                return Ok(MapElitesArchive::new(archive.resolution));
            }
        }

        // Check descriptor compatibility
        let current_desc1 = default_descriptor_1();
        let current_desc2 = default_descriptor_2();
        let current_desc3 = default_descriptor_3();
        if archive.descriptor_1 != current_desc1
            || archive.descriptor_2 != current_desc2
            || archive.descriptor_3 != current_desc3
        {
            eprintln!(
                "⚠️  Archive descriptors mismatch: loaded ({}, {}, {}) != current ({}, {}, {}).",
                archive.descriptor_1,
                archive.descriptor_2,
                archive.descriptor_3,
                current_desc1,
                current_desc2,
                current_desc3
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
        assert_eq!(archive.capacity(), 1000);
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
        individual.desc_turn_rate = 0.5;
        individual.desc_coverage = 0.5;
        individual.fitness = 100.0;

        assert!(archive.insert(individual));
        assert_eq!(archive.filled_cells(), 1);
    }

    #[test]
    fn test_better_fitness_replacement() {
        let mut archive = MapElitesArchive::new(10);

        let mut individual1 = Individual::new_random(0);
        individual1.desc_turn_rate = 0.5;
        individual1.desc_coverage = 0.5;
        individual1.fitness = 100.0;

        assert!(archive.insert(individual1));

        let mut individual2 = Individual::new_random(1);
        individual2.desc_turn_rate = 0.5;
        individual2.desc_coverage = 0.5;
        individual2.fitness = 200.0;

        assert!(archive.insert(individual2));
        assert_eq!(archive.filled_cells(), 1);
        assert_eq!(archive.best_fitness, 200.0);
    }

    #[test]
    fn test_worse_fitness_rejection() {
        let mut archive = MapElitesArchive::new(10);

        let mut individual1 = Individual::new_random(0);
        individual1.desc_turn_rate = 0.5;
        individual1.desc_coverage = 0.5;
        individual1.fitness = 200.0;

        assert!(archive.insert(individual1));

        let mut individual2 = Individual::new_random(1);
        individual2.desc_turn_rate = 0.5;
        individual2.desc_coverage = 0.5;
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
            individual.desc_turn_rate = i as f32 / 10.0;
            individual.desc_coverage = i as f32 / 10.0;
            individual.fitness = (i * 100) as f32;
            archive.insert(individual);
        }

        let population = archive.generate_population(50, 0.1, 0.5);
        assert_eq!(population.len(), 50);
    }
}
