//! Lightweight CPU-based Feed-Forward Neural Network for MAP-Elites
//!
//! This module provides a simple neural network that can be instantiated from
//! a flat genome vector (weights + biases). No external ML crates required.

use serde::{Deserialize, Serialize};

use crate::snake::STATE_SIZE;

/// Network architecture constants
pub const INPUT_SIZE: usize = STATE_SIZE; // 34 inputs (17 current + 17 previous frame)
pub const HIDDEN_SIZE: usize = 128; // First hidden layer (was 64)
pub const HIDDEN2_SIZE: usize = 64; // Second hidden layer (new)
pub const OUTPUT_SIZE: usize = 3; // Left, Straight, Right

/// Total number of parameters in the genome
/// [0]           ih weights:  INPUT_SIZE * HIDDEN_SIZE  = 34*128 = 4352
/// [4352]        h1 biases:   HIDDEN_SIZE               = 128
/// [4480]        h1h2 weights: HIDDEN_SIZE * HIDDEN2_SIZE = 128*64 = 8192
/// [12672]       h2 biases:   HIDDEN2_SIZE              = 64
/// [12736]       ho weights:  HIDDEN2_SIZE * OUTPUT_SIZE = 64*3  = 192
/// [12928]       o biases:    OUTPUT_SIZE               = 3
/// TOTAL = 12931 parameters
pub const GENOME_SIZE: usize = (INPUT_SIZE * HIDDEN_SIZE)
    + HIDDEN_SIZE
    + (HIDDEN_SIZE * HIDDEN2_SIZE)
    + HIDDEN2_SIZE
    + (HIDDEN2_SIZE * OUTPUT_SIZE)
    + OUTPUT_SIZE;

/// RGB Color representation for genetic inheritance
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct GenomeColor {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl GenomeColor {
    /// Create a random color
    pub fn random() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        Self {
            r: rng.gen::<f64>(),
            g: rng.gen::<f64>(),
            b: rng.gen::<f64>(),
        }
    }

    /// Interpolate between two colors (for crossover)
    pub fn lerp(&self, other: &GenomeColor, t: f64) -> Self {
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
        }
    }

    /// Apply small random variation (jitter) for mutation
    pub fn mutate(&self, strength: f64) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut jitter = |val: f64| -> f64 {
            let delta = (rng.gen::<f64>() * 2.0 - 1.0) * strength;
            (val + delta).clamp(0.0, 1.0)
        };

        Self {
            r: jitter(self.r),
            g: jitter(self.g),
            b: jitter(self.b),
        }
    }

    /// Convert to Bevy Color
    pub fn to_bevy_color(&self) -> bevy::prelude::Color {
        bevy::prelude::Color::rgb(self.r as f32, self.g as f32, self.b as f32)
    }
}

impl Default for GenomeColor {
    fn default() -> Self {
        Self::random()
    }
}

/// Action output by the brain
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    Left = 0,
    Straight = 1,
    Right = 2,
}

impl Default for Action {
    fn default() -> Self {
        Action::Straight
    }
}

/// Lightweight CPU-based Feed-Forward Neural Network
///
/// Architecture: 34 inputs -> 128 hidden1 (ReLU) -> 64 hidden2 (ReLU) -> 3 outputs (argmax)
/// The genome is a flat vector containing all weights and biases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Brain {
    /// Genome: flat vector of all weights and biases
    pub genome: Vec<f64>,
    /// Input to hidden1 layer weights [HIDDEN_SIZE x INPUT_SIZE]
    weights_ih: Vec<f64>,
    /// Hidden1 layer biases [HIDDEN_SIZE]
    biases_h1: Vec<f64>,
    /// Hidden1 to hidden2 layer weights [HIDDEN2_SIZE x HIDDEN_SIZE]
    weights_h1h2: Vec<f64>,
    /// Hidden2 layer biases [HIDDEN2_SIZE]
    biases_h2: Vec<f64>,
    /// Hidden2 to output layer weights [OUTPUT_SIZE x HIDDEN2_SIZE]
    weights_ho: Vec<f64>,
    /// Output layer biases [OUTPUT_SIZE]
    biases_o: Vec<f64>,
}

impl Brain {
    /// Create a new brain with random weights
    pub fn new_random() -> Self {
        let mut rng = rand::thread_rng();
        use rand::Rng;

        let genome: Vec<f64> = (0..GENOME_SIZE)
            .map(|_| rng.gen::<f64>() * 2.0 - 1.0) // Xavier-like initialization
            .collect();

        Self::from_genome(&genome)
    }

    /// Create a brain from a flat genome vector
    pub fn from_genome(genome: &[f64]) -> Self {
        assert_eq!(
            genome.len(),
            GENOME_SIZE,
            "Genome size mismatch: expected {}, got {}",
            GENOME_SIZE,
            genome.len()
        );

        let mut offset = 0;

        // Extract weights_ih [HIDDEN_SIZE * INPUT_SIZE]
        let weights_ih: Vec<f64> = genome[offset..offset + INPUT_SIZE * HIDDEN_SIZE].to_vec();
        offset += INPUT_SIZE * HIDDEN_SIZE;

        // Extract biases_h1 [HIDDEN_SIZE]
        let biases_h1: Vec<f64> = genome[offset..offset + HIDDEN_SIZE].to_vec();
        offset += HIDDEN_SIZE;

        // Extract weights_h1h2 [HIDDEN2_SIZE * HIDDEN_SIZE]
        let weights_h1h2: Vec<f64> = genome[offset..offset + HIDDEN_SIZE * HIDDEN2_SIZE].to_vec();
        offset += HIDDEN_SIZE * HIDDEN2_SIZE;

        // Extract biases_h2 [HIDDEN2_SIZE]
        let biases_h2: Vec<f64> = genome[offset..offset + HIDDEN2_SIZE].to_vec();
        offset += HIDDEN2_SIZE;

        // Extract weights_ho [OUTPUT_SIZE * HIDDEN2_SIZE]
        let weights_ho: Vec<f64> = genome[offset..offset + HIDDEN2_SIZE * OUTPUT_SIZE].to_vec();
        offset += HIDDEN2_SIZE * OUTPUT_SIZE;

        // Extract biases_o [OUTPUT_SIZE]
        let biases_o: Vec<f64> = genome[offset..offset + OUTPUT_SIZE].to_vec();

        Self {
            genome: genome.to_vec(),
            weights_ih,
            biases_h1,
            weights_h1h2,
            biases_h2,
            weights_ho,
            biases_o,
        }
    }

    /// Forward pass: compute action from input state
    ///
    /// Input: 34-dimensional state vector (17 current + 17 previous frame sensors)
    /// Output: Action (Left, Straight, Right)
    pub fn predict(&self, input: &[f32; STATE_SIZE]) -> Action {
        let output = self.forward(input);
        self.argmax(&output)
    }

    /// Forward pass returning raw output values (for debugging/analysis)
    pub fn forward(&self, input: &[f32; STATE_SIZE]) -> [f64; OUTPUT_SIZE] {
        // Layer 1: INPUT → HIDDEN1 (ReLU)
        let mut hidden1 = [0.0f64; HIDDEN_SIZE];
        for h in 0..HIDDEN_SIZE {
            let mut sum = self.biases_h1[h];
            for i in 0..INPUT_SIZE {
                sum += self.weights_ih[h * INPUT_SIZE + i] * input[i] as f64;
            }
            hidden1[h] = relu(sum);
        }

        // Layer 2: HIDDEN1 → HIDDEN2 (ReLU)
        let mut hidden2 = [0.0f64; HIDDEN2_SIZE];
        for h in 0..HIDDEN2_SIZE {
            let mut sum = self.biases_h2[h];
            for i in 0..HIDDEN_SIZE {
                sum += self.weights_h1h2[h * HIDDEN_SIZE + i] * hidden1[i];
            }
            hidden2[h] = relu(sum);
        }

        // Output layer: HIDDEN2 → OUTPUT (linear, we use argmax)
        let mut output = [0.0; OUTPUT_SIZE];
        for o in 0..OUTPUT_SIZE {
            let mut sum = self.biases_o[o];
            for h in 0..HIDDEN2_SIZE {
                sum += self.weights_ho[o * HIDDEN2_SIZE + h] * hidden2[h];
            }
            output[o] = sum;
        }

        output
    }

    /// Get the action with highest output value
    fn argmax(&self, output: &[f64; OUTPUT_SIZE]) -> Action {
        let mut max_idx = 0;
        let mut max_val = output[0];

        for (i, &val) in output.iter().enumerate().skip(1) {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        match max_idx {
            0 => Action::Left,
            1 => Action::Straight,
            2 => Action::Right,
            _ => Action::Straight, // Fallback
        }
    }

    /// Get the genome reference
    pub fn get_genome(&self) -> &[f64] {
        &self.genome
    }

    /// Create a mutated copy of this brain
    pub fn mutate(&self, mutation_rate: f64, mutation_strength: f64) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let new_genome: Vec<f64> = self
            .genome
            .iter()
            .map(|&w| {
                if rng.gen::<f64>() < mutation_rate {
                    let mutation = rng.gen::<f64>() * 2.0 - 1.0;
                    w + mutation * mutation_strength
                } else {
                    w
                }
            })
            .collect();

        Self::from_genome(&new_genome)
    }

    /// Crossover between two brains
    pub fn crossover(&self, other: &Brain) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let new_genome: Vec<f64> = self
            .genome
            .iter()
            .zip(other.genome.iter())
            .map(|(&a, &b)| if rng.gen::<f64>() < 0.5 { a } else { b })
            .collect();

        Self::from_genome(&new_genome)
    }
}

/// ReLU activation function
#[inline]
fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// Individual in the population with its genome and tracking data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Individual {
    /// Unique identifier
    pub id: usize,
    /// The brain/genome
    pub brain: Brain,
    /// Genetic color (RGB) for visual identification
    pub color: GenomeColor,
    /// Archive color (from parent cell fitness: blue→green gradient)
    pub archive_color: GenomeColor,
    /// Fitness score (apples * 1000 + frames)
    pub fitness: f64,
    /// Behavioral descriptor 1: Congestion tolerance (spaces tight vs open)
    pub congestion: f64,
    /// Behavioral descriptor 2: Agility (turn frequency)
    pub agility: f64,
    /// Frames survived
    pub frames_survived: u32,
    /// Apples eaten
    pub apples_eaten: u32,
    /// Whether this individual is currently alive in simulation
    pub is_alive: bool,
}

impl Individual {
    /// Create a new individual with a random brain and random color
    pub fn new_random(id: usize) -> Self {
        Self {
            id,
            brain: Brain::new_random(),
            color: GenomeColor::random(),
            archive_color: GenomeColor::default(),
            fitness: 0.0,
            congestion: 0.0,
            agility: 0.0,
            frames_survived: 0,
            apples_eaten: 0,
            is_alive: true,
        }
    }

    /// Create an individual from a genome (used in evolution)
    #[allow(dead_code)]
    pub fn from_genome(id: usize, genome: &[f64]) -> Self {
        Self {
            id,
            brain: Brain::from_genome(genome),
            color: GenomeColor::random(),
            archive_color: GenomeColor::default(),
            fitness: 0.0,
            congestion: 0.0,
            agility: 0.0,
            frames_survived: 0,
            apples_eaten: 0,
            is_alive: true,
        }
    }

    /// Create an individual from a genome with archive color (from parent cell fitness)
    pub fn from_genome_with_archive_color(
        id: usize,
        genome: &[f64],
        color: GenomeColor,
        archive_color: GenomeColor,
    ) -> Self {
        Self {
            id,
            brain: Brain::from_genome(genome),
            color,
            archive_color,
            fitness: 0.0,
            congestion: 0.0,
            agility: 0.0,
            frames_survived: 0,
            apples_eaten: 0,
            is_alive: true,
        }
    }

    /// Reset for a new evaluation
    pub fn reset(&mut self) {
        self.fitness = 0.0;
        self.congestion = 0.0;
        self.agility = 0.0;
        self.frames_survived = 0;
        self.apples_eaten = 0;
        self.is_alive = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genome_size() {
        let expected = (INPUT_SIZE * HIDDEN_SIZE)
            + HIDDEN_SIZE
            + (HIDDEN_SIZE * HIDDEN2_SIZE)
            + HIDDEN2_SIZE
            + (HIDDEN2_SIZE * OUTPUT_SIZE)
            + OUTPUT_SIZE;
        assert_eq!(GENOME_SIZE, expected);
        println!("Genome size: {}", GENOME_SIZE);
    }

    #[test]
    fn test_brain_creation() {
        let brain = Brain::new_random();
        assert_eq!(brain.genome.len(), GENOME_SIZE);
    }

    #[test]
    fn test_brain_prediction() {
        let brain = Brain::new_random();
        let input = [0.5f32; STATE_SIZE];
        let action = brain.predict(&input);
        assert!(matches!(
            action,
            Action::Left | Action::Straight | Action::Right
        ));
    }

    #[test]
    fn test_mutation() {
        let brain = Brain::new_random();
        let mutated = brain.mutate(0.1, 0.5);

        // Check that some weights changed
        let changes: usize = brain
            .genome
            .iter()
            .zip(mutated.genome.iter())
            .filter(|pair| (*pair.0 - *pair.1).abs() > 1e-10)
            .count();

        // With 10% mutation rate on ~13k genes, expect ~1300 changes
        assert!(changes > 0);
    }

    #[test]
    fn test_crossover() {
        let brain1 = Brain::new_random();
        let brain2 = Brain::new_random();
        let child = brain1.crossover(&brain2);

        // Child should have genes from both parents
        let from_parent1: usize = brain1
            .genome
            .iter()
            .zip(child.genome.iter())
            .filter(|pair| (*pair.0 - *pair.1).abs() < 1e-10)
            .count();

        assert!(from_parent1 > 0 && from_parent1 < GENOME_SIZE);
    }
}
