use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    tensor::{backend::Backend, Tensor},
};
use serde::{Deserialize, Serialize};

use crate::snake::STATE_SIZE;

/// DQN Model using Burn
/// Architecture: 34 inputs -> 128 hidden -> 3 outputs (Q-values for [Left, Right, Straight])
#[derive(Module, Debug)]
pub struct DqnModel<B: Backend> {
    pub input: Linear<B>,
    pub hidden: Linear<B>,
    pub output: Linear<B>,
    pub activation: Relu,
}

impl<B: Backend> DqnModel<B> {
    /// Create a new DQN model with the given device
    pub fn new(device: &B::Device) -> Self {
        let input = LinearConfig::new(STATE_SIZE as usize, 128).init(device);
        let hidden = LinearConfig::new(128, 128).init(device);
        let output = LinearConfig::new(128, 3).init(device);

        Self {
            input,
            hidden,
            output,
            activation: Relu::new(),
        }
    }

    /// Forward pass through the network
    /// Input: [batch_size, 34] - normalized sensor readings with frame stacking
    /// Output: [batch_size, 3] - Q-values for each action
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.input.forward(input);
        let x = self.activation.forward(x);
        let x = self.hidden.forward(x);
        let x = self.activation.forward(x);
        self.output.forward(x)
    }

    /// Forward pass for a single state (inference)
    /// Note: Not used in hot loop - use forward_batch for batch inference to avoid GPU sync overhead
    #[allow(dead_code)]
    pub fn forward_single(&self, state: [f32; STATE_SIZE], device: &B::Device) -> [f32; 3] {
        let input = Tensor::<B, 2>::from_floats([state], device);
        let output = self.forward(input);
        let data = output.to_data();
        let values = data.as_slice::<f32>().unwrap();
        [values[0], values[1], values[2]]
    }

    /// Forward pass for batch inference
    /// states: Vec of [f32; 34] sensor readings
    /// Returns: Vec of [f32; 3] Q-values
    pub fn forward_batch(&self, states: &[[f32; STATE_SIZE]], device: &B::Device) -> Vec<[f32; 3]> {
        if states.is_empty() {
            return Vec::new();
        }

        // Convert to tensor [batch_size, STATE_SIZE] using TensorData for correct dimensions
        let flat: Vec<f32> = states.iter().flatten().copied().collect();
        let batch_size = states.len();
        let state_data = burn::tensor::TensorData::new(flat, [batch_size, STATE_SIZE]);
        let input = Tensor::<B, 2>::from_data(state_data, device);

        let output = self.forward(input);
        let data = output.to_data();
        let values = data.as_slice::<f32>().unwrap();

        // Convert back to Vec<[f32; 3]>
        (0..batch_size)
            .map(|i| {
                let offset = i * 3;
                [values[offset], values[offset + 1], values[offset + 2]]
            })
            .collect()
    }
}

/// Serializable model state for saving/loading
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelState {
    pub input_weights: Vec<f32>,
    pub input_bias: Vec<f32>,
    pub hidden_weights: Vec<f32>,
    pub hidden_bias: Vec<f32>,
    pub output_weights: Vec<f32>,
    pub output_bias: Vec<f32>,
}

impl ModelState {
    /// Save to JSON file
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load from JSON file
    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let state: ModelState = serde_json::from_str(&json)?;
        Ok(state)
    }
}
