use bevy::prelude::*;
use burn::{
    backend::{
        wgpu::{Wgpu, WgpuDevice}, // Rimosso AutoGraphicsApi non necessario
        Autodiff,
    },
    module::Module,
    nn::loss::MseLoss,
    optim::{
        adaptor::OptimizerAdaptor, Adam, AdamConfig, GradientsParams, Optimizer, SimpleOptimizer,
    },
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{backend::Backend, Int, Tensor},
};
use std::collections::VecDeque;

use crate::model::DqnModel;
use crate::snake::{BATCH_SIZE, MEMORY_SIZE, STATE_SIZE, TARGET_UPDATE_FREQ};

// FIX 1: Definizione Backend aggiornata per Burn 0.16
// Wgpu ora ha default corretti (GraphicsApi, f32, i32) che evitano l'errore di FloatElement
type MyBackend = Autodiff<Wgpu>;
type MyDevice = <MyBackend as Backend>::Device;

pub type Experience = ([f32; STATE_SIZE], usize, f32, [f32; STATE_SIZE], bool);

#[derive(Clone, Debug)]
pub struct AgentConfig {
    pub learning_rate: f64,
    pub gamma: f32,
    pub epsilon_start: f32,
    pub epsilon_min: f32,
    pub epsilon_decay: f32,
}

impl AgentConfig {
    pub fn new(_snake_count: usize) -> Self {
        Self {
            learning_rate: 1e-4,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_min: 0.01,
            epsilon_decay: 0.995,
        }
    }
}

pub struct ReplayBuffer {
    pub buffer: VecDeque<Experience>,
    pub capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, experience: Experience) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }

    pub fn sample(&self, batch_size: usize) -> Vec<Experience> {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        self.buffer
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .choose_multiple(&mut rng, batch_size)
            .cloned()
            .collect()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

#[derive(Resource)]
pub struct DqnAgent {
    pub config: AgentConfig,
    pub online_model: DqnModel<MyBackend>,
    pub target_model: DqnModel<MyBackend>,
    pub optimizer: OptimizerAdaptor<Adam, DqnModel<MyBackend>, MyBackend>,
    pub replay_buffer: ReplayBuffer,
    pub epsilon: f32,
    pub loss: f32,
    pub target_update_counter: usize,
    pub iterations: u32,
    pub device: MyDevice,
}

// FIX 2: Implementazione unsafe di Sync per Bevy
// I tensori Burn/WGPU usano interni non-Sync (RefCell/OnceCell).
// Poiché Bevy gestisce l'accesso esclusivo tramite ResMut<DqnAgent>,
// e non stiamo usando l'agente su thread multipli raw, questo è accettabile.
unsafe impl Sync for DqnAgent {}

impl DqnAgent {
    pub fn new(config: AgentConfig) -> Self {
        // FIX 3: Sostituito BestAvailable (deprecato) con Default
        let device = WgpuDevice::default();
        println!("🚀 Initializing DqnAgent on device: {:?}", device);

        let online_model = DqnModel::new(&device);
        let target_model = online_model.clone();

        // Configurazione Optimizer
        let optimizer = AdamConfig::new().init();

        Self {
            config: config.clone(),
            online_model,
            target_model,
            optimizer,
            replay_buffer: ReplayBuffer::new(MEMORY_SIZE),
            epsilon: config.epsilon_start,
            loss: 0.0,
            target_update_counter: 0,
            iterations: 0,
            device,
        }
    }

    pub fn select_action(&self, state: [f32; STATE_SIZE]) -> usize {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        if rng.gen::<f32>() < self.epsilon {
            rng.gen_range(0..3)
        } else {
            // Inference diretta
            let q_values = self.online_model.forward_single(state, &self.device);
            q_values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        }
    }

    /// Batch action selection for multiple states
    /// Returns a vector of action indices, one for each input state
    pub fn select_actions_batch(&self, states: Vec<[f32; STATE_SIZE]>) -> Vec<usize> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let batch_size = states.len();

        // Vector to store results
        let mut actions = vec![0; batch_size];

        // Track which indices need neural network inference (non-random)
        let mut indices_to_infer: Vec<usize> = Vec::new();
        let mut states_to_infer: Vec<[f32; STATE_SIZE]> = Vec::new();

        for (i, _) in states.iter().enumerate() {
            if rng.gen::<f32>() < self.epsilon {
                // Random action for exploration
                actions[i] = rng.gen_range(0..3);
            } else {
                // Mark for neural network inference
                indices_to_infer.push(i);
                states_to_infer.push(states[i]);
            }
        }

        // If nothing to infer (all random), return immediately
        if states_to_infer.is_empty() {
            return actions;
        }

        // BATCH INFERENCE: Single GPU pass for all non-random states
        let q_values_batch = self
            .online_model
            .forward_batch(&states_to_infer, &self.device);

        // Assign computed actions to correct indices
        for (batch_idx, q_values) in q_values_batch.iter().enumerate() {
            let original_idx = indices_to_infer[batch_idx];

            // Argmax: select action with highest Q-value
            let best_action = q_values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            actions[original_idx] = best_action;
        }

        actions
    }

    pub fn remember(&mut self, experience: Experience) {
        self.replay_buffer.push(experience);
    }

    pub fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.config.epsilon_decay).max(self.config.epsilon_min);
    }

    /// TRAINING SU GPU (Double DQN)
    pub fn train(&mut self) {
        if self.replay_buffer.len() < BATCH_SIZE {
            return;
        }

        let batch = self.replay_buffer.sample(BATCH_SIZE);

        let mut states = Vec::with_capacity(BATCH_SIZE * STATE_SIZE);
        let mut next_states = Vec::with_capacity(BATCH_SIZE * STATE_SIZE);
        let mut actions = Vec::with_capacity(BATCH_SIZE);
        let mut rewards = Vec::with_capacity(BATCH_SIZE);
        let mut dones = Vec::with_capacity(BATCH_SIZE);

        for (s, a, r, ns, d) in batch {
            states.extend_from_slice(&s);
            next_states.extend_from_slice(&ns);
            actions.push(a as i32);
            rewards.push(r);
            dones.push(if d { 1.0 } else { 0.0 });
        }

        // Creazione tensori con TensorData per specificare correttamente le dimensioni
        let state_data = burn::tensor::TensorData::new(states, [BATCH_SIZE, STATE_SIZE]);
        let state_tensor = Tensor::<MyBackend, 2>::from_data(state_data, &self.device);

        let next_state_data = burn::tensor::TensorData::new(next_states, [BATCH_SIZE, STATE_SIZE]);
        let next_state_tensor = Tensor::<MyBackend, 2>::from_data(next_state_data, &self.device);

        let action_data = burn::tensor::TensorData::new(actions, [BATCH_SIZE, 1]);
        let action_tensor = Tensor::<MyBackend, 2, Int>::from_data(action_data, &self.device);

        let reward_data = burn::tensor::TensorData::new(rewards, [BATCH_SIZE]);
        let reward_tensor = Tensor::<MyBackend, 1>::from_data(reward_data, &self.device);

        let done_data = burn::tensor::TensorData::new(dones, [BATCH_SIZE]);
        let done_tensor = Tensor::<MyBackend, 1>::from_data(done_data, &self.device);

        // Double DQN Logic
        let next_q_target = self
            .target_model
            .forward(next_state_tensor.clone())
            .detach();
        let next_q_online = self.online_model.forward(next_state_tensor).detach();
        let next_actions = next_q_online.argmax(1).reshape([BATCH_SIZE, 1]);

        let max_next_q = next_q_target.gather(1, next_actions).squeeze(1);
        let target_q =
            reward_tensor + (max_next_q * done_tensor.neg().add_scalar(1.0) * self.config.gamma);

        let q_values = self.online_model.forward(state_tensor);
        let current_q = q_values.gather(1, action_tensor).squeeze(1);

        let loss = MseLoss::new().forward(
            current_q,
            target_q.detach(),
            burn::nn::loss::Reduction::Mean,
        );

        self.loss = loss.to_data().as_slice::<f32>().unwrap()[0];

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.online_model);
        self.online_model =
            self.optimizer
                .step(self.config.learning_rate, self.online_model.clone(), grads);

        self.target_update_counter += 1;
        if self.target_update_counter >= TARGET_UPDATE_FREQ {
            self.target_model = self.online_model.clone();
            self.target_update_counter = 0;
        }

        self.iterations += 1;
    }

    /// SALVATAGGIO MODELLO
    pub fn save(&mut self, path: &str) -> std::io::Result<()> {
        println!("💾 Saving model via Burn Recorder to: {}", path);
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();

        self.online_model
            .clone()
            .save_file(path, &recorder)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        Ok(())
    }

    /// CARICAMENTO MODELLO
    pub fn load(path: &str) -> std::io::Result<Self> {
        println!("📂 Loading model via Burn Recorder from: {}", path);
        let config = AgentConfig::new(1);
        let mut agent = DqnAgent::new(config);

        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();

        // Carica i pesi nel modello online
        agent.online_model = agent
            .online_model
            .load_file(path, &recorder, &agent.device)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        // Sincronizza il target model
        agent.target_model = agent.online_model.clone();

        Ok(agent)
    }
}
