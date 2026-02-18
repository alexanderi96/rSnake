use bevy::prelude::*;
use burn::{
    backend::{ndarray::NdArrayDevice, Autodiff, NdArray},
    module::Module, // ← questa riga mancava
    nn::loss::MseLoss,
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{Int, Tensor},
};
use std::collections::VecDeque;

use crate::model::DqnModel;
use crate::snake::{BATCH_SIZE, MEMORY_SIZE, STATE_SIZE, TARGET_UPDATE_FREQ};

type MyBackend = Autodiff<NdArray>;
type MyDevice = NdArrayDevice;

// OptimizerAdaptor e Adam sono inferiti dal tipo restituito da AdamConfig::new().init()
type MyOptimizer =
    burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, DqnModel<MyBackend>, MyBackend>;

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
        use rand::seq::index;
        let mut rng = rand::thread_rng();
        let len = self.buffer.len();
        index::sample(&mut rng, len, batch_size)
            .iter()
            .map(|i| self.buffer[i])
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
    pub optimizer: MyOptimizer,
    pub replay_buffer: ReplayBuffer,
    pub epsilon: f32,
    pub loss: f32,
    pub target_update_counter: usize,
    pub iterations: u32,
    pub device: MyDevice,
}

// NdArray usa RefCell internamente → non è Sync di default.
// Bevy accede tramite ResMut<DqnAgent> (accesso esclusivo), quindi è sicuro.
unsafe impl Sync for DqnAgent {}

impl DqnAgent {
    pub fn new(config: AgentConfig) -> Self {
        let device = NdArrayDevice::Cpu;
        println!("🚀 Initializing DqnAgent on NdArray CPU");

        let online_model = DqnModel::new(&device);
        let target_model = online_model.clone();
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
        self.select_actions_batch(vec![state])[0]
    }

    pub fn select_actions_batch(&self, states: Vec<[f32; STATE_SIZE]>) -> Vec<usize> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let batch_size = states.len();
        let mut actions = vec![0usize; batch_size];
        let mut indices_to_infer: Vec<usize> = Vec::new();
        let mut states_to_infer: Vec<[f32; STATE_SIZE]> = Vec::new();

        for (i, _) in states.iter().enumerate() {
            if rng.gen::<f32>() < self.epsilon {
                actions[i] = rng.gen_range(0..3);
            } else {
                indices_to_infer.push(i);
                states_to_infer.push(states[i]);
            }
        }

        if states_to_infer.is_empty() {
            return actions;
        }

        let q_values_batch = self
            .online_model
            .forward_batch(&states_to_infer, &self.device);

        for (batch_idx, q_values) in q_values_batch.iter().enumerate() {
            let original_idx = indices_to_infer[batch_idx];
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
            dones.push(if d { 1.0f32 } else { 0.0 });
        }

        let state_tensor = Tensor::<MyBackend, 2>::from_data(
            burn::tensor::TensorData::new(states, [BATCH_SIZE, STATE_SIZE]),
            &self.device,
        );
        let next_state_tensor = Tensor::<MyBackend, 2>::from_data(
            burn::tensor::TensorData::new(next_states, [BATCH_SIZE, STATE_SIZE]),
            &self.device,
        );
        let action_tensor = Tensor::<MyBackend, 2, Int>::from_data(
            burn::tensor::TensorData::new(actions, [BATCH_SIZE, 1]),
            &self.device,
        );
        let reward_tensor = Tensor::<MyBackend, 1>::from_data(
            burn::tensor::TensorData::new(rewards, [BATCH_SIZE]),
            &self.device,
        );
        let done_tensor = Tensor::<MyBackend, 1>::from_data(
            burn::tensor::TensorData::new(dones, [BATCH_SIZE]),
            &self.device,
        );

        // Double DQN — target model (separato per stabilità)
        let next_q_target = self
            .target_model
            .forward(next_state_tensor.clone())
            .detach();

        // Un solo forward pass online su [next_states; states] concatenati
        let combined_input = Tensor::cat(vec![next_state_tensor, state_tensor], 0);
        let all_q_online = self.online_model.forward(combined_input);

        let next_q_online = all_q_online.clone().slice([0..BATCH_SIZE, 0..3]).detach();
        let q_values = all_q_online.slice([BATCH_SIZE..BATCH_SIZE * 2, 0..3]);

        let next_actions = next_q_online.argmax(1).reshape([BATCH_SIZE, 1]);
        let max_next_q = next_q_target.gather(1, next_actions).squeeze(1);
        let target_q =
            reward_tensor + (max_next_q * done_tensor.neg().add_scalar(1.0) * self.config.gamma);

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

    pub fn save(&mut self, path: &str) -> std::io::Result<()> {
        println!("💾 Saving model to: {}", path);
        // NamedMpkFileRecorder è il recorder file-based disponibile senza feature wgpu
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        self.online_model
            .clone()
            .save_file(path, &recorder)
            .map_err(|e: burn::record::RecorderError| {
                std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
            })?;
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        println!("📂 Loading model from: {}", path);
        let config = AgentConfig::new(1);
        let mut agent = DqnAgent::new(config);
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        agent.online_model = agent
            .online_model
            .load_file(path, &recorder, &agent.device)
            .map_err(|e: burn::record::RecorderError| {
                std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
            })?;
        agent.target_model = agent.online_model.clone();
        Ok(agent)
    }
}
