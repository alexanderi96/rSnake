use bevy::prelude::*;
use burn::{
    backend::{
        wgpu::{AutoGraphicsApi, Wgpu, WgpuDevice},
        Autodiff,
    },
    module::Module,
    nn::loss::MseLoss,
    optim::{Adam, AdamConfig, GradientsParams, Optimizer},
    record::{BinFileRecorder, FullPrecisionSettings, Recorder}, // Necessario per salvare
    tensor::{backend::Backend, Tensor},
};
use std::collections::VecDeque;

use crate::model::DqnModel;
use crate::snake::{BATCH_SIZE, MEMORY_SIZE, STATE_SIZE, TARGET_UPDATE_FREQ};

// Definiamo il backend: WGPU con differenziazione automatica
type MyBackend = Autodiff<Wgpu<AutoGraphicsApi, f32, i32>>;
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
    pub optimizer: Adam<MyBackend>,
    pub replay_buffer: ReplayBuffer,
    pub epsilon: f32,
    pub loss: f32,
    pub target_update_counter: usize,
    pub iterations: u32,
    pub device: MyDevice,
}

impl DqnAgent {
    pub fn new(config: AgentConfig) -> Self {
        // Inizializza la GPU
        let device = WgpuDevice::BestAvailable;
        println!("🚀 Initializing DqnAgent on device: {:?}", device);

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

        let state_tensor = Tensor::<MyBackend, 2>::from_floats(states.as_slice(), &self.device)
            .reshape([BATCH_SIZE, STATE_SIZE]);
        let next_state_tensor =
            Tensor::<MyBackend, 2>::from_floats(next_states.as_slice(), &self.device)
                .reshape([BATCH_SIZE, STATE_SIZE]);
        let action_tensor = Tensor::<MyBackend, 1>::from_ints(actions.as_slice(), &self.device)
            .reshape([BATCH_SIZE, 1]);
        let reward_tensor = Tensor::<MyBackend, 1>::from_floats(rewards.as_slice(), &self.device);
        let done_tensor = Tensor::<MyBackend, 1>::from_floats(dones.as_slice(), &self.device);

        // Double DQN Logic
        let next_q_target = self
            .target_model
            .forward(next_state_tensor.clone())
            .detach();
        let next_q_online = self.online_model.forward(next_state_tensor).detach();
        let next_actions = next_q_online.argmax(1).reshape([BATCH_SIZE, 1]);

        let max_next_q = next_q_target.gather(1, next_actions).squeeze(1);
        let target_q = reward_tensor + (max_next_q * (1.0 - done_tensor) * self.config.gamma);

        let q_values = self.online_model.forward(state_tensor);
        let current_q = q_values.gather(1, action_tensor).squeeze(1);

        let loss = MseLoss::new().forward(
            current_q,
            target_q.detach(),
            burn::nn::loss::Reduction::Mean,
        );

        self.loss = loss.to_data().value[0];

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
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        println!("💾 Saving model via Burn Recorder to: {}", path);
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();

        // Burn salva in formato binario compresso (gzip) di default con BinFileRecorder
        self.online_model
            .save_file(path, &recorder)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        Ok(())
    }

    /// CARICAMENTO MODELLO
    pub fn load(path: &str) -> std::io::Result<Self> {
        // Nota: Load è complesso perché DqnAgent contiene lo stato dell'optimizer e altro.
        // Per semplicità qui carichiamo solo i pesi del modello in un agente nuovo.
        // Idealmente dovresti salvare l'intera struct o ricaricare i pesi su un agente esistente.

        println!("📂 Loading model via Burn Recorder from: {}", path);
        let config = AgentConfig::new(1); // Config default o passata
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
