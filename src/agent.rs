use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    grad_clipping::GradientClippingConfig,
    module::Module,
    optim::decay::WeightDecayConfig,
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{Int, Tensor},
};

use crate::buffer::{PrioritizedReplayBuffer, Transition};
use crate::model::DqnModel;
use crate::snake::{BATCH_SIZE, MEMORY_SIZE, STATE_SIZE, TARGET_UPDATE_FREQ};

type MyBackend = Autodiff<Wgpu>;
type MyDevice = WgpuDevice;

// OptimizerAdaptor e Adam sono inferiti dal tipo restituito da AdamConfig::new().init()
type MyOptimizer =
    burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, DqnModel<MyBackend>, MyBackend>;

pub type Experience = Transition;

#[derive(Clone, Debug)]
pub struct AgentConfig {
    pub learning_rate: f64,
    pub gamma: f32,
}

impl AgentConfig {
    pub fn new(_snake_count: usize) -> Self {
        Self {
            learning_rate: 1e-4,
            gamma: 0.99,
        }
    }
}

/// DQN Agent con Prioritized Experience Replay
pub struct DqnAgent {
    pub config: AgentConfig,
    pub online_model: DqnModel<MyBackend>,
    pub target_model: DqnModel<MyBackend>,
    pub optimizer: MyOptimizer,
    pub replay_buffer: PrioritizedReplayBuffer,
    pub loss: f32,
    pub target_update_counter: usize,
    pub iterations: u32,
    pub device: MyDevice,
    // Metriche diagnostiche per tracciamento
    pub generation_q_values: Vec<f32>,
    pub generation_losses: Vec<f32>,
}

impl DqnAgent {
    pub fn new(config: AgentConfig) -> Self {
        let device = WgpuDevice::default();
        println!("🚀 Initializing DqnAgent on Wgpu GPU with PER");

        let online_model = DqnModel::new(&device);
        let target_model = online_model.clone();

        // Configura Adam con Clipping e Decay
        let optimizer = AdamConfig::new()
            .with_weight_decay(Some(WeightDecayConfig::new(1e-5)))
            .with_grad_clipping(Some(GradientClippingConfig::Value(1.0)))
            .init();

        Self {
            config: config.clone(),
            online_model,
            target_model,
            optimizer,
            replay_buffer: PrioritizedReplayBuffer::new(MEMORY_SIZE),
            loss: 0.0,
            target_update_counter: 0,
            iterations: 0,
            device,
            generation_q_values: Vec::new(),
            generation_losses: Vec::new(),
        }
    }

    pub fn select_action(&mut self, state: [f32; STATE_SIZE], epsilon: f32) -> usize {
        self.select_actions_batch(vec![(state, epsilon)])[0]
    }

    pub fn select_actions_batch(
        &mut self,
        states_with_eps: Vec<([f32; STATE_SIZE], f32)>,
    ) -> Vec<usize> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let batch_size = states_with_eps.len();
        let mut actions = vec![0usize; batch_size];
        let mut indices_to_infer: Vec<usize> = Vec::new();
        let mut states_to_infer: Vec<[f32; STATE_SIZE]> = Vec::new();

        for (i, (state, eps)) in states_with_eps.iter().enumerate() {
            if rng.gen::<f32>() < *eps {
                actions[i] = rng.gen_range(0..3);
            } else {
                indices_to_infer.push(i);
                states_to_infer.push(*state);
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
            // Trova l'azione con Q-value massimo in modo sicuro
            let mut best_action = 0;
            let mut best_value = f32::NEG_INFINITY;
            for (i, &v) in q_values.iter().enumerate() {
                if v.is_finite() && v > best_value {
                    best_value = v;
                    best_action = i;
                }
            }
            actions[original_idx] = best_action;

            // Traccia il Q-value medio per questa inferenza
            let avg_q: f32 = q_values.iter().filter(|&&v| v.is_finite()).sum::<f32>()
                / q_values.iter().filter(|&&v| v.is_finite()).count().max(1) as f32;
            self.generation_q_values.push(avg_q);
        }

        actions
    }

    pub fn remember(&mut self, experience: Experience) {
        self.replay_buffer.push(experience);
    }

    /// Calcola le statistiche Q-value per la generazione corrente
    pub fn get_generation_q_stats(&self) -> f32 {
        if self.generation_q_values.is_empty() {
            return 0.0;
        }
        self.generation_q_values.iter().sum::<f32>() / self.generation_q_values.len() as f32
    }

    /// Calcola le statistiche loss per la generazione corrente
    pub fn get_generation_loss_stats(&self) -> (f32, f32, f32) {
        if self.generation_losses.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let min = self
            .generation_losses
            .iter()
            .fold(f32::INFINITY, |a, &b| a.min(b));
        let max = self
            .generation_losses
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let avg = self.generation_losses.iter().sum::<f32>() / self.generation_losses.len() as f32;
        (min, max, avg)
    }

    /// Reset delle metriche di generazione
    pub fn reset_generation_metrics(&mut self) {
        self.generation_q_values.clear();
        self.generation_losses.clear();
    }

    pub fn train(&mut self) {
        if self.replay_buffer.len() < BATCH_SIZE {
            return;
        }

        // Campiona dal buffer PER (restituisce anche indici e pesi IS)
        let (batch, tree_indices, is_weights) = self.replay_buffer.sample(BATCH_SIZE);

        if batch.is_empty() {
            return;
        }

        let mut states = Vec::with_capacity(BATCH_SIZE * STATE_SIZE);
        let mut next_states = Vec::with_capacity(BATCH_SIZE * STATE_SIZE);
        let mut actions = Vec::with_capacity(BATCH_SIZE);
        let mut rewards = Vec::with_capacity(BATCH_SIZE);
        let mut dones = Vec::with_capacity(BATCH_SIZE);

        for t in &batch {
            states.extend_from_slice(&t.state);
            next_states.extend_from_slice(&t.next_state);
            actions.push(t.action as i32);
            rewards.push(t.reward);
            dones.push(if t.done { 1.0f32 } else { 0.0 });
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
            burn::tensor::TensorData::new(rewards.clone(), [BATCH_SIZE]),
            &self.device,
        );
        let done_tensor = Tensor::<MyBackend, 1>::from_data(
            burn::tensor::TensorData::new(dones, [BATCH_SIZE]),
            &self.device,
        );

        // Pesi IS come tensore per la loss ponderata
        let is_weights_tensor = Tensor::<MyBackend, 1>::from_data(
            burn::tensor::TensorData::new(is_weights.clone(), [BATCH_SIZE]),
            &self.device,
        );

        // Double DQN — target model (separato per stabilità)
        let next_q_target = self
            .target_model
            .forward(next_state_tensor.clone())
            .detach();

        // Forward pass combinato per efficienza
        let combined_input = Tensor::cat(vec![next_state_tensor.clone(), state_tensor.clone()], 0);
        let all_q_online = self.online_model.forward(combined_input);

        let next_q_online = all_q_online.clone().slice([0..BATCH_SIZE, 0..3]).detach();
        let q_values = all_q_online.slice([BATCH_SIZE..BATCH_SIZE * 2, 0..3]);

        let next_actions = next_q_online.argmax(1).reshape([BATCH_SIZE, 1]);
        let max_next_q = next_q_target.gather(1, next_actions).squeeze(1);
        let target_q =
            reward_tensor + (max_next_q * done_tensor.neg().add_scalar(1.0) * self.config.gamma);

        let current_q = q_values.gather(1, action_tensor).squeeze(1);

        // Calcola TD-Error assoluto per aggiornamento priorità
        let td_errors = target_q.clone().detach().sub(current_q.clone()).abs();

        // Calcola MSE loss per ogni elemento del batch
        // MSE = (current_q - target_q)^2
        let diff = current_q.sub(target_q.detach());
        let loss_per_element = diff.clone().mul(diff);

        // Moltiplica per i pesi IS per weighted loss
        let weighted_loss = loss_per_element.mul(is_weights_tensor);
        let mean_weighted_loss = weighted_loss.mean();

        // Estrai loss in modo sicuro
        let loss_data = mean_weighted_loss.to_data();
        let loss_val = match loss_data.as_slice::<f32>() {
            Ok(slice) => slice.first().copied().unwrap_or(0.0),
            Err(_) => 0.0,
        };
        self.loss = loss_val;
        self.generation_losses.push(self.loss);

        // Backpropagation
        let grads = mean_weighted_loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.online_model);
        self.online_model =
            self.optimizer
                .step(self.config.learning_rate, self.online_model.clone(), grads);

        self.target_update_counter += 1;
        if self.target_update_counter >= TARGET_UPDATE_FREQ {
            self.target_model = self.online_model.clone();
            self.target_update_counter = 0;
        }

        // Aggiorna le priorità nel buffer PER
        let td_errors_data = td_errors.to_data();
        let td_errors_vec: Vec<f32> = match td_errors_data.as_slice::<f32>() {
            Ok(slice) => slice.to_vec(),
            Err(_) => Vec::new(),
        };

        self.replay_buffer
            .update_priorities(&tree_indices, &td_errors_vec);

        self.iterations += 1;
    }

    pub fn save(&mut self, path: &str) -> std::io::Result<()> {
        println!("💾 Saving model to: {}", path);
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
