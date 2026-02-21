use std::collections::VecDeque;

/// Transizione memorizzata nel buffer
#[derive(Clone, Debug)]
pub struct Transition {
    pub state: [f32; 34],
    pub action: usize,
    pub reward: f32,
    pub next_state: [f32; 34],
    pub done: bool,
}

impl Transition {
    pub fn new(
        state: [f32; 34],
        action: usize,
        reward: f32,
        next_state: [f32; 34],
        done: bool,
    ) -> Self {
        Self {
            state,
            action,
            reward,
            next_state,
            done,
        }
    }
}

/// Alias per retrocompatibilità
pub type Experience = Transition;

/// Struttura SumTree per O(log n) campionamento e aggiornamento priorità
#[derive(Debug, Clone)]
pub struct SumTree<T> {
    /// L'albero binario completo: dimensione 2 * capacity - 1
    tree: Vec<f32>,
    /// Dati memorizzati (transizioni): dimensione capacity
    data: Vec<T>,
    /// Puntatore ciclico al prossimo elemento da inserire
    write_index: usize,
    /// Numero di elementi attualmente memorizzati
    size: usize,
    /// Capacità massima
    capacity: usize,
}

impl<T: Clone> SumTree<T> {
    /// Crea un nuovo SumTree con la capacità specificata
    pub fn new(capacity: usize) -> Self {
        let tree_size = 2 * capacity - 1;
        Self {
            tree: vec![0.0; tree_size],
            data: Vec::with_capacity(capacity),
            write_index: 0,
            size: 0,
            capacity,
        }
    }

    /// Restituisce la capacità massima
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Restituisce il numero di elementi attualmente memorizzati
    pub fn len(&self) -> usize {
        self.size
    }

    /// Verifica se è vuoto
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Restituisce la priorità totale (valore alla radice)
    pub fn total_priority(&self) -> f32 {
        self.tree[0]
    }

    /// Aggiunge un elemento con la priorità data
    /// Restituisce l'indice della foglia dove è stato inserito
    pub fn add(&mut self, priority: f32, value: T) -> usize {
        // Calcola l'indice della foglia
        let leaf_index = self.write_index + self.capacity - 1;

        // Aggiorna il dato
        if self.data.len() < self.capacity {
            self.data.push(value);
        } else {
            self.data[self.write_index] = value;
        }

        // Aggiorna l'albero
        self.update(leaf_index, priority);

        // Avanza il puntatore ciclico
        self.write_index = (self.write_index + 1) % self.capacity;
        if self.size < self.capacity {
            self.size += 1;
        }

        leaf_index
    }

    /// Aggiorna la priorità di una foglia e propaga verso la radice
    pub fn update(&mut self, tree_index: usize, priority: f32) {
        let delta = priority - self.tree[tree_index];
        self.tree[tree_index] = priority;

        // Propaga verso l'alto
        let mut idx = tree_index;
        while idx > 0 {
            idx = (idx - 1) / 2;
            self.tree[idx] += delta;
        }
    }

    /// Trova la foglia corrispondente al valore v (campionamento O(log n))
    /// Restituisce (tree_index, priority, data)
    pub fn get_leaf(&self, mut v: f32) -> (usize, f32, T) {
        let mut parent = 0;

        loop {
            let left = 2 * parent + 1;
            let right = left + 1;

            if left >= self.tree.len() {
                // Siamo alla foglia
                break;
            }

            if v < self.tree[left] {
                parent = left;
            } else {
                v -= self.tree[left];
                parent = right;
            }
        }

        let data_index = parent - (self.capacity - 1);
        let data = if data_index < self.data.len() {
            self.data[data_index].clone()
        } else {
            // Fallback: usa l'ultimo elemento
            self.data.last().cloned().unwrap()
        };

        (parent, self.tree[parent], data)
    }

    /// Aggiorna le priorità per un batch di indici
    pub fn update_batch(&mut self, indices: &[usize], priorities: &[f32]) {
        for (idx, priority) in indices.iter().zip(priorities.iter()) {
            self.update(*idx, *priority);
        }
    }
}

/// Prioritized Experience Replay Buffer con SumTree
#[derive(Clone, Debug)]
pub struct PrioritizedReplayBuffer {
    /// SumTree per memorizzare priorità e transizioni
    sum_tree: SumTree<Transition>,
    /// Parametro alpha: quanto usare la priorità (0 = uniforme, 1 = solo priorità)
    pub alpha: f32,
    /// Parametro beta: compensazione del bias per Importance Sampling
    pub beta: f32,
    /// Incremento di beta per ogni campionamento
    pub beta_increment: f32,
    /// Piccolo valore per evitare priorità zero
    pub epsilon: f32,
    /// Priorità massima per nuove transizioni
    pub max_priority: f32,
}

impl PrioritizedReplayBuffer {
    /// Crea un nuovo buffer PER
    pub fn new(capacity: usize) -> Self {
        Self {
            sum_tree: SumTree::new(capacity),
            alpha: 0.6,
            beta: 0.4,
            beta_increment: 0.001,
            epsilon: 1e-5,
            max_priority: 1.0,
        }
    }

    /// Crea un buffer PER con parametri personalizzati
    pub fn with_params(
        capacity: usize,
        alpha: f32,
        beta: f32,
        beta_increment: f32,
        epsilon: f32,
    ) -> Self {
        Self {
            sum_tree: SumTree::new(capacity),
            alpha,
            beta,
            beta_increment,
            epsilon,
            max_priority: 1.0,
        }
    }

    /// Restituisce il numero di elementi nel buffer
    pub fn len(&self) -> usize {
        self.sum_tree.len()
    }

    /// Verifica se il buffer è vuoto
    pub fn is_empty(&self) -> bool {
        self.sum_tree.is_empty()
    }

    /// Inserisce una nuova transizione con priorità massima
    pub fn push(&mut self, experience: Transition) {
        self.sum_tree.add(self.max_priority, experience);
    }

    /// Campiona un batch dal buffer usando distribuzione prioritaria
    /// Restituisce (batch, tree_indices, is_weights)
    pub fn sample(&mut self, batch_size: usize) -> (Vec<Transition>, Vec<usize>, Vec<f32>) {
        let total_priority = self.sum_tree.total_priority();

        // CORREZIONE 1: Aggiungi un controllo per is_nan()
        if total_priority == 0.0 || total_priority.is_nan() {
            return (Vec::new(), Vec::new(), Vec::new());
        }

        let segment_size = total_priority / batch_size as f32;
        let mut batch = Vec::with_capacity(batch_size);
        let mut indices = Vec::with_capacity(batch_size);
        let mut weights = Vec::with_capacity(batch_size);

        use rand::Rng;
        let mut rng = rand::thread_rng();

        for i in 0..batch_size {
            let start = i as f32 * segment_size;
            let end = (i + 1) as f32 * segment_size;

            // CORREZIONE 2: Metti in sicurezza il range
            let v = if start < end && !start.is_nan() && !end.is_nan() {
                rng.gen_range(start..end)
            } else {
                start // Fallback sicuro
            };

            let (tree_index, priority, transition) = self.sum_tree.get_leaf(v);

            // CORREZIONE 3: Evita che probability sia esattamente 0.0
            let size = self.sum_tree.len() as f32;
            let probability = (priority / total_priority).max(1e-10);

            // Ora questo peso non esploderà in Infinity
            let weight = (size * probability).powf(-self.beta);

            batch.push(transition);
            indices.push(tree_index);
            weights.push(weight);
        }

        let max_weight = weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        if max_weight > 0.0 && !max_weight.is_infinite() {
            for w in &mut weights {
                *w /= max_weight;
            }
        } else {
            // Fallback se max_weight è anomalo
            weights.fill(1.0);
        }

        self.beta = (self.beta + self.beta_increment).min(1.0);

        (batch, indices, weights)
    }

    /// Aggiorna le priorità per le transizioni campionate
    pub fn update_priorities(&mut self, tree_indices: &[usize], td_errors: &[f32]) {
        for (&idx, &td_error) in tree_indices.iter().zip(td_errors.iter()) {
            // Nuova priorità: (|TD_error| + epsilon)^alpha
            let priority = (td_error.abs() + self.epsilon).powf(self.alpha);

            // Aggiorna max_priority se necessario
            if priority > self.max_priority {
                self.max_priority = priority;
            }

            self.sum_tree.update(idx, priority);
        }
    }

    /// Analizza la distribuzione dei reward nel buffer
    pub fn analyze_reward_distribution(&self) -> (f32, f32, f32) {
        let len = self.sum_tree.len();
        if len == 0 {
            return (0.0, 0.0, 0.0);
        }

        let mut positive = 0;
        let mut negative = 0;
        let mut neutral = 0;

        for t in &self.sum_tree.data {
            if t.reward > 0.0 {
                positive += 1;
            } else if t.reward < 0.0 {
                negative += 1;
            } else {
                neutral += 1;
            }
        }

        let len_f = len as f32;
        (
            positive as f32 / len_f,
            negative as f32 / len_f,
            neutral as f32 / len_f,
        )
    }

    /// Restituisce statistiche utili del buffer PER
    pub fn get_stats(&self) -> BufferStats {
        let len = self.sum_tree.len();
        let capacity = self.sum_tree.capacity();
        BufferStats {
            len,
            capacity,
            fullness: len as f32 / capacity as f32,
            alpha: self.alpha,
            beta: self.beta,
            max_priority: self.max_priority,
            epsilon: self.epsilon,
        }
    }
}

/// Statistiche del buffer PER
#[derive(Clone, Debug)]
pub struct BufferStats {
    pub len: usize,
    pub capacity: usize,
    pub fullness: f32,
    pub alpha: f32,
    pub beta: f32,
    pub max_priority: f32,
    pub epsilon: f32,
}

/// Replay Buffer standard (uniforme) - mantenuto per compatibilità
#[derive(Clone, Debug)]
pub struct ReplayBuffer {
    pub buffer: VecDeque<Transition>,
    pub capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, experience: Transition) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }

    pub fn sample(&self, batch_size: usize) -> Vec<Transition> {
        use rand::seq::index;
        let mut rng = rand::thread_rng();
        let len = self.buffer.len();
        if len == 0 {
            return Vec::new();
        }
        index::sample(&mut rng, len, batch_size)
            .iter()
            .map(|i| self.buffer[i].clone())
            .collect()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn analyze_reward_distribution(&self) -> (f32, f32, f32) {
        let len = self.buffer.len();
        if len == 0 {
            return (0.0, 0.0, 0.0);
        }

        let mut positive = 0;
        let mut negative = 0;
        let mut neutral = 0;

        for t in &self.buffer {
            if t.reward > 0.0 {
                positive += 1;
            } else if t.reward < 0.0 {
                negative += 1;
            } else {
                neutral += 1;
            }
        }

        let len_f = len as f32;
        (
            positive as f32 / len_f,
            negative as f32 / len_f,
            neutral as f32 / len_f,
        )
    }
}
