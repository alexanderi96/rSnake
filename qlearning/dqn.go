package qlearning

import (
	"encoding/gob"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"sync"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func init() {
	gob.Register(&tensor.Dense{})
	gob.Register(map[string]*tensor.Dense{})
}

const (
	// Network parameters
	BatchSize        = 64
	ReplayBufferSize = 10000
	HiddenLayer1Size = 128 // Increased for better representation
	HiddenLayer2Size = 64  // Increased for better representation
	InputFeatures    = 24  // Matrix 3x8: [walls][body][food] x [backLeft, left, frontLeft, front, frontRight, right, backRight, back]
	OutputActions    = 3
	GradientClip     = 1.0 // Gradient clipping threshold
	DropoutRate      = 0.2 // Dropout for regularization

	// Training parameters
	TargetUpdateFreq = 1000 // Steps between target network updates
	NumTargets       = 3    // Number of target networks
	RewardScale      = 0.1  // Scale factor for rewards
	RewardClip       = 1.0  // Maximum absolute reward value

	// File paths
	DataDir     = "data"
	WeightsFile = DataDir + "/dqn_weights.gob"
)

// Transition rappresenta un singolo step nell'ambiente
type Transition struct {
	State     []float64
	Action    int
	Reward    float64
	NextState []float64
	Done      bool
}

// ReplayBuffer memorizza le esperienze per il training
type ReplayBuffer struct {
	buffer   []Transition
	maxSize  int
	position int
	size     int
}

// DQN rappresenta la rete neurale
type DQN struct {
	g          *gorgonia.ExprGraph
	w1, w2, w3 *gorgonia.Node
	b1, b2, b3 *gorgonia.Node
	pred       *gorgonia.Node
	vm         gorgonia.VM
	solver     gorgonia.Solver
	mu         sync.Mutex // Protegge le operazioni sul graph
}

// Agent rappresenta l'agente DQN
type Agent struct {
	dqn           *DQN   // Online network
	targetDQNs    []*DQN // Multiple target networks
	currentTarget int    // Indice del target network corrente
	updateFreq    int    // Frequenza di aggiornamento dei target networks
	stepCount     int    // Contatore degli step per aggiornamento periodico
	replayBuffer  *ReplayBuffer
	LearningRate  float64
	Discount      float64
	episodeCount  int

	// Reward processing
	rewardStats struct {
		history   []float64
		mean      float64
		std       float64
		clipValue float64
	}
}

const (
	NumTargetNetworks = 3    // Numero di target networks
	UpdateFrequency   = 1000 // Frequenza di aggiornamento dei target networks
	RewardClipValue   = 1.0  // Valore massimo per il clipping delle rewards
)

// NewReplayBuffer crea un nuovo buffer di replay
func NewReplayBuffer(maxSize int) *ReplayBuffer {
	return &ReplayBuffer{
		buffer:   make([]Transition, maxSize),
		maxSize:  maxSize,
		position: 0,
		size:     0,
	}
}

// Add aggiunge una transizione al buffer
func (b *ReplayBuffer) Add(t Transition) {
	b.buffer[b.position] = t
	b.position = (b.position + 1) % b.maxSize
	if b.size < b.maxSize {
		b.size++
	}
}

// Sample restituisce un batch casuale di transizioni
func (b *ReplayBuffer) Sample(batchSize int) []Transition {
	if batchSize > b.size {
		batchSize = b.size
	}

	batch := make([]Transition, batchSize)
	for i := 0; i < batchSize; i++ {
		idx := rand.Intn(b.size)
		batch[i] = b.buffer[idx]
	}
	return batch
}

// NewDQN crea una nuova rete DQN
func NewDQN() *DQN {
	g := gorgonia.NewGraph()

	// Input layer -> First Hidden layer (128)
	w1 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithShape(InputFeatures, HiddenLayer1Size),
		gorgonia.WithInit(gorgonia.GlorotU(1.0)))

	b1 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithShape(1, HiddenLayer1Size),
		gorgonia.WithInit(gorgonia.Zeroes()))

	// First Hidden layer -> Second Hidden layer (32)
	w2 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithShape(HiddenLayer1Size, HiddenLayer2Size),
		gorgonia.WithInit(gorgonia.GlorotU(1.0)))

	b2 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithShape(1, HiddenLayer2Size),
		gorgonia.WithInit(gorgonia.Zeroes()))

	// Second Hidden layer -> Output layer
	w3 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithShape(HiddenLayer2Size, OutputActions),
		gorgonia.WithInit(gorgonia.GlorotU(1.0)))

	b3 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithShape(1, OutputActions),
		gorgonia.WithInit(gorgonia.Zeroes()))

	dqn := &DQN{
		g:      g,
		w1:     w1,
		w2:     w2,
		w3:     w3,
		b1:     b1,
		b2:     b2,
		b3:     b3,
		solver: gorgonia.NewAdamSolver(gorgonia.WithL2Reg(1e-6), gorgonia.WithLearnRate(0.001)),
	}

	dqn.vm = gorgonia.NewTapeMachine(g)
	return dqn
}

// Forward esegue un forward pass attraverso la rete
func (dqn *DQN) Forward(states []float64) ([]float64, error) {
	if len(states) == 0 {
		return nil, fmt.Errorf("empty states input")
	}

	dqn.mu.Lock()
	defer dqn.mu.Unlock()

	// Reset VM and create new graph for clean state
	dqn.vm.Reset()
	g := gorgonia.NewGraph()
	dqn.g = g

	batchSize := len(states) / InputFeatures
	if batchSize == 0 {
		batchSize = 1
		if len(states) < InputFeatures {
			paddedStates := make([]float64, InputFeatures)
			copy(paddedStates, states)
			states = paddedStates
		}
	}

	if len(states) != batchSize*InputFeatures {
		return nil, fmt.Errorf("invalid input shape: got %d values, expected %d", len(states), batchSize*InputFeatures)
	}

	// Create and validate input tensor
	statesTensor := tensor.New(tensor.WithBacking(states), tensor.WithShape(batchSize, InputFeatures))
	if statesTensor == nil {
		return nil, fmt.Errorf("failed to create input tensor")
	}

	statesNode := gorgonia.NodeFromAny(g, statesTensor)
	if statesNode == nil {
		return nil, fmt.Errorf("failed to create input node")
	}

	// Recreate weight nodes
	w1Node := gorgonia.NodeFromAny(g, dqn.w1.Value())
	w2Node := gorgonia.NodeFromAny(g, dqn.w2.Value())
	w3Node := gorgonia.NodeFromAny(g, dqn.w3.Value())
	b1Node := gorgonia.NodeFromAny(g, dqn.b1.Value())
	b2Node := gorgonia.NodeFromAny(g, dqn.b2.Value())
	b3Node := gorgonia.NodeFromAny(g, dqn.b3.Value())

	if w1Node == nil || w2Node == nil || w3Node == nil ||
		b1Node == nil || b2Node == nil || b3Node == nil {
		return nil, fmt.Errorf("failed to create weight/bias nodes")
	}

	// Helper function per il broadcasting del bias
	expandBias := func(bias *gorgonia.Node, size int) (*gorgonia.Node, error) {
		ones := tensor.New(tensor.WithShape(size, 1), tensor.WithBacking(make([]float64, size)))
		for i := range ones.Data().([]float64) {
			ones.Data().([]float64)[i] = 1.0
		}
		onesNode := gorgonia.NodeFromAny(g, ones)
		if onesNode == nil {
			return nil, fmt.Errorf("failed to create ones node")
		}
		return gorgonia.Mul(onesNode, bias)
	}

	var err error
	// First Hidden layer con ReLU
	h1 := gorgonia.Must(gorgonia.Mul(statesNode, w1Node))
	expandedBias1, err := expandBias(b1Node, batchSize)
	if err != nil {
		return nil, fmt.Errorf("failed to expand bias1: %v", err)
	}
	h1 = gorgonia.Must(gorgonia.Add(h1, expandedBias1))
	h1 = gorgonia.Must(gorgonia.Rectify(h1))

	// Second Hidden layer con ReLU
	h2 := gorgonia.Must(gorgonia.Mul(h1, w2Node))
	expandedBias2, err := expandBias(b2Node, batchSize)
	if err != nil {
		return nil, fmt.Errorf("failed to expand bias2: %v", err)
	}
	h2 = gorgonia.Must(gorgonia.Add(h2, expandedBias2))
	h2 = gorgonia.Must(gorgonia.Rectify(h2))

	// Output layer
	output := gorgonia.Must(gorgonia.Mul(h2, w3Node))
	expandedBias3, err := expandBias(b3Node, batchSize)
	if err != nil {
		return nil, fmt.Errorf("failed to expand bias3: %v", err)
	}
	pred := gorgonia.Must(gorgonia.Add(output, expandedBias3))

	// Run forward pass
	if err := dqn.vm.RunAll(); err != nil {
		return nil, fmt.Errorf("forward pass error: %v", err)
	}

	// Get predictions
	predValue := pred.Value()
	if predValue == nil {
		// Return default Q-values instead of error
		defaultQValues := make([]float64, batchSize*OutputActions)
		for i := 0; i < len(defaultQValues); i++ {
			defaultQValues[i] = 0.0
		}
		return defaultQValues, nil
	}

	predTensor, ok := predValue.(*tensor.Dense)
	if !ok {
		return nil, fmt.Errorf("invalid prediction tensor type")
	}

	predictions := make([]float64, batchSize*OutputActions)
	copy(predictions, predTensor.Data().([]float64))

	return predictions, nil
}

// NewAgent crea un nuovo agente DQN
func NewAgent(learningRate, discount float64) *Agent {
	agent := &Agent{
		dqn:           NewDQN(),
		targetDQNs:    make([]*DQN, NumTargetNetworks),
		currentTarget: 0,
		updateFreq:    UpdateFrequency,
		stepCount:     0,
		replayBuffer:  NewReplayBuffer(ReplayBufferSize),
		LearningRate:  learningRate,
		Discount:      discount,
		episodeCount:  0,
	}

	// Inizializza i target networks
	for i := 0; i < NumTargetNetworks; i++ {
		agent.targetDQNs[i] = NewDQN()
	}

	// Inizializza le statistiche delle reward
	agent.rewardStats.history = make([]float64, 0, 1000)
	agent.rewardStats.clipValue = RewardClipValue

	agent.LoadWeights(WeightsFile)
	return agent
}

// processReward applica clipping, scaling e normalizzazione alla reward
func (a *Agent) processReward(rawReward float64) float64 {
	// 1. Apply reward scaling
	scaledReward := rawReward * RewardScale

	// 2. Clipping
	clippedReward := math.Max(-a.rewardStats.clipValue,
		math.Min(a.rewardStats.clipValue, scaledReward))

	// 3. Aggiorna statistiche con finestra mobile
	a.rewardStats.history = append(a.rewardStats.history, clippedReward)
	windowSize := 1000
	if len(a.rewardStats.history) > windowSize {
		a.rewardStats.history = a.rewardStats.history[1:]
	}

	// 4. Calcola statistiche con Exponential Moving Average
	if len(a.rewardStats.history) == 1 {
		a.rewardStats.mean = clippedReward
		a.rewardStats.std = 1.0 // Inizializza con valore non zero
	} else {
		alpha := 0.01 // Fattore di smoothing per EMA
		a.rewardStats.mean = (1-alpha)*a.rewardStats.mean + alpha*clippedReward

		// Calcola deviazione standard con EMA
		diff := clippedReward - a.rewardStats.mean
		variance := diff * diff
		a.rewardStats.std = math.Sqrt((1-alpha)*a.rewardStats.std*a.rewardStats.std + alpha*variance)
	}

	// 5. Normalizzazione con epsilon per stabilità numerica
	epsilon := 1e-8
	normalizedReward := (clippedReward - a.rewardStats.mean) / (a.rewardStats.std + epsilon)

	// 6. Clip normalized reward per evitare valori estremi
	return math.Max(-5.0, math.Min(5.0, normalizedReward))
}

// GetQValues returns Q-values for all actions in the given state
func (a *Agent) GetQValues(state []float64) ([]float64, error) {
	if a.dqn == nil {
		return nil, fmt.Errorf("DQN not initialized")
	}

	if len(state) == 0 {
		return nil, fmt.Errorf("empty state input")
	}

	if len(state) != InputFeatures {
		return nil, fmt.Errorf("invalid state length: got %d, expected %d", len(state), InputFeatures)
	}

	qValues, err := a.dqn.Forward(state)
	if err != nil {
		return nil, fmt.Errorf("forward pass failed: %v", err)
	}

	if len(qValues) != OutputActions {
		return nil, fmt.Errorf("invalid Q-values length: got %d, expected %d", len(qValues), OutputActions)
	}

	return qValues, nil
}

const (
	InitialEpsilon = 1.0   // Valore iniziale di epsilon
	MinEpsilon     = 0.01  // Valore minimo di epsilon
	EpsilonDecay   = 0.995 // Fattore di decadimento di epsilon
)

// GetAction seleziona un'azione usando softmax con temperatura dinamica
func (a *Agent) GetAction(state []float64, numActions int) int {
	qValues, err := a.GetQValues(state)
	if err != nil {
		action := rand.Intn(numActions)
		log.Printf("Error getting Q-values (%v). Using random action: %d", err, action)
		return action
	}

	if len(qValues) != numActions {
		action := rand.Intn(numActions)
		log.Printf("Invalid number of Q-values (got %d, expected %d). Using random action: %d",
			len(qValues), numActions, action)
		return action
	}

	// Calcola la policy usando softmax
	policy := a.softmax(qValues, 0.5) // temperatura fissa a 0.5 per bilanciare esplorazione/sfruttamento

	// Campiona un'azione dalla distribuzione di probabilità
	r := rand.Float64()
	cumulativeProb := 0.0
	for i, prob := range policy {
		cumulativeProb += prob
		if r <= cumulativeProb {
			log.Printf("Choosing action %d with probability %.4f", i, prob)
			return i
		}
	}

	// Fallback all'azione con probabilità più alta
	bestAction := 0
	maxProb := policy[0]
	for i := 1; i < len(policy); i++ {
		if policy[i] > maxProb {
			maxProb = policy[i]
			bestAction = i
		}
	}

	return bestAction
}

// softmax converte Q-values in una distribuzione di probabilità
func (a *Agent) softmax(qValues []float64, temperature float64) []float64 {
	policy := make([]float64, len(qValues))
	maxQ := qValues[0]
	for _, q := range qValues {
		if q > maxQ {
			maxQ = q
		}
	}

	var sum float64
	for i, q := range qValues {
		policy[i] = math.Exp((q - maxQ) / temperature)
		sum += policy[i]
	}

	for i := range policy {
		policy[i] /= sum
	}
	return policy
}

// Update esegue un passo di aggiornamento DQN
func (a *Agent) Update(state []float64, action int, reward float64, nextState []float64, numActions int) {
	// Process reward
	processedReward := a.processReward(reward)

	// Increment step counter
	a.stepCount++

	// Add to replay buffer
	a.replayBuffer.Add(Transition{
		State:     state,
		Action:    action,
		Reward:    processedReward,
		NextState: nextState,
		Done:      false,
	})

	if a.replayBuffer.size < BatchSize {
		return
	}

	batch := a.replayBuffer.Sample(BatchSize)
	a.trainOnBatch(batch)

	// Periodic update of target networks
	if a.stepCount%a.updateFreq == 0 {
		// Rotate to next target network
		a.currentTarget = (a.currentTarget + 1) % len(a.targetDQNs)
		// Hard update instead of soft update
		copyWeights(a.targetDQNs[a.currentTarget], a.dqn)
	}
}

// trainOnBatch esegue un passo di training su un batch di transizioni
func (a *Agent) trainOnBatch(batch []Transition) {
	// Reset VM and create new graph for training
	a.dqn.vm.Reset()
	g := gorgonia.NewGraph()
	a.dqn.g = g

	states := make([]float64, 0, len(batch)*InputFeatures)
	nextStates := make([]float64, 0, len(batch)*InputFeatures)
	actions := make([]int, 0, len(batch))
	rewards := make([]float64, 0, len(batch))

	for _, transition := range batch {
		states = append(states, transition.State...)
		nextStates = append(nextStates, transition.NextState...)
		actions = append(actions, transition.Action)
		rewards = append(rewards, transition.Reward)
	}

	// Create input tensors
	statesTensor := tensor.New(tensor.WithBacking(states), tensor.WithShape(len(batch), InputFeatures))
	statesNode := gorgonia.NodeFromAny(g, statesTensor)

	// Recreate weight nodes in the new graph
	w1Node := gorgonia.NodeFromAny(g, a.dqn.w1.Value())
	w2Node := gorgonia.NodeFromAny(g, a.dqn.w2.Value())
	w3Node := gorgonia.NodeFromAny(g, a.dqn.w3.Value())
	b1Node := gorgonia.NodeFromAny(g, a.dqn.b1.Value())
	b2Node := gorgonia.NodeFromAny(g, a.dqn.b2.Value())
	b3Node := gorgonia.NodeFromAny(g, a.dqn.b3.Value())

	// Helper function per il broadcasting del bias
	expandBias := func(bias *gorgonia.Node, size int) (*gorgonia.Node, error) {
		ones := tensor.New(tensor.WithShape(size, 1), tensor.WithBacking(make([]float64, size)))
		for i := range ones.Data().([]float64) {
			ones.Data().([]float64)[i] = 1.0
		}
		onesNode := gorgonia.NodeFromAny(g, ones)
		if onesNode == nil {
			return nil, fmt.Errorf("failed to create ones node")
		}
		return gorgonia.Mul(onesNode, bias)
	}

	// Forward pass with dropout during training
	var err error
	// First Hidden layer con ReLU e Dropout
	h1 := gorgonia.Must(gorgonia.Mul(statesNode, w1Node))
	expandedBias1, err := expandBias(b1Node, len(batch))
	if err != nil {
		log.Printf("Error expanding bias1: %v", err)
		return
	}
	h1 = gorgonia.Must(gorgonia.Add(h1, expandedBias1))
	h1 = gorgonia.Must(gorgonia.Rectify(h1))
	h1 = gorgonia.Must(gorgonia.Dropout(h1, DropoutRate))

	// Second Hidden layer con ReLU e Dropout
	h2 := gorgonia.Must(gorgonia.Mul(h1, w2Node))
	expandedBias2, err := expandBias(b2Node, len(batch))
	if err != nil {
		log.Printf("Error expanding bias2: %v", err)
		return
	}
	h2 = gorgonia.Must(gorgonia.Add(h2, expandedBias2))
	h2 = gorgonia.Must(gorgonia.Rectify(h2))
	h2 = gorgonia.Must(gorgonia.Dropout(h2, DropoutRate))

	// Output layer
	output := gorgonia.Must(gorgonia.Mul(h2, w3Node))
	expandedBias3, err := expandBias(b3Node, len(batch))
	if err != nil {
		log.Printf("Error expanding bias3: %v", err)
		return
	}
	pred := gorgonia.Must(gorgonia.Add(output, expandedBias3))

	if err := a.dqn.vm.RunAll(); err != nil {
		log.Printf("Error during forward pass: %v", err)
		return
	}

	predValue := pred.Value()
	if predValue == nil {
		log.Printf("Nil prediction value during training")
		return
	}

	currentQValues := predValue.Data().([]float64)

	// Get next state Q-values from current target network
	nextQValues, err := a.targetDQNs[a.currentTarget].Forward(nextStates)
	if err != nil {
		log.Printf("Error getting next state Q-values: %v", err)
		return
	}

	// Double DQN: use online network to select actions, target network to evaluate them
	onlineNextQValues, err := a.dqn.Forward(nextStates)
	if err != nil {
		log.Printf("Error getting online next state Q-values: %v", err)
		return
	}

	targetQValues := make([]float64, len(batch)*OutputActions)
	copy(targetQValues, currentQValues)

	for i := 0; i < len(batch); i++ {
		// Double DQN: select action using online network
		bestAction := 0
		maxQ := math.Inf(-1)
		for j := 0; j < OutputActions; j++ {
			if onlineNextQValues[i*OutputActions+j] > maxQ {
				maxQ = onlineNextQValues[i*OutputActions+j]
				bestAction = j
			}
		}

		// Use target network to evaluate the selected action
		targetQ := nextQValues[i*OutputActions+bestAction]
		targetQValues[i*OutputActions+actions[i]] = rewards[i] + a.Discount*targetQ
	}

	targetTensor := tensor.New(tensor.WithBacking(targetQValues), tensor.WithShape(len(batch), OutputActions))
	targetNode := gorgonia.NodeFromAny(g, targetTensor)

	// MSE Loss
	diff := gorgonia.Must(gorgonia.Sub(pred, targetNode))
	loss := gorgonia.Must(gorgonia.Mean(gorgonia.Must(gorgonia.Square(diff))))

	// Backpropagation con gradient clipping
	nodes := gorgonia.Nodes{a.dqn.w1, a.dqn.w2, a.dqn.w3, a.dqn.b1, a.dqn.b2, a.dqn.b3}
	gorgonia.Grad(loss, nodes...)

	if err := a.dqn.vm.RunAll(); err != nil {
		log.Printf("Error during backprop: %v", err)
		return
	}

	grads := gorgonia.NodesToValueGrads(nodes)

	// Simple gradient clipping
	for _, grad := range grads {
		if grad == nil {
			continue
		}
		if t, ok := grad.(tensor.Tensor); ok {
			data := t.Data().([]float64)
			for i := range data {
				if math.Abs(data[i]) > GradientClip {
					data[i] *= GradientClip / math.Abs(data[i])
				}
			}
		}
	}

	a.dqn.solver.Step(grads)
	a.dqn.vm.Reset()

}

// copyWeights esegue un soft update dei pesi dal DQN principale al target network
func copyWeights(target, source *DQN) {
	tau := 0.01 // Fattore di soft update (piccolo per aggiornamenti graduali)
	copyTensor(target.w1.Value().(*tensor.Dense), source.w1.Value().(*tensor.Dense), tau)
	copyTensor(target.w2.Value().(*tensor.Dense), source.w2.Value().(*tensor.Dense), tau)
	copyTensor(target.w3.Value().(*tensor.Dense), source.w3.Value().(*tensor.Dense), tau)
	copyTensor(target.b1.Value().(*tensor.Dense), source.b1.Value().(*tensor.Dense), tau)
	copyTensor(target.b2.Value().(*tensor.Dense), source.b2.Value().(*tensor.Dense), tau)
	copyTensor(target.b3.Value().(*tensor.Dense), source.b3.Value().(*tensor.Dense), tau)
}

// copyTensor esegue un soft update dei pesi: θ_target = τ*θ_online + (1-τ)*θ_target
func copyTensor(target, source *tensor.Dense, tau float64) {
	targetData := target.Data().([]float64)
	sourceData := source.Data().([]float64)
	for i := range targetData {
		targetData[i] = tau*sourceData[i] + (1-tau)*targetData[i]
	}
}

// Cleanup releases resources used by the Agent
func (a *Agent) Cleanup() {
	if a.dqn != nil {
		a.dqn.Cleanup()
	}
	for _, target := range a.targetDQNs {
		if target != nil {
			target.Cleanup()
		}
	}
}

// IncrementEpisode incrementa il contatore degli episodi
func (a *Agent) IncrementEpisode() {
	a.episodeCount++
}

// Cleanup releases resources used by the DQN
func (dqn *DQN) Cleanup() {
	if dqn.vm != nil {
		dqn.vm.Close()
	}
}

// SaveWeights salva i pesi del DQN su file
func (a *Agent) SaveWeights(filename string) error {
	if err := os.MkdirAll(DataDir, 0755); err != nil {
		return fmt.Errorf("failed to create data directory: %v", err)
	}

	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create weights file: %v", err)
	}
	defer f.Close()

	enc := gob.NewEncoder(f)
	weights := map[string]*tensor.Dense{
		"w1": a.dqn.w1.Value().(*tensor.Dense),
		"w2": a.dqn.w2.Value().(*tensor.Dense),
		"w3": a.dqn.w3.Value().(*tensor.Dense),
		"b1": a.dqn.b1.Value().(*tensor.Dense),
		"b2": a.dqn.b2.Value().(*tensor.Dense),
		"b3": a.dqn.b3.Value().(*tensor.Dense),
	}

	if err := enc.Encode(weights); err != nil {
		return fmt.Errorf("failed to encode weights: %v", err)
	}

	return nil
}

// LoadWeights carica i pesi del DQN da file
func (a *Agent) LoadWeights(filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("failed to open weights file: %v", err)
	}
	defer f.Close()

	var weights map[string]*tensor.Dense
	dec := gob.NewDecoder(f)
	if err := dec.Decode(&weights); err != nil {
		return fmt.Errorf("failed to decode weights: %v", err)
	}

	if w1, ok := weights["w1"]; ok {
		tensor.Copy(a.dqn.w1.Value().(*tensor.Dense), w1)
		for _, target := range a.targetDQNs {
			tensor.Copy(target.w1.Value().(*tensor.Dense), w1)
		}
	}
	if w2, ok := weights["w2"]; ok {
		tensor.Copy(a.dqn.w2.Value().(*tensor.Dense), w2)
		for _, target := range a.targetDQNs {
			tensor.Copy(target.w2.Value().(*tensor.Dense), w2)
		}
	}
	if w3, ok := weights["w3"]; ok {
		tensor.Copy(a.dqn.w3.Value().(*tensor.Dense), w3)
		for _, target := range a.targetDQNs {
			tensor.Copy(target.w3.Value().(*tensor.Dense), w3)
		}
	}
	if b1, ok := weights["b1"]; ok {
		tensor.Copy(a.dqn.b1.Value().(*tensor.Dense), b1)
		for _, target := range a.targetDQNs {
			tensor.Copy(target.b1.Value().(*tensor.Dense), b1)
		}
	}
	if b2, ok := weights["b2"]; ok {
		tensor.Copy(a.dqn.b2.Value().(*tensor.Dense), b2)
		for _, target := range a.targetDQNs {
			tensor.Copy(target.b2.Value().(*tensor.Dense), b2)
		}
	}
	if b3, ok := weights["b3"]; ok {
		tensor.Copy(a.dqn.b3.Value().(*tensor.Dense), b3)
		for _, target := range a.targetDQNs {
			tensor.Copy(target.b3.Value().(*tensor.Dense), b3)
		}
	}

	return nil
}
