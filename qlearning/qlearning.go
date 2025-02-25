package qlearning

import (
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"os"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func init() {
	// Register tensor types with gob
	gob.Register(&tensor.Dense{})
	gob.Register(map[string]*tensor.Dense{})
}

const (
	// Learning parameters
	LearningRate   = 0.001
	Gamma          = 0.99
	InitialEpsilon = 1.0
	EpsilonDecay   = 0.9975 // Faster decay
	MinEpsilon     = 0.01

	DataDir     = "data"
	WeightsFile = DataDir + "/dqn_weights.gob"

	// DQN hyperparameters
	BatchSize        = 64     // Increased batch size
	ReplayBufferSize = 100000 // Larger buffer
	TargetUpdateFreq = 1000   // More frequent target updates
	HiddenLayerSize  = 128    // Larger hidden layer
	MinReplaySize    = 1000
	InputFeatures    = 7 // Current dir, food dir, distances (3), food dist, length
	OutputActions    = 3 // left, forward, right
)

// Transition represents a single step in the environment
type Transition struct {
	State     []float64
	Action    int
	Reward    float64
	NextState []float64
	Done      bool
}

// ReplayBuffer stores experience for training
type ReplayBuffer struct {
	buffer   []Transition
	maxSize  int
	position int
	size     int
}

// DQN represents the deep Q-network
type DQN struct {
	g      *gorgonia.ExprGraph
	w1, w2 *gorgonia.Node
	b1, b2 *gorgonia.Node
	pred   *gorgonia.Node
	vm     gorgonia.VM
	solver gorgonia.Solver
}

// Agent represents a DQN agent
type Agent struct {
	dqn             *DQN
	targetDQN       *DQN
	replayBuffer    *ReplayBuffer
	LearningRate    float64
	Discount        float64
	Epsilon         float64
	InitialEpsilon  float64
	MinEpsilon      float64
	EpsilonDecay    float64
	TrainingEpisode int
	updateCounter   int
}

// NewReplayBuffer creates a new replay buffer
func NewReplayBuffer(maxSize int) *ReplayBuffer {
	return &ReplayBuffer{
		buffer:   make([]Transition, maxSize),
		maxSize:  maxSize,
		position: 0,
		size:     0,
	}
}

// Add adds a transition to the buffer
func (b *ReplayBuffer) Add(t Transition) {
	b.buffer[b.position] = t
	b.position = (b.position + 1) % b.maxSize
	if b.size < b.maxSize {
		b.size++
	}
}

// Sample returns a random batch of transitions
func (b *ReplayBuffer) Sample(batchSize int) []Transition {
	if batchSize > b.size {
		batchSize = b.size
	}

	batch := make([]Transition, batchSize)

	// Reservoir sampling for uniform random selection
	for i := 0; i < batchSize; i++ {
		idx := rand.Intn(b.size)
		batch[i] = b.buffer[idx]
	}

	return batch
}

// NewDQN creates a new DQN
func NewDQN() *DQN {
	g := gorgonia.NewGraph()

	// Initialize weights and biases
	w1 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithShape(HiddenLayerSize, InputFeatures),
		gorgonia.WithInit(gorgonia.GlorotU(1.0)))

	b1 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithShape(HiddenLayerSize, 1),
		gorgonia.WithInit(gorgonia.Zeroes()))

	w2 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithShape(OutputActions, HiddenLayerSize),
		gorgonia.WithInit(gorgonia.GlorotU(1.0)))

	b2 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithShape(OutputActions, 1),
		gorgonia.WithInit(gorgonia.Zeroes()))

	return &DQN{
		g:      g,
		w1:     w1,
		w2:     w2,
		b1:     b1,
		b2:     b2,
		vm:     gorgonia.NewTapeMachine(g),
		solver: gorgonia.NewRMSPropSolver(),
	}
}

// Forward performs a forward pass through the network
func (dqn *DQN) Forward(states []float64) ([]float64, error) {
	batchSize := len(states) / InputFeatures
	if batchSize == 0 && len(states) == InputFeatures {
		batchSize = 1
	}

	// Convert states to tensor and create graph
	g := dqn.g
	statesTensor := tensor.New(tensor.WithBacking(states), tensor.WithShape(batchSize, InputFeatures))
	statesNode := gorgonia.NodeFromAny(g, statesTensor)

	// First layer
	h1 := gorgonia.Must(gorgonia.Mul(statesNode, gorgonia.Must(gorgonia.Transpose(dqn.w1))))
	// Create bias matrix for batch
	b1Data := make([]float64, batchSize*HiddenLayerSize)
	b1Value := dqn.b1.Value().Data().([]float64)
	for i := 0; i < batchSize; i++ {
		copy(b1Data[i*HiddenLayerSize:], b1Value)
	}
	b1Tensor := tensor.New(tensor.WithBacking(b1Data), tensor.WithShape(batchSize, HiddenLayerSize))
	b1Node := gorgonia.NodeFromAny(g, b1Tensor)
	h1 = gorgonia.Must(gorgonia.Add(h1, b1Node))
	h1 = gorgonia.Must(gorgonia.Rectify(h1))

	// Output layer
	output := gorgonia.Must(gorgonia.Mul(h1, gorgonia.Must(gorgonia.Transpose(dqn.w2))))
	// Create bias matrix for batch
	b2Data := make([]float64, batchSize*OutputActions)
	b2Value := dqn.b2.Value().Data().([]float64)
	for i := 0; i < batchSize; i++ {
		copy(b2Data[i*OutputActions:], b2Value)
	}
	b2Tensor := tensor.New(tensor.WithBacking(b2Data), tensor.WithShape(batchSize, OutputActions))
	b2Node := gorgonia.NodeFromAny(g, b2Tensor)
	pred := gorgonia.Must(gorgonia.Add(output, b2Node))

	// Run forward pass
	if err := dqn.vm.RunAll(); err != nil {
		return nil, fmt.Errorf("forward pass error: %v", err)
	}
	dqn.vm.Reset()

	// Extract predictions
	predValue := pred.Value()
	if predValue == nil {
		return nil, fmt.Errorf("nil prediction value")
	}

	predTensor, ok := predValue.(*tensor.Dense)
	if !ok {
		return nil, fmt.Errorf("invalid prediction tensor type")
	}

	predictions := make([]float64, batchSize*OutputActions)
	copy(predictions, predTensor.Data().([]float64))

	return predictions, nil
}

// NewAgent creates a new DQN agent
func NewAgent(learningRate, discount, epsilon float64) *Agent {
	agent := &Agent{
		dqn:             NewDQN(),
		targetDQN:       NewDQN(),
		replayBuffer:    NewReplayBuffer(ReplayBufferSize),
		LearningRate:    learningRate,
		Discount:        discount,
		Epsilon:         InitialEpsilon,
		InitialEpsilon:  InitialEpsilon,
		MinEpsilon:      MinEpsilon,
		EpsilonDecay:    EpsilonDecay,
		TrainingEpisode: 0,
		updateCounter:   0,
	}

	// Try to load existing weights
	if err := agent.LoadWeights(WeightsFile); err == nil {
		agent.LearningRate = learningRate
		agent.Discount = discount
	}

	return agent
}

// GetAction selects an action using epsilon-greedy policy
func (a *Agent) GetAction(state string, numActions int) int {
	// Update epsilon with decay
	if a.Epsilon > a.MinEpsilon {
		a.Epsilon = a.InitialEpsilon * math.Pow(a.EpsilonDecay, float64(a.TrainingEpisode))
		if a.Epsilon < a.MinEpsilon {
			a.Epsilon = a.MinEpsilon
		}
	}

	// Convert state string to float64 slice
	stateVec := a.stateToVector(state)

	// Exploration: random action
	if rand.Float64() < a.Epsilon {
		return rand.Intn(numActions)
	}

	// Exploitation: best action from DQN
	qValues, err := a.dqn.Forward(stateVec)
	if err != nil {
		// Fallback to random action on error
		return rand.Intn(numActions)
	}

	// Find action with maximum Q-value
	maxQ := math.Inf(-1)
	bestAction := 0
	for action, qValue := range qValues {
		if qValue > maxQ {
			maxQ = qValue
			bestAction = action
		}
	}

	return bestAction
}

// Update performs a DQN update step
func (a *Agent) Update(state string, action int, reward float64, nextState string, numActions int) {
	stateVec := a.stateToVector(state)
	nextStateVec := a.stateToVector(nextState)

	// Add transition to replay buffer
	a.replayBuffer.Add(Transition{
		State:     stateVec,
		Action:    action,
		Reward:    reward,
		NextState: nextStateVec,
		Done:      false,
	})

	// Only train if we have enough samples
	if a.replayBuffer.size < MinReplaySize {
		return
	}

	// Sample batch
	batch := a.replayBuffer.Sample(BatchSize)
	a.trainOnBatch(batch)

	// Update target network periodically
	a.updateCounter++
	if a.updateCounter >= TargetUpdateFreq {
		a.updateTargetNetwork()
		a.updateCounter = 0
	}
}

// trainOnBatch performs a training step on a batch of transitions
func (a *Agent) trainOnBatch(batch []Transition) {
	g := a.dqn.g

	// Prepare batch data
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

	// Get current Q-values
	currentQValues, err := a.dqn.Forward(states)
	if err != nil {
		return
	}

	// Get target Q-values
	nextQValues, err := a.targetDQN.Forward(nextStates)
	if err != nil {
		return
	}

	// Calculate target Q-values using Double DQN
	targetQValues := make([]float64, len(batch)*OutputActions)

	// Get next state Q-values from primary network
	nextQValuesPrimary, err := a.dqn.Forward(nextStates)
	if err != nil {
		return
	}

	for i := 0; i < len(batch); i++ {
		baseIdx := i * OutputActions

		// Use primary network to select action
		bestAction := 0
		for j := 1; j < OutputActions; j++ {
			if nextQValuesPrimary[baseIdx+j] > nextQValuesPrimary[baseIdx+bestAction] {
				bestAction = j
			}
		}

		// Use target network to evaluate action
		target := rewards[i]
		if !batch[i].Done {
			target += a.Discount * nextQValues[baseIdx+bestAction]
		}

		// Set target for the taken action
		for j := 0; j < OutputActions; j++ {
			targetQValues[baseIdx+j] = currentQValues[baseIdx+j]
			if j == actions[i] {
				targetQValues[baseIdx+j] = target
			}
		}
	}

	// Convert to tensors for loss calculation
	targetTensor := tensor.New(tensor.WithBacking(targetQValues), tensor.WithShape(len(batch), OutputActions))
	targetNode := gorgonia.NodeFromAny(g, targetTensor)

	currentTensor := tensor.New(tensor.WithBacking(currentQValues), tensor.WithShape(len(batch), OutputActions))
	currentNode := gorgonia.NodeFromAny(g, currentTensor)

	// Calculate loss
	diff := gorgonia.Must(gorgonia.Sub(currentNode, targetNode))
	squaredDiff := gorgonia.Must(gorgonia.Square(diff))
	loss := gorgonia.Must(gorgonia.Mean(squaredDiff))

	// Backpropagate and update weights
	gorgonia.Grad(loss, a.dqn.w1, a.dqn.w2, a.dqn.b1, a.dqn.b2)
	a.dqn.vm.RunAll()
	nodes := gorgonia.Nodes{a.dqn.w1, a.dqn.w2, a.dqn.b1, a.dqn.b2}
	grads := gorgonia.NodesToValueGrads(nodes)
	a.dqn.solver.Step(grads)
	a.dqn.vm.Reset()
}

// updateTargetNetwork copies weights from DQN to target network
func (a *Agent) updateTargetNetwork() {
	tensor.Copy(a.targetDQN.w1.Value().(*tensor.Dense), a.dqn.w1.Value().(*tensor.Dense))
	tensor.Copy(a.targetDQN.w2.Value().(*tensor.Dense), a.dqn.w2.Value().(*tensor.Dense))
	tensor.Copy(a.targetDQN.b1.Value().(*tensor.Dense), a.dqn.b1.Value().(*tensor.Dense))
	tensor.Copy(a.targetDQN.b2.Value().(*tensor.Dense), a.dqn.b2.Value().(*tensor.Dense))
}

// stateToVector converts a state string to a vector of float64
func (a *Agent) stateToVector(state string) []float64 {
	var currentDir, foodDir, distAhead, distLeft, distRight, foodDistNorm int
	var dangerPattern string
	var lengthNorm int

	fmt.Sscanf(state, "%d:%d:%d:%d:%d:%d:%s:%d",
		&currentDir, &foodDir, &distAhead, &distLeft, &distRight,
		&foodDistNorm, &dangerPattern, &lengthNorm)

	return []float64{
		float64(currentDir),
		float64(foodDir),
		float64(distAhead),
		float64(distLeft),
		float64(distRight),
		float64(foodDistNorm),
		float64(lengthNorm),
	}
}

// IncrementEpisode increments the training episode counter
func (a *Agent) IncrementEpisode() {
	a.TrainingEpisode++
	if a.TrainingEpisode < 1000 {
		a.Epsilon = math.Max(0.1, a.InitialEpsilon*math.Exp(-float64(a.TrainingEpisode)/500))
	} else {
		a.Epsilon = math.Max(0.05, a.InitialEpsilon*math.Exp(-float64(a.TrainingEpisode)/1000))
	}
}

// SaveWeights saves the DQN weights to a file
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
		"b1": a.dqn.b1.Value().(*tensor.Dense),
		"b2": a.dqn.b2.Value().(*tensor.Dense),
	}

	if err := enc.Encode(weights); err != nil {
		return fmt.Errorf("failed to encode weights: %v", err)
	}

	return nil
}

// LoadWeights loads the DQN weights from a file
func (a *Agent) LoadWeights(filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // Use default weights if file doesn't exist
		}
		return fmt.Errorf("failed to open weights file: %v", err)
	}
	defer f.Close()

	var weights map[string]*tensor.Dense
	dec := gob.NewDecoder(f)
	if err := dec.Decode(&weights); err != nil {
		return fmt.Errorf("failed to decode weights: %v", err)
	}

	// Update network weights
	if w1, ok := weights["w1"]; ok {
		tensor.Copy(a.dqn.w1.Value().(*tensor.Dense), w1)
	}
	if w2, ok := weights["w2"]; ok {
		tensor.Copy(a.dqn.w2.Value().(*tensor.Dense), w2)
	}
	if b1, ok := weights["b1"]; ok {
		tensor.Copy(a.dqn.b1.Value().(*tensor.Dense), b1)
	}
	if b2, ok := weights["b2"]; ok {
		tensor.Copy(a.dqn.b2.Value().(*tensor.Dense), b2)
	}

	return nil
}
