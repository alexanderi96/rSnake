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

// PrioritizedReplayBuffer stores experience with priorities
type PrioritizedReplayBuffer struct {
	buffer     []Transition
	priorities []float64
	alpha      float64 // How much to emphasize priorities
	beta       float64 // How much to correct importance sampling bias
	maxSize    int
	position   int
	size       int
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
	replayBuffer    *PrioritizedReplayBuffer
	LearningRate    float64
	Discount        float64
	Epsilon         float64
	InitialEpsilon  float64
	MinEpsilon      float64
	EpsilonDecay    float64
	TrainingEpisode int
	updateCounter   int
}

// NewPrioritizedReplayBuffer creates a new prioritized replay buffer
func NewPrioritizedReplayBuffer(maxSize int) *PrioritizedReplayBuffer {
	return &PrioritizedReplayBuffer{
		buffer:     make([]Transition, maxSize),
		priorities: make([]float64, maxSize),
		alpha:      0.6,
		beta:       0.4,
		maxSize:    maxSize,
		position:   0,
		size:       0,
	}
}

// Add adds a transition to the buffer with maximum priority
func (b *PrioritizedReplayBuffer) Add(t Transition) {
	maxPriority := 1.0
	if b.size > 0 {
		maxPriority = b.getMaxPriority()
	}

	b.buffer[b.position] = t
	b.priorities[b.position] = maxPriority

	b.position = (b.position + 1) % b.maxSize
	if b.size < b.maxSize {
		b.size++
	}
}

func (b *PrioritizedReplayBuffer) getMaxPriority() float64 {
	maxPriority := b.priorities[0]
	for i := 1; i < b.size; i++ {
		if b.priorities[i] > maxPriority {
			maxPriority = b.priorities[i]
		}
	}
	return maxPriority
}

// Sample returns a batch of transitions based on priorities
func (b *PrioritizedReplayBuffer) Sample(batchSize int) ([]Transition, []int, []float64) {
	if batchSize > b.size {
		batchSize = b.size
	}

	batch := make([]Transition, batchSize)
	indices := make([]int, batchSize)
	weights := make([]float64, batchSize)

	// Calculate selection probabilities
	sum := 0.0
	for i := 0; i < b.size; i++ {
		sum += math.Pow(b.priorities[i], b.alpha)
	}

	probabilities := make([]float64, b.size)
	for i := 0; i < b.size; i++ {
		probabilities[i] = math.Pow(b.priorities[i], b.alpha) / sum
	}

	// Sample based on probabilities
	for i := 0; i < batchSize; i++ {
		r := rand.Float64()
		cumProb := 0.0
		for j := 0; j < b.size; j++ {
			cumProb += probabilities[j]
			if r < cumProb {
				indices[i] = j
				batch[i] = b.buffer[j]
				weights[i] = math.Pow(float64(b.size)*probabilities[j], -b.beta)
				break
			}
		}
	}

	// Normalize weights
	maxWeight := weights[0]
	for i := 1; i < len(weights); i++ {
		if weights[i] > maxWeight {
			maxWeight = weights[i]
		}
	}
	for i := range weights {
		weights[i] /= maxWeight
	}

	return batch, indices, weights
}

// UpdatePriorities updates priorities after learning
func (b *PrioritizedReplayBuffer) UpdatePriorities(indices []int, errors []float64) {
	for i, idx := range indices {
		b.priorities[idx] = math.Pow(errors[i]+1e-5, b.alpha)
	}

	// Gradually increase beta to reduce bias
	b.beta = math.Min(1.0, b.beta+0.001)
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
	// Create a new graph for this forward pass
	g := gorgonia.NewGraph()

	// Convert states to tensor
	batchSize := len(states) / InputFeatures
	if batchSize == 0 && len(states) == InputFeatures {
		batchSize = 1
	}
	statesTensor := tensor.New(tensor.WithBacking(states), tensor.WithShape(batchSize, InputFeatures))
	statesNode := gorgonia.NodeFromAny(g, statesTensor)

	// Create new weight nodes in the graph
	w1 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(HiddenLayerSize, InputFeatures))
	b1 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(HiddenLayerSize, 1))
	w2 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(OutputActions, HiddenLayerSize))
	b2 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(OutputActions, 1))

	// Copy weights from the stored nodes
	if err := gorgonia.Let(w1, dqn.w1.Value()); err != nil {
		return nil, fmt.Errorf("failed to copy w1: %v", err)
	}
	if err := gorgonia.Let(b1, dqn.b1.Value()); err != nil {
		return nil, fmt.Errorf("failed to copy b1: %v", err)
	}
	if err := gorgonia.Let(w2, dqn.w2.Value()); err != nil {
		return nil, fmt.Errorf("failed to copy w2: %v", err)
	}
	if err := gorgonia.Let(b2, dqn.b2.Value()); err != nil {
		return nil, fmt.Errorf("failed to copy b2: %v", err)
	}

	// First layer
	// Reshape input for batch processing
	statesReshaped := gorgonia.Must(gorgonia.Reshape(statesNode, tensor.Shape{batchSize, InputFeatures}))

	// First layer computation
	h1 := gorgonia.Must(gorgonia.Mul(statesReshaped, gorgonia.Must(gorgonia.Transpose(w1))))

	// Create bias matrix for broadcasting
	b1Data := make([]float64, batchSize*HiddenLayerSize)
	b1Value := b1.Value().Data().([]float64)
	for i := 0; i < batchSize; i++ {
		for j := 0; j < HiddenLayerSize; j++ {
			b1Data[i*HiddenLayerSize+j] = b1Value[j]
		}
	}
	b1Tensor := tensor.New(tensor.WithBacking(b1Data), tensor.WithShape(batchSize, HiddenLayerSize))
	b1Node := gorgonia.NodeFromAny(g, b1Tensor)

	h1 = gorgonia.Must(gorgonia.Add(h1, b1Node))
	h1 = gorgonia.Must(gorgonia.Rectify(h1))

	// Output layer computation
	output := gorgonia.Must(gorgonia.Mul(h1, gorgonia.Must(gorgonia.Transpose(w2))))

	// Create bias matrix for broadcasting
	b2Data := make([]float64, batchSize*OutputActions)
	b2Value := b2.Value().Data().([]float64)
	for i := 0; i < batchSize; i++ {
		for j := 0; j < OutputActions; j++ {
			b2Data[i*OutputActions+j] = b2Value[j]
		}
	}
	b2Tensor := tensor.New(tensor.WithBacking(b2Data), tensor.WithShape(batchSize, OutputActions))
	b2Node := gorgonia.NodeFromAny(g, b2Tensor)

	pred := gorgonia.Must(gorgonia.Add(output, b2Node))

	// Create and run VM
	vm := gorgonia.NewTapeMachine(g)
	defer vm.Close()

	if err := vm.RunAll(); err != nil {
		return nil, fmt.Errorf("forward pass error: %v", err)
	}

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
		replayBuffer:    NewPrioritizedReplayBuffer(ReplayBufferSize),
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
		Done:      false, // We don't have this information, assume not done
	})

	// Only train if we have enough samples
	if a.replayBuffer.size < MinReplaySize {
		return
	}

	// Sample batch with priorities
	batch, indices, weights := a.replayBuffer.Sample(BatchSize)
	tdErrors := a.trainOnBatch(batch, weights)

	// Update priorities based on TD errors
	a.replayBuffer.UpdatePriorities(indices, tdErrors)

	// Update target network periodically
	a.updateCounter++
	if a.updateCounter >= TargetUpdateFreq {
		a.updateTargetNetwork()
		a.updateCounter = 0
	}
}

// trainOnBatch performs a training step on a batch of transitions and returns TD errors
func (a *Agent) trainOnBatch(batch []Transition, weights []float64) []float64 {
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
		return make([]float64, len(batch)) // Return zero TD errors on error
	}

	// Get target Q-values
	nextQValues, err := a.targetDQN.Forward(nextStates)
	if err != nil {
		return make([]float64, len(batch)) // Return zero TD errors on error
	}

	// Calculate target Q-values using Double DQN
	targetQValues := make([]float64, len(batch)*OutputActions)
	tdErrors := make([]float64, len(batch))

	// Get next state Q-values from both networks
	nextQValuesPrimary, err := a.dqn.Forward(nextStates)
	if err != nil {
		return tdErrors
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
				// Calculate TD error for priority update
				tdErrors[i] = math.Abs(target - currentQValues[baseIdx+j])
			}
		}
	}

	// Convert to tensors for loss calculation
	targetTensor := tensor.New(tensor.WithBacking(targetQValues), tensor.WithShape(len(batch), OutputActions))
	targetNode := gorgonia.NodeFromAny(g, targetTensor)

	currentTensor := tensor.New(tensor.WithBacking(currentQValues), tensor.WithShape(len(batch), OutputActions))
	currentNode := gorgonia.NodeFromAny(g, currentTensor)

	// Apply importance sampling weights to loss calculation
	// Broadcast weights to match the shape of squared differences
	weightsExpanded := make([]float64, len(batch)*OutputActions)
	for i := 0; i < len(batch); i++ {
		for j := 0; j < OutputActions; j++ {
			weightsExpanded[i*OutputActions+j] = weights[i]
		}
	}
	weightsTensor := tensor.New(tensor.WithBacking(weightsExpanded), tensor.WithShape(len(batch), OutputActions))
	weightsNode := gorgonia.NodeFromAny(g, weightsTensor)

	diff := gorgonia.Must(gorgonia.Sub(currentNode, targetNode))
	squaredDiff := gorgonia.Must(gorgonia.Square(diff))
	weightedDiff := gorgonia.Must(gorgonia.HadamardProd(squaredDiff, weightsNode))
	loss := gorgonia.Must(gorgonia.Mean(weightedDiff))

	// Backpropagate and update weights
	gorgonia.Grad(loss, a.dqn.w1, a.dqn.w2, a.dqn.b1, a.dqn.b2)
	a.dqn.vm.RunAll()
	nodes := gorgonia.Nodes{a.dqn.w1, a.dqn.w2, a.dqn.b1, a.dqn.b2}
	grads := gorgonia.NodesToValueGrads(nodes)
	a.dqn.solver.Step(grads)
	a.dqn.vm.Reset()

	return tdErrors
}

// updateTargetNetwork copies weights from DQN to target network
func (a *Agent) updateTargetNetwork() {
	// Create a new DQN with the same architecture
	a.targetDQN = NewDQN()

	// Copy weights
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
