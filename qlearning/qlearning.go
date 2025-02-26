package qlearning

import (
	"encoding/gob"
	"fmt"
	"log"
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
	LearningRate   = 0.001 // Reduced for more stable learning
	Gamma          = 0.99  // Increased for better future reward consideration
	InitialEpsilon = 1.0
	EpsilonDecay   = 0.997 // Slower decay
	MinEpsilon     = 0.01  // Lower minimum for better exploitation
	TauSoftUpdate  = 0.001 // Soft update rate for target network

	DataDir     = "data"
	WeightsFile = DataDir + "/dqn_weights.gob"

	// DQN hyperparameters
	BatchSize        = 64     // Smaller batch for more frequent updates
	ReplayBufferSize = 100000 // Keep large buffer
	TargetUpdateFreq = 100    // More frequent soft updates
	HiddenLayer1Size = 32     // First hidden layer (reduced from 128)
	HiddenLayer2Size = 16     // Second hidden layer (reduced from 64)
	DropoutRate      = 0.2    // Dropout probability
	MinReplaySize    = 1000   // Start training earlier
	InputFeatures    = 5      // Current dir, food dir, distances (3)
	OutputActions    = 3      // left, forward, right
	GradientClip     = 1.0    // Maximum gradient norm
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
	g                                          *gorgonia.ExprGraph
	w1, w2, w3                                 *gorgonia.Node
	b1, b2, b3                                 *gorgonia.Node
	bn1Scale, bn1Bias, bn2Scale, bn2Bias       *gorgonia.Node
	bn1Mean, bn1Variance, bn2Mean, bn2Variance *gorgonia.Node
	dropoutMask1, dropoutMask2                 *gorgonia.Node
	pred                                       *gorgonia.Node
	vm                                         gorgonia.VM
	solver                                     gorgonia.Solver
	training                                   bool
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

	// Create and verify first layer weights
	w1 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithShape(InputFeatures, HiddenLayer1Size),
		gorgonia.WithInit(gorgonia.GlorotU(1.0)))

	// Verify w1 shape
	w1Shape := w1.Shape()
	if w1Shape[0] != InputFeatures || w1Shape[1] != HiddenLayer1Size {
		log.Fatalf("W1 has incorrect shape. Expected (%d, %d), got %v", InputFeatures, HiddenLayer1Size, w1Shape)
	}

	w2 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithShape(HiddenLayer1Size, HiddenLayer2Size),
		gorgonia.WithInit(gorgonia.GlorotU(1.0)))

	w3 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithShape(HiddenLayer2Size, OutputActions),
		gorgonia.WithInit(gorgonia.GlorotU(1.0)))

	// Initialize biases
	b1 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithShape(1, HiddenLayer1Size),
		gorgonia.WithInit(gorgonia.Zeroes()))

	b2 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithShape(1, HiddenLayer2Size),
		gorgonia.WithInit(gorgonia.Zeroes()))

	b3 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithShape(1, OutputActions),
		gorgonia.WithInit(gorgonia.Zeroes()))

	// Initialize batch normalization parameters
	bn1Scale := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithShape(1, HiddenLayer1Size),
		gorgonia.WithInit(gorgonia.Ones()))

	bn1Bias := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithShape(1, HiddenLayer1Size),
		gorgonia.WithInit(gorgonia.Zeroes()))

	bn2Scale := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithShape(1, HiddenLayer2Size),
		gorgonia.WithInit(gorgonia.Ones()))

	bn2Bias := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithShape(1, HiddenLayer2Size),
		gorgonia.WithInit(gorgonia.Zeroes()))

	dqn := &DQN{
		g:        g,
		w1:       w1,
		w2:       w2,
		w3:       w3,
		b1:       b1,
		b2:       b2,
		b3:       b3,
		bn1Scale: bn1Scale,
		bn1Bias:  bn1Bias,
		bn2Scale: bn2Scale,
		bn2Bias:  bn2Bias,
		solver:   gorgonia.NewAdamSolver(gorgonia.WithL2Reg(1e-6)), // Added L2 regularization
		training: true,
	}

	// Create VM after all nodes are added to graph
	dqn.vm = gorgonia.NewTapeMachine(g)

	return dqn
}

// Forward performs a forward pass through the network
func (dqn *DQN) Forward(states []float64) ([]float64, error) {
	// Calculate batch size and ensure proper reshaping
	var batchSize int
	if len(states) == InputFeatures {
		batchSize = 1
	} else {
		batchSize = len(states) / InputFeatures
	}

	// Use existing graph
	g := dqn.g

	// Normalize input states to [0,1]
	normalizedStates := make([]float64, len(states))
	for i := 0; i < len(states); i++ {
		if i < 2 {
			// First two values are directions (0-4)
			normalizedStates[i] = states[i] / 4.0
		} else {
			// Last three values are distances (0-5)
			normalizedStates[i] = states[i] / 5.0
		}
	}

	// Convert states to tensor and create graph
	statesTensor := tensor.New(tensor.WithBacking(normalizedStates), tensor.WithShape(batchSize, InputFeatures))

	// Verify input tensor shape
	if statesTensor.Shape()[1] != InputFeatures {
		log.Fatalf("Input tensor has incorrect shape. Expected (_, %d), got %v", InputFeatures, statesTensor.Shape())
	}

	statesNode := gorgonia.NodeFromAny(g, statesTensor)

	// First hidden layer with batch normalization
	h1 := gorgonia.Must(gorgonia.Mul(statesNode, dqn.w1))
	h1 = gorgonia.Must(gorgonia.Add(h1, expandBias(dqn.b1, batchSize, HiddenLayer1Size)))

	h1 = gorgonia.Must(gorgonia.Rectify(h1))

	if dqn.training {
		// Apply dropout during training
		mask1 := tensor.New(tensor.WithShape(batchSize, HiddenLayer1Size), tensor.WithBacking(generateDropoutMask(batchSize*HiddenLayer1Size)))
		dqn.dropoutMask1 = gorgonia.NodeFromAny(g, mask1)
		h1 = gorgonia.Must(gorgonia.HadamardProd(h1, dqn.dropoutMask1))
	}

	// Second hidden layer
	h2 := gorgonia.Must(gorgonia.Mul(h1, dqn.w2))
	h2 = gorgonia.Must(gorgonia.Add(h2, expandBias(dqn.b2, batchSize, HiddenLayer2Size)))

	h2 = gorgonia.Must(gorgonia.Rectify(h2))

	if dqn.training {
		mask2 := tensor.New(tensor.WithShape(batchSize, HiddenLayer2Size), tensor.WithBacking(generateDropoutMask(batchSize*HiddenLayer2Size)))
		dqn.dropoutMask2 = gorgonia.NodeFromAny(g, mask2)
		h2 = gorgonia.Must(gorgonia.HadamardProd(h2, dqn.dropoutMask2))
	}

	// Output layer
	output := gorgonia.Must(gorgonia.Mul(h2, dqn.w3))
	pred := gorgonia.Must(gorgonia.Add(output, expandBias(dqn.b3, batchSize, OutputActions)))

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
func (a *Agent) GetAction(state []float64, numActions int) int {
	// Update epsilon with decay
	if a.Epsilon > a.MinEpsilon {
		a.Epsilon = a.InitialEpsilon * math.Pow(a.EpsilonDecay, float64(a.TrainingEpisode))
		if a.Epsilon < a.MinEpsilon {
			a.Epsilon = a.MinEpsilon
		}
	}

	// Exploration: random action
	if rand.Float64() < a.Epsilon {
		return rand.Intn(numActions)
	}

	// Exploitation: best action from DQN
	qValues, err := a.dqn.Forward(state)
	if err != nil {
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
func (a *Agent) Update(state []float64, action int, reward float64, nextState []float64, numActions int) {
	// Add transition to replay buffer
	a.replayBuffer.Add(Transition{
		State:     state,
		Action:    action,
		Reward:    reward,
		NextState: nextState,
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
		a.softUpdateTargetNetwork()
		a.updateCounter = 0
	}
}

// Helper function to generate dropout mask
func generateDropoutMask(size int) []float64 {
	mask := make([]float64, size)
	for i := range mask {
		if rand.Float64() > DropoutRate {
			mask[i] = 1.0 / (1.0 - DropoutRate)
		}
	}
	return mask
}

// Helper function to expand bias for batch operations
func expandBias(bias *gorgonia.Node, batchSize, size int) *gorgonia.Node {
	// Create a tensor of ones for broadcasting
	ones := tensor.New(tensor.WithShape(batchSize, 1), tensor.WithBacking(make([]float64, batchSize)))
	for i := range ones.Data().([]float64) {
		ones.Data().([]float64)[i] = 1.0
	}
	onesNode := gorgonia.NodeFromAny(bias.Graph(), ones)

	// Broadcast bias to match batch size
	return gorgonia.Must(gorgonia.Mul(onesNode, bias))
}

// huberLoss implements the Huber loss function
func huberLoss(pred, target *gorgonia.Node, delta float64) *gorgonia.Node {
	diff := gorgonia.Must(gorgonia.Sub(pred, target))
	absDiff := gorgonia.Must(gorgonia.Abs(diff))

	quadratic := gorgonia.Must(gorgonia.Square(diff))
	quadratic = gorgonia.Must(gorgonia.Mul(quadratic, gorgonia.NewScalar(pred.Graph(), tensor.Float64, gorgonia.WithValue(0.5))))

	linear := gorgonia.Must(gorgonia.Sub(absDiff, gorgonia.NewScalar(pred.Graph(), tensor.Float64, gorgonia.WithValue(delta/2.0))))
	linear = gorgonia.Must(gorgonia.Mul(linear, gorgonia.NewScalar(pred.Graph(), tensor.Float64, gorgonia.WithValue(delta))))

	condition := gorgonia.Must(gorgonia.Lte(absDiff, gorgonia.NewScalar(pred.Graph(), tensor.Float64, gorgonia.WithValue(delta)), false))
	loss := gorgonia.Must(gorgonia.Add(
		gorgonia.Must(gorgonia.Mul(condition, quadratic)),
		gorgonia.Must(gorgonia.Mul(
			gorgonia.Must(gorgonia.Sub(
				gorgonia.NewScalar(pred.Graph(), tensor.Float64, gorgonia.WithValue(1.0)),
				condition)),
			linear))))

	return gorgonia.Must(gorgonia.Mean(loss))
}

// trainOnBatch performs a training step on a batch of transitions
func (a *Agent) trainOnBatch(batch []Transition) {
	g := a.dqn.g
	a.dqn.training = true
	defer func() { a.dqn.training = false }()

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

	// Calculate loss using Huber loss
	loss := huberLoss(currentNode, targetNode, 1.0)

	// Backpropagate and update weights with gradient clipping
	nodes := gorgonia.Nodes{a.dqn.w1, a.dqn.w2, a.dqn.w3, a.dqn.b1, a.dqn.b2, a.dqn.b3,
		a.dqn.bn1Scale, a.dqn.bn1Bias, a.dqn.bn2Scale, a.dqn.bn2Bias}
	gorgonia.Grad(loss, nodes...)

	// Run backprop
	if err := a.dqn.vm.RunAll(); err != nil {
		log.Printf("Error during backprop: %v", err)
		return
	}

	// Get gradients
	grads := gorgonia.NodesToValueGrads(nodes)

	// Clip gradients
	for _, grad := range grads {
		if grad == nil {
			continue
		}
		if t, ok := grad.(tensor.Tensor); ok {
			data := t.Data().([]float64)
			for i := range data {
				if data[i] > GradientClip {
					data[i] = GradientClip
				} else if data[i] < -GradientClip {
					data[i] = -GradientClip
				}
			}
		}
	}

	// Update weights
	a.dqn.solver.Step(grads)
	a.dqn.vm.Reset()

	// Soft update target network
	a.softUpdateTargetNetwork()
}

// softUpdateTargetNetwork performs soft update of target network parameters
func (a *Agent) softUpdateTargetNetwork() {
	w1Target := a.targetDQN.w1.Value().(*tensor.Dense)
	w2Target := a.targetDQN.w2.Value().(*tensor.Dense)
	w3Target := a.targetDQN.w3.Value().(*tensor.Dense)
	b1Target := a.targetDQN.b1.Value().(*tensor.Dense)
	b2Target := a.targetDQN.b2.Value().(*tensor.Dense)
	b3Target := a.targetDQN.b3.Value().(*tensor.Dense)

	w1Current := a.dqn.w1.Value().(*tensor.Dense)
	w2Current := a.dqn.w2.Value().(*tensor.Dense)
	w3Current := a.dqn.w3.Value().(*tensor.Dense)
	b1Current := a.dqn.b1.Value().(*tensor.Dense)
	b2Current := a.dqn.b2.Value().(*tensor.Dense)
	b3Current := a.dqn.b3.Value().(*tensor.Dense)

	// θ_target = τ*θ_current + (1-τ)*θ_target
	softUpdate(w1Target, w1Current, TauSoftUpdate)
	softUpdate(w2Target, w2Current, TauSoftUpdate)
	softUpdate(w3Target, w3Current, TauSoftUpdate)
	softUpdate(b1Target, b1Current, TauSoftUpdate)
	softUpdate(b2Target, b2Current, TauSoftUpdate)
	softUpdate(b3Target, b3Current, TauSoftUpdate)
}

// softUpdate performs the soft update operation for a single tensor
func softUpdate(target, current *tensor.Dense, tau float64) {
	targetData := target.Data().([]float64)
	currentData := current.Data().([]float64)

	for i := range targetData {
		targetData[i] = tau*currentData[i] + (1-tau)*targetData[i]
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
		"w1":        a.dqn.w1.Value().(*tensor.Dense),
		"w2":        a.dqn.w2.Value().(*tensor.Dense),
		"w3":        a.dqn.w3.Value().(*tensor.Dense),
		"b1":        a.dqn.b1.Value().(*tensor.Dense),
		"b2":        a.dqn.b2.Value().(*tensor.Dense),
		"b3":        a.dqn.b3.Value().(*tensor.Dense),
		"bn1_scale": a.dqn.bn1Scale.Value().(*tensor.Dense),
		"bn1_bias":  a.dqn.bn1Bias.Value().(*tensor.Dense),
		"bn2_scale": a.dqn.bn2Scale.Value().(*tensor.Dense),
		"bn2_bias":  a.dqn.bn2Bias.Value().(*tensor.Dense),
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

	// Update network weights and parameters
	if w1, ok := weights["w1"]; ok {
		tensor.Copy(a.dqn.w1.Value().(*tensor.Dense), w1)
		tensor.Copy(a.targetDQN.w1.Value().(*tensor.Dense), w1)
	}
	if w2, ok := weights["w2"]; ok {
		tensor.Copy(a.dqn.w2.Value().(*tensor.Dense), w2)
		tensor.Copy(a.targetDQN.w2.Value().(*tensor.Dense), w2)
	}
	if w3, ok := weights["w3"]; ok {
		tensor.Copy(a.dqn.w3.Value().(*tensor.Dense), w3)
		tensor.Copy(a.targetDQN.w3.Value().(*tensor.Dense), w3)
	}
	if b1, ok := weights["b1"]; ok {
		tensor.Copy(a.dqn.b1.Value().(*tensor.Dense), b1)
		tensor.Copy(a.targetDQN.b1.Value().(*tensor.Dense), b1)
	}
	if b2, ok := weights["b2"]; ok {
		tensor.Copy(a.dqn.b2.Value().(*tensor.Dense), b2)
		tensor.Copy(a.targetDQN.b2.Value().(*tensor.Dense), b2)
	}
	if b3, ok := weights["b3"]; ok {
		tensor.Copy(a.dqn.b3.Value().(*tensor.Dense), b3)
		tensor.Copy(a.targetDQN.b3.Value().(*tensor.Dense), b3)
	}
	if bn1Scale, ok := weights["bn1_scale"]; ok {
		tensor.Copy(a.dqn.bn1Scale.Value().(*tensor.Dense), bn1Scale)
		tensor.Copy(a.targetDQN.bn1Scale.Value().(*tensor.Dense), bn1Scale)
	}
	if bn1Bias, ok := weights["bn1_bias"]; ok {
		tensor.Copy(a.dqn.bn1Bias.Value().(*tensor.Dense), bn1Bias)
		tensor.Copy(a.targetDQN.bn1Bias.Value().(*tensor.Dense), bn1Bias)
	}
	if bn2Scale, ok := weights["bn2_scale"]; ok {
		tensor.Copy(a.dqn.bn2Scale.Value().(*tensor.Dense), bn2Scale)
		tensor.Copy(a.targetDQN.bn2Scale.Value().(*tensor.Dense), bn2Scale)
	}
	if bn2Bias, ok := weights["bn2_bias"]; ok {
		tensor.Copy(a.dqn.bn2Bias.Value().(*tensor.Dense), bn2Bias)
		tensor.Copy(a.targetDQN.bn2Bias.Value().(*tensor.Dense), bn2Bias)
	}

	return nil
}
