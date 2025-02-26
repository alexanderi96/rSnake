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
	gob.Register(&tensor.Dense{})
	gob.Register(map[string]*tensor.Dense{})
}

const (
	// Parametri di apprendimento
	LearningRate   = 0.005 // Increased for faster learning
	Gamma          = 0.95  // Reduced to focus more on immediate rewards
	InitialEpsilon = 1.0
	EpsilonDecay   = 0.99 // More aggressive decay
	MinEpsilon     = 0.01

	// Parametri DQN
	BatchSize        = 32
	ReplayBufferSize = 5000 // Further reduced
	HiddenLayerSize  = 12   // Increased for more complex state
	InputFeatures    = 7    // 4 per food direction one-hot + 3 per danger flags
	OutputActions    = 3
	GradientClip     = 0.5 // Reduced for more stable learning

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
	g      *gorgonia.ExprGraph
	w1, w2 *gorgonia.Node
	b1, b2 *gorgonia.Node
	pred   *gorgonia.Node
	vm     gorgonia.VM
	solver gorgonia.Solver
}

// Agent rappresenta l'agente DQN
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
}

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

	// Input layer -> Hidden layer
	w1 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithShape(InputFeatures, HiddenLayerSize),
		gorgonia.WithInit(gorgonia.GlorotU(1.0)))

	b1 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithShape(1, HiddenLayerSize),
		gorgonia.WithInit(gorgonia.Zeroes()))

	// Hidden layer -> Output layer
	w2 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithShape(HiddenLayerSize, OutputActions),
		gorgonia.WithInit(gorgonia.GlorotU(1.0)))

	b2 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithShape(1, OutputActions),
		gorgonia.WithInit(gorgonia.Zeroes()))

	dqn := &DQN{
		g:      g,
		w1:     w1,
		w2:     w2,
		b1:     b1,
		b2:     b2,
		solver: gorgonia.NewAdamSolver(gorgonia.WithL2Reg(1e-6)),
	}

	dqn.vm = gorgonia.NewTapeMachine(g)
	return dqn
}

// Forward esegue un forward pass attraverso la rete
func (dqn *DQN) Forward(states []float64) ([]float64, error) {
	batchSize := len(states) / InputFeatures
	if batchSize == 0 {
		batchSize = 1
	}

	g := dqn.g

	// Gli input sono già normalizzati (one-hot e booleani)
	normalizedStates := make([]float64, len(states))
	copy(normalizedStates, states)

	statesTensor := tensor.New(tensor.WithBacking(normalizedStates), tensor.WithShape(batchSize, InputFeatures))
	statesNode := gorgonia.NodeFromAny(g, statesTensor)

	// Helper function per il broadcasting del bias
	expandBias := func(bias *gorgonia.Node, size int) (*gorgonia.Node, error) {
		ones := tensor.New(tensor.WithShape(size, 1), tensor.WithBacking(make([]float64, size)))
		for i := range ones.Data().([]float64) {
			ones.Data().([]float64)[i] = 1.0
		}
		onesNode := gorgonia.NodeFromAny(g, ones)
		return gorgonia.Mul(onesNode, bias)
	}

	// Hidden layer con ReLU
	h1 := gorgonia.Must(gorgonia.Mul(statesNode, dqn.w1))
	expandedBias1 := gorgonia.Must(expandBias(dqn.b1, batchSize))
	h1 = gorgonia.Must(gorgonia.Add(h1, expandedBias1))
	h1 = gorgonia.Must(gorgonia.Rectify(h1))

	// Output layer
	output := gorgonia.Must(gorgonia.Mul(h1, dqn.w2))
	expandedBias2 := gorgonia.Must(expandBias(dqn.b2, batchSize))
	pred := gorgonia.Must(gorgonia.Add(output, expandedBias2))

	if err := dqn.vm.RunAll(); err != nil {
		return nil, fmt.Errorf("forward pass error: %v", err)
	}
	dqn.vm.Reset()

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

// NewAgent crea un nuovo agente DQN
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
	}

	if err := agent.LoadWeights(WeightsFile); err == nil {
		agent.LearningRate = learningRate
		agent.Discount = discount
	}

	return agent
}

// GetAction seleziona un'azione usando la policy epsilon-greedy
func (a *Agent) GetAction(state []float64, numActions int) int {
	if a.Epsilon > a.MinEpsilon {
		a.Epsilon = a.InitialEpsilon * math.Pow(a.EpsilonDecay, float64(a.TrainingEpisode))
		if a.Epsilon < a.MinEpsilon {
			a.Epsilon = a.MinEpsilon
		}
	}

	if rand.Float64() < a.Epsilon {
		return rand.Intn(numActions)
	}

	qValues, err := a.dqn.Forward(state)
	if err != nil {
		return rand.Intn(numActions)
	}

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

// Update esegue un passo di aggiornamento DQN
func (a *Agent) Update(state []float64, action int, reward float64, nextState []float64, numActions int) {
	a.replayBuffer.Add(Transition{
		State:     state,
		Action:    action,
		Reward:    reward,
		NextState: nextState,
		Done:      false,
	})

	if a.replayBuffer.size < BatchSize {
		return
	}

	batch := a.replayBuffer.Sample(BatchSize)
	a.trainOnBatch(batch)
}

// trainOnBatch esegue un passo di training su un batch di transizioni
func (a *Agent) trainOnBatch(batch []Transition) {
	g := a.dqn.g
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

	currentQValues, err := a.dqn.Forward(states)
	if err != nil {
		return
	}

	nextQValues, err := a.targetDQN.Forward(nextStates)
	if err != nil {
		return
	}

	targetQValues := make([]float64, len(batch)*OutputActions)
	copy(targetQValues, currentQValues)

	for i := 0; i < len(batch); i++ {
		maxQ := math.Inf(-1)
		for j := 0; j < OutputActions; j++ {
			if nextQValues[i*OutputActions+j] > maxQ {
				maxQ = nextQValues[i*OutputActions+j]
			}
		}

		targetQValues[i*OutputActions+actions[i]] = rewards[i] + a.Discount*maxQ
	}

	targetTensor := tensor.New(tensor.WithBacking(targetQValues), tensor.WithShape(len(batch), OutputActions))
	targetNode := gorgonia.NodeFromAny(g, targetTensor)

	currentTensor := tensor.New(tensor.WithBacking(currentQValues), tensor.WithShape(len(batch), OutputActions))
	currentNode := gorgonia.NodeFromAny(g, currentTensor)

	// MSE Loss
	diff := gorgonia.Must(gorgonia.Sub(currentNode, targetNode))
	loss := gorgonia.Must(gorgonia.Mean(gorgonia.Must(gorgonia.Square(diff))))

	// Backpropagation con gradient clipping
	nodes := gorgonia.Nodes{a.dqn.w1, a.dqn.w2, a.dqn.b1, a.dqn.b2}
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

	// Soft update del target network
	copyWeights(a.targetDQN, a.dqn)
}

// copyWeights copia i pesi dal DQN principale al target network
func copyWeights(target, source *DQN) {
	tau := 0.001 // Soft update rate
	copyTensor(target.w1.Value().(*tensor.Dense), source.w1.Value().(*tensor.Dense), tau)
	copyTensor(target.w2.Value().(*tensor.Dense), source.w2.Value().(*tensor.Dense), tau)
	copyTensor(target.b1.Value().(*tensor.Dense), source.b1.Value().(*tensor.Dense), tau)
	copyTensor(target.b2.Value().(*tensor.Dense), source.b2.Value().(*tensor.Dense), tau)
}

// copyTensor esegue un soft update dei pesi
func copyTensor(target, source *tensor.Dense, tau float64) {
	targetData := target.Data().([]float64)
	sourceData := source.Data().([]float64)
	for i := range targetData {
		targetData[i] = tau*sourceData[i] + (1-tau)*targetData[i]
	}
}

// IncrementEpisode incrementa il contatore degli episodi
func (a *Agent) IncrementEpisode() {
	a.TrainingEpisode++
	if a.TrainingEpisode < 1000 {
		a.Epsilon = math.Max(0.1, a.InitialEpsilon*math.Exp(-float64(a.TrainingEpisode)/500))
	} else {
		a.Epsilon = math.Max(0.05, a.InitialEpsilon*math.Exp(-float64(a.TrainingEpisode)/1000))
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
		"b1": a.dqn.b1.Value().(*tensor.Dense),
		"b2": a.dqn.b2.Value().(*tensor.Dense),
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
		tensor.Copy(a.targetDQN.w1.Value().(*tensor.Dense), w1)
	}
	if w2, ok := weights["w2"]; ok {
		tensor.Copy(a.dqn.w2.Value().(*tensor.Dense), w2)
		tensor.Copy(a.targetDQN.w2.Value().(*tensor.Dense), w2)
	}
	if b1, ok := weights["b1"]; ok {
		tensor.Copy(a.dqn.b1.Value().(*tensor.Dense), b1)
		tensor.Copy(a.targetDQN.b1.Value().(*tensor.Dense), b1)
	}
	if b2, ok := weights["b2"]; ok {
		tensor.Copy(a.dqn.b2.Value().(*tensor.Dense), b2)
		tensor.Copy(a.targetDQN.b2.Value().(*tensor.Dense), b2)
	}

	return nil
}
