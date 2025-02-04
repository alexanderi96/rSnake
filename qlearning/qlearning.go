package qlearning

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
)

// QTable stores Q-values for state-action pairs
type QTable map[string][]float64

// Agent represents a Q-learning agent
type Agent struct {
	QTable          QTable
	LearningRate    float64
	Discount        float64
	Epsilon         float64
	InitialEpsilon  float64
	MinEpsilon      float64
	EpsilonDecay    float64
	TrainingEpisode int
}

// NewAgent creates a new Q-learning agent
func NewAgent(learningRate, discount, epsilon float64) *Agent {
	return &Agent{
		QTable:          make(QTable),
		LearningRate:    learningRate,
		Discount:        discount,
		Epsilon:         0.9, // Start with high exploration
		InitialEpsilon:  0.9,
		MinEpsilon:      0.1,
		EpsilonDecay:    0.995, // Decay rate per episode
		TrainingEpisode: 0,
	}
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

	// Exploration: random action
	if rand.Float64() < a.Epsilon {
		return rand.Intn(numActions)
	}

	// Exploitation: best known action
	return a.getBestAction(state, numActions)
}

// IncrementEpisode increments the training episode counter
func (a *Agent) IncrementEpisode() {
	a.TrainingEpisode++
}

// Update updates Q-value for a state-action pair
func (a *Agent) Update(state string, action int, reward float64, nextState string, numActions int) {
	// Initialize Q-values if state not seen before
	if _, exists := a.QTable[state]; !exists {
		a.QTable[state] = make([]float64, numActions)
	}
	if _, exists := a.QTable[nextState]; !exists {
		a.QTable[nextState] = make([]float64, numActions)
	}

	// Q-learning update formula
	currentQ := a.QTable[state][action]
	maxNextQ := a.getMaxQValue(nextState)

	// Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
	a.QTable[state][action] = currentQ + a.LearningRate*(reward+a.Discount*maxNextQ-currentQ)
}

// getBestAction returns the action with highest Q-value for given state
func (a *Agent) getBestAction(state string, numActions int) int {
	if _, exists := a.QTable[state]; !exists {
		a.QTable[state] = make([]float64, numActions)
	}

	bestAction := 0
	maxQ := math.Inf(-1)

	for action, qValue := range a.QTable[state] {
		if qValue > maxQ {
			maxQ = qValue
			bestAction = action
		}
	}

	return bestAction
}

// getMaxQValue returns the maximum Q-value for given state
func (a *Agent) getMaxQValue(state string) float64 {
	if _, exists := a.QTable[state]; !exists {
		return 0
	}

	maxQ := math.Inf(-1)
	for _, qValue := range a.QTable[state] {
		if qValue > maxQ {
			maxQ = qValue
		}
	}

	return maxQ
}

// SaveQTable saves the Q-table to a file
func (a *Agent) SaveQTable(filename string) error {
	data, err := json.MarshalIndent(a.QTable, "", "  ")
	if err != nil {
		return fmt.Errorf("error marshaling QTable: %v", err)
	}

	err = os.WriteFile(filename, data, 0644)
	if err != nil {
		return fmt.Errorf("error writing QTable to file: %v", err)
	}

	return nil
}

// LoadQTable loads the Q-table from a file
func (a *Agent) LoadQTable(filename string) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		if os.IsNotExist(err) {
			// If file doesn't exist, start with empty QTable
			a.QTable = make(QTable)
			return nil
		}
		return fmt.Errorf("error reading QTable file: %v", err)
	}

	err = json.Unmarshal(data, &a.QTable)
	if err != nil {
		return fmt.Errorf("error unmarshaling QTable: %v", err)
	}

	return nil
}
