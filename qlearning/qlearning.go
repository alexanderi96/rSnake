package qlearning

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
)

// QTableManager handles the shared Q-table across multiple agents
type QTableManager struct {
	table map[string][]float64
	mu    sync.RWMutex
}

// NewQTableManager creates a new QTableManager
func NewQTableManager() *QTableManager {
	manager := &QTableManager{
		table: make(map[string][]float64),
	}

	// Try to load existing state
	if data, err := os.ReadFile("qtable.json"); err == nil {
		var state AgentState
		if err := json.Unmarshal(data, &state); err == nil {
			manager.table = state.QTable
		}
	}

	return manager
}

// GetQValues returns Q-values for a state, creating new entry if needed
func (m *QTableManager) GetQValues(state string, numActions int) []float64 {
	m.mu.RLock()
	qValues, exists := m.table[state]
	m.mu.RUnlock()

	if !exists {
		m.mu.Lock()
		m.table[state] = make([]float64, numActions)
		qValues = m.table[state]
		m.mu.Unlock()
	}

	return qValues
}

// UpdateQValue updates a specific Q-value in the table
func (m *QTableManager) UpdateQValue(state string, action int, value float64) {
	m.mu.Lock()
	if _, exists := m.table[state]; !exists {
		m.table[state] = make([]float64, 4) // Assuming 4 actions for snake
	}
	m.table[state][action] = value
	m.mu.Unlock()
}

// SaveQTable saves the Q-table to file
func (m *QTableManager) SaveQTable() error {
	m.mu.RLock()
	state := AgentState{
		QTable: m.table,
	}
	m.mu.RUnlock()

	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("error marshaling QTable: %v", err)
	}

	return os.WriteFile("qtable.json", data, 0644)
}

// Agent represents a Q-learning agent
type Agent struct {
	manager         *QTableManager
	LearningRate    float64
	Discount        float64
	Epsilon         float64
	InitialEpsilon  float64
	MinEpsilon      float64
	EpsilonDecay    float64
	TrainingEpisode int
}

// NewAgent creates a new Q-learning agent with shared Q-table
func NewAgent(manager *QTableManager, learningRate, discount float64) *Agent {
	return &Agent{
		manager:         manager,
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
	// Get current and next state Q-values
	currentQ := a.manager.GetQValues(state, numActions)[action]
	maxNextQ := a.getMaxQValue(nextState)

	// Q-learning update formula
	newQ := currentQ + a.LearningRate*(reward+a.Discount*maxNextQ-currentQ)

	// Update Q-value in shared table
	a.manager.UpdateQValue(state, action, newQ)
}

// getBestAction returns the action with highest Q-value for given state
func (a *Agent) getBestAction(state string, numActions int) int {
	qValues := a.manager.GetQValues(state, numActions)

	bestAction := 0
	maxQ := math.Inf(-1)

	for action, qValue := range qValues {
		if qValue > maxQ {
			maxQ = qValue
			bestAction = action
		}
	}

	return bestAction
}

// getMaxQValue returns the maximum Q-value for given state
func (a *Agent) getMaxQValue(state string) float64 {
	qValues := a.manager.GetQValues(state, 4) // Assuming 4 actions for snake

	maxQ := math.Inf(-1)
	for _, qValue := range qValues {
		if qValue > maxQ {
			maxQ = qValue
		}
	}

	return maxQ
}

// AgentState represents the complete state to be saved
type AgentState struct {
	QTable map[string][]float64 `json:"qtable"`
}
