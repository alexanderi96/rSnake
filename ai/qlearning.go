package ai

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
)

type State struct {
	RelativeFoodDir [2]int  // Food direction relative to head (x, y)
	FoodDistance    int     // Manhattan distance to food
	DangerDirs      [4]bool // Danger in each direction (up, right, down, left)
	LastAction      Action  // Previous action taken
	LastDistance    int     // Previous distance to food
}

// NewState creates a new state with initialized values
func NewState(foodDir [2]int, foodDist int, dangers [4]bool) State {
	return State{
		RelativeFoodDir: foodDir,
		FoodDistance:    foodDist,
		DangerDirs:      dangers,
		LastAction:      -1, // No previous action
		LastDistance:    foodDist,
	}
}

type Action int

const (
	Up Action = iota
	Right
	Down
	Left
)

type QTable map[string]map[Action]float64

type QLearning struct {
	QTable       QTable
	LearningRate float64
	Discount     float64
	Epsilon      float64
}

func NewQLearning() *QLearning {
	q := &QLearning{
		QTable:       make(QTable),
		LearningRate: 0.1,
		Discount:     0.9,
		Epsilon:      0.1,
	}

	// Try to load existing Q-table
	if err := q.LoadQTable("qtable.json"); err != nil {
		// If file doesn't exist, use the new Q-table
		q.SaveQTable("qtable.json") // Create initial empty file
	}

	return q
}

// SaveQTable saves the Q-table to a file
func (q *QLearning) SaveQTable(filename string) error {
	// Ensure directory exists
	dir := filepath.Dir(filename)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	// Marshal Q-table to JSON
	data, err := json.MarshalIndent(q.QTable, "", "  ")
	if err != nil {
		return err
	}

	// Write to file
	return os.WriteFile(filename, data, 0644)
}

// LoadQTable loads the Q-table from a file
func (q *QLearning) LoadQTable(filename string) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		return err
	}

	return json.Unmarshal(data, &q.QTable)
}

func (q *QLearning) getStateKey(s State) string {
	return string(rune(s.RelativeFoodDir[0])) + string(rune(s.RelativeFoodDir[1])) +
		string(rune(boolToInt(s.DangerDirs[0]))) +
		string(rune(boolToInt(s.DangerDirs[1]))) +
		string(rune(boolToInt(s.DangerDirs[2]))) +
		string(rune(boolToInt(s.DangerDirs[3])))
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

func (q *QLearning) GetAction(state State) Action {
	// Exploration: random action
	if rand.Float64() < q.Epsilon {
		return Action(rand.Intn(4))
	}

	// Exploitation: best known action
	return q.getBestAction(state)
}

func (q *QLearning) getBestAction(state State) Action {
	stateKey := q.getStateKey(state)
	if _, exists := q.QTable[stateKey]; !exists {
		q.QTable[stateKey] = make(map[Action]float64)
		for action := Up; action <= Left; action++ {
			q.QTable[stateKey][action] = 0
		}
	}

	bestAction := Up
	bestValue := math.Inf(-1)
	for action, value := range q.QTable[stateKey] {
		if value > bestValue {
			bestValue = value
			bestAction = action
		}
	}

	return bestAction
}

func (q *QLearning) Update(state State, action Action, nextState State) float64 {
	stateKey := q.getStateKey(state)
	nextStateKey := q.getStateKey(nextState)

	// Calculate reward based on state transition
	var reward float64

	// Base reward for moving towards/away from food
	distanceChange := nextState.FoodDistance - state.FoodDistance
	if distanceChange < 0 {
		// Got closer to food
		reward = 0.5
	} else if distanceChange > 0 {
		// Got further from food
		reward = -0.3
	}

	// Additional rewards/penalties
	if nextState.FoodDistance == 0 {
		// Found food
		reward = 1.0
	} else if nextState.DangerDirs[action] {
		// Moved towards danger
		reward = -1.0
	}

	// Initialize Q-values if not exists
	if _, exists := q.QTable[stateKey]; !exists {
		q.QTable[stateKey] = make(map[Action]float64)
		for a := Up; a <= Left; a++ {
			q.QTable[stateKey][a] = 0
		}
	}
	if _, exists := q.QTable[nextStateKey]; !exists {
		q.QTable[nextStateKey] = make(map[Action]float64)
		for a := Up; a <= Left; a++ {
			q.QTable[nextStateKey][a] = 0
		}
	}

	// Get max Q-value for next state
	maxNextQ := math.Inf(-1)
	for _, value := range q.QTable[nextStateKey] {
		if value > maxNextQ {
			maxNextQ = value
		}
	}

	// Q-learning update formula
	currentQ := q.QTable[stateKey][action]
	q.QTable[stateKey][action] = currentQ + q.LearningRate*(reward+q.Discount*maxNextQ-currentQ)

	return reward
}
