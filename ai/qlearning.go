package ai

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"sync"
)

type State struct {
	RelativeFoodDir [2]int  // Food direction relative to head (x, y)
	FoodDistance    int     // Manhattan distance to food
	DangerDirs      [4]bool // Danger in each direction (up, right, down, left)
	CurrentDir      [2]int  // Current direction of movement (x, y)
}

func NewState(foodDir [2]int, foodDist int, dangers [4]bool, currentDir [2]int) State {
	return State{
		RelativeFoodDir: foodDir,
		FoodDistance:    foodDist,
		DangerDirs:      dangers,
		CurrentDir:      currentDir,
	}
}

type Action int

const (
	Right Action = iota
	Left
	Straight
)

type QTable map[string]map[Action]float64

type QLearning struct {
	QTable       QTable
	LearningRate float64
	Discount     float64
	Epsilon      float64
	mutex        sync.RWMutex
}

func NewQLearning(parentTable QTable, mutationRate float64) *QLearning {
	q := &QLearning{
		QTable:       make(QTable),
		LearningRate: 0.1,
		Discount:     0.9,
		Epsilon:      0.1,
	}

	// Create data directory if it doesn't exist
	if err := os.MkdirAll("data", 0755); err != nil {
		// If we can't create the directory, just log the error and continue
		println("Warning: Could not create data directory:", err)
	}

	// Try to load existing Q-table
	if err := q.LoadQTable("data/qtable.json"); err != nil {
		// If loading fails, use parent table if available
		if parentTable != nil {
			q.QTable = q.createMutatedTable(parentTable, mutationRate)
		}
		// If no parent table and no saved table, q.QTable is already initialized as empty map
	}

	return q
}

func (q *QLearning) SaveQTable(filename string) error {
	q.mutex.RLock()
	defer q.mutex.RUnlock()

	data, err := json.MarshalIndent(q.QTable, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(filename, data, 0644)
}

func (q *QLearning) LoadQTable(filename string) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		return err
	}

	q.mutex.Lock()
	defer q.mutex.Unlock()

	return json.Unmarshal(data, &q.QTable)
}

func (q *QLearning) createMutatedTable(parentTable QTable, mutationRate float64) QTable {
	newTable := make(QTable)
	for state, actions := range parentTable {
		newTable[state] = make(map[Action]float64)
		for action, value := range actions {
			// Add a small random mutation
			mutation := (rand.Float64()*2 - 1) * mutationRate * math.Abs(value)
			newTable[state][action] = value + mutation
		}
	}
	return newTable
}

func (q *QLearning) getStateKey(s State) string {
	return string(rune(s.RelativeFoodDir[0])) + string(rune(s.RelativeFoodDir[1])) +
		string(rune(boolToInt(s.DangerDirs[0]))) +
		string(rune(boolToInt(s.DangerDirs[1]))) +
		string(rune(boolToInt(s.DangerDirs[2]))) +
		string(rune(boolToInt(s.DangerDirs[3]))) +
		string(rune(s.CurrentDir[0])) + string(rune(s.CurrentDir[1]))
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
		return Action(rand.Intn(3)) // Right, Left, or Straight
	}

	// Exploitation: best known action
	return q.getBestAction(state)
}

func (q *QLearning) getBestAction(state State) Action {
	q.mutex.RLock()
	defer q.mutex.RUnlock()

	stateKey := q.getStateKey(state)
	if _, exists := q.QTable[stateKey]; !exists {
		q.mutex.RUnlock()
		q.mutex.Lock()
		// Check again in case another goroutine initialized it
		if _, exists := q.QTable[stateKey]; !exists {
			q.QTable[stateKey] = make(map[Action]float64)
			for action := Right; action <= Straight; action++ {
				q.QTable[stateKey][action] = 0
			}
		}
		q.mutex.Unlock()
		q.mutex.RLock()
	}

	bestAction := Right
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

	// Calculate reward
	reward := q.calculateReward(state, action, nextState)

	q.mutex.Lock()
	defer q.mutex.Unlock()

	// Initialize Q-values if not exists
	if _, exists := q.QTable[stateKey]; !exists {
		q.QTable[stateKey] = make(map[Action]float64)
		for a := Right; a <= Straight; a++ {
			if a < 2 && nextState.DangerDirs[a] { // Only check dangers for Right/Left
				q.QTable[stateKey][a] = -0.5 // Start with negative value for dangerous actions
			} else {
				q.QTable[stateKey][a] = 0.1 // Small positive value for safe actions
			}
		}
	}
	if _, exists := q.QTable[nextStateKey]; !exists {
		q.QTable[nextStateKey] = make(map[Action]float64)
		for a := Right; a <= Straight; a++ {
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

func (q *QLearning) calculateReward(state State, action Action, nextState State) float64 {
	var reward float64

	// Moving closer/farther from food
	distanceChange := nextState.FoodDistance - state.FoodDistance
	if distanceChange < 0 {
		// Moving closer to food
		reward += 1.0
	} else if distanceChange > 0 {
		// Moving away from food
		reward -= 0.5
	}

	// Getting food
	if nextState.FoodDistance == 0 {
		reward += 2.0
	}

	// Death penalty
	if nextState.DangerDirs[action] {
		reward -= 1.0
	}

	return reward
}
