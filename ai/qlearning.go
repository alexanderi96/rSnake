package ai

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sync"

	"github.com/google/uuid"
)

// Global mutex for file access synchronization
var (
	fileMutex  sync.RWMutex
	tableMutex sync.RWMutex
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
	UUID         string
	QTable       QTable
	LearningRate float64
	Discount     float64
	Epsilon      float64
	TotalReward  float64
	GamesPlayed  int
}

// Breed creates a new QLearning agent from two parents
func Breed(parent1, parent2 *QLearning) *QLearning {
	child := &QLearning{
		UUID:         uuid.New().String(),
		QTable:       make(QTable),
		LearningRate: 0.1,
		Discount:     0.9,
		Epsilon:      0.1,
		TotalReward:  0,
		GamesPlayed:  0,
	}

	// Combine Q-tables from parents randomly
	allStates := make(map[string]bool)
	for state := range parent1.QTable {
		allStates[state] = true
	}
	for state := range parent2.QTable {
		allStates[state] = true
	}

	for state := range allStates {
		child.QTable[state] = make(map[Action]float64)
		for action := Up; action <= Left; action++ {
			// Randomly choose between parents or create a mixed value
			if rand.Float64() < 0.5 {
				if val, ok := parent1.QTable[state][action]; ok {
					child.QTable[state][action] = val
				} else if val, ok := parent2.QTable[state][action]; ok {
					child.QTable[state][action] = val
				}
			} else {
				val1, ok1 := parent1.QTable[state][action]
				val2, ok2 := parent2.QTable[state][action]
				if ok1 && ok2 {
					// Create a mixed value with random mutation
					mix := (val1 + val2) / 2
					mutation := (rand.Float64()*2 - 1) * 0.1 * math.Abs(mix)
					child.QTable[state][action] = mix + mutation
				} else if ok1 {
					child.QTable[state][action] = val1
				} else if ok2 {
					child.QTable[state][action] = val2
				}
			}
		}
	}

	return child
}

func NewQLearning(parentTable QTable, mutationRate float64) *QLearning {
	q := &QLearning{
		UUID:         uuid.New().String(),
		QTable:       make(QTable),
		LearningRate: 0.1,
		Discount:     0.9,
		Epsilon:      0.1,
		TotalReward:  0,
		GamesPlayed:  0,
	}

	if parentTable != nil {
		// Create a mutated copy of the parent table
		q.QTable = q.createMutatedTable(parentTable, mutationRate)
	} else {
		// Try to load existing Q-table if no parent is provided
		filename := GetQTableFilename("", q.UUID) // Empty game UUID for initial creation
		if err := q.LoadQTable(filename); err != nil {
			// If file doesn't exist, use the new Q-table
			q.SaveQTable(filename) // Create initial empty file
		}
	}

	return q
}

// createMutatedTable creates a copy of the parent table with slight mutations
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

// GetQTableFilename returns the filename for a Q-table based on agent ID
func GetQTableFilename(gameUUID, agentUUID string) string {
	return filepath.Join("data", "games", gameUUID, "agents", agentUUID+".json")
}

// SaveQTable saves the Q-table to a file with concurrent access protection
func (q *QLearning) SaveQTable(filename string) error {
	fileMutex.Lock()
	defer fileMutex.Unlock()

	// Ensure directory exists
	dir := filepath.Dir(filename)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	// Load existing Q-table if it exists to merge values
	existingTable := make(QTable)
	if data, err := os.ReadFile(filename); err == nil {
		json.Unmarshal(data, &existingTable)
		q.mergeQTables(existingTable)
	}

	// Marshal Q-table to JSON
	data, err := json.MarshalIndent(q.QTable, "", "  ")
	if err != nil {
		return err
	}

	// Write to file
	return os.WriteFile(filename, data, 0644)
}

// LoadQTable loads the Q-table from a file with concurrent access protection
func (q *QLearning) LoadQTable(filename string) error {
	fileMutex.RLock()
	defer fileMutex.RUnlock()

	data, err := os.ReadFile(filename)
	if err != nil {
		return err
	}

	tableMutex.Lock()
	defer tableMutex.Unlock()
	return json.Unmarshal(data, &q.QTable)
}

// mergeQTables combines the current Q-table with another Q-table
// keeping only the better values
func (q *QLearning) mergeQTables(other QTable) {
	tableMutex.Lock()
	defer tableMutex.Unlock()

	// Only merge if the current performance is better than previous
	if q.TotalReward <= 0 {
		return // Don't save poor performances
	}

	for state, actions := range other {
		if _, exists := q.QTable[state]; !exists {
			q.QTable[state] = make(map[Action]float64)
		}

		for action, value := range actions {
			currentValue, exists := q.QTable[state][action]
			if !exists || value > currentValue {
				// Keep the better value
				q.QTable[state][action] = value
			}
		}
	}
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
	tableMutex.RLock()
	defer tableMutex.RUnlock()

	stateKey := q.getStateKey(state)
	if _, exists := q.QTable[stateKey]; !exists {
		tableMutex.RUnlock()
		tableMutex.Lock()
		// Check again in case another goroutine initialized it
		if _, exists := q.QTable[stateKey]; !exists {
			q.QTable[stateKey] = make(map[Action]float64)
			for action := Up; action <= Left; action++ {
				q.QTable[stateKey][action] = 0
			}
		}
		tableMutex.Unlock()
		tableMutex.RLock()
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
	} else {
		// Got further from food or stayed same distance
		reward = -0.3
	}

	// Additional rewards/penalties
	if nextState.FoodDistance == 0 {
		// Found food
		reward = 1.0
	} else if nextState.DangerDirs[action] {
		// Check if it's a self-collision (danger in the direction we're moving)
		reward = -1.5 // More severe penalty for self-collision
	}

	tableMutex.Lock()
	defer tableMutex.Unlock()

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

	// Update total reward
	q.TotalReward += reward

	// Only save if we have a good performance
	if reward > 0 && q.GamesPlayed%50 == 0 { // Save good performances more frequently
		go func() {
			if err := q.SaveQTable(GetQTableFilename("", q.UUID)); err != nil {
				return // Ignore errors during periodic saves
			}
		}()
	}

	return reward
}
