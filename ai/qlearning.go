package ai

import (
	"encoding/json"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"time"

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
	CurrentDir      [2]int  // Current direction of movement (x, y)
}

// NewState creates a new state with initialized values
func NewState(foodDir [2]int, foodDist int, dangers [4]bool, currentDir [2]int) State {
	return State{
		RelativeFoodDir: foodDir,
		FoodDistance:    foodDist,
		DangerDirs:      dangers,
		LastAction:      -1, // No previous action
		LastDistance:    foodDist,
		CurrentDir:      currentDir,
	}
}

type Action int

const (
	Forward Action = iota
	ForwardRight
	ForwardLeft
)

type QTable map[string]map[Action]float64

// Copy creates a deep copy of the QTable
func (qt QTable) Copy() QTable {
	newTable := make(QTable)
	for state, actions := range qt {
		newTable[state] = make(map[Action]float64)
		for action, value := range actions {
			newTable[state][action] = value
		}
	}
	return newTable
}

type QLearning struct {
	UUID               string
	QTable             QTable
	LearningRate       float64
	Discount           float64
	Epsilon            float64
	TotalReward        float64
	GamesPlayed        int
	MutationEfficiency float64
	LastMutationTime   time.Time
	Parents            []string
	Generation         int
	Fitness            float64
	MutationHistory    []MutationRecord
}

type MutationRecord struct {
	Timestamp time.Time `json:"timestamp"`
	StateKey  string    `json:"state_key"`
	Action    Action    `json:"action"`
	OldValue  float64   `json:"old_value"`
	NewValue  float64   `json:"new_value"`
	Reward    float64   `json:"reward"`
	Effective bool      `json:"effective"`
}

// Breed creates a new QLearning agent from two parents
func Breed(parent1, parent2 *QLearning) *QLearning {
	// Calculate new generation number
	childGeneration := max(parent1.Generation, parent2.Generation) + 1

	child := &QLearning{
		UUID:         uuid.New().String(),
		QTable:       make(QTable),
		LearningRate: 0.1,
		Discount:     0.9,
		Epsilon:      0.1,
		TotalReward:  0,
		GamesPlayed:  0,
		// Generation:   1, // Initialize first generation
		Generation: childGeneration,
		Parents:    []string{parent1.UUID, parent2.UUID},
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
		for action := Forward; action <= ForwardLeft; action++ {
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
	totalMutations := 0
	effectiveMutations := 0

	for state, actions := range parentTable {
		newTable[state] = make(map[Action]float64)
		for action, value := range actions {
			oldValue := value
			// Add a small random mutation
			mutation := (rand.Float64()*2 - 1) * mutationRate * math.Abs(value)
			newValue := value + mutation
			newTable[state][action] = newValue

			totalMutations++
			// Consider a mutation effective if it changes the value significantly
			if math.Abs(newValue-oldValue) > 0.1*math.Abs(oldValue) {
				effectiveMutations++
				q.MutationHistory = append(q.MutationHistory, MutationRecord{
					Timestamp: time.Now(),
					StateKey:  state,
					Action:    action,
					OldValue:  oldValue,
					NewValue:  newValue,
					Effective: true,
				})
			}
		}
	}

	// Update mutation efficiency
	if totalMutations > 0 {
		q.MutationEfficiency = float64(effectiveMutations) / float64(totalMutations)
	}
	q.LastMutationTime = time.Now()

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

// Mutate applies random mutations to the Q-table values
func (q *QLearning) Mutate(mutationRate float64) {
	tableMutex.Lock()
	defer tableMutex.Unlock()

	for state, actions := range q.QTable {
		for action, value := range actions {
			// Add a small random mutation
			mutation := (rand.Float64()*2 - 1) * mutationRate * math.Abs(value)
			q.QTable[state][action] = value + mutation
		}
	}
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
		string(rune(boolToInt(s.DangerDirs[3]))) +
		string(rune(s.CurrentDir[0])) + string(rune(s.CurrentDir[1]))
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func (q *QLearning) GetAction(state State) Action {
	// Exploration: random action
	if rand.Float64() < q.Epsilon {
		return Action(rand.Intn(3))
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
			for action := Forward; action <= ForwardLeft; action++ {
				q.QTable[stateKey][action] = 0
			}
		}
		tableMutex.Unlock()
		tableMutex.RLock()
	}

	bestAction := Forward
	bestValue := math.Inf(-1)
	for action, value := range q.QTable[stateKey] {
		if value > bestValue {
			bestValue = value
			bestAction = action
		}
	}

	return bestAction
}

func (q *QLearning) GetQTable() QTable {
	tableMutex.RLock()
	defer tableMutex.RUnlock()
	return q.QTable.Copy()
}

func (q *QLearning) Update(state State, action Action, nextState State) float64 {
	// Adaptive learning rate based on performance
	if q.TotalReward > 0 {
		// Decrease learning rate as performance improves
		q.LearningRate = math.Max(0.01, q.LearningRate*0.999)
	} else {
		// Increase learning rate if performance is poor
		q.LearningRate = math.Min(0.5, q.LearningRate*1.001)
	}
	stateKey := q.getStateKey(state)
	nextStateKey := q.getStateKey(nextState)

	// Enhanced reward calculation
	reward := q.calculateReward(state, action, nextState)

	// No need for direction penalty since we only allow forward movements

	// Track mutation effectiveness
	if len(q.MutationHistory) > 0 {
		lastMutation := &q.MutationHistory[len(q.MutationHistory)-1]
		if !lastMutation.Effective && reward > 0 {
			lastMutation.Effective = true
			lastMutation.Reward = reward
		}
	}

	tableMutex.Lock()
	defer tableMutex.Unlock()

	// Initialize Q-values if not exists with smart initialization
	if _, exists := q.QTable[stateKey]; !exists {
		q.QTable[stateKey] = make(map[Action]float64)
		for a := Forward; a <= ForwardLeft; a++ {
			// Initialize based on danger directions
			if nextState.DangerDirs[a] {
				q.QTable[stateKey][a] = -0.5 // Start with negative value for dangerous actions
			} else {
				q.QTable[stateKey][a] = 0.1 // Small positive value for safe actions
			}
		}
	}
	if _, exists := q.QTable[nextStateKey]; !exists {
		q.QTable[nextStateKey] = make(map[Action]float64)
		for a := Forward; a <= ForwardLeft; a++ {
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

	// Update total reward and fitness
	q.TotalReward += reward
	q.Fitness = q.calculateFitness()

	// Periodic save with performance tracking
	if (reward > 0 && q.GamesPlayed%50 == 0) || q.Fitness > q.previousBestFitness() {
		go func() {
			if err := q.SaveQTable(GetQTableFilename("", q.UUID)); err != nil {
				log.Printf("Failed to save Q-table: %v", err)
				return
			}
		}()
	}

	return reward
}

func (q *QLearning) calculateReward(state State, action Action, nextState State) float64 {
	var reward float64

	// Base reward for moving towards/away from food
	distanceChange := nextState.FoodDistance - state.FoodDistance
	if distanceChange < 0 {
		// Got closer to food
		reward = 0.5 + float64(state.FoodDistance-nextState.FoodDistance)*0.1 // Additional reward based on distance improvement
	} else {
		// Got further from food or stayed same distance
		reward = -0.3 - float64(nextState.FoodDistance-state.FoodDistance)*0.05 // Increased penalty for getting further
	}

	// Additional rewards/penalties
	if nextState.FoodDistance == 0 {
		// Found food
		reward = 1.0 + float64(q.GamesPlayed)/1000 // Bonus increases with experience
	} else if nextState.DangerDirs[action] {
		// Check if it's a self-collision
		reward = -1.5
		// Additional penalty if it's a repeated mistake
		if q.isRepeatedMistake(state, action) {
			reward *= 1.2
		}
	}

	// Bonus for exploring new states
	if _, exists := q.QTable[q.getStateKey(state)]; !exists {
		reward += 0.1 // Small bonus for exploration
	}

	return reward
}

func (q *QLearning) calculateFitness() float64 {
	// Combine multiple factors for fitness calculation
	avgReward := q.TotalReward / float64(q.GamesPlayed+1)
	survivalRate := 1.0 - float64(len(q.MutationHistory))/float64(q.GamesPlayed+1)
	learningProgress := q.LearningRate * q.MutationEfficiency

	return avgReward*0.5 + survivalRate*0.3 + learningProgress*0.2
}

func (q *QLearning) previousBestFitness() float64 {
	filename := GetQTableFilename("", q.UUID)
	if data, err := os.ReadFile(filename); err == nil {
		var prevTable QTable
		if err := json.Unmarshal(data, &prevTable); err == nil {
			// Simple heuristic: sum of positive Q-values
			var sum float64
			for _, actions := range prevTable {
				for _, value := range actions {
					if value > 0 {
						sum += value
					}
				}
			}
			return sum
		}
	}
	return 0
}

func (q *QLearning) isRepeatedMistake(state State, action Action) bool {
	stateKey := q.getStateKey(state)
	count := 0
	for _, record := range q.MutationHistory {
		if record.StateKey == stateKey && record.Action == action && !record.Effective {
			count++
		}
	}
	return count >= 3 // Consider it a repeated mistake if it happened 3 or more times
}
