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

	if err := os.MkdirAll("data", 0755); err != nil {
		println("Warning: Could not create data directory:", err)
	}

	if err := q.LoadQTable("data/qtable.json"); err != nil {
		if parentTable != nil {
			q.QTable = q.createMutatedTable(parentTable, mutationRate)
		}
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
	if rand.Float64() < q.Epsilon {
		return Action(rand.Intn(3))
	}
	return q.getBestAction(state)
}

func (q *QLearning) getBestAction(state State) Action {
	q.mutex.RLock()
	defer q.mutex.RUnlock()
	stateKey := q.getStateKey(state)
	if _, exists := q.QTable[stateKey]; !exists {
		q.mutex.RUnlock()
		q.mutex.Lock()
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
	reward := q.calculateReward(state, action, nextState)

	q.mutex.Lock()
	defer q.mutex.Unlock()

	if _, exists := q.QTable[stateKey]; !exists {
		q.QTable[stateKey] = make(map[Action]float64)
		for a := Right; a <= Straight; a++ {
			q.QTable[stateKey][a] = 0.1
		}
	}

	if _, exists := q.QTable[nextStateKey]; !exists {
		q.QTable[nextStateKey] = make(map[Action]float64)
		for a := Right; a <= Straight; a++ {
			q.QTable[nextStateKey][a] = 0
		}
	}

	maxNextQ := math.Inf(-1)
	for _, value := range q.QTable[nextStateKey] {
		if value > maxNextQ {
			maxNextQ = value
		}
	}

	currentQ := q.QTable[stateKey][action]
	q.QTable[stateKey][action] = currentQ + q.LearningRate*(reward+q.Discount*maxNextQ-currentQ)

	return reward
}

func (q *QLearning) calculateReward(state State, action Action, nextState State) float64 {
	reward := 0.0
	distanceChange := nextState.FoodDistance - state.FoodDistance
	if distanceChange < 0 {
		reward += 1.0
	} else if distanceChange > 0 {
		reward -= 0.5
	}
	if nextState.FoodDistance == 0 {
		reward += 2.0
	}
	newDirIndex := absoluteDirection(action, state.CurrentDir)
	if nextState.DangerDirs[newDirIndex] {
		reward -= 1.0
	}
	return reward
}

// absoluteDirection restituisce l'indice della direzione assoluta
// corrispondente all'azione scelta, dato l'attuale direzione del serpente.
func absoluteDirection(action Action, currentDir [2]int) int {
	// Mappa delle direzioni assolute: 0=Su, 1=Destra, 2=Giù, 3=Sinistra
	// Assumiamo che currentDir sia uno di questi vettori:
	// Su: [0, -1], Destra: [1, 0], Giù: [0, 1], Sinistra: [-1, 0]
	var currentIndex int
	switch currentDir {
	case [2]int{0, -1}: // Su
		currentIndex = 0
	case [2]int{1, 0}: // Destra
		currentIndex = 1
	case [2]int{0, 1}: // Giù
		currentIndex = 2
	case [2]int{-1, 0}: // Sinistra
		currentIndex = 3
	default:
		currentIndex = 0 // Default a "Su"
	}

	// Determina la nuova direzione assoluta in base all'azione:
	// Se l'azione è Straight, mantieni la stessa direzione.
	// Se l'azione è Right, ruota di 90° in senso orario.
	// Se l'azione è Left, ruota di 90° in senso antiorario.
	var newIndex int
	switch action {
	case Straight:
		newIndex = currentIndex
	case Right:
		newIndex = (currentIndex + 1) % 4
	case Left:
		newIndex = (currentIndex + 3) % 4 // equivalente a -1 modulo 4
	default:
		newIndex = currentIndex
	}
	return newIndex
}

func NewState(relativeFoodDir [2]int, foodDistance int, dangerDirs [4]bool, currentDir [2]int) State {
	return State{
		RelativeFoodDir: relativeFoodDir,
		FoodDistance:    foodDistance,
		DangerDirs:      dangerDirs,
		CurrentDir:      currentDir,
	}
}
