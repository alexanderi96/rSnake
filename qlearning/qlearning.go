package qlearning

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
)

// In qlearning.go
const (
	InitialEpsilon = 0.95
	MinEpsilon     = 0.1   // Mantiene più esplorazione
	EpsilonDecay   = 0.995 // Decay più lento

	DataDir    = "data"
	QtableFile = DataDir + "/qtable.json"
)

// QTable memorizza i valori Q per coppie stato-azione.
type QTable map[string][]float64

// Agent rappresenta un agente di Q-learning.
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

// NewAgent crea un nuovo agente di Q-learning.
func NewAgent(learningRate, discount, epsilon float64) *Agent {
	agent := &Agent{
		QTable:          make(QTable),
		LearningRate:    learningRate,
		Discount:        discount,
		Epsilon:         InitialEpsilon, // Partenza con alta esplorazione
		InitialEpsilon:  InitialEpsilon,
		MinEpsilon:      MinEpsilon,
		EpsilonDecay:    EpsilonDecay, // Decay rallentato: ad esempio, dopo 100 episodi, epsilon decadrà meno
		TrainingEpisode: 0,
	}

	// Prova a caricare uno stato esistente
	if err := agent.LoadQTable(QtableFile); err == nil {
		// Aggiorna i parametri di apprendimento, mantenendo lo stato caricato
		agent.LearningRate = learningRate
		agent.Discount = discount
		agent.InitialEpsilon = InitialEpsilon
		agent.MinEpsilon = MinEpsilon
		agent.EpsilonDecay = EpsilonDecay
	}

	return agent
}

// GetAction seleziona un'azione usando una politica epsilon-greedy.
func (a *Agent) GetAction(state string, numActions int) int {
	// Aggiorna epsilon con il decadimento
	if a.Epsilon > a.MinEpsilon {
		a.Epsilon = a.InitialEpsilon * math.Pow(a.EpsilonDecay, float64(a.TrainingEpisode))
		if a.Epsilon < a.MinEpsilon {
			a.Epsilon = a.MinEpsilon
		}
	}

	// Esplorazione: azione casuale
	if rand.Float64() < a.Epsilon {
		return rand.Intn(numActions)
	}

	// Sfruttamento: azione con il massimo valore Q
	return a.getBestAction(state, numActions)
}

// IncrementEpisode incrementa il contatore degli episodi di training.
func (a *Agent) IncrementEpisode() {
	a.TrainingEpisode++
	// Decadimento più rapido iniziale, poi più lento
	if a.TrainingEpisode < 1000 {
		a.Epsilon = math.Max(0.1, a.InitialEpsilon*math.Exp(-float64(a.TrainingEpisode)/500))
	} else {
		a.Epsilon = math.Max(0.05, a.InitialEpsilon*math.Exp(-float64(a.TrainingEpisode)/1000))
	}
}

// Update aggiorna il valore Q per una coppia stato-azione.
func (a *Agent) Update(state string, action int, reward float64, nextState string, numActions int) {
	// Inizializza i valori Q se lo stato non è stato ancora visto
	if _, exists := a.QTable[state]; !exists {
		a.QTable[state] = make([]float64, numActions)
	}
	if _, exists := a.QTable[nextState]; !exists {
		a.QTable[nextState] = make([]float64, numActions)
	}

	// Formula di aggiornamento del Q-learning:
	// Q(s,a) = Q(s,a) + α [r + γ * max_a' Q(s',a') - Q(s,a)]
	currentQ := a.QTable[state][action]
	maxNextQ := a.getMaxQValue(nextState)
	a.QTable[state][action] = currentQ + a.LearningRate*(reward+a.Discount*maxNextQ-currentQ)
}

// getBestAction restituisce l'azione con il valore Q più alto per uno stato dato.
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

// getMaxQValue restituisce il massimo valore Q per uno stato dato.
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

// AgentState rappresenta lo stato completo dell'agente da salvare.
type AgentState struct {
	QTable          QTable  `json:"qtable"`
	Epsilon         float64 `json:"epsilon"`
	TrainingEpisode int     `json:"training_episode"`
}

// SaveQTable salva lo stato dell'agente su un file.
func (a *Agent) SaveQTable(filename string) error {
	// Ensure data directory exists
	if err := os.MkdirAll(DataDir, 0755); err != nil {
		return fmt.Errorf("failed to create data directory: %v", err)
	}

	state := AgentState{
		QTable:          a.QTable,
		Epsilon:         a.Epsilon,
		TrainingEpisode: a.TrainingEpisode,
	}

	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("error marshaling QTable: %v", err)
	}

	if err := os.WriteFile(filename, data, 0644); err != nil {
		return fmt.Errorf("error writing QTable to file: %v", err)
	}

	return nil
}

// LoadQTable carica lo stato dell'agente da un file.
func (a *Agent) LoadQTable(filename string) error {
	if a.QTable == nil {
		a.QTable = make(QTable)
	}

	data, err := os.ReadFile(filename)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // Se il file non esiste, si usa la QTable vuota
		}
		return fmt.Errorf("error reading QTable file: %v", err)
	}

	var state AgentState
	err = json.Unmarshal(data, &state)
	if err != nil {
		return fmt.Errorf("error unmarshaling QTable: %v", err)
	}

	if state.QTable != nil {
		a.QTable = state.QTable
		a.Epsilon = state.Epsilon
		a.TrainingEpisode = state.TrainingEpisode
	}

	return nil
}
