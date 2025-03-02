package main

import (
	"fmt"
	"math"
	"math/rand"
	"snake-game/qlearning"
)

// SnakeAgent rappresenta l'agente che gioca a Snake usando Q-learning.
type SnakeAgent struct {
	agent *qlearning.Agent
	game  *Game
}

// NewSnakeAgent crea un nuovo agente per il gioco.
func NewSnakeAgent(game *Game) *SnakeAgent {
	agent := qlearning.NewAgent(0.5, 0.8) // learningRate, discount
	return &SnakeAgent{
		agent: agent,
		game:  game,
	}
}

// getManhattanDistance calcola la distanza Manhattan tra la testa del serpente e il cibo
func (sa *SnakeAgent) getManhattanDistance() int {
	head := sa.game.GetSnake().Body[0]
	food := sa.game.food
	return abs(head.X-food.X) + abs(head.Y-food.Y)
}

// getState restituisce il vettore di stato come matrice 3x8 appiattita
func (sa *SnakeAgent) getState() []float64 {
	// Ottiene la matrice 3x8 appiattita [muri][corpo][cibo] x [backLeft, left, frontLeft, front, frontRight, right, backRight, back]
	return sa.game.GetStateInfo()
}

// relativeActionToAbsolute converte un'azione relativa in una direzione assoluta.
// Le azioni relative sono definite come:
//
//	0: ruota a sinistra
//	1: vai avanti
//	2: ruota a destra
func (sa *SnakeAgent) relativeActionToAbsolute(relativeAction int) Point {
	currentDir := sa.game.GetCurrentDirection()
	var newDir Direction
	switch relativeAction {
	case 0: // ruota a sinistra
		newDir = currentDir.TurnLeft()
	case 1: // vai avanti
		newDir = currentDir
	case 2: // ruota a destra
		newDir = currentDir.TurnRight()
	default:
		newDir = currentDir // fallback
	}
	return newDir.ToPoint()
}

// Update esegue un passo di decisione e aggiornamento Q-learning.
func (sa *SnakeAgent) Update() {
	if sa.game.GetSnake().Dead {
		return
	}

	currentState := sa.getState()
	// Get Q-values for all actions before choosing one
	qValues, err := sa.agent.GetQValues(currentState)
	var action int

	if err != nil {
		// Se c'è un errore nel calcolo dei Q-values, fai una scelta casuale
		action = rand.Intn(3)
		qValues = make([]float64, 3) // Q-values nulli per scaling reward
		fmt.Printf("Errore nel calcolo Q-values: %v. Usando azione casuale: %d\n", err, action)
	} else {
		action = sa.agent.GetAction(currentState, 3)
	}
	newDir := sa.relativeActionToAbsolute(action)

	// Applica l'azione e calcola il reward
	oldScore := sa.game.GetSnake().Score
	sa.game.SetLastAction(action) // Salva l'azione eseguita
	sa.game.GetSnake().SetDirection(newDir)
	sa.game.Update()

	// Sistema di reward con scaling basato sulla qualità dell'azione
	reward := sa.calculateReward(oldScore, action, qValues)

	// Aggiorna i Q-values
	newState := sa.getState()
	sa.agent.Update(currentState, action, reward, newState, 3)
}

func (sa *SnakeAgent) calculateReward(oldScore int, chosenAction int, qValues []float64) float64 {
	snake := sa.game.GetSnake()

	// Reward base semplificato
	baseReward := -0.01 // Piccola penalità per step per incoraggiare movimento efficiente

	// Morte
	if snake.Dead {
		return -1.0
	}

	// Cibo mangiato
	if snake.Score > oldScore {
		baseReward = 1.0
	}

	// Calcola entropia della policy per incoraggiare esplorazione
	policy := sa.softmax(qValues, 0.1) // temperatura bassa per policy più sharp
	entropy := sa.calculateEntropy(policy)

	// Bonus di esplorazione basato sull'entropia
	explorationBonus := 0.1 * entropy

	// Reward finale
	return baseReward + explorationBonus
}

// softmax converte Q-values in una distribuzione di probabilità
func (sa *SnakeAgent) softmax(qValues []float64, temperature float64) []float64 {
	policy := make([]float64, len(qValues))
	maxQ := qValues[0]
	for _, q := range qValues {
		if q > maxQ {
			maxQ = q
		}
	}

	var sum float64
	for i, q := range qValues {
		policy[i] = math.Exp((q - maxQ) / temperature)
		sum += policy[i]
	}

	for i := range policy {
		policy[i] /= sum
	}
	return policy
}

// calculateEntropy calcola l'entropia della policy
func (sa *SnakeAgent) calculateEntropy(policy []float64) float64 {
	var entropy float64
	for _, p := range policy {
		if p > 0 {
			entropy -= p * math.Log(p)
		}
	}
	return entropy
}

// Reset prepara l'agente per una nuova partita mantenendo le conoscenze apprese.
func (sa *SnakeAgent) Reset() {
	existingStats := sa.game.Stats
	width := sa.game.Grid.Width
	height := sa.game.Grid.Height

	// Cleanup existing game resources
	if sa.game != nil {
		sa.game = nil
	}

	// Create new game
	sa.game = NewGame(width, height)
	sa.game.Stats = existingStats
	sa.agent.IncrementEpisode()
}

// Cleanup releases all resources used by the agent
func (sa *SnakeAgent) Cleanup() {
	if sa.agent != nil {
		sa.agent.Cleanup()
		sa.agent = nil
	}
	if sa.game != nil {
		sa.game = nil
	}
}

// SaveWeights salva i pesi della rete neurale su file.
func (sa *SnakeAgent) SaveWeights() error {
	return sa.agent.SaveWeights(qlearning.WeightsFile)
}
