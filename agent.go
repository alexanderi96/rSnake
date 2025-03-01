package main

import (
	"fmt"
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
	agent := qlearning.NewAgent(0.5, 0.8, 0.95)
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
	baseReward := -0.005 // Penalità minima per step

	// Morte - penalità significativa ma non eccessiva
	if snake.Dead {
		return -2.0
	}

	// Cibo mangiato - premio base più contenuto ma con bonus per lunghezza
	if snake.Score > oldScore {
		baseReward = 5.0
		baseReward += float64(snake.Score) * 0.2
	}

	// Analisi stato
	state := sa.game.GetStateInfo()
	walls := state[0:8]  // Primi 8 valori: muri
	body := state[8:16]  // Secondi 8 valori: corpo
	food := state[16:24] // Ultimi 8 valori: cibo

	// Indici per la direzione scelta
	directionIndex := 3 // front
	switch chosenAction {
	case 0: // sinistra
		directionIndex = 1
	case 2: // destra
		directionIndex = 5
	}

	// 1. Vicinanza al cibo (0.0 - 1.0) * 1.5
	foodReward := food[directionIndex] * 1.5

	// 2. Pericoli (-2.0 - 0.0)
	wallDanger := walls[directionIndex] * -1.0
	bodyDanger := body[directionIndex] * -1.0
	dangerPenalty := wallDanger + bodyDanger

	// 3. Spazio libero circostante (bonus per aree aperte)
	freeSpace := 0.0
	for i := 0; i < 8; i++ {
		if walls[i] == 0 && body[i] == 0 {
			freeSpace += 0.1
		}
	}

	// Calcola il reward base
	baseReward += foodReward + dangerPenalty + freeSpace

	// Scala il reward basato sulla qualità dell'azione
	bestQValue := qValues[0]
	worstQValue := qValues[0]
	for _, qValue := range qValues {
		if qValue > bestQValue {
			bestQValue = qValue
		}
		if qValue < worstQValue {
			worstQValue = qValue
		}
	}

	chosenQValue := qValues[chosenAction]

	// Se c'è una differenza significativa tra le azioni
	if bestQValue != worstQValue {
		// Normalizza il Q-value dell'azione scelta
		normalizedQValue := (chosenQValue - worstQValue) / (bestQValue - worstQValue)

		// Scala il reward:
		// - Se è la migliore azione (normalizedQValue = 1): moltiplica per 1.2
		// - Se è la peggiore azione (normalizedQValue = 0): moltiplica per 0.8
		// - Per azioni intermedie: scala proporzionalmente
		scaleFactor := 0.8 + (0.4 * normalizedQValue)
		baseReward *= scaleFactor
	}

	return baseReward
}

// GetEpsilon returns the current epsilon value
func (sa *SnakeAgent) GetEpsilon() float64 {
	return sa.agent.Epsilon
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
