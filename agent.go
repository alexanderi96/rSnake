package main

import (
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

// getState restituisce il vettore di stato semplificato come []float64
func (sa *SnakeAgent) getState() []float64 {
	// Ottiene la direzione relativa del cibo rispetto alla direzione corrente
	foodDir := float64(sa.game.GetRelativeFoodDirection())

	// Distanze dai pericoli nelle varie direzioni
	distAhead, distLeft, distRight := sa.game.GetDangers()

	// Stato semplificato: solo direzione cibo e pericoli
	return []float64{
		foodDir,            // direzione relativa del cibo (0-3)
		float64(distAhead), // distanza ostacoli avanti (0-5)
		float64(distLeft),  // distanza ostacoli sinistra (0-5)
		float64(distRight), // distanza ostacoli destra (0-5)
	}
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
	action := sa.agent.GetAction(currentState, 3)
	newDir := sa.relativeActionToAbsolute(action)

	// Applica l'azione e calcola il reward
	oldScore := sa.game.GetSnake().Score
	sa.game.GetSnake().SetDirection(newDir)
	sa.game.Update()

	// Sistema di reward semplificato
	reward := sa.calculateReward(oldScore)

	// Aggiorna i Q-values
	newState := sa.getState()
	sa.agent.Update(currentState, action, reward, newState, 3)
}

func (sa *SnakeAgent) calculateReward(oldScore int) float64 {
	snake := sa.game.GetSnake()

	// Reward principale per aver mangiato il cibo
	if snake.Score > oldScore {
		return 1.0 + float64(snake.Score)*0.1 // Bonus progressivo base
	}

	// Penalità per morte
	if snake.Dead {
		return -1.0
	}

	// Piccolo reward per sopravvivenza
	return 0.01
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

	sa.game = NewGame(width, height)
	sa.game.Stats = existingStats
	sa.agent.IncrementEpisode()
}

// SaveWeights salva i pesi della rete neurale su file.
func (sa *SnakeAgent) SaveWeights() error {
	return sa.agent.SaveWeights(qlearning.WeightsFile)
}
