package main

import (
	"snake-game/qlearning"
)

// SnakeAgent rappresenta l'agente che gioca a Snake usando Q-learning.
type SnakeAgent struct {
	agent            *qlearning.Agent
	game             *Game
	previousDistance int // Per il reward shaping basato sulla direzione
}

// NewSnakeAgent crea un nuovo agente per il gioco.
func NewSnakeAgent(game *Game) *SnakeAgent {
	agent := qlearning.NewAgent(0.5, 0.8, 0.95)
	sa := &SnakeAgent{
		agent: agent,
		game:  game,
	}
	sa.previousDistance = sa.getManhattanDistance() // Inizializza la distanza iniziale
	return sa
}

// getManhattanDistance calcola la distanza Manhattan tra la testa del serpente e il cibo
func (sa *SnakeAgent) getManhattanDistance() int {
	head := sa.game.GetSnake().Body[0]
	food := sa.game.food
	return abs(head.X-food.X) + abs(head.Y-food.Y)
}

// getState restituisce il vettore di stato semplificato come []float64
func (sa *SnakeAgent) getState() []float64 {
	// One-hot encoding per la direzione del cibo
	foodDir := sa.game.GetRelativeFoodDirection()
	foodDirOneHot := make([]float64, 4)
	foodDirOneHot[foodDir] = 1.0

	// Flag di pericolo immediato
	dangerAhead, dangerLeft, dangerRight := sa.game.GetDangers()

	// Converte i bool in float64
	dangers := []float64{
		boolToFloat64(dangerAhead),
		boolToFloat64(dangerLeft),
		boolToFloat64(dangerRight),
	}

	// Combina i vettori
	return append(foodDirOneHot, dangers...)
}

// boolToFloat64 converte un bool in float64
func boolToFloat64(b bool) float64 {
	if b {
		return 1.0
	}
	return 0.0
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
	currentDistance := sa.getManhattanDistance()

	// Penalità forte per morte
	if snake.Dead {
		return -10.0
	}

	// Reward fortemente aumentato per aver mangiato il cibo
	if snake.Score > oldScore {
		sa.previousDistance = currentDistance // Reset della distanza dopo aver mangiato
		return 10.0                           // Aumentato da 5.0 a 10.0
	}

	// Reward shaping basato sulla direzione e distanza
	reward := -0.01 // Piccola penalità base per ogni step

	// Aggiusta il reward in base alla direzione con penalità aumentata
	if currentDistance < sa.previousDistance {
		reward += 0.3 // Bonus per avvicinamento al cibo
	} else if currentDistance > sa.previousDistance {
		reward -= 0.3 // Penalità aumentata per allontanamento dal cibo (da -0.1 a -0.3)
	}

	// Aggiungi reward basato sulla distanza assoluta
	// Usa una funzione inversamente proporzionale alla distanza
	// Normalizzata rispetto alla dimensione della griglia per mantenere i valori in scala
	gridSize := float64(sa.game.Grid.Width + sa.game.Grid.Height)
	distanceReward := 0.2 * (1.0 - float64(currentDistance)/gridSize)
	reward += distanceReward

	sa.previousDistance = currentDistance
	return reward
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
	sa.previousDistance = sa.getManhattanDistance() // Inizializza la distanza per il nuovo episodio
	sa.agent.IncrementEpisode()
}

// SaveWeights salva i pesi della rete neurale su file.
func (sa *SnakeAgent) SaveWeights() error {
	return sa.agent.SaveWeights(qlearning.WeightsFile)
}
