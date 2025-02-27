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
	// Ottiene i valori dettagliati della direzione del cibo (-1, 0, 1)
	foodAhead, foodLeft, foodRight := sa.game.GetDetailedFoodDirections()

	// Vettore delle direzioni del cibo
	foodDirs := []float64{foodAhead, foodLeft, foodRight}

	// Flag di pericolo immediato
	dangerAhead, dangerLeft, dangerRight := sa.game.GetDangers()
	dangers := []float64{
		boolToFloat64(dangerAhead),
		boolToFloat64(dangerLeft),
		boolToFloat64(dangerRight),
	}

	// Combina i vettori
	return append(foodDirs, dangers...)
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
	sa.game.SetLastAction(action) // Salva l'azione eseguita
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

	if snake.Dead {
		return -1.0 // Penalità fissa per morte
	}
	if snake.Score > oldScore {
		return 1.0 // Reward fisso per cibo
	}

	// Ottiene i valori dettagliati della direzione del cibo per la direzione scelta
	foodAhead, foodLeft, foodRight := sa.game.GetDetailedFoodDirections()

	// Determina quale direzione è stata scelta basandosi sull'azione relativa
	var chosenDirection float64
	switch action := sa.game.GetLastAction(); action {
	case 1: // vai avanti
		chosenDirection = foodAhead
	case 0: // ruota a sinistra
		chosenDirection = foodLeft
	case 2: // ruota a destra
		chosenDirection = foodRight
	}

	// Reward basato sulla direzione scelta
	switch chosenDirection {
	case 1.0:
		return 0.1 // Premio piccolo per aver scelto la via più breve
	case 0.0:
		return 0.0 // Neutrale per una direzione valida ma non ottimale
	default:
		return -0.1 // Penalità piccola per aver scelto una direzione sbagliata
	}
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
