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

// isFoodInDangerDirection verifica se il cibo è in una direzione pericolosa
func isFoodInDangerDirection(foodAhead, foodLeft, foodRight float64,
	dangerAhead, dangerLeft, dangerRight bool) bool {
	return (foodAhead == 1.0 && dangerAhead) ||
		(foodLeft == 1.0 && dangerLeft) ||
		(foodRight == 1.0 && dangerRight)
}

func (sa *SnakeAgent) calculateReward(oldScore int) float64 {
	snake := sa.game.GetSnake()
	reward := -0.01 // Piccola penalità per ogni step per incentivare percorsi più brevi

	// Morte
	if snake.Dead {
		return -3.0 // Aumentata penalità per la morte
	}

	// Cibo mangiato (reward proporzionale alla lunghezza)
	if snake.Score > oldScore {
		baseReward := 2.0
		lengthBonus := float64(snake.Score) * 0.1 // Bonus crescente con la lunghezza
		return baseReward + lengthBonus
	}

	// Ottiene informazioni sulla direzione del cibo e sui pericoli
	foodAhead, foodLeft, foodRight := sa.game.GetDetailedFoodDirections()
	dangerAhead, dangerLeft, dangerRight := sa.game.GetDangers()

	// Determina la direzione scelta e il relativo pericolo
	var chosenFoodDir float64
	var chosenDanger bool

	switch action := sa.game.GetLastAction(); action {
	case 1: // avanti
		chosenFoodDir = foodAhead
		chosenDanger = dangerAhead
	case 0: // sinistra
		chosenFoodDir = foodLeft
		chosenDanger = dangerLeft
	case 2: // destra
		chosenFoodDir = foodRight
		chosenDanger = dangerRight
	}

	// Caso speciale: il cibo è in una direzione pericolosa
	if isFoodInDangerDirection(foodAhead, foodLeft, foodRight, dangerAhead, dangerLeft, dangerRight) {
		if !chosenDanger {
			reward += 0.8 // Aumentato premio per aver evitato il pericolo
		}
	}

	// Penalità per aver scelto una direzione pericolosa
	if chosenDanger {
		reward -= 1.5 // Aumentata penalità per scelte pericolose
	}

	// Reward basato sulla direzione del cibo scelta
	switch chosenFoodDir {
	case 1.0: // Direzione ottimale verso il cibo
		reward += 1.0
	case 0.0: // Direzione neutra
		reward += 0.0
	default: // Direzione che allontana dal cibo
		reward -= 0.5
	}

	// Reward basato sulla distanza dal cibo
	oldDist := sa.getManhattanDistance()
	newDist := abs(snake.Body[0].X-sa.game.food.X) + abs(snake.Body[0].Y-sa.game.food.Y)
	if newDist < oldDist {
		reward += 0.3 // Premio per avvicinarsi al cibo
	} else if newDist > oldDist {
		reward -= 0.2 // Penalità per allontanarsi dal cibo
	}

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
