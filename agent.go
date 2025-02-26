package main

import (
	"math"

	"snake-game/qlearning"
)

// SnakeAgent rappresenta l'agente che gioca a Snake usando Q-learning.
type SnakeAgent struct {
	agent        *qlearning.Agent
	game         *Game
	foodReward   float64
	deathPenalty float64
}

// NewSnakeAgent crea un nuovo agente per il gioco.
func NewSnakeAgent(game *Game) *SnakeAgent {
	agent := qlearning.NewAgent(0.5, 0.8, 0.95) // Learning rate aumentato per apprendimento più rapido
	return &SnakeAgent{
		agent:        agent,
		game:         game,
		foodReward:   500.0,  // Default reward values
		deathPenalty: -250.0, // Can be overridden by curriculum
	}
}

// SetRewardValues imposta i valori di reward per la fase corrente
func (sa *SnakeAgent) SetRewardValues(foodReward, deathPenalty float64) {
	sa.foodReward = foodReward
	sa.deathPenalty = deathPenalty
}

// getState restituisce il vettore di stato direttamente come []float64
func (sa *SnakeAgent) getState() []float64 {
	snake := sa.game.GetSnake()
	head := snake.GetHead()
	food := sa.game.food

	// Ottiene la direzione del cibo usando il sensore
	foodDir := float64(sa.game.GetFoodDirection())

	// Distanze dai pericoli nelle varie direzioni
	distAhead, distLeft, distRight := sa.game.GetDangers()

	// Distanza normalizzata dal cibo
	foodDist := math.Sqrt(math.Pow(float64(head.X-food.X), 2) + math.Pow(float64(head.Y-food.Y), 2))
	foodDistNorm := math.Min(foodDist/5.0, 10.0)

	// Lunghezza del serpente (normalizzata)
	lengthNorm := math.Min(float64(len(snake.Body))/4.0, 5.0)

	// Pericoli immediati nelle direzioni cardinali
	currentDir := sa.game.GetCurrentDirection()
	dirPoint := currentDir.ToPoint()
	leftDir := Point{X: dirPoint.Y, Y: -dirPoint.X}  // Ruota 90° a sinistra
	rightDir := Point{X: -dirPoint.Y, Y: dirPoint.X} // Ruota 90° a destra

	dangerAhead := sa.game.checkCollision(Point{X: head.X + dirPoint.X, Y: head.Y + dirPoint.Y}) != NoCollision
	dangerLeft := sa.game.checkCollision(Point{X: head.X + leftDir.X, Y: head.Y + leftDir.Y}) != NoCollision
	dangerRight := sa.game.checkCollision(Point{X: head.X + rightDir.X, Y: head.Y + rightDir.Y}) != NoCollision

	return []float64{
		float64(currentDir),
		foodDir,
		float64(distAhead),
		float64(distLeft),
		float64(distRight),
		foodDistNorm,
		lengthNorm,
		boolToFloat64(dangerAhead),
		boolToFloat64(dangerLeft),
		boolToFloat64(dangerRight),
	}
}

// boolToFloat64 converte un booleano in float64
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

	// Reward base per sopravvivenza
	reward := 1.0

	// Reward per mangiare cibo
	if snake.Score > oldScore {
		reward = sa.foodReward
		sa.game.Steps = 0
	}

	// Penalty per morte
	if snake.Dead {
		reward = sa.deathPenalty
	}

	// Penalty per stagnazione
	if sa.game.Steps > 100 {
		reward *= 0.95 // Decay graduale del reward
	}

	return reward
}

// Reset prepara l'agente per una nuova partita mantenendo le conoscenze apprese.
func (sa *SnakeAgent) Reset() {
	// Preserve the existing stats
	existingStats := sa.game.Stats

	width := sa.game.Grid.Width
	height := sa.game.Grid.Height
	sa.game = NewGame(width, height)

	// Restore the stats
	sa.game.Stats = existingStats

	sa.agent.IncrementEpisode()
}

// SaveWeights salva i pesi della rete neurale su file.
func (sa *SnakeAgent) SaveWeights() error {
	return sa.agent.SaveWeights(qlearning.WeightsFile)
}
