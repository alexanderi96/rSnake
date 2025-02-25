package main

import (
	"fmt"
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

// getState costruisce uno stato più dettagliato che include:
// - Direzione del cibo
// - Distanze dai muri e dal proprio corpo
// - Configurazione dei pericoli nelle 8 celle circostanti
// - Lunghezza del serpente normalizzata
// - Distanza normalizzata dal cibo
func (sa *SnakeAgent) getState() string {
	snake := sa.game.GetSnake()
	head := snake.GetHead()
	food := sa.game.food

	// Direzione del cibo (0=sopra, 1=destra, 2=sotto, 3=sinistra)
	foodDirX := food.X - head.X
	foodDirY := food.Y - head.Y
	foodDir := -1

	if math.Abs(float64(foodDirX)) > math.Abs(float64(foodDirY)) {
		if foodDirX > 0 {
			foodDir = 1 // destra
		} else {
			foodDir = 3 // sinistra
		}
	} else {
		if foodDirY > 0 {
			foodDir = 2 // giù
		} else {
			foodDir = 0 // su
		}
	}

	// Distanze dai pericoli nelle varie direzioni
	distAhead, distLeft, distRight := sa.game.GetDangers()

	// Distanza normalizzata dal cibo
	foodDist := math.Sqrt(math.Pow(float64(head.X-food.X), 2) + math.Pow(float64(head.Y-food.Y), 2))
	foodDistNorm := int(math.Min(foodDist/5.0, 10.0))

	// Lunghezza del serpente (normalizzata)
	length := math.Min(float64(len(snake.Body)), 20.0)
	lengthNorm := int(length / 4.0)

	// Configurazione pericoli vicini (8 celle circostanti)
	dangerN := sa.game.checkCollision(Point{X: head.X, Y: head.Y - 1}) != NoCollision
	dangerNE := sa.game.checkCollision(Point{X: head.X + 1, Y: head.Y - 1}) != NoCollision
	dangerE := sa.game.checkCollision(Point{X: head.X + 1, Y: head.Y}) != NoCollision
	dangerSE := sa.game.checkCollision(Point{X: head.X + 1, Y: head.Y + 1}) != NoCollision
	dangerS := sa.game.checkCollision(Point{X: head.X, Y: head.Y + 1}) != NoCollision
	dangerSW := sa.game.checkCollision(Point{X: head.X - 1, Y: head.Y + 1}) != NoCollision
	dangerW := sa.game.checkCollision(Point{X: head.X - 1, Y: head.Y}) != NoCollision
	dangerNW := sa.game.checkCollision(Point{X: head.X - 1, Y: head.Y - 1}) != NoCollision

	dangerPattern := fmt.Sprintf("%d%d%d%d%d%d%d%d",
		boolToInt(dangerN), boolToInt(dangerNE),
		boolToInt(dangerE), boolToInt(dangerSE),
		boolToInt(dangerS), boolToInt(dangerSW),
		boolToInt(dangerW), boolToInt(dangerNW))

	// Stato finale: combina tutte le informazioni
	state := fmt.Sprintf("%d:%d:%d:%d:%d:%d:%s:%d",
		int(sa.game.GetCurrentDirection()), foodDir,
		distAhead, distLeft, distRight,
		foodDistNorm, dangerPattern, lengthNorm)

	return state
}

// boolToInt converte un booleano in intero
func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

// relativeActionToAbsolute converte un'azione relativa in una direzione assoluta.
// Le azioni relative sono definite come:
//
//	0: ruota a sinistra
//	1: vai avanti
//	2: ruota a destra
func (sa *SnakeAgent) relativeActionToAbsolute(relativeAction int) Direction {
	currentDir := sa.game.GetCurrentDirection()
	switch relativeAction {
	case 0: // ruota a sinistra
		return currentDir.TurnLeft()
	case 1: // vai avanti
		return currentDir
	case 2: // ruota a destra
		return currentDir.TurnRight()
	default:
		return currentDir // fallback
	}
}

// Update esegue un passo di decisione e aggiornamento Q-learning.
func (sa *SnakeAgent) Update() {
	if sa.game.GetSnake().Dead {
		return
	}

	currentState := sa.getState()
	// Usa 3 possibili azioni relative.
	action := sa.agent.GetAction(currentState, 3)

	// Converte l'azione relativa in una direzione assoluta.
	newDir := sa.relativeActionToAbsolute(action).ToPoint()

	// Salva il punteggio corrente per calcolare il reward.
	oldScore := sa.game.GetSnake().Score
	oldLength := len(sa.game.GetSnake().Body)

	// Applica l'azione.
	sa.game.GetSnake().SetDirection(newDir)
	sa.game.Update()

	// Calcola il reward.
	reward := sa.calculateReward(oldScore, oldLength)

	// Aggiorna i Q-values.
	newState := sa.getState()
	sa.agent.Update(currentState, action, reward, newState, 3)
}

func (sa *SnakeAgent) calculateReward(oldScore, oldLength int) float64 {
	snake := sa.game.GetSnake()
	head := snake.GetHead()
	previousHead := snake.GetPreviousHead()
	food := sa.game.food

	// Base reward for being alive
	reward := 1.0

	// Reward for eating food
	if snake.Score > oldScore {
		reward += sa.foodReward
		sa.game.Steps = 0 // Reset stagnation counter
	}

	// Reward for moving towards/away from food
	oldDist := math.Sqrt(math.Pow(float64(previousHead.X-food.X), 2) + math.Pow(float64(previousHead.Y-food.Y), 2))
	newDist := math.Sqrt(math.Pow(float64(head.X-food.X), 2) + math.Pow(float64(head.Y-food.Y), 2))
	reward += (oldDist - newDist) * 10 // Simple distance-based reward

	// Penalty for death
	if snake.Dead {
		reward = sa.deathPenalty
	}

	// Penalty for stagnation
	stepsWithoutFood := sa.game.Steps - oldLength*10
	if stepsWithoutFood > 50 {
		reward -= float64(stepsWithoutFood-50) * 0.1 // Linear penalty
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
