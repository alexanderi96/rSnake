package main

import (
	"fmt"

	"snake-game/qlearning"
)

// SnakeAgent rappresenta l'agente che gioca a Snake usando Q-learning.
type SnakeAgent struct {
	agent *qlearning.Agent
	game  *Game
}

// NewSnakeAgent crea un nuovo agente per il gioco.
func NewSnakeAgent(game *Game) *SnakeAgent {
	agent := qlearning.NewAgent(0.5, 0.8, 0.95) // Learning rate aumentato per apprendimento più rapido
	return &SnakeAgent{
		agent: agent,
		game:  game,
	}
}

// getState costruisce lo stato secondo il formato:
// state := fmt.Sprintf("%d:%v:%v:%v", foodDir, dangerAhead, dangerLeft, dangerRight)
// dove:
//   - foodDir è la direzione del cibo relativa allo snake:
//     0: cibo davanti, 1: cibo a destra, 2: cibo dietro, 3: cibo a sinistra
//   - dangerAhead, dangerLeft, dangerRight indicano se c'è pericolo nelle direzioni relative.
func (sa *SnakeAgent) getState() string {
	// Ottieni la direzione assoluta del cibo
	absoluteFoodDir := int(sa.game.GetFoodDirection())

	// Ottieni la direzione attuale dello snake in forma cardinale
	currentDir := int(sa.game.GetCurrentDirection())

	// Calcola la direzione del cibo relativa allo snake
	// La formula standard: (absoluteFoodDir - snakeDir + 4) % 4
	foodDir := (absoluteFoodDir - currentDir + 4) % 4

	// Ottieni i pericoli relativi alla direzione corrente
	dangerAhead, dangerLeft, dangerRight := sa.game.GetDangers()

	// Stato finale: una stringa che contiene la posizione del cibo (relativa) e i flag per il pericolo
	state := fmt.Sprintf("%d:%v:%v:%v", foodDir, dangerAhead, dangerLeft, dangerRight)
	return state
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
	if snake.Dead {
		return -1000.0
	}

	reward := 0.0

	// Reward significativo per il cibo
	if snake.Score > oldScore {
		reward += 500.0                       // Reward base per il cibo
		reward += 50.0 * float64(snake.Score) // Bonus per punteggio totale
	}

	// Reward/penalità per la distanza dal cibo
	oldDist := sa.getManhattanDistance(snake.GetPreviousHead(), sa.game.food)
	newDist := sa.getManhattanDistance(snake.GetHead(), sa.game.food)
	distDiff := oldDist - newDist

	if distDiff > 0 {
		reward += 10.0 * float64(distDiff) // Reward per avvicinamento
	} else {
		reward -= 5.0 * float64(-distDiff) // Penalità per allontanamento
	}

	// Penalità per stagnazione
	stepsWithoutFood := sa.game.Steps - oldLength*10
	if stepsWithoutFood > 30 {
		reward -= float64(stepsWithoutFood) * 0.1
	}

	return reward
}

// getManhattanDistance calcola la distanza Manhattan tra due punti, considerando il wrapping della griglia.
func (sa *SnakeAgent) getManhattanDistance(p1, p2 Point) int {
	dx := abs(p1.X - p2.X)
	dy := abs(p1.Y - p2.Y)

	if dx > sa.game.Grid.Width/2 {
		dx = sa.game.Grid.Width - dx
	}
	if dy > sa.game.Grid.Height/2 {
		dy = sa.game.Grid.Height - dy
	}

	return dx + dy
}

// abs restituisce il valore assoluto di un intero.
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
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

// SaveQTable salva la QTable su file.
func (sa *SnakeAgent) SaveQTable() error {
	return sa.agent.SaveQTable(qlearning.QtableFile)
}
