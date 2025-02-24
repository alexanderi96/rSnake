package main

import (
	"fmt"
	"math"

	"snake-game/qlearning"
)

// SnakeAgent rappresenta l'agente che gioca a Snake usando Q-learning.
type SnakeAgent struct {
	agent    *qlearning.Agent
	game     *Game
	maxScore int
}

// NewSnakeAgent crea un nuovo agente per il gioco.
func NewSnakeAgent(game *Game) *SnakeAgent {
	agent := qlearning.NewAgent(0.5, 0.8, 0.95) // Learning rate aumentato per apprendimento più rapido
	return &SnakeAgent{
		agent:    agent,
		game:     game,
		maxScore: 0,
	}
}

// getState costruisce uno stato più ricco che include:
// - Direzione del cibo relativa
// - Pericoli nelle direzioni relative
// - Lunghezza del serpente
// - Distanza dal cibo
func (sa *SnakeAgent) getState() string {
	// Direzione del cibo relativa allo snake
	foodDir := sa.game.GetRelativeFoodDirection()

	// Pericoli nelle direzioni relative
	dangerAhead, dangerLeft, dangerRight := sa.game.GetDangers()

	// Lunghezza del serpente
	snakeLength := len(sa.game.GetSnake().Body)

	// Distanza dal cibo
	foodDist := sa.getManhattanDistance(sa.game.GetSnake().GetHead(), sa.game.food)

	// Stato finale: include più informazioni per una rappresentazione più ricca
	state := fmt.Sprintf("%d:%v:%v:%v:%d:%d:%d",
		foodDir,                              // direzione del cibo
		dangerAhead, dangerLeft, dangerRight, // pericoli
		snakeLength,                        // lunghezza del serpente
		foodDist,                           // distanza dal cibo
		int(sa.game.GetCurrentDirection()), // direzione corrente
	)
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

	// Aggiorna il punteggio massimo se necessario.
	if sa.game.GetSnake().Score > sa.maxScore {
		sa.maxScore = sa.game.GetSnake().Score
	}
}

// calculateReward calcola il reward in base a diversi fattori:
// - Consumo di cibo (aumento del punteggio)
// - Variazione della distanza Manhattan rispetto al cibo
// - Penalità per stagnazione (passi senza mangiare)
// - Bonus o penalità basati sul confronto tra punteggio corrente e average score
// - Bonus o penalità basati sul confronto tra durata corrente e average duration
func (sa *SnakeAgent) calculateReward(oldScore, oldLength int) float64 {
	snake := sa.game.GetSnake()
	if snake.Dead {
		return -500.0 // Penalità ridotta per la morte
	}

	reward := 0.0

	// Reward base per sopravvivenza
	reward += 1.0

	// Reward per cibo mangiato aumentato
	if snake.Score > oldScore {
		reward += 1000.0                           // Reward base aumentato per il cibo
		reward += 100.0 * float64(len(snake.Body)) // Bonus maggiore per lunghezza
	}

	// Reward per avvicinamento al cibo
	oldDist := sa.getManhattanDistance(snake.GetPreviousHead(), sa.game.food)
	newDist := sa.getManhattanDistance(snake.GetHead(), sa.game.food)
	distDiff := oldDist - newDist

	if distDiff > 0 {
		// Reward aumentato per avvicinamento
		reward += 30.0 * float64(distDiff)
	} else {
		// Penalità ridotta per allontanamento
		reward -= 5.0 * float64(-distDiff)
	}

	// Penalità per stagnazione più leggera
	stepsWithoutFood := sa.game.Steps - oldLength*10
	if stepsWithoutFood > 0 {
		stagnationPenalty := -5.0 * math.Log(float64(stepsWithoutFood))
		reward += stagnationPenalty
	}

	// Bonus per efficienza aumentato
	efficiency := float64(snake.Score) / float64(sa.game.Steps)
	reward += 100.0 * efficiency

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
	// Salva lo stato corrente prima del reset
	err := sa.agent.SaveQTable(qlearning.QtableFile)
	if err != nil {
		fmt.Printf("Error saving QTable: %v\n", err)
	}

	// Salva le statistiche
	if err := sa.game.Stats.SaveToFile(); err != nil {
		fmt.Printf("Error saving stats: %v\n", err)
	}

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
