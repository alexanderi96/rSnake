package main

import (
	"fmt"

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
	agent := qlearning.NewAgent(0.3, 0.1, 0.99)
	return &SnakeAgent{
		agent:    agent,
		game:     game,
		maxScore: 0,
	}
}

// getState costruisce lo stato secondo il formato:
// state := fmt.Sprintf("%d:%v:%v:%v", foodDir, dangerAhead, dangerLeft, dangerRight)
// dove:
//   - foodDir è la direzione del cibo relativa allo snake:
//     0: cibo davanti, 1: cibo a destra, 2: cibo dietro, 3: cibo a sinistra
//   - dangerAhead, dangerLeft, dangerRight indicano se c'è pericolo nelle direzioni relative.
func (sa *SnakeAgent) getState() string {
	// Ottieni la direzione assoluta del cibo.
	// foodUp, foodRight, foodDown, foodLeft := sa.game.GetFoodDirection()
	// absoluteFoodDir := -1
	// switch {
	// case foodUp:
	// 	absoluteFoodDir = 0 // cibo in alto (UP)
	// case foodRight:
	// 	absoluteFoodDir = 1 // cibo a destra
	// case foodDown:
	// 	absoluteFoodDir = 2 // cibo in basso (DOWN)
	// case foodLeft:
	// 	absoluteFoodDir = 3 // cibo a sinistra
	// default:
	// 	absoluteFoodDir = 0 // default (non dovrebbe verificarsi)
	// }

	// Ottieni la direzione attuale dello snake in forma cardinale.
	// currentDir := sa.game.GetCurrentDirection()
	// snakeDir := int(currentDir)

	// Calcola la direzione del cibo relativa allo snake.
	// La formula standard: (absoluteFoodDir - snakeDir + 4) % 4
	foodDir := sa.game.GetRelativeFoodDirection()

	// Ottieni i pericoli relativi alla direzione corrente.
	// GetDangers restituisce (dangerAhead, dangerLeft, dangerRight)
	dangerAhead, dangerLeft, dangerRight := sa.game.GetDangers()

	// Stato finale: una stringa che contiene la posizione del cibo (relativa) e i flag per il pericolo.
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
		return -1000.0
	}

	reward := 0.0

	// 1. Reward per aver mangiato (score aumenta)
	if snake.Score > oldScore {
		reward += 1000.0 + 100.0*float64(snake.Score)
	}

	// 2. Modifica del reward in base alla variazione della distanza Manhattan dal cibo
	oldDist := sa.getManhattanDistance(snake.GetPreviousHead(), sa.game.food)
	newDist := sa.getManhattanDistance(snake.GetHead(), sa.game.food)
	distReward := 30.0 * float64(oldDist-newDist)
	reward += distReward

	// 3. Penalità per stagnazione: se sono passati molti step senza mangiare
	stepsWithoutFood := sa.game.Steps - oldLength*10
	stagnationPenalty := -1.0 * float64(stepsWithoutFood)
	if stepsWithoutFood > 30 { // Se il numero di step senza cibo supera una soglia, penalizza ulteriormente
		stagnationPenalty -= 20.0
	}
	reward += stagnationPenalty

	// 4. Bonus/Penalità basati sulla durata
	avgDuration := sa.game.Stats.GetAverageDuration() // Media delle durate (in secondi)
	currentDuration := sa.game.ElapsedTime()          // Durata corrente della partita (in secondi)

	// Bonus basato sul punteggio corrente, indipendente dalla media
	reward += 50.0 * float64(snake.Score)

	// Bonus per la velocità di gioco, ma con un impatto minore
	if currentDuration < avgDuration {
		reward += 20.0 // Bonus fisso per essere più veloce della media
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
