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
	agent := qlearning.NewAgent(0.5, 0.95, 0.2)
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

func (sa *SnakeAgent) calculateReward(oldScore, oldLength int) float64 {
	snake := sa.game.GetSnake()
	if snake.Dead {
		return -1000.0 // Penalità fissa alta
	}

	// Reward per aver mangiato
	if snake.Score > oldScore {
		return 500.0 + 50.0*float64(snake.Score)
	}

	// Reward per avvicinamento al cibo
	oldDist := sa.getManhattanDistance(snake.GetPreviousHead(), sa.game.food)
	newDist := sa.getManhattanDistance(snake.GetHead(), sa.game.food)
	distReward := 20.0 * float64(oldDist-newDist)

	// Penalità stagnazione
	stepsWithoutFood := sa.game.Steps - oldLength*10
	stagnationPenalty := -0.1 * float64(stepsWithoutFood)
	if stepsWithoutFood > 50 {
		stagnationPenalty -= 10.0 // Penalità aggiuntiva dopo 50 passi senza cibo
	}

	return distReward + stagnationPenalty
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
	// err := sa.agent.SaveQTable(qlearning.QtableFile)
	// if err != nil {
	// 	fmt.Printf("Error saving QTable: %v\n", err)
	// }

	width := sa.game.Grid.Width
	height := sa.game.Grid.Height
	sa.game = NewGame(width, height)
	sa.agent.IncrementEpisode()
}

// SaveQTable salva la QTable su file.
func (sa *SnakeAgent) SaveQTable() error {
	return sa.agent.SaveQTable(qlearning.QtableFile)
}
