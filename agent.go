package main

import (
	"fmt"

	"snake-game/qlearning"
)

type SnakeAgent struct {
	agent    *qlearning.Agent
	game     *Game
	maxScore int
}

func NewSnakeAgent(game *Game) *SnakeAgent {
	agent := qlearning.NewAgent(0.1, 0.9, 0.1) // learning rate, discount, epsilon
	return &SnakeAgent{
		agent:    agent,
		game:     game,
		maxScore: 0,
	}
}

func (sa *SnakeAgent) getState() string {
	// Ottieni la direzione del cibo
	foodUp, foodRight, foodDown, foodLeft := sa.game.GetFoodDirection()
	var foodDir Direction
	switch {
	case foodUp:
		foodDir = UP
	case foodRight:
		foodDir = RIGHT
	case foodDown:
		foodDir = DOWN
	case foodLeft:
		foodDir = LEFT
	default:
		foodDir = NONE
	}

	// Ottieni la direzione corrente dello snake
	snakeDir := sa.game.GetCurrentDirection()

	// snakeLength := len(sa.game.GetSnake().Body)

	dangerUp, dangerDown, dangerLeft, dangerRight := sa.game.GetDangers()

	state := fmt.Sprintf("%d:%d:%v:%v:%v:%v", foodDir, snakeDir, dangerUp, dangerDown, dangerLeft, dangerRight)
	return state
}

func (sa *SnakeAgent) Update() {
	if sa.game.GetSnake().Dead {
		return
	}

	currentState := sa.getState()
	// Qui indichiamo 3 possibili azioni relative: 0 (sinistra), 1 (dritto), 2 (destra)
	action := sa.agent.GetAction(currentState, 3)

	// Ottieni la direzione corrente dello snake (in forma relativa, usando ad esempio GetCurrentDirection)
	currentDir := sa.game.GetCurrentDirection()

	// Mappa l'azione relativa nella nuova direzione assoluta
	newRelativeDirection := sa.mapRelativeAction(currentDir, action)
	newDir := newRelativeDirection.ToPoint()

	// Salva stato attuale per il reward
	oldScore := sa.game.GetSnake().Score
	oldLength := len(sa.game.GetSnake().Body)

	// Applica la nuova direzione (SetDirection già previene inversioni a 180°)
	sa.game.GetSnake().SetDirection(newDir)
	sa.game.Update()

	// Calcola il reward
	reward := sa.calculateReward(oldScore, oldLength)

	// Aggiorna la Q-table
	newState := sa.getState()
	sa.agent.Update(currentState, action, reward, newState, 3)

	if sa.game.GetSnake().Score > sa.maxScore {
		sa.maxScore = sa.game.GetSnake().Score
	}
}

func (sa *SnakeAgent) calculateReward(oldScore, oldLength int) float64 {
	snake := sa.game.GetSnake()
	// Calcola la distanza di Manhattan dalla testa al cibo prima e dopo il movimento
	oldDist := sa.getManhattanDistance(sa.game.GetSnake().GetPreviousHead(), sa.game.food)
	newDist := sa.getManhattanDistance(snake.GetHead(), sa.game.food)

	// Caso in cui lo snake muore
	if snake.Dead {
		return -100.0
	}

	// Caso in cui lo snake ha mangiato il cibo (score incrementato)
	if snake.Score > oldScore {
		return 100.0
	}

	// Calcola uno scaling basato sulla lunghezza dello snake:
	// Ad esempio, se lo snake è lungo, scale aumenta (qui usiamo snakeLength/10, modificabile)
	snakeLength := len(snake.Body)
	scale := 1.0 + float64(snakeLength)/10.0

	// Calcola il reward in base alla variazione di distanza
	// Se il movimento avvicina il cibo, (oldDist - newDist) > 0, quindi reward positivo.
	// Se il movimento allontana il cibo, (oldDist - newDist) < 0, reward negativo.
	reward := float64(oldDist-newDist) * scale

	return reward
}

// getManhattanDistance calculates the Manhattan distance between two points
func (sa *SnakeAgent) getManhattanDistance(p1, p2 Point) int {
	dx := abs(p1.X - p2.X)
	dy := abs(p1.Y - p2.Y)

	// Consider grid wrapping
	if dx > sa.game.Grid.Width/2 {
		dx = sa.game.Grid.Width - dx
	}
	if dy > sa.game.Grid.Height/2 {
		dy = sa.game.Grid.Height - dy
	}

	return dx + dy
}

// abs returns the absolute value of x
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// Reset prepares the agent for a new game while keeping learned knowledge
func (sa *SnakeAgent) Reset() {
	// Save QTable
	err := sa.agent.SaveQTable("qtable.json")
	if err != nil {
		fmt.Printf("Error saving QTable: %v\n", err)
	}

	// Create new game with same dimensions
	width := sa.game.Grid.Width
	height := sa.game.Grid.Height
	sa.game = NewGame(width, height)

	// Increment episode counter for epsilon decay
	sa.agent.IncrementEpisode()
}

// SaveQTable saves the current QTable to file
func (sa *SnakeAgent) SaveQTable() error {
	return sa.agent.SaveQTable("qtable.json")
}

// mapRelativeAction converte un'azione relativa in una direzione assoluta,
// in base alla direzione corrente dello snake.
func (sa *SnakeAgent) mapRelativeAction(current Direction, action int) Direction {
	// Assumiamo le seguenti azioni relative:
	// 0 = gira a sinistra, 1 = vai dritto, 2 = gira a destra.
	// Se per caso l'agente restituisce un valore fuori range, lo consideriamo come "vai dritto".
	if action < 0 || action > 2 {
		action = 1
	}

	// Utilizziamo la funzione GetCurrentDirection per ottenere la direzione corrente
	switch current {
	case UP:
		if action == 0 {
			return LEFT
		} else if action == 1 {
			return UP
		} else { // action == 2
			return RIGHT
		}
	case RIGHT:
		if action == 0 {
			return UP // girando a sinistra da RIGHT diventa UP
		} else if action == 1 {
			return RIGHT
		} else { // action == 2
			return DOWN
		}
	case DOWN:
		if action == 0 {
			return RIGHT
		} else if action == 1 {
			return DOWN
		} else { // action == 2
			return LEFT
		}
	case LEFT:
		if action == 0 {
			return DOWN
		} else if action == 1 {
			return LEFT
		} else { // action == 2
			return UP // girando a destra da LEFT diventa UP
		}
	default:
		return current
	}
}
