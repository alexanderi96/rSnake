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

// getState restituisce il vettore di stato come matrice 3x8 appiattita
func (sa *SnakeAgent) getState() []float64 {
	// Ottiene la matrice 3x8 appiattita [muri][corpo][cibo] x [backLeft, left, frontLeft, front, frontRight, right, backRight, back]
	return sa.game.GetStateInfo()
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
	reward := -0.01 // Piccola penalità per ogni step per incentivare percorsi più brevi

	// Morte
	if snake.Dead {
		return -3.0
	}

	// Cibo mangiato (reward proporzionale alla lunghezza)
	if snake.Score > oldScore {
		baseReward := 10.0
		lengthBonus := float64(snake.Score) * 0.1
		return baseReward + lengthBonus
	}

	// Ottiene la matrice di stato
	state := sa.game.GetStateInfo()
	walls := state[0:8]  // Primi 8 valori: muri
	body := state[8:16]  // Secondi 8 valori: corpo
	food := state[16:24] // Ultimi 8 valori: cibo

	// Determina gli indici per la direzione scelta e le direzioni associate
	var mainIndex, diagLeftIndex, diagRightIndex, backIndex int
	switch action := sa.game.GetLastAction(); action {
	case 1: // avanti
		mainIndex = 3      // front
		diagLeftIndex = 2  // frontLeft
		diagRightIndex = 4 // frontRight
		backIndex = 7      // back
	case 0: // sinistra
		mainIndex = 1     // left
		diagLeftIndex = 2 // frontLeft
		backIndex = 0     // backLeft
	case 2: // destra
		mainIndex = 5      // right
		diagRightIndex = 4 // frontRight
		backIndex = 6      // backRight
	}

	// Reward basato sulla direzione scelta
	if food[mainIndex] > 0 { // Ha scelto una direzione con cibo
		reward += 1.0
	} else if walls[mainIndex] > 0 || body[mainIndex] > 0 { // Ha scelto una direzione pericolosa
		reward -= 1.5
	}

	// Reward aggiuntivo basato sulle diagonali e direzione posteriore
	dangerNearby := false
	foodNearby := false

	// Controlla le diagonali
	if diagLeftIndex >= 0 {
		if walls[diagLeftIndex] > 0 || body[diagLeftIndex] > 0 {
			dangerNearby = true
		}
		if food[diagLeftIndex] > 0 {
			foodNearby = true
		}
	}
	if diagRightIndex >= 0 {
		if walls[diagRightIndex] > 0 || body[diagRightIndex] > 0 {
			dangerNearby = true
		}
		if food[diagRightIndex] > 0 {
			foodNearby = true
		}
	}

	// Controlla la direzione posteriore
	if walls[backIndex] > 0 || body[backIndex] > 0 {
		dangerNearby = true
	}
	if food[backIndex] > 0 {
		foodNearby = true
	}

	if dangerNearby {
		reward -= 0.5 // Penalità se ci sono pericoli nelle vicinanze
	}
	if foodNearby {
		reward += 0.3 // Bonus se c'è cibo nelle vicinanze
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
