package main

import (
	"log"

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
	// Ottiene la direzione corrente e la direzione relativa del cibo
	currentDir := sa.game.GetCurrentDirection()
	foodDir := float64(sa.game.GetRelativeFoodDirection())

	// Distanze dai pericoli nelle varie direzioni (già limitate a 5 celle in GetDangers)
	distAhead, distLeft, distRight := sa.game.GetDangers()

	return []float64{
		float64(currentDir), // direzione attuale (0-3)
		foodDir,             // direzione del cibo relativa alla direzione corrente (0-3)
		float64(distAhead),  // distanza ostacoli (0-5)
		float64(distLeft),   // distanza ostacoli (0-5)
		float64(distRight),  // distanza ostacoli (0-5)
	}
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
		log.Printf("Snake is dead, skipping update")
		return
	}

	currentState := sa.getState()
	action := sa.agent.GetAction(currentState, 3)
	newDir := sa.relativeActionToAbsolute(action)

	log.Printf("Current state: %v", currentState)
	log.Printf("Chosen action: %d (direction: %v)", action, newDir)

	// Applica l'azione e calcola il reward
	oldScore := sa.game.GetSnake().Score
	sa.game.GetSnake().SetDirection(newDir)
	sa.game.Update()

	// Sistema di reward semplificato
	reward := sa.calculateReward(oldScore)
	log.Printf("Reward received: %.2f", reward)

	// Aggiorna i Q-values
	newState := sa.getState()
	sa.agent.Update(currentState, action, reward, newState, 3)
}

func (sa *SnakeAgent) calculateReward(oldScore int) float64 {
	snake := sa.game.GetSnake()

	// Reward base per sopravvivenza
	reward := 1.0

	// Calcola la distanza di Manhattan prima e dopo il movimento
	oldHead := snake.GetPreviousHead()
	newHead := snake.GetHead()
	food := sa.game.GetFood()

	oldDistance := manhattanDistance(oldHead, food, sa.game.Grid.Width, sa.game.Grid.Height)
	newDistance := manhattanDistance(newHead, food, sa.game.Grid.Width, sa.game.Grid.Height)

	// Premia la riduzione della distanza dal cibo
	if newDistance < oldDistance {
		reward += 5.0
	} else if newDistance > oldDistance {
		reward -= 2.0
	}

	// Reward per mangiare cibo
	if snake.Score > oldScore {
		reward = sa.foodReward
		sa.game.Steps = 0
	}

	// Penalty per morte
	if snake.Dead {
		reward = sa.deathPenalty
	}

	// Penalty graduale per stagnazione
	if sa.game.Steps > 100 {
		reward *= 0.95 // Decay più aggressivo per evitare loop
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
