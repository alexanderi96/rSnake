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
	agent := qlearning.NewAgent(0.5, 0.8, 0.95) // Learning rate aumentato per apprendimento più rapido
	return &SnakeAgent{
		agent: agent,
		game:  game,
	}
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

	// Base reward MOLTO più basso per sopravvivenza
	reward := 0.01

	// Analisi del movimento rispetto allo stato
	state := sa.getState()
	currentDir := int(state[0]) // direzione attuale (0-3)
	foodDir := int(state[1])    // direzione cibo (0-3)
	distAhead := state[2]       // distanza ostacoli
	distLeft := state[3]
	distRight := state[4]

	// Calcola la differenza di direzione (considerando la circolarità 0-3)
	dirDiff := (currentDir - foodDir + 4) % 4

	// Reward più aggressivi per movimento verso il cibo
	switch dirDiff {
	case 0: // Allineato con il cibo
		reward += 0.4 * (distAhead / 5.0) // Aumentato per incentivare l'allineamento
	case 1: // Necessaria svolta a destra
		if distRight > 2 {
			reward += 0.25 // Svolta sicura - reward aumentato
		} else {
			reward += 0.1 // Svolta rischiosa - reward aumentato
		}
	case 3: // Necessaria svolta a sinistra
		if distLeft > 2 {
			reward += 0.25 // Svolta sicura - reward aumentato
		} else {
			reward += 0.1 // Svolta rischiosa - reward aumentato
		}
	case 2: // Direzione opposta al cibo
		reward -= 0.1 // Penalità maggiore per scoraggiare direzione opposta
	}

	// Reward aumentato per mangiare il cibo
	if snake.Score > oldScore {
		reward = 1.5 // Aumentato per incentivare la ricerca del cibo
		sa.game.Steps = 0
	}

	// Penalty ridotta per morte
	if snake.Dead {
		reward = -0.8 // Ridotta per incoraggiare più rischi
	}

	return reward
}

// GetEpsilon returns the current epsilon value
func (sa *SnakeAgent) GetEpsilon() float64 {
	return sa.agent.Epsilon
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
