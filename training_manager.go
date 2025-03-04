package main

import (
	"fmt"
	"sync"
	"time"
)

// GameState rappresenta lo stato del gioco che viene condiviso tra i thread
type GameState struct {
	Snake     *Snake
	Food      Point
	Score     int
	Dead      bool
	Width     int
	Height    int
	Direction Point
}

// TrainingManager gestisce il training degli agenti in thread separati
type TrainingManager struct {
	agentPool      *AgentPool
	gameStateChan  []chan GameState
	controlChan    chan bool // per segnali di controllo (es. stop)
	wg             sync.WaitGroup
	mutex          sync.RWMutex
	isTraining     bool
	updateInterval time.Duration
	bestScore      int
	totalScore     int
	episodeCount   int
}

// NewTrainingManager crea un nuovo training manager
func NewTrainingManager(numAgents int, gameWidth, gameHeight int, updateInterval time.Duration) *TrainingManager {
	// Crea un canale per ogni agente
	gameStateChannels := make([]chan GameState, numAgents)
	for i := 0; i < numAgents; i++ {
		gameStateChannels[i] = make(chan GameState, 1)
	}

	return &TrainingManager{
		agentPool:      NewAgentPool(numAgents, gameWidth, gameHeight),
		gameStateChan:  gameStateChannels,
		controlChan:    make(chan bool, 1),
		updateInterval: updateInterval,
	}
}

// StartTraining avvia il loop di training in un goroutine separato
func (tm *TrainingManager) StartTraining() {
	tm.mutex.Lock()
	if tm.isTraining {
		tm.mutex.Unlock()
		return
	}
	tm.isTraining = true
	tm.mutex.Unlock()

	tm.wg.Add(1)
	go tm.trainingLoop()
}

// StopTraining ferma il training
func (tm *TrainingManager) StopTraining() {
	tm.mutex.Lock()
	if !tm.isTraining {
		tm.mutex.Unlock()
		return
	}
	tm.isTraining = false
	tm.mutex.Unlock()

	tm.controlChan <- true
	tm.wg.Wait()
}

// UpdateGameState aggiorna lo stato del gioco nel canale per ogni agente
func (tm *TrainingManager) UpdateGameState() {
	tm.mutex.RLock()
	if !tm.isTraining {
		tm.mutex.RUnlock()
		return
	}
	tm.mutex.RUnlock()

	agents := tm.agentPool.GetAllAgents()
	for i, agent := range agents {
		snake := agent.game.GetSnake()
		state := GameState{
			Snake:     snake,
			Food:      agent.game.food,
			Score:     snake.Score,
			Dead:      snake.Dead,
			Width:     agent.game.Grid.Width,
			Height:    agent.game.Grid.Height,
			Direction: snake.Direction,
		}

		// Invio non bloccante dello stato per ogni agente
		select {
		case tm.gameStateChan[i] <- state:
		default:
			// Se il canale è pieno, scartiamo lo stato
		}
	}
}

// trainingLoop è il loop principale di training che viene eseguito in goroutine separate per ogni agente
func (tm *TrainingManager) trainingLoop() {
	defer tm.wg.Done()

	ticker := time.NewTicker(tm.updateInterval)
	defer ticker.Stop()

	agents := tm.agentPool.GetAllAgents()
	for i, agent := range agents {
		// Avvia un goroutine separato per ogni agente
		go func(agentIndex int, currentAgent *SnakeAgent) {
			for {
				select {
				case <-tm.controlChan:
					// Salva i pesi prima di uscire
					if err := tm.agentPool.SaveWeights(); err != nil {
						fmt.Printf("Error saving final weights: %v\n", err)
					}
					return
				case state := <-tm.gameStateChan[agentIndex]:
					tm.mutex.Lock()
					// Aggiorna lo stato del gioco
					snake := currentAgent.game.GetSnake()
					snake.Body = state.Snake.Body
					snake.Direction = state.Direction
					snake.Dead = state.Dead
					snake.Score = state.Score
					currentAgent.game.food = state.Food
					currentAgent.game.Grid.Width = state.Width
					currentAgent.game.Grid.Height = state.Height

					// Esegui l'update dell'agente
					if !snake.Dead {
						currentAgent.Update()
					} else {
						// Aggiorna le statistiche quando il serpente muore
						tm.updateStats(snake.Score)
						// Resetta per il prossimo episodio
						tm.ResetGame(agentIndex)
					}
					tm.mutex.Unlock()
				case <-ticker.C:
					// Timeout per evitare blocchi
					continue
				}
			}
		}(i, agent)
	}
}

// GetGame restituisce il gioco dell'agente specificato in modo thread-safe
func (tm *TrainingManager) GetGame(agentIndex int) *Game {
	tm.mutex.RLock()
	defer tm.mutex.RUnlock()
	if agent := tm.agentPool.GetAgent(agentIndex); agent != nil {
		return agent.game
	}
	return nil
}

// updateStats aggiorna le statistiche di training
func (tm *TrainingManager) updateStats(score int) {
	tm.totalScore += score
	if score > tm.bestScore {
		tm.bestScore = score
	}

	tm.episodeCount++

	// Salva i pesi automaticamente ogni 500 episodi
	if tm.episodeCount%500 == 0 {
		go func() {
			if err := tm.agentPool.SaveWeights(); err != nil {
				fmt.Printf("Failed to save weights at episode %d: %v\n", tm.episodeCount, err)
			} else {
				fmt.Printf("Weights automatically saved at episode %d\n", tm.episodeCount)
			}
		}()
	}

	// Reset statistiche per il prossimo batch
	if tm.episodeCount%50 == 0 {
		tm.totalScore = 0
	}
}

// ResetGame resetta il gioco per l'agente specificato in modo thread-safe
func (tm *TrainingManager) ResetGame(agentIndex int) {
	agent := tm.agentPool.GetAgent(agentIndex)
	if agent == nil {
		return
	}

	// Preserva le statistiche esistenti
	existingStats := agent.game.Stats

	// Crea un nuovo gioco con le stesse dimensioni
	width := agent.game.Grid.Width
	height := agent.game.Grid.Height
	agent.game = NewGame(width, height)

	// Ripristina le statistiche
	agent.game.Stats = existingStats

	// Resetta l'agente
	agent.Reset()
}
