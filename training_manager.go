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

// TrainingManager gestisce il training dell'agente in un thread separato
type TrainingManager struct {
	game           *Game
	agent          *SnakeAgent
	gameStateChan  chan GameState
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
func NewTrainingManager(game *Game, agent *SnakeAgent, updateInterval time.Duration) *TrainingManager {
	return &TrainingManager{
		game:           game,
		agent:          agent,
		gameStateChan:  make(chan GameState, 1), // buffer di 1 per evitare blocchi
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

// UpdateGameState aggiorna lo stato del gioco nel canale
func (tm *TrainingManager) UpdateGameState() {
	tm.mutex.RLock()
	if !tm.isTraining {
		tm.mutex.RUnlock()
		return
	}
	tm.mutex.RUnlock()

	snake := tm.game.GetSnake()
	state := GameState{
		Snake:     snake,
		Food:      tm.game.food,
		Score:     snake.Score,
		Dead:      snake.Dead,
		Width:     tm.game.Grid.Width,
		Height:    tm.game.Grid.Height,
		Direction: snake.Direction,
	}

	// Invio non bloccante dello stato
	select {
	case tm.gameStateChan <- state:
	default:
		// Se il canale è pieno, scartiamo lo stato
	}
}

// trainingLoop è il loop principale di training che viene eseguito in un goroutine separato
func (tm *TrainingManager) trainingLoop() {
	defer tm.wg.Done()

	ticker := time.NewTicker(tm.updateInterval)
	defer ticker.Stop()

	for {
		select {
		case <-tm.controlChan:
			// Salva i pesi prima di uscire
			if err := tm.agent.SaveWeights(); err != nil {
				fmt.Printf("Error saving final weights: %v\n", err)
			}
			return
		case state := <-tm.gameStateChan:
			tm.mutex.Lock()
			// Aggiorna lo stato del gioco
			snake := tm.game.GetSnake()
			snake.Body = state.Snake.Body
			snake.Direction = state.Direction
			snake.Dead = state.Dead
			snake.Score = state.Score
			tm.game.food = state.Food
			tm.game.Grid.Width = state.Width
			tm.game.Grid.Height = state.Height

			// Esegui l'update dell'agente
			if !snake.Dead {
				tm.agent.Update()
			} else {
				// Aggiorna le statistiche quando il serpente muore
				tm.updateStats(snake.Score)
				// Resetta per il prossimo episodio
				tm.ResetGame()
			}
			tm.mutex.Unlock()
		case <-ticker.C:
			// Timeout per evitare blocchi
			continue
		}
	}
}

// GetGame restituisce il gioco corrente in modo thread-safe
func (tm *TrainingManager) GetGame() *Game {
	tm.mutex.RLock()
	defer tm.mutex.RUnlock()
	return tm.game
}

// updateStats aggiorna le statistiche di training
func (tm *TrainingManager) updateStats(score int) {
	tm.totalScore += score
	if score > tm.bestScore {
		tm.bestScore = score
	}

	tm.episodeCount++

	// Salva i pesi automaticamente ogni 100 episodi
	if tm.episodeCount%100 == 0 {
		// Prova a salvare fino a 3 volte in caso di errore
		var err error
		for attempts := 0; attempts < 3; attempts++ {
			err = tm.agent.SaveWeights()
			if err == nil {
				fmt.Printf("Weights automatically saved at episode %d\n", tm.episodeCount)
				break
			}
			time.Sleep(100 * time.Millisecond) // Breve attesa tra i tentativi
		}
		if err != nil {
			fmt.Printf("Failed to save weights at episode %d after 3 attempts: %v\n", tm.episodeCount, err)
		}
	}

	// Reset statistiche per il prossimo batch
	if tm.episodeCount%50 == 0 {
		tm.totalScore = 0
	}
}

// ResetGame resetta il gioco in modo thread-safe
func (tm *TrainingManager) ResetGame() {
	// Preserva le statistiche esistenti
	existingStats := tm.game.Stats

	// Crea un nuovo gioco con le stesse dimensioni
	width := tm.game.Grid.Width
	height := tm.game.Grid.Height
	tm.game = NewGame(width, height)

	// Ripristina le statistiche
	tm.game.Stats = existingStats

	// Resetta l'agente
	tm.agent.Reset()
}
