package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"runtime/pprof"
	"time"

	rl "github.com/gen2brain/raylib-go/raylib"
)

var shouldBeProfiled = true

func main() {
	if shouldBeProfiled {
		f, err := os.Create("cpu.pprof")
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	speed := flag.Int("speed", 1, "Game speed in milliseconds (lower = faster)")
	flag.Parse()

	rl.InitWindow(900, 600, "Snake Game")
	rl.SetWindowMinSize(600, 600)
	rl.SetWindowState(rl.FlagWindowResizable)
	defer rl.CloseWindow()

	rl.SetTargetFPS(999)

	// Calculate initial grid dimensions
	width := rl.GetScreenWidth() / 20
	height := rl.GetScreenHeight() / 20

	// Numero di agenti per il training parallelo
	const numAgents = 4

	// Crea il training manager con gli agenti
	trainingManager := NewTrainingManager(numAgents, width, height, time.Duration(*speed)*time.Millisecond)
	renderer := NewRenderer()

	// Inizializza il renderer con le statistiche del primo agente
	if game := trainingManager.GetGame(0); game != nil {
		renderer.stats = game.Stats
	}

	// Avvia il training
	trainingManager.StartTraining()
	defer trainingManager.StopTraining()

	var gameStartTime time.Time
	currentAgentIndex := 0 // Indice dell'agente correntemente visualizzato

	// Main game loop
	for !rl.WindowShouldClose() {
		game := trainingManager.GetGame(currentAgentIndex)
		if game == nil {
			continue
		}

		snake := game.GetSnake()

		// Start timing when snake is alive
		if !snake.Dead && gameStartTime.IsZero() {
			gameStartTime = time.Now()
		}

		// Check if snake is dead
		if snake.Dead {
			// Calculate policy entropy for the current agent
			state := game.GetStateInfo()
			agent := trainingManager.agentPool.GetAgent(currentAgentIndex)
			if agent != nil {
				qValues, err := agent.agent.GetQValues(state)
				policyEntropy := 0.0
				if err == nil {
					policy := agent.softmax(qValues, 0.1)
					policyEntropy = agent.calculateEntropy(policy)
				}

				// Save stats before reset
				renderer.stats.AddGame(snake.Score, gameStartTime, time.Now(), policyEntropy)
			}
			gameStartTime = time.Time{} // Reset start time
		}

		// Handle window resize for all agents
		if rl.IsWindowResized() {
			renderer.UpdateDimensions()
			newWidth := rl.GetScreenWidth() / 20
			newHeight := rl.GetScreenHeight() / 20

			// Aggiorna tutti gli agenti
			for i := 0; i < numAgents; i++ {
				if game := trainingManager.GetGame(i); game != nil {
					// Verifica e correggi la posizione del serpente e del cibo
					snake := game.GetSnake()
					snake.Mutex.Lock()
					for j := 0; j < len(snake.Body); j++ {
						if snake.Body[j].X >= newWidth {
							snake.Body[j].X = newWidth - 1
						}
						if snake.Body[j].Y >= newHeight {
							snake.Body[j].Y = newHeight - 1
						}
					}
					snake.Mutex.Unlock()

					// Correggi la posizione del cibo
					if game.food.X >= newWidth {
						game.food.X = newWidth - 1
					}
					if game.food.Y >= newHeight {
						game.food.Y = newHeight - 1
					}

					// Aggiorna le dimensioni della griglia
					game.Grid.Width = newWidth
					game.Grid.Height = newHeight
				}
			}
		}

		// Switch between agents with number keys
		for i := 0; i < numAgents; i++ {
			if rl.IsKeyPressed(rl.KeyOne + int32(i)) {
				currentAgentIndex = i
				if game := trainingManager.GetGame(i); game != nil {
					renderer.stats = game.Stats
				}
			}
		}

		// Update game states
		trainingManager.UpdateGameState()

		// Handle quit
		if rl.IsKeyPressed(rl.KeyQ) || rl.IsKeyPressed(rl.KeyEscape) {
			if err := trainingManager.agentPool.SaveWeights(); err != nil {
				fmt.Printf("Error saving weights: %v\n", err)
			}
			if err := renderer.stats.SaveToFile(); err != nil {
				fmt.Printf("Error saving stats: %v\n", err)
			}
			break
		}

		// Render all agents
		renderer.DrawMultiAgent(trainingManager)
	}
}
